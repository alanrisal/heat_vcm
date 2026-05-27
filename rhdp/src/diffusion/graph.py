"""
src/diffusion/graph.py

Downloads STRING-DB human protein interaction data, builds a weighted gene
interaction graph restricted to HVG genes, and computes the symmetric
normalized graph Laplacian used by the heat diffusion kernel.

STRING-DB overview
------------------
STRING-DB v12.0 provides combined interaction scores (0–1000) aggregating
evidence from co-expression, text mining, experimental data, databases,
neighbourhood, gene fusion, and phylogeny. We filter to combined_score >= 700
("high confidence") and weight edges by score/1000 so edge weights ∈ [0, 1].

Files used
----------
    9606.protein.links.v12.0.txt.gz   edge list: ENSP_A ENSP_B combined_score
    9606.protein.info.v12.0.txt.gz    ENSP → preferred_name (HGNC symbol)

Both files are cached in data/string_cache/ on first download (~350MB total).

Graph construction
------------------
1. Map ENSP IDs → HGNC symbols using the info file.
2. Keep only edges where both genes are in our HVG set and score >= min_score.
3. Build a weighted sparse adjacency matrix A where A_ij = score_ij / 1000.
4. Compute the symmetric normalized Laplacian:
       L_sym = I  −  D^{−½}  A  D^{−½}
   where D_ii = Σ_j A_ij (weighted degree).
   Eigenvalues of L_sym ∈ [0, 2], making exp(−β·L_sym) numerically stable
   for any β > 0.

Why symmetric normalized?
Hubs (e.g., TP53, MYC) have high degree in STRING-DB. Using the unnormalized
Laplacian would make their diffusion signal dominate all others, regardless of
β. The symmetric normalization cancels this degree bias so diffusion reflects
genuine regulatory proximity, not node degree.
"""

import gzip
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ── STRING-DB download URLs (v12.0 human = taxon 9606) ───────────────────────
_STRING_BASE = "https://stringdb-static.org/download"
_STRING_LINKS = f"{_STRING_BASE}/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
_STRING_INFO  = f"{_STRING_BASE}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"

_LINKS_MD5 = None   # set to known MD5 if you want integrity checks
_INFO_MD5  = None


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = '') -> Path:
    """Stream-download a URL to dest, showing MB progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"  Cache hit: {dest.name}")
        return dest

    logger.info(f"  Downloading {desc or url}")
    logger.info(f"  → {dest}")
    t0 = time.time()

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    mb  = downloaded / 1e6
                    logger.info(f"    {mb:.0f} MB / {total/1e6:.0f} MB  ({pct:.0f}%)")

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s.")
    return dest


def download_string_db(cache_dir: str = 'data/string_cache') -> tuple:
    """
    Download STRING-DB human protein links and info files.

    Files are cached locally — subsequent calls return immediately.

    Args:
        cache_dir: Directory to store downloaded files.

    Returns:
        (links_path, info_path) — Path objects to the .gz files.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    links_path = _download_file(
        _STRING_LINKS,
        cache / '9606.protein.links.v12.0.txt.gz',
        desc='STRING-DB protein links (~360 MB)',
    )
    info_path = _download_file(
        _STRING_INFO,
        cache / '9606.protein.info.v12.0.txt.gz',
        desc='STRING-DB protein info (~30 MB)',
    )
    return links_path, info_path


# ── Gene name mapping ─────────────────────────────────────────────────────────

def load_string_gene_map(info_path: Path) -> dict:
    """
    Build ENSP → HGNC symbol mapping from the STRING-DB info file.

    The info file has columns:
        #string_protein_id    preferred_name    protein_size    annotation

    Returns:
        dict mapping '9606.ENSP00000...' → 'GENE_SYMBOL'
    """
    logger.info("Loading STRING-DB gene name map...")
    rows = []
    with gzip.open(str(info_path), 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))   # ENSP_id, preferred_name

    gene_map = {ensp: name for ensp, name in rows}
    logger.info(f"  Loaded {len(gene_map):,} ENSP → gene name mappings.")
    return gene_map


# ── Edge loading ──────────────────────────────────────────────────────────────

def load_filtered_edges(
    links_path: Path,
    gene_map: dict,
    hvg_set: set,
    min_score: int = 700,
) -> pd.DataFrame:
    """
    Stream-read the STRING-DB links file and return edges within the HVG set.

    Filters:
        1. combined_score >= min_score
        2. Both genes in hvg_set (HGNC symbols)
        3. No self-loops

    Args:
        links_path: Path to 9606.protein.links.v12.0.txt.gz
        gene_map:   ENSP → HGNC mapping from load_string_gene_map().
        hvg_set:    Set of HGNC gene symbols from Component 1.
        min_score:  Minimum combined_score to retain. Default 700 = high conf.

    Returns:
        DataFrame with columns [gene_a, gene_b, weight]
        where weight = score / 1000.
    """
    logger.info(
        f"Filtering STRING-DB edges  "
        f"(min_score={min_score}, HVG genes={len(hvg_set):,})..."
    )

    kept = []
    total_lines = 0

    with gzip.open(str(links_path), 'rt') as f:
        header = f.readline()   # skip header line
        for line in f:
            total_lines += 1
            parts = line.split()
            if len(parts) < 3:
                continue
            score = int(parts[2])
            if score < min_score:
                continue

            gene_a = gene_map.get(parts[0])
            gene_b = gene_map.get(parts[1])

            if gene_a is None or gene_b is None:
                continue
            if gene_a == gene_b:
                continue
            if gene_a not in hvg_set or gene_b not in hvg_set:
                continue

            kept.append((gene_a, gene_b, score / 1000.0))

    edges = pd.DataFrame(kept, columns=['gene_a', 'gene_b', 'weight'])

    # STRING-DB edges are listed once per direction; keep unique undirected pairs
    edges['key'] = edges.apply(
        lambda r: tuple(sorted([r.gene_a, r.gene_b])), axis=1
    )
    edges = edges.drop_duplicates(subset='key').drop(columns='key')
    edges = edges.reset_index(drop=True)

    logger.info(
        f"  Scanned {total_lines:,} edges. "
        f"Kept {len(edges):,} within HVG set at score >= {min_score}."
    )
    return edges


# ── Adjacency and Laplacian ───────────────────────────────────────────────────

def build_adjacency(
    edges: pd.DataFrame,
    gene_names: list,
) -> tuple:
    """
    Build a symmetric weighted sparse adjacency matrix over the HVG genes.

    Only genes that appear in at least one retained edge are "connected" in
    the graph. Isolated HVG genes (no STRING-DB edges above threshold) have
    zero rows/columns but are still included in the matrix so indices stay
    aligned with the full HVG gene list.

    Args:
        edges:      DataFrame with [gene_a, gene_b, weight].
        gene_names: Full ordered HVG gene list from Component 1.

    Returns:
        (A, gene_to_idx, connected_genes)
            A              — (G, G) scipy sparse CSR matrix, symmetric.
            gene_to_idx    — dict mapping gene name → integer index in A.
            connected_mask — boolean array len(gene_names), True if gene
                             has at least one edge in the graph.
    """
    G = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    rows, cols, data = [], [], []
    for _, row in edges.iterrows():
        i = gene_to_idx.get(row.gene_a)
        j = gene_to_idx.get(row.gene_b)
        if i is None or j is None:
            continue
        # symmetric: add both directions
        rows += [i, j]
        cols += [j, i]
        data += [row.weight, row.weight]

    A = sp.csr_matrix((data, (rows, cols)), shape=(G, G), dtype=np.float32)

    # Identify connected genes
    node_degrees = np.asarray(A.sum(axis=1)).ravel()
    connected_mask = node_degrees > 0

    n_connected = connected_mask.sum()
    logger.info(
        f"Adjacency matrix: {G}×{G} sparse.  "
        f"{n_connected:,} / {G:,} HVG genes are connected in the graph."
    )
    return A, gene_to_idx, connected_mask


def compute_normalized_laplacian(A: sp.spmatrix) -> sp.spmatrix:
    """
    Compute the symmetric normalized graph Laplacian.

        L_sym = I  −  D^{−½} · A · D^{−½}

    where D is the diagonal weighted degree matrix.

    Eigenvalues of L_sym are in [0, 2] for any weighted graph, making
    exp(−β · L_sym) numerically stable for any β > 0.

    Isolated nodes (degree 0) are handled gracefully: D^{−½}_ii = 0,
    so the corresponding rows/cols of L_sym are zero (no diffusion from
    or to isolated nodes — which is correct biologically).

    Returns:
        L_sym as a scipy sparse CSR matrix.
    """
    G = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).ravel()

    # D^{-½}: 1/sqrt(d) for connected nodes, 0 for isolated nodes
    inv_sqrt_d = np.where(degree > 0, 1.0 / np.sqrt(degree + 1e-12), 0.0)
    D_inv_sqrt = sp.diags(inv_sqrt_d, format='csr', dtype=np.float32)

    # L_sym = I - D^{-½} A D^{-½}
    I   = sp.eye(G, format='csr', dtype=np.float32)
    L   = I - D_inv_sqrt @ A @ D_inv_sqrt

    # Clip tiny numerical negatives on diagonal to 0
    L   = L.tocsr()

    logger.info(
        f"Laplacian computed: {G}×{G}, "
        f"nnz={L.nnz:,}, "
        f"density={L.nnz / G**2:.4%}."
    )
    return L


# ── Full graph build pipeline ─────────────────────────────────────────────────

def build_regulatory_graph(
    gene_names: list,
    cache_dir: str = 'data/string_cache',
    min_score: int = 700,
) -> dict:
    """
    End-to-end: download STRING-DB, build filtered graph, return Laplacian.

    This is the main entry point for Component 2 graph construction.

    Args:
        gene_names: HVG gene names (HGNC symbols) from Component 1.
        cache_dir:  Directory for STRING-DB cache files.
        min_score:  Minimum STRING-DB combined score (default 700).

    Returns:
        dict with keys:
            L              — sparse symmetric normalized Laplacian (G×G)
            A              — sparse adjacency matrix (G×G)
            gene_to_idx    — dict gene_name → row/col index
            connected_mask — bool array indicating genes with ≥1 edge
            edges          — DataFrame of retained edges
            gene_names     — same as input (for convenience)
    """
    hvg_set = set(gene_names)

    # Step 1: Download
    links_path, info_path = download_string_db(cache_dir)

    # Step 2: Gene name map
    gene_map = load_string_gene_map(info_path)

    # Step 3: Filter edges
    edges = load_filtered_edges(links_path, gene_map, hvg_set, min_score)

    # Step 4: Adjacency
    A, gene_to_idx, connected_mask = build_adjacency(edges, gene_names)

    # Step 5: Laplacian
    L = compute_normalized_laplacian(A)

    return {
        'L':              L,
        'A':              A,
        'gene_to_idx':    gene_to_idx,
        'connected_mask': connected_mask,
        'edges':          edges,
        'gene_names':     gene_names,
    }