"""
src/programs/nmf.py

Fits NMF to log-normalized control cell expression to extract K gene programs.

NMF factorization:  X  ≈  W · H
    X  shape (n_cells, n_genes)  — input expression matrix
    W  shape (n_cells, K)        — cell-level program activity scores
    H  shape (K, n_genes)        — program-gene loading vectors

Each row of H defines one gene program.
High-loading genes in that row are the "members" of the program.
Each column of W tells you how active each program is in a given cell.

Design notes:
  - init='nndsvda' is deterministic and avoids the random-restart instability
    of random initialization. It uses a double SVD to seed H and W near a
    good local minimum.
  - alpha_H adds mild L2 regularization to H, discouraging programs from
    spreading loading weight uniformly across all genes (which produces
    biologically uninterpretable, diffuse programs).
  - We do NOT center or scale the data before NMF. NMF requires non-negative
    input, and mean-centering would introduce negative values.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


# ── Core NMF Fitting ──────────────────────────────────────────────────────────

def fit_nmf(
    adata: ad.AnnData,
    n_programs: int = 50,
    random_state: int = 42,
    max_iter: int = 1000,
    alpha_H: float = 0.1,
    l1_ratio: float = 0.0,
) -> tuple:
    """
    Fit NMF on log-normalized control cell expression.

    Args:
        adata:        Preprocessed control AnnData (HVG subset, log-normalized).
        n_programs:   K — number of programs to extract.
        random_state: Seed. nndsvda init is deterministic so this mainly
                      affects any tie-breaking in the solver.
        max_iter:     Maximum coordinate descent iterations.
        alpha_H:      L2 regularization weight on H (gene loadings).
                      Higher = more sparse/focused programs.
                      Range to try: 0.0 (no reg) → 1.0 (strong reg).
        l1_ratio:     Mixing parameter for L1/L2 regularization on H.
                      0.0 = pure L2, 1.0 = pure L1.
                      L1 encourages harder sparsity; L2 encourages softer.

    Returns:
        (model, W, H) where:
            model — fitted sklearn NMF object (exposes reconstruction_err_, n_iter_)
            W     — cell activity matrix, shape (n_cells, K), np.ndarray
            H     — program-gene loading matrix, shape (K, n_genes), np.ndarray
    """
    # Extract dense matrix. NMF cannot accept sparse input.
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray().astype(np.float32)
    else:
        X = np.array(adata.X, dtype=np.float32)

    # Guard: NMF requires X >= 0. Log-normalized data should satisfy this,
    # but floating-point issues occasionally produce tiny negative values.
    min_val = X.min()
    if min_val < 0:
        logger.warning(
            f"Expression matrix has min value {min_val:.6f} < 0. "
            f"Clipping to 0 before NMF (this should be a very small correction)."
        )
        X = np.clip(X, 0, None)

    logger.info(
        f"Fitting NMF: K={n_programs}, shape={X.shape}, "
        f"alpha_H={alpha_H}, l1_ratio={l1_ratio}"
    )
    t0 = time.time()

    model = NMF(
        n_components=n_programs,
        init='nndsvda',      # Non-negative double SVD: deterministic, stable
        solver='cd',         # Coordinate descent: memory-efficient, fast
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=0.0,         # No regularization on cell activities W
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        tol=1e-4,
        verbose=0,
    )

    W = model.fit_transform(X)  # (n_cells, K)
    H = model.components_       # (K, n_genes)

    elapsed = time.time() - t0
    logger.info(
        f"NMF done in {elapsed:.1f}s. "
        f"Reconstruction error: {model.reconstruction_err_:.4f}. "
        f"Iterations: {model.n_iter_}/{max_iter}."
    )

    if model.n_iter_ == max_iter:
        logger.warning(
            f"NMF hit max_iter={max_iter} without converging. "
            f"Consider increasing max_iter or loosening tol."
        )

    return model, W, H


# ── Program Summary ───────────────────────────────────────────────────────────

def build_program_dataframe(
    H: np.ndarray,
    gene_names: list,
    n_top: int = 100,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of the top-loading genes for each program.

    This is the primary human-readable output of NMF. For biological
    verification, you inspect the top genes of each program and ask:
    do these genes share a known biological function?

    Args:
        H:          Program-gene loading matrix, shape (K, n_genes).
        gene_names: Ordered list of gene names (columns of H).
        n_top:      Number of top genes to record per program.

    Returns:
        DataFrame with columns: [program_id, rank, gene, loading_score]
    """
    records = []
    for k in range(H.shape[0]):
        scores = H[k]
        top_idx = np.argsort(scores)[::-1][:n_top]
        for rank, idx in enumerate(top_idx):
            records.append({
                'program_id': k,
                'rank': rank + 1,
                'gene': gene_names[idx],
                'loading_score': float(scores[idx]),
            })
    return pd.DataFrame(records)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_nmf_results(
    W: np.ndarray,
    H: np.ndarray,
    gene_names: list,
    cell_barcodes: list,
    output_dir: str,
) -> None:
    """
    Persist NMF outputs to disk in both binary (.npy) and human-readable (.csv) formats.

    Saved files:
        H_matrix.npy     — raw H matrix, fast to reload (K × n_genes)
        W_matrix.npy     — raw W matrix, fast to reload (n_cells × K)
        H_matrix.csv     — H with gene names as columns, program IDs as index
        W_matrix.csv     — W with cell barcodes as index, program IDs as columns
        top_genes.csv    — top 100 genes per program (tidy format)
        gene_names.txt   — ordered gene list (one per line)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = H.shape[0]
    program_ids = [f'Program_{k}' for k in range(K)]

    # Binary — for fast loading in downstream components
    np.save(out / 'H_matrix.npy', H)
    np.save(out / 'W_matrix.npy', W)

    # CSV — human-readable
    H_df = pd.DataFrame(H, index=program_ids, columns=gene_names)
    H_df.to_csv(out / 'H_matrix.csv')

    W_df = pd.DataFrame(W, index=cell_barcodes, columns=program_ids)
    W_df.to_csv(out / 'W_matrix.csv')

    top_genes_df = build_program_dataframe(H, gene_names, n_top=100)
    top_genes_df.to_csv(out / 'top_genes.csv', index=False)

    with open(out / 'gene_names.txt', 'w') as f:
        f.write('\n'.join(gene_names))

    logger.info(f"NMF results saved to '{out}/'")


def load_nmf_results(output_dir: str) -> dict:
    """
    Load previously saved NMF results from disk.

    Returns:
        Dict with keys: H, W, gene_names, top_genes
    """
    out = Path(output_dir)

    H = np.load(out / 'H_matrix.npy')
    W = np.load(out / 'W_matrix.npy')

    with open(out / 'gene_names.txt') as f:
        gene_names = [line.strip() for line in f.readlines()]

    top_genes = pd.read_csv(out / 'top_genes.csv')

    logger.info(f"Loaded NMF results: H={H.shape}, W={W.shape}, {len(gene_names)} genes.")
    return {'H': H, 'W': W, 'gene_names': gene_names, 'top_genes': top_genes}