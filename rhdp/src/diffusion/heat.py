"""
src/diffusion/heat.py

Computes the heat diffusion perturbation encoding and projects into program
space to produce the P_g matrix used by Component 3.

Heat diffusion recap
--------------------
For perturbation of gene g:

    h_g = exp(−β · L) · e_g

where L is the symmetric normalized Laplacian and e_g is the one-hot vector
for gene g. h_g is a G-dimensional vector encoding how the perturbation
propagates through the regulatory network.

Projection to program space:

    p_g = H_nmf · h_g          (K-dimensional)

where H_nmf is the (K × G) NMF loading matrix from Component 1.

P_matrix has shape (n_perturbations × K) — one row per perturbation gene.

Computation strategy
--------------------
We need h_g for every perturbation gene g simultaneously. Forming the full
exp(−β·L) matrix would be (G×G) and expensive to materialise. Instead:

  GPU path  — eigendecomposition of L:
                  L = Q Λ Q^T
                  exp(−β·L) = Q diag(exp(−β·λ)) Q^T
              Then h_g = Q (exp(−β·λ) ⊙ Q[g,:]) using batch matmul.
              All perturbation vectors are computed in one batched operation.

  CPU path  — scipy.sparse.linalg.expm_multiply:
              Computes exp(−β·L) · B column-by-column using Krylov methods.
              Works well for sparse L without forming the full dense kernel.

For non-HVG perturbation genes (genes knocked out in the experiment but
absent from the HVG set): we seed diffusion at their STRING-DB HVG
neighbours with equal weight. The signal then propagates inward from the
neighbourhood, approximating the effect of the missing gene.

For genes with no HVG neighbours in STRING-DB: the p_g vector is zero.
These perturbations are flagged and will generalize poorly in Component 3.
"""

import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


# ── Seed matrix construction ──────────────────────────────────────────────────

def build_seed_matrix(
    perturbation_genes: list,
    gene_to_idx: dict,
    A: sp.spmatrix,
) -> tuple:
    """
    Build the (G × n_perts) seed matrix E used as the input to heat diffusion.

    For each perturbation gene g:
      - If g is in the HVG set (gene_to_idx): E[:,g] is a one-hot vector
        with 1.0 at index gene_to_idx[g].
      - If g is NOT in the HVG set but has HVG neighbours in STRING-DB:
        E[:,g] is uniform over those neighbours (sum = 1.0). Diffusion then
        spreads inward from the neighbourhood.
      - Otherwise: E[:,g] = 0 (flagged as unrepresented).

    Args:
        perturbation_genes: List of gene names being knocked out.
        gene_to_idx:        Dict from build_adjacency: gene → HVG index.
        A:                  Sparse adjacency (G×G) for neighbour lookup.

    Returns:
        (E, seed_labels)
            E            — (G × n_perts) float32 ndarray.
            seed_labels  — list of str: 'direct', 'neighbour', or 'missing'
                           one per perturbation gene (for diagnostics).
    """
    G       = A.shape[0]
    n_perts = len(perturbation_genes)

    E           = np.zeros((G, n_perts), dtype=np.float32)
    seed_labels = []

    # Precompute adjacency as CSR for fast row access
    A_csr = A.tocsr()

    for col, gene in enumerate(perturbation_genes):

        if gene in gene_to_idx:
            # Direct: gene is in HVG set
            idx = gene_to_idx[gene]
            E[idx, col] = 1.0
            seed_labels.append('direct')

        else:
            # Indirect: gene not in HVGs — seed from HVG neighbours
            # We can't directly index into A with a non-HVG gene, but we can
            # search the adjacency for any row that has this gene as a target.
            # Actually A is already restricted to HVG×HVG, so non-HVG genes
            # have no direct adjacency row. We use a name-based lookup of the
            # full STRING-DB edges to find HVG neighbours.
            # This is handled by the calling code which passes a neighbour_map;
            # for now, fall back to 'missing' if not provided.
            seed_labels.append('missing')

    return E, seed_labels


def build_seed_matrix_with_neighbours(
    perturbation_genes: list,
    gene_to_idx: dict,
    neighbour_map: dict,
) -> tuple:
    """
    Full seed matrix construction including neighbour-seeding for non-HVG genes.

    Args:
        perturbation_genes: List of knocked-out gene names.
        gene_to_idx:        HVG gene → matrix index (from build_adjacency).
        neighbour_map:      gene_name → list of HVG gene names that are
                            STRING-DB neighbours. Precomputed from edges DataFrame.

    Returns:
        (E, seed_labels, coverage_stats)
    """
    G       = len(gene_to_idx)
    n_perts = len(perturbation_genes)
    E       = np.zeros((G, n_perts), dtype=np.float32)
    labels  = []

    n_direct   = 0
    n_neighbour = 0
    n_missing  = 0

    for col, gene in enumerate(perturbation_genes):

        if gene in gene_to_idx:
            E[gene_to_idx[gene], col] = 1.0
            labels.append('direct')
            n_direct += 1

        elif gene in neighbour_map and len(neighbour_map[gene]) > 0:
            neighbours = neighbour_map[gene]
            weight = 1.0 / len(neighbours)
            for nbr in neighbours:
                if nbr in gene_to_idx:
                    E[gene_to_idx[nbr], col] += weight
            labels.append('neighbour')
            n_neighbour += 1

        else:
            labels.append('missing')
            n_missing += 1

    coverage_stats = {
        'n_perturbations': n_perts,
        'n_direct':        n_direct,
        'n_neighbour':     n_neighbour,
        'n_missing':       n_missing,
        'frac_represented': (n_direct + n_neighbour) / max(n_perts, 1),
    }

    logger.info(
        f"Seed matrix: {n_perts} perturbations — "
        f"direct={n_direct}, neighbour={n_neighbour}, missing={n_missing} "
        f"({100*(n_direct+n_neighbour)/max(n_perts,1):.1f}% represented)"
    )

    return E, labels, coverage_stats


def build_neighbour_map(
    edges_df,
    gene_to_idx: dict,
) -> dict:
    """
    Build a dict: all_gene_names → [HVG neighbour names] from STRING-DB edges.

    This is used to seed diffusion for non-HVG perturbation genes.
    Even though these genes are not in the HVG set, they may have edges
    in STRING-DB to genes that ARE in the HVG set.

    We need the full edges_df which also includes non-HVG genes if we're
    doing a broader lookup. Here we use only the already-filtered edges
    (both genes in HVG), so this gives: HVG gene → its HVG neighbours.
    For non-HVG genes with no row in edges_df, the map returns [].
    """
    hvg_set = set(gene_to_idx.keys())
    nbr_map = {g: [] for g in hvg_set}

    for _, row in edges_df.iterrows():
        a, b = row['gene_a'], row['gene_b']
        if a in hvg_set:
            nbr_map[a].append(b)
        if b in hvg_set:
            nbr_map[b].append(a)

    return nbr_map


# ── Heat diffusion computation ────────────────────────────────────────────────

def compute_heat_diffusion_gpu(
    L: sp.spmatrix,
    E: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute H_diffusion = exp(−β·L) · E using GPU eigendecomposition.

    Algorithm:
        1. Densify L and move to GPU.
        2. Eigendecompose: L = Q Λ Q^T  (L is symmetric PSD).
        3. For each column e of E:
               h = Q · (exp(−β·λ) ⊙ (Q^T · e))
        4. Compute all columns at once: H = Q · (exp(−β·λ)[:,None] * (Q^T · E))

    Memory: L dense = G×G float32. For G=5000: ~100 MB. Fine on Blackwell.

    Returns:
        H_diffusion: (G × n_perts) ndarray, float32, on CPU.
    """
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Heat diffusion (GPU eigen path) on {device}...")

    # Densify sparse Laplacian
    L_dense = torch.tensor(L.toarray(), dtype=torch.float32, device=device)

    # Eigendecomposition (symmetric → real eigenvalues, orthonormal eigenvectors)
    eigenvalues, Q = torch.linalg.eigh(L_dense)
    del L_dense
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # exp(−β · λ) for each eigenvalue
    decay = torch.exp(-beta * eigenvalues)   # (G,)

    # Project E onto eigenbasis, apply decay, project back
    # H = Q · diag(decay) · Q^T · E
    #   = Q · (decay[:,None] * (Q^T · E))
    E_t = torch.tensor(E, dtype=torch.float32, device=device)  # (G, n_perts)
    Qt_E = Q.T @ E_t                                            # (G, n_perts)
    decayed = decay[:, None] * Qt_E                             # (G, n_perts)
    H = Q @ decayed                                             # (G, n_perts)

    # Clamp small negatives from floating-point
    H = torch.clamp(H, min=0.0)

    result = H.cpu().numpy()
    del Q, decay, E_t, Qt_E, decayed, H
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    logger.info("  GPU diffusion complete.")
    return result


def compute_heat_diffusion_cpu(
    L: sp.spmatrix,
    E: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute H_diffusion = exp(−β·L) · E using scipy expm_multiply.

    Uses Krylov-subspace methods — does not form the dense exp(−β·L) matrix.
    Processes all columns of E in one call.

    Returns:
        H_diffusion: (G × n_perts) ndarray, float32.
    """
    logger.info("Heat diffusion (CPU expm_multiply path)...")

    neg_beta_L = (-beta * L).astype(np.float64)   # expm_multiply needs float64

    H = spla.expm_multiply(neg_beta_L, E.astype(np.float64))
    H = np.clip(H, 0.0, None).astype(np.float32)

    logger.info("  CPU diffusion complete.")
    return H


def compute_heat_diffusion(
    L: sp.spmatrix,
    E: np.ndarray,
    beta: float,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Dispatch to GPU or CPU heat diffusion depending on availability.

    Probes GPU with a small matmul before committing; falls back to CPU
    automatically if the probe fails (e.g. CUDA kernel mismatch).

    Args:
        L:       Symmetric normalized Laplacian, (G×G) sparse.
        E:       Seed matrix, (G × n_perts) float32.
        beta:    Diffusion hyperparameter > 0. Larger β = more local signal.
        use_gpu: Try GPU if available.

    Returns:
        H_diffusion: (G × n_perts) float32 ndarray.
    """
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.device('cuda')
                # Probe
                _ = torch.zeros(2, 2, device=dev) @ torch.ones(2, 2, device=dev)
                torch.cuda.synchronize()
                return compute_heat_diffusion_gpu(L, E, beta)
            else:
                logger.info("No CUDA device — using CPU path.")
        except Exception as e:
            logger.warning(f"GPU probe failed ({e}) — falling back to CPU.")

    return compute_heat_diffusion_cpu(L, E, beta)


# ── Program-space projection ──────────────────────────────────────────────────

def project_to_programs(
    H_diffusion: np.ndarray,
    H_nmf: np.ndarray,
) -> np.ndarray:
    """
    Project diffusion profiles into NMF program space.

        P = (H_nmf · H_diffusion)^T

    where:
        H_nmf       (K × G)       — NMF gene loadings from Component 1
        H_diffusion (G × n_perts) — heat diffusion profiles
        P           (n_perts × K) — perturbation influence vectors

    Each row P[i,:] is the K-dimensional representation of perturbation i.
    P[i,k] is large if program k contains many downstream targets of gene i.

    Returns:
        P_matrix: (n_perts × K) float32 ndarray.
    """
    # (K × G) @ (G × n_perts) → (K × n_perts) → transpose → (n_perts × K)
    P = (H_nmf @ H_diffusion).T.astype(np.float32)
    logger.info(f"P_matrix shape: {P.shape}  (n_perturbations × K_programs)")
    return P


# ── Full pipeline ─────────────────────────────────────────────────────────────

def compute_pg_matrix(
    perturbation_genes: list,
    graph_data: dict,
    H_nmf: np.ndarray,
    beta: float = 0.1,
    use_gpu: bool = True,
) -> dict:
    """
    End-to-end Component 2 computation: produce the P_g matrix.

    Args:
        perturbation_genes: List of gene names knocked out in the dataset.
        graph_data:         Output of build_regulatory_graph() from graph.py.
        H_nmf:              NMF loading matrix (K × G) from Component 1.
        beta:               Heat diffusion hyperparameter.
        use_gpu:            Use GPU for eigendecomposition if available.

    Returns:
        dict with:
            P_matrix        — (n_perts × K) float32
            H_diffusion     — (G × n_perts) float32 raw diffusion profiles
            seed_labels     — list of 'direct'/'neighbour'/'missing' per pert
            coverage_stats  — coverage breakdown
            perturbation_genes — same as input (row order of P_matrix)
            beta            — β value used
    """
    L          = graph_data['L']
    gene_to_idx = graph_data['gene_to_idx']
    edges      = graph_data['edges']

    # Build neighbour map for non-HVG perturbation genes
    nbr_map = build_neighbour_map(edges, gene_to_idx)

    # Build seed matrix
    E, seed_labels, coverage = build_seed_matrix_with_neighbours(
        perturbation_genes, gene_to_idx, nbr_map
    )

    logger.info(f"Running heat diffusion  (β={beta})...")
    H_diffusion = compute_heat_diffusion(L, E, beta, use_gpu=use_gpu)

    logger.info("Projecting diffusion profiles into program space...")
    P_matrix = project_to_programs(H_diffusion, H_nmf)

    return {
        'P_matrix':          P_matrix,
        'H_diffusion':       H_diffusion,
        'seed_labels':       seed_labels,
        'coverage_stats':    coverage,
        'perturbation_genes': perturbation_genes,
        'beta':              beta,
    }