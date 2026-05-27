"""
src/model/dataset.py

Builds training data for Component 3: (p_g, Δ_g) pairs where
    p_g  — K-dim perturbation influence vector from Component 2
    Δ_g  — K-dim program-activity delta (perturbed mean − control mean)

Data flow
---------
1. Load the GWPS-processed Replogle K562 h5ad
   (clean gene-level perturbation labels, no guide coordinates)
2. Subset expression to the same 5,000 HVGs used in Component 1
3. Apply the same log-normalization pipeline
4. For each perturbation gene g:
      - Collect perturbed cells → mean expression μ_g (G-dim)
      - NNLS-project μ_g to program space → w_g (K-dim)
      - delta_g = w_g − ctrl_mean_w          (K-dim)
5. Load P_matrix from Component 2 → p_g for each gene
6. Match: (p_g, delta_g) for every gene present in both P_matrix and the dataset

Why project the mean rather than individual cells?
  Training only needs mean deltas (target = mean shift in program space).
  NNLS on a single mean vector is O(K·G) — fast.
  Individual cell projection is needed only at inference for population generation.

Gene splits
-----------
Split is performed at the GENE level, not the cell level.
  train / val / test = 70% / 15% / 15% by default.
No gene should appear in more than one split.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import Dataset
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_CONTROL_LABELS = {
    'control', 'non-targeting', 'ctrl', 'neg_ctrl',
    'CONTROL', 'non_targeting', 'NonTargeting',
    'non-targeting_ctrl',
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_and_preprocess(
    data_path: str,
    hvg_gene_names: list,
    perturbation_col: str = 'perturbation',
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Load the GWPS-processed Replogle K562 h5ad and apply the same
    normalization pipeline used in Component 1.

    Args:
        data_path:       Path to the .h5ad file with clean gene labels.
        hvg_gene_names:  Ordered list of HVG gene names from Component 1.
                         Used to subset the expression matrix.
        perturbation_col: obs column containing perturbation gene labels.
        target_sum:      Library-size normalization target (match Component 1).

    Returns:
        AnnData subset to HVGs, log-normalized.
        obs column 'perturbation' is guaranteed to be present.
    """
    logger.info(f"Loading dataset from: {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes.")

    # Convert Ensembl IDs to HGNC symbols if needed (auto-detected)
    from src.data.loader import map_ensembl_to_symbols
    adata = map_ensembl_to_symbols(adata)

    # ── Exclude housekeeping genes (must match Component 1 filtering) ─────────
    excl_prefixes = ('MT-', 'mt-', 'RPS', 'RPL', 'Rps', 'Rpl')
    excl_exact    = {'MALAT1', 'NEAT1', 'XIST', 'TSIX'}
    keep_mask = np.array([
        not (g.startswith(excl_prefixes) or g in excl_exact)
        for g in adata.var_names
    ])
    n_excl = (~keep_mask).sum()
    if n_excl > 0:
        import gc
        adata = adata[:, keep_mask].copy()
        gc.collect()
        logger.info(f"  Excluded {n_excl:,} housekeeping genes (MT-*, RPS*, RPL*, etc.).")

    # Resolve perturbation column
    if perturbation_col not in adata.obs.columns:
        candidates = ['perturbation', 'gene', 'target', 'gene_name',
                      'target_gene', 'perturbation_name']
        for c in candidates:
            if c in adata.obs.columns:
                n_uniq = adata.obs[c].nunique()
                if 2 <= n_uniq <= 5000:
                    perturbation_col = c
                    logger.info(f"  Using obs column '{c}' for perturbation labels.")
                    break
        else:
            raise ValueError(
                f"Could not find a perturbation column.\n"
                f"Available obs columns: {list(adata.obs.columns)}\n"
                f"Pass --perturbation_col COLUMN_NAME"
            )

    adata.obs['perturbation'] = adata.obs[perturbation_col].astype(str)

    # ── Subset to HVGs ────────────────────────────────────────────────────────
    # Only keep genes that appear in the HVG list; maintain the same order.
    available_hvgs = [g for g in hvg_gene_names if g in adata.var_names]
    n_missing = len(hvg_gene_names) - len(available_hvgs)
    if n_missing > 0:
        logger.warning(
            f"  {n_missing}/{len(hvg_gene_names)} HVG genes absent from dataset. "
            f"These will be treated as zero-expression."
        )

    adata = adata[:, available_hvgs].copy()
    logger.info(f"  After HVG subset: {adata.n_obs:,} × {adata.n_vars:,}")

    # ── Normalization (match Component 1 exactly) ─────────────────────────────
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # If some HVGs were missing, reindex to the full HVG list (fill with 0)
    if n_missing > 0:
        adata = _reindex_to_full_hvg(adata, hvg_gene_names)

    logger.info(f"  Normalization complete. Final: {adata.n_obs:,} × {adata.n_vars:,}")
    return adata


def _reindex_to_full_hvg(adata: ad.AnnData, full_gene_names: list) -> ad.AnnData:
    """Fill missing HVGs with zeros so the matrix aligns with H_nmf exactly."""
    import scipy.sparse as sp
    G_full = len(full_gene_names)
    present = set(adata.var_names)
    gene_to_col = {g: i for i, g in enumerate(full_gene_names)}

    if sp.issparse(adata.X):
        X_old = adata.X.toarray()
    else:
        X_old = np.array(adata.X)

    X_new = np.zeros((adata.n_obs, G_full), dtype=np.float32)
    for j, gene in enumerate(adata.var_names):
        if gene in gene_to_col:
            X_new[:, gene_to_col[gene]] = X_old[:, j]

    new_var = pd.DataFrame(index=full_gene_names)
    return ad.AnnData(X=X_new, obs=adata.obs.copy(), var=new_var)


# ── Program-space projection ──────────────────────────────────────────────────

def project_mean_to_programs(
    mean_expr: np.ndarray,
    H_nmf: np.ndarray,
) -> np.ndarray:
    """
    Project a single mean expression vector to program activity space via NNLS.

    Solves: w = argmin ||mean_expr - w · H_nmf||²  s.t. w >= 0

    Args:
        mean_expr: (G,) mean expression vector.
        H_nmf:     (K, G) NMF gene loading matrix from Component 1.

    Returns:
        w: (K,) program activity vector.
    """
    # NNLS: A·x = b  →  H_nmf^T · w = mean_expr^T
    w, _ = nnls(H_nmf.T, mean_expr.astype(np.float64))
    return w.astype(np.float32)


def compute_program_activities_batch(
    X: np.ndarray,
    H_nmf: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Project a batch of cells to program activities.

    Used for individual control cells (needed at inference for population
    generation). For training, prefer project_mean_to_programs.

    Args:
        X:     (n_cells, G) log-normalized expression matrix.
        H_nmf: (K, G) NMF loadings.

    Returns:
        W: (n_cells, K) program activity matrix.
    """
    n = X.shape[0]
    K = H_nmf.shape[0]
    W = np.zeros((n, K), dtype=np.float32)
    H_T = H_nmf.T.astype(np.float64)   # G × K

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            w, _ = nnls(H_T, X[i].astype(np.float64))
            W[i] = w.astype(np.float32)

        if (start // batch_size) % 5 == 0:
            logger.info(f"  Projecting cells: {end}/{n}")

    return W


# ── Training pair construction ────────────────────────────────────────────────

def build_training_pairs(
    adata: ad.AnnData,
    H_nmf: np.ndarray,
    P_matrix: np.ndarray,
    perturbation_genes_in_P: list,
    min_cells_per_pert: int = 5,
) -> dict:
    """
    Build (p_g, delta_g) training pairs for all perturbation genes that have
    both a P_matrix row and observed expression data.

    Args:
        adata:                  Preprocessed AnnData (all cells, HVG subset).
        H_nmf:                  (K × G) from Component 1.
        P_matrix:               (n_hvg × K) from Component 2.
        perturbation_genes_in_P: Ordered list of gene names (rows of P_matrix).
        min_cells_per_pert:     Minimum cells required to include a perturbation.

    Returns:
        dict with keys:
            gene_names  — list of gene names (length N_matched)
            p_vectors   — (N_matched, K) float32 — p_g from Component 2
            deltas      — (N_matched, K) float32 — Δ_g = w_pert − w_ctrl
            ctrl_mean_w — (K,)  float32 — mean control program activity
            n_cells     — list of int — cells per perturbation
    """
    # ── Control mean ──────────────────────────────────────────────────────────
    ctrl_mask = adata.obs['perturbation'].isin(_CONTROL_LABELS)
    n_ctrl    = ctrl_mask.sum()
    if n_ctrl == 0:
        raise ValueError(
            "No control cells found. Check that _CONTROL_LABELS matches "
            f"the labels in your dataset. "
            f"Sample labels: {adata.obs['perturbation'].unique()[:10].tolist()}"
        )

    logger.info(f"Control cells: {n_ctrl:,}")
    X_ctrl = (
        adata[ctrl_mask].X.toarray()
        if hasattr(adata[ctrl_mask].X, 'toarray')
        else np.array(adata[ctrl_mask].X)
    ).astype(np.float32)
    ctrl_mean_expr = X_ctrl.mean(axis=0)            # (G,)
    ctrl_mean_w    = project_mean_to_programs(ctrl_mean_expr, H_nmf)  # (K,)
    logger.info(f"Control mean program activity computed: shape {ctrl_mean_w.shape}")

    # ── Build P_matrix lookup ─────────────────────────────────────────────────
    p_gene_to_row = {g: i for i, g in enumerate(perturbation_genes_in_P)}

    # ── Per-perturbation deltas ───────────────────────────────────────────────
    pert_labels = adata.obs['perturbation'].unique()
    pert_labels = [l for l in pert_labels if l not in _CONTROL_LABELS]
    logger.info(f"Perturbation labels in dataset: {len(pert_labels):,}")

    gene_names = []
    p_vectors  = []
    deltas     = []
    n_cells_list = []

    skipped_no_p    = 0
    skipped_few_cells = 0

    for gene in sorted(pert_labels):
        # Must have a p_g vector
        if gene not in p_gene_to_row:
            skipped_no_p += 1
            continue

        # Must have enough cells
        cell_mask = adata.obs['perturbation'] == gene
        n_cells   = cell_mask.sum()
        if n_cells < min_cells_per_pert:
            skipped_few_cells += 1
            continue

        # Mean expression of perturbed cells → program space
        X_pert = (
            adata[cell_mask].X.toarray()
            if hasattr(adata[cell_mask].X, 'toarray')
            else np.array(adata[cell_mask].X)
        ).astype(np.float32)
        pert_mean_expr = X_pert.mean(axis=0)
        pert_mean_w    = project_mean_to_programs(pert_mean_expr, H_nmf)

        delta = pert_mean_w - ctrl_mean_w        # (K,)

        gene_names.append(gene)
        p_vectors.append(P_matrix[p_gene_to_row[gene]])
        deltas.append(delta)
        n_cells_list.append(n_cells)

    logger.info(
        f"Training pairs built: {len(gene_names):,} genes matched.\n"
        f"  Skipped (no p_g in P_matrix): {skipped_no_p:,}\n"
        f"  Skipped (< {min_cells_per_pert} cells): {skipped_few_cells:,}"
    )

    if not gene_names:
        raise ValueError(
            "No training pairs could be constructed.\n"
            "Check that the gene names in P_matrix (Component 2) match "
            "the perturbation labels in the dataset.\n"
            f"Sample dataset labels: {pert_labels[:10]}\n"
            f"Sample P_matrix genes: {perturbation_genes_in_P[:10]}"
        )

    return {
        'gene_names':  gene_names,
        'p_vectors':   np.stack(p_vectors, axis=0).astype(np.float32),
        'deltas':      np.stack(deltas, axis=0).astype(np.float32),
        'ctrl_mean_w': ctrl_mean_w,
        'n_cells':     n_cells_list,
    }


# ── Train/val/test split ──────────────────────────────────────────────────────

def make_gene_splits(
    gene_names: list,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> dict:
    """
    Split genes into train / val / test sets.

    Split is at the gene level — no gene appears in more than one split.
    This tests the model's ability to generalise to held-out perturbations.

    Returns:
        dict with 'train', 'val', 'test' → lists of gene names
    """
    n = len(gene_names)
    test_frac = 1.0 - train_frac - val_frac

    train_genes, temp_genes = train_test_split(
        gene_names, train_size=train_frac, random_state=seed
    )
    val_size_of_temp = val_frac / (val_frac + test_frac)
    val_genes, test_genes = train_test_split(
        temp_genes, train_size=val_size_of_temp, random_state=seed
    )

    logger.info(
        f"Gene split: train={len(train_genes)}, "
        f"val={len(val_genes)}, test={len(test_genes)}"
    )
    return {'train': train_genes, 'val': val_genes, 'test': test_genes}


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PerturbationDataset(Dataset):
    """
    PyTorch Dataset of (p_g, delta_g) pairs for a given gene split.

    Args:
        pairs_data: Output of build_training_pairs().
        gene_subset: List of gene names to include (e.g. train split).
    """

    def __init__(self, pairs_data: dict, gene_subset: list):
        gene_to_idx = {g: i for i, g in enumerate(pairs_data['gene_names'])}
        indices = [gene_to_idx[g] for g in gene_subset if g in gene_to_idx]

        if not indices:
            raise ValueError(
                f"None of the {len(gene_subset)} requested genes found in pairs_data."
            )

        self.gene_names = [pairs_data['gene_names'][i] for i in indices]
        self.p_vectors  = torch.tensor(
            pairs_data['p_vectors'][indices], dtype=torch.float32
        )
        self.deltas = torch.tensor(
            pairs_data['deltas'][indices], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.gene_names)

    def __getitem__(self, idx: int) -> tuple:
        return self.p_vectors[idx], self.deltas[idx]