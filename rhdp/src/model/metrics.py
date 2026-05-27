"""
src/model/metrics.py

Virtual Cell Challenge evaluation metrics.

    MAE  — Mean Absolute Error on mean gene expression
    DES  — Differential Expression Score (Pearson r of log-fold-changes)
    PDS  — Perturbation Discrimination Score (population-level separability)

All three operate in gene expression space (G-dimensional), not program space,
so predictions must be decoded back to genes before evaluation.

Decoding
--------
Given program delta Δ (K-dim) and control mean program activity w_ctrl (K-dim):

    predicted_mean_expr = (w_ctrl + Δ) @ H_nmf    (G-dim)

For population generation (needed for PDS):

    For each control cell c with activity w_c (K-dim):
        predicted_pert_expr[c] = (w_c + Δ) @ H_nmf    (G-dim)

This produces a cloud of predicted perturbed cells whose variation comes
from the natural heterogeneity of the control population.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_delta_to_expr(
    delta: np.ndarray,
    ctrl_mean_w: np.ndarray,
    H_nmf: np.ndarray,
) -> np.ndarray:
    """
    Decode a program-space delta to predicted mean gene expression.

    predicted_expr = (ctrl_mean_w + delta) @ H_nmf

    Args:
        delta:      (K,) predicted program delta.
        ctrl_mean_w:(K,) mean control program activity from training data.
        H_nmf:      (K, G) NMF gene loading matrix.

    Returns:
        predicted_mean_expr: (G,) predicted mean expression.
    """
    predicted_w    = ctrl_mean_w + delta              # (K,)
    predicted_w    = np.clip(predicted_w, 0, None)    # enforce non-negativity
    predicted_expr = predicted_w @ H_nmf              # (K,) @ (K, G) → (G,)
    return predicted_expr.astype(np.float32)


def generate_population(
    delta: np.ndarray,
    ctrl_W: np.ndarray,
    H_nmf: np.ndarray,
) -> np.ndarray:
    """
    Generate a predicted perturbed cell population by applying delta to
    individual control cells.

    Args:
        delta:   (K,) predicted program delta for one perturbation.
        ctrl_W:  (n_ctrl, K) individual control cell program activities.
        H_nmf:   (K, G) NMF gene loading matrix.

    Returns:
        population: (n_ctrl, G) predicted perturbed expression for each cell.
    """
    perturbed_W   = np.clip(ctrl_W + delta[None, :], 0, None)  # (n_ctrl, K)
    population    = perturbed_W @ H_nmf                          # (n_ctrl, G)
    return population.astype(np.float32)


# ── MAE ───────────────────────────────────────────────────────────────────────

def compute_mae(
    predicted_mean: np.ndarray,
    observed_mean:  np.ndarray,
) -> float:
    """
    Mean Absolute Error between predicted and observed mean expression.

    Both arrays should be (G,) — one value per gene.
    """
    return float(np.mean(np.abs(predicted_mean - observed_mean)))


# ── DES ───────────────────────────────────────────────────────────────────────

def compute_des(
    predicted_mean: np.ndarray,
    observed_mean:  np.ndarray,
    ctrl_mean:      np.ndarray,
    eps:            float = 1e-6,
) -> float:
    """
    Differential Expression Score — Pearson correlation of predicted vs
    observed log-fold-changes relative to control.

    LFC = log2((expr + eps) / (ctrl + eps))

    A high DES means the model correctly identifies which genes go up and
    down, and by how much — the key signal for biological interpretability.

    Returns:
        Pearson r ∈ [−1, 1]. Higher is better.
    """
    predicted_lfc = np.log2((predicted_mean + eps) / (ctrl_mean + eps))
    observed_lfc  = np.log2((observed_mean  + eps) / (ctrl_mean + eps))

    # Remove genes with near-zero variance in both (uninformative)
    mask = (np.abs(observed_lfc) > 0.01) | (np.abs(predicted_lfc) > 0.01)
    if mask.sum() < 10:
        return float('nan')

    r, _ = pearsonr(predicted_lfc[mask], observed_lfc[mask])
    return float(r)


# ── PDS ───────────────────────────────────────────────────────────────────────

def compute_pds_pair(
    pop_a: np.ndarray,
    pop_b: np.ndarray,
    n_features: int = 50,
) -> float:
    """
    Perturbation Discrimination Score for a single pair of perturbations.

    Trains a logistic regression classifier to distinguish pop_a from pop_b,
    returns AUROC. An AUROC near 1.0 means the two populations are clearly
    distinct; near 0.5 means they are indistinguishable.

    Uses PCA-reduced representation for speed (n_features components).

    Args:
        pop_a: (n_cells_a, G) predicted population for perturbation A.
        pop_b: (n_cells_b, G) predicted population for perturbation B.

    Returns:
        AUROC ∈ [0, 1].
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X      = np.vstack([pop_a, pop_b])
    y      = np.array([0] * len(pop_a) + [1] * len(pop_b))

    # Reduce to n_features dimensions for speed
    n_comp = min(n_features, X.shape[1], X.shape[0] - 1)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=n_comp, random_state=42)
    X_pca  = pca.fit_transform(X_sc)

    clf  = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    clf.fit(X_pca, y)
    proba = clf.predict_proba(X_pca)[:, 1]

    return float(roc_auc_score(y, proba))


def compute_pds_batch(
    predicted_pops: list,
    observed_pops:  list,
    gene_names:     list,
    n_pairs:        int = 200,
    seed:           int = 42,
) -> dict:
    """
    Compute PDS over a random sample of perturbation pairs.

    For each sampled pair (A, B):
        pds_pred = AUROC(classify predicted_A vs predicted_B)
        pds_obs  = AUROC(classify observed_A vs observed_B)

    A good model has pds_pred ≈ pds_obs.
    The final PDS score is Pearson r(pds_pred, pds_obs) over all pairs.

    Args:
        predicted_pops: list of (n_cells, G) arrays, one per gene.
        observed_pops:  list of (n_cells, G) arrays, one per gene.
        gene_names:     gene name for each population.
        n_pairs:        number of random pairs to evaluate.

    Returns:
        dict with keys: pearson_r, mean_pred_auroc, mean_obs_auroc, pair_results
    """
    rng   = np.random.default_rng(seed)
    n     = len(gene_names)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    # Sample random pairs
    all_pairs  = [(i, j) for i in range(n) for j in range(i + 1, n)]
    chosen_idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
    pairs      = [all_pairs[i] for i in chosen_idx]

    pred_aurocs = []
    obs_aurocs  = []
    pair_records = []

    for i, j in pairs:
        pa = compute_pds_pair(predicted_pops[i], predicted_pops[j])
        oa = compute_pds_pair(observed_pops[i],  observed_pops[j])
        pred_aurocs.append(pa)
        obs_aurocs.append(oa)
        pair_records.append({
            'gene_a': gene_names[i],
            'gene_b': gene_names[j],
            'pred_auroc': pa,
            'obs_auroc':  oa,
        })

    pearson_r = float(pearsonr(pred_aurocs, obs_aurocs)[0])

    return {
        'pearson_r':       pearson_r,
        'mean_pred_auroc': float(np.mean(pred_aurocs)),
        'mean_obs_auroc':  float(np.mean(obs_aurocs)),
        'pair_results':    pd.DataFrame(pair_records),
    }


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_test_set(
    model,
    test_dataset,
    pairs_data:      dict,
    H_nmf:           np.ndarray,
    ctrl_W:          np.ndarray,
    adata,
    device:          torch.device,
    n_pds_pairs:     int = 100,
    compute_pds_flag: bool = True,
) -> dict:
    """
    Run all three metrics on the held-out test genes.

    Args:
        model:        Trained ProgramDeltaNetwork (eval mode).
        test_dataset: PerturbationDataset for test genes.
        pairs_data:   Full training pairs dict (for gene lookup).
        H_nmf:        (K, G) NMF loadings.
        ctrl_W:       (n_ctrl, K) control cell program activities.
        adata:        Full AnnData (for observed expression lookup).
        device:       Torch device.
        n_pds_pairs:  Number of pairs for PDS computation.

    Returns:
        dict with per-gene and aggregated MAE, DES, PDS results.
    """
    import torch
    model.eval()

    ctrl_mean_w    = pairs_data['ctrl_mean_w']
    gene_to_idx    = {g: i for i, g in enumerate(pairs_data['gene_names'])}

    mae_scores  = []
    des_scores  = []
    gene_results = []

    predicted_pops = []
    observed_pops  = []
    test_gene_names = []

    # Control mean expression for LFC computation
    ctrl_mask = adata.obs['perturbation'].isin(
        {'control', 'non-targeting', 'ctrl', 'neg_ctrl',
         'CONTROL', 'non_targeting', 'NonTargeting'}
    )
    X_ctrl    = (
        adata[ctrl_mask].X.toarray()
        if hasattr(adata[ctrl_mask].X, 'toarray')
        else np.array(adata[ctrl_mask].X)
    ).astype(np.float32)
    ctrl_mean_expr = X_ctrl.mean(axis=0)     # (G,)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            p_g, _ = test_dataset[idx]
            gene   = test_dataset.gene_names[idx]

            # Predict delta
            pred_delta = model(p_g.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

            # Decode to gene space
            pred_mean  = decode_delta_to_expr(pred_delta, ctrl_mean_w, H_nmf)

            # Observed mean expression for this gene
            cell_mask  = adata.obs['perturbation'] == gene
            X_pert     = (
                adata[cell_mask].X.toarray()
                if hasattr(adata[cell_mask].X, 'toarray')
                else np.array(adata[cell_mask].X)
            ).astype(np.float32)
            obs_mean   = X_pert.mean(axis=0)    # (G,)

            # MAE
            mae = compute_mae(pred_mean, obs_mean)

            # DES
            des = compute_des(pred_mean, obs_mean, ctrl_mean_expr)

            mae_scores.append(mae)
            des_scores.append(des)
            gene_results.append({
                'gene': gene, 'mae': mae, 'des': des,
                'n_cells': int(cell_mask.sum()),
            })

            # Population for PDS
            pop = generate_population(pred_delta, ctrl_W, H_nmf)
            predicted_pops.append(pop)
            observed_pops.append(X_pert)
            test_gene_names.append(gene)

    results = {
        'mean_mae': float(np.nanmean(mae_scores)),
        'mean_des': float(np.nanmean(des_scores)),
        'per_gene': pd.DataFrame(gene_results),
    }

    # PDS (slower — skip if only a few test genes)
    if compute_pds_flag and len(test_gene_names) >= 4:
        logger.info(f"Computing PDS over {n_pds_pairs} random pairs...")
        pds_result = compute_pds_batch(
            predicted_pops, observed_pops, test_gene_names, n_pds_pairs
        )
        results['pds'] = pds_result['pearson_r']
        results['pds_details'] = pds_result
    else:
        results['pds'] = float('nan')

    logger.info(
        f"Test metrics — "
        f"MAE={results['mean_mae']:.4f}  "
        f"DES={results['mean_des']:.4f}  "
        f"PDS={results['pds']:.4f}"
    )
    return results