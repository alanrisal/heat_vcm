"""
src/data/loader.py

Handles loading the Replogle K562 Perturb-seq dataset, isolating control
cells, and running the standard single-cell preprocessing pipeline.

The output of this module is a clean, HVG-subset, log-normalized AnnData
of control cells only — the input to NMF in Component 1.
"""

import logging
import anndata as ad
import numpy as np
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_replogle_k562(data_path: str = None, use_pertpy: bool = True) -> ad.AnnData:
    """
    Load the Replogle K562 Perturb-seq dataset.

    Tries pertpy first (downloads automatically on first run, then caches).
    Falls back to a local .h5ad file if pertpy fails or use_pertpy=False.

    Args:
        data_path:  Path to a local .h5ad file. Required if use_pertpy=False.
        use_pertpy: Attempt to load via pertpy.

    Returns:
        AnnData with raw counts, all perturbations included.
    """
    if use_pertpy:
        try:
            import pertpy as pt
            logger.info("Loading K562 dataset via pertpy (will download on first run)...")
            adata = pt.data.replogle_2022_k562_essential()
            logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes via pertpy.")
            return adata
        except Exception as e:
            logger.warning(f"pertpy loading failed: {e}. Falling back to local file.")

    if data_path is None:
        raise ValueError(
            "Either set use_pertpy=True or provide a data_path to a local .h5ad file.\n"
            "You can download the dataset from: https://plus.figshare.com/articles/dataset/"
            "Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq/"
            "22307680"
        )

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    logger.info(f"Loading data from {path}...")
    adata = sc.read_h5ad(str(path))
    logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes from disk.")
    return adata


# ── Control Cell Extraction ───────────────────────────────────────────────────

# Known labels used for unperturbed/non-targeting control cells
# across common Perturb-seq dataset conventions.
_CONTROL_LABELS = {
    'control', 'non-targeting', 'ctrl', 'neg_ctrl',
    'CONTROL', 'non_targeting', 'NonTargeting', 'non-targeting_ctrl',
}


def extract_control_cells(
    adata: ad.AnnData,
    perturbation_key: str = 'gene',
) -> ad.AnnData:
    """
    Isolate unperturbed control cells from the full dataset.

    Scans common perturbation column names and control label conventions
    so this works across different versions of the Replogle dataset.

    Args:
        adata:            Full AnnData (all perturbations).
        perturbation_key: Column in adata.obs holding perturbation labels.
                          Will search alternatives if not found.

    Returns:
        AnnData of control cells only (copy, not a view).
    """
    # Resolve the perturbation column
    if perturbation_key not in adata.obs.columns:
        alternatives = ['perturbation', 'gene', 'condition', 'perturbation_name', 'guide_ids', 'gene_name']
        for alt in alternatives:
            if alt in adata.obs.columns:
                perturbation_key = alt
                logger.info(f"Using '{perturbation_key}' as perturbation column.")
                break
        else:
            raise KeyError(
                f"Could not find a perturbation column.\n"
                f"Tried: {[perturbation_key] + alternatives}\n"
                f"Available obs columns: {list(adata.obs.columns)}"
            )

    unique_labels = set(adata.obs[perturbation_key].astype(str).unique())
    found = unique_labels.intersection(_CONTROL_LABELS)

    if not found:
        logger.error(
            f"No control labels found in column '{perturbation_key}'.\n"
            f"First 30 unique values: {sorted(unique_labels)[:30]}\n"
            f"Expected one of: {_CONTROL_LABELS}"
        )
        raise ValueError(
            "Could not identify control cells. "
            "Check the perturbation_key or manually inspect adata.obs."
        )

    # If multiple control labels match (edge case), take all of them
    ctrl_label = found.pop()
    logger.info(f"Control label identified: '{ctrl_label}'")

    mask = adata.obs[perturbation_key].astype(str) == ctrl_label
    ctrl_adata = adata[mask].copy()

    logger.info(
        f"Extracted {ctrl_adata.n_obs:,} control cells "
        f"({100 * ctrl_adata.n_obs / adata.n_obs:.1f}% of total dataset)."
    )
    return ctrl_adata


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_control_cells(
    ctrl_adata: ad.AnnData,
    n_top_genes: int = 5000,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """
    Standard single-cell preprocessing pipeline for control cells.

    Pipeline:
        1. Basic QC filtering  — remove very low-quality cells and rarely
                                 detected genes (conservative thresholds).
        2. Library-size norm   — equalize sequencing depth across cells.
        3. Log1p transform     — stabilize variance, compress dynamic range.
        4. HVG selection       — retain the most informative genes for NMF.

    NMF requires non-negative input, which log-normalized data satisfies.

    Args:
        ctrl_adata:   AnnData of control cells (raw counts expected).
        n_top_genes:  Number of highly variable genes to retain for NMF.
        target_sum:   Library-size normalization target (reads per cell).

    Returns:
        Preprocessed AnnData, HVG subset, log-normalized.
        Layers preserved:
            'counts'   — raw counts (before any transformation)
            'log_norm' — log-normalized full-gene expression (before HVG subset)
    """
    adata = ctrl_adata.copy()

    # Preserve raw counts before any transformation
    adata.layers['counts'] = adata.X.copy()

    # ── QC filtering ──────────────────────────────────────────────────────────
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    sc.pp.filter_cells(adata, min_genes=200)   # remove empty/damaged cells
    sc.pp.filter_genes(adata, min_cells=10)    # remove very rare genes

    logger.info(
        f"QC filtering: {n_cells_before:,} → {adata.n_obs:,} cells, "
        f"{n_genes_before:,} → {adata.n_vars:,} genes."
    )

    # ── Library-size normalization ────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=target_sum)
    logger.info(f"Library-size normalized to {target_sum:.0f} counts per cell.")

    # ── Log1p transformation ──────────────────────────────────────────────────
    sc.pp.log1p(adata)
    logger.info("Applied log1p transformation.")

    # Store log-normalized expression over ALL genes before HVG subsetting.
    # This layer is needed later if we want to decode predictions back to
    # full gene space using genes outside the HVG set.
    adata.layers['log_norm'] = adata.X.copy()

    # ── Highly variable gene selection ────────────────────────────────────────
    # seurat_v3 selects HVGs based on mean-variance trend in count space.
    # span=1.0 uses a global loess fit (more stable with fewer cells).
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        span=1.0,
    )
    n_hvg = adata.var['highly_variable'].sum()
    logger.info(f"Selected {n_hvg:,} highly variable genes (target: {n_top_genes:,}).")

    # Subset to HVGs only — this is the matrix NMF will see
    adata = adata[:, adata.var['highly_variable']].copy()

    logger.info(
        f"Preprocessing complete. Final matrix: "
        f"{adata.n_obs:,} cells × {adata.n_vars:,} HVGs."
    )
    return adata