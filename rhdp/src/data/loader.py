"""
src/data/loader.py

Handles loading the Replogle K562 Perturb-seq dataset, isolating control
cells, and running the standard single-cell preprocessing pipeline.

The output of this module is a clean, HVG-subset, log-normalized AnnData
of control cells only — the input to NMF in Component 1.
"""

import gc
import logging
import anndata as ad
import numpy as np
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)


# ── RAM Monitoring ────────────────────────────────────────────────────────────

def log_ram_usage(label: str = '') -> float:
    """
    Log current process RSS memory usage.

    Uses psutil if available; falls back to a no-op if not installed.
    Returns usage in GB (0.0 if psutil unavailable).
    """
    try:
        import psutil, os
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
        tag = f'  [{label}]' if label else ''
        logger.info(f'RAM usage{tag}: {rss_gb:.2f} GB')
        return rss_gb
    except ImportError:
        return 0.0


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
    Memory-efficient preprocessing pipeline for control cells.

    Memory design
    -------------
    The original ordering (norm → log → store log_norm layer → HVG → subset)
    kept three full-gene copies of the control matrix alive simultaneously,
    which causes OOM on datasets with ~600k+ control cells.

    The correct ordering is also the memory-efficient one:

        QC filter → HVG selection (raw counts) → subset to HVGs
        → store counts layer (HVG-size only) → normalize → log1p

    seurat_v3 explicitly requires raw counts for HVG selection, so selecting
    HVGs before normalization is both statistically correct and avoids ever
    storing any full-gene layers.

    Side-effect contract
    --------------------
    This function does NOT copy ctrl_adata internally. It operates on the
    object passed in. The caller must:
        1. Ensure ctrl_adata is already a standalone copy (not an AnnData view).
        2. `del ctrl_adata; gc.collect()` immediately after this call returns.
    Both conditions are satisfied by the runner (01_extract_programs.py).

    Layers in the returned AnnData
    --------------------------------
        'counts'  — raw UMI counts for the HVG subset (stored before
                    normalization; useful for any future count-based steps).
        .X        — log1p-normalized expression for the HVG subset.
                    This is the matrix NMF receives. No separate log_norm
                    layer is stored — .X is already the log-normalized data.

    Args:
        ctrl_adata:   AnnData of control cells (raw counts in .X).
                      Modified in-place during QC + HVG steps.
        n_top_genes:  Number of highly variable genes to retain.
        target_sum:   Library-size normalization target.

    Returns:
        New AnnData — HVG subset, log-normalized, with 'counts' layer.
        The returned object is independent of ctrl_adata (not a view).
    """
    adata = ctrl_adata   # no copy — see docstring

    # ── QC filtering (in-place, shrinks the matrix) ───────────────────────────
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)

    logger.info(
        f"QC filtering: {n_cells_before:,} → {adata.n_obs:,} cells, "
        f"{n_genes_before:,} → {adata.n_vars:,} genes."
    )

    # ── HVG selection on RAW COUNTS ───────────────────────────────────────────
    # seurat_v3 requires count-space data. Running this before normalization
    # is statistically correct AND avoids storing any full-gene normalized copy.
    # span=1.0 uses a global loess fit (more stable on large, uniform datasets).
    #
    # Fallback: if scikit-misc is not installed or the data is too sparse for
    # the loess fit, we fall back to 'cell_ranger'. A temporary normalized copy
    # is used only for HVG detection; adata.X (raw counts) is left untouched.
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
            span=1.0,
        )
        logger.info("HVG selection: used seurat_v3 (count-space).")
    except (ModuleNotFoundError, ValueError) as e:
        logger.warning(
            f"seurat_v3 HVG selection failed ({type(e).__name__}: {e}). "
            f"Falling back to cell_ranger flavor. "
            f"Install scikit-misc for seurat_v3: pip install scikit-misc"
        )
        _tmp = adata.copy()
        sc.pp.normalize_total(_tmp, target_sum=target_sum)
        sc.pp.log1p(_tmp)
        sc.pp.highly_variable_genes(_tmp, n_top_genes=n_top_genes, flavor='cell_ranger')
        adata.var['highly_variable'] = _tmp.var['highly_variable'].values
        del _tmp
        gc.collect()
        logger.info("HVG selection: used cell_ranger fallback (log-normalized space).")

    n_hvg = adata.var['highly_variable'].sum()
    logger.info(f"Selected {n_hvg:,} highly variable genes (target: {n_top_genes:,}).")

    # ── Subset to HVGs immediately ────────────────────────────────────────────
    # Everything from this point forward operates on the HVG-subset matrix.
    # The .copy() here is the ONLY allocation of a full control-cell matrix
    # in this function, and it is HVG-sized (n_cells × n_hvg), not full-gene.
    adata = adata[:, adata.var['highly_variable']].copy()
    gc.collect()

    # ── Store raw counts (HVG subset only, not full-gene) ─────────────────────
    adata.layers['counts'] = adata.X.copy()

    # ── Library-size normalization (in-place) ─────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=target_sum)
    logger.info(f"Library-size normalized to {target_sum:.0f} counts per cell.")

    # ── Log1p transform (in-place) ────────────────────────────────────────────
    sc.pp.log1p(adata)
    logger.info("Applied log1p transformation.")

    # .X is now log-normalized — NMF reads it directly from here.
    # No separate log_norm layer is needed (it would just duplicate .X).

    logger.info(
        f"Preprocessing complete. "
        f"Final matrix: {adata.n_obs:,} cells × {adata.n_vars:,} HVGs. "
        f"Layers: ['counts']. .X = log-normalized."
    )
    return adata