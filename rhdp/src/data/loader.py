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
import pandas as pd
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Ensembl → HGNC Symbol Conversion ─────────────────────────────────────────

# Candidate column names where the h5ad var dataframe might store HGNC symbols
_SYMBOL_COLUMN_CANDIDATES = [
    'gene_name', 'gene_names', 'gene_symbols', 'symbol',
    'hgnc_symbol', 'name', 'Symbol', 'Gene', 'gene',
]


def _is_ensembl(var_names) -> bool:
    """Return True if the majority of var_names look like Ensembl gene IDs."""
    sample = list(var_names[:20])
    n_ensg = sum(1 for g in sample if str(g).startswith('ENSG'))
    return n_ensg > len(sample) * 0.5


def _find_symbol_column(var_df) -> str | None:
    """
    Find which var column contains HGNC gene symbols.

    Checks known candidate column names first, then falls back to detecting
    any column where most values look like gene symbols (not Ensembl IDs,
    not chromosomes, not pure numbers).
    """
    # Check known candidates first
    for col in _SYMBOL_COLUMN_CANDIDATES:
        if col in var_df.columns:
            sample = var_df[col].dropna().astype(str).iloc[:20]
            # Must not look like Ensembl IDs or chromosome names
            n_ensg = sum(1 for v in sample if v.startswith('ENSG'))
            n_chr  = sum(1 for v in sample if v.startswith('chr'))
            if n_ensg == 0 and n_chr == 0 and len(sample) > 0:
                return col

    # Fallback: scan all object/string columns for symbol-like values
    for col in var_df.columns:
        if var_df[col].dtype != object:
            continue
        sample = var_df[col].dropna().astype(str).iloc[:20]
        if len(sample) == 0:
            continue
        n_ensg  = sum(1 for v in sample if v.startswith('ENSG'))
        n_chr   = sum(1 for v in sample if v.startswith('chr'))
        n_alpha = sum(1 for v in sample if v[0].isalpha() and not v.startswith('ENSG'))
        if n_ensg == 0 and n_chr == 0 and n_alpha > len(sample) * 0.7:
            return col

    return None


def map_ensembl_to_symbols(
    adata: ad.AnnData,
    cache_path: str = None,   # kept for API compatibility, no longer used
) -> ad.AnnData:
    """
    Convert Ensembl gene IDs in adata.var_names to HGNC symbols.

    Reads the symbol directly from a column in adata.var — no network call,
    no external dependency. The symbol column is auto-detected from common
    names ('gene_name', 'gene_symbols', 'symbol', etc.).

    If the var dataframe contains no symbol column, raises a clear error
    rather than falling back to the network.

    When two Ensembl IDs share the same HGNC symbol, the gene with higher
    mean expression is kept and duplicates are dropped.

    Args:
        adata:      AnnData with Ensembl IDs in var_names.
        cache_path: Ignored. Kept for backwards compatibility only.

    Returns:
        AnnData with HGNC symbols as var_names.
        Returns adata unchanged if var_names are already gene symbols.
    """
    if not _is_ensembl(adata.var_names):
        logger.info("var_names do not look like Ensembl IDs — skipping conversion.")
        return adata

    logger.info(
        f"Detected Ensembl IDs in var_names ({adata.n_vars:,} genes). "
        f"Reading HGNC symbols from var dataframe..."
    )

    # ── Find the symbol column ────────────────────────────────────────────────
    sym_col = _find_symbol_column(adata.var)

    if sym_col is None:
        raise ValueError(
            "var_names are Ensembl IDs but no HGNC symbol column was found "
            f"in adata.var.\nAvailable var columns: {list(adata.var.columns)}\n"
            "Add a 'gene_name' column to adata.var containing HGNC symbols "
            "and re-run, or use a dataset that includes gene symbols."
        )

    logger.info(f"  Using var column '{sym_col}' for HGNC symbols.")

    symbols_raw = adata.var[sym_col].astype(str).values   # (n_genes,)
    n_before    = len(symbols_raw)

    # ── Drop genes with missing or Ensembl-like symbols ──────────────────────
    valid_mask = np.array([
        bool(s) and not s.startswith('ENSG') and s != 'nan' and s != ''
        for s in symbols_raw
    ])
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.info(
            f"  Dropping {n_invalid:,} genes with missing or unmapped symbols."
        )

    # ── Resolve duplicate symbols ─────────────────────────────────────────────
    # When two Ensembl IDs share a symbol, keep the gene with higher mean
    # expression. This avoids duplicated var_names and discards the lower-
    # signal isoform.
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        # Compute per-gene means without densifying the full matrix
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
    else:
        mean_expr = np.array(X).mean(axis=0)

    symbol_to_best: dict = {}   # symbol → (gene_index, mean_expr)
    for idx, (sym, valid, me) in enumerate(zip(symbols_raw, valid_mask, mean_expr)):
        if not valid:
            continue
        if sym not in symbol_to_best or me > symbol_to_best[sym][1]:
            symbol_to_best[sym] = (idx, me)

    final_indices = sorted(idx for idx, _ in symbol_to_best.values())
    final_symbols = [symbols_raw[i] for i in final_indices]

    n_dupes   = (n_before - n_invalid) - len(final_indices)
    n_dropped = n_before - len(final_indices)

    # ── Rebuild AnnData with new var_names ────────────────────────────────────
    # Subset using boolean index — avoids materialising a dense copy.
    keep_mask = np.zeros(n_before, dtype=bool)
    keep_mask[final_indices] = True
    adata_sub = adata[:, keep_mask].copy()

    # Reassign var_names to HGNC symbols
    adata_sub.var_names = final_symbols
    adata_sub.var_names_make_unique()   # safety: should be unique already

    logger.info(
        f"  Conversion complete: {n_before:,} Ensembl IDs → "
        f"{len(final_symbols):,} unique HGNC symbols "
        f"({n_invalid:,} unmapped dropped, {n_dupes:,} duplicates resolved)."
    )
    return adata_sub


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

def load_replogle_k562(
    data_path: str = None,
    use_pertpy: bool = True,
) -> ad.AnnData:
    """
    Load the Replogle K562 Perturb-seq dataset.

    If var_names are Ensembl IDs (ENSG...), automatically converts them to
    HGNC symbols by reading the gene symbol column already present in
    adata.var. No network call or external dependency required.

    Args:
        data_path:  Path to a local .h5ad file. Required if use_pertpy=False.
        use_pertpy: Attempt to load via pertpy.

    Returns:
        AnnData with HGNC symbols in var_names, all perturbations included.
    """
    if use_pertpy:
        try:
            import pertpy as pt
            logger.info("Loading K562 dataset via pertpy (will download on first run)...")
            adata = pt.data.replogle_2022_k562_essential()
            logger.info(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes via pertpy.")
            adata = map_ensembl_to_symbols(adata)
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
    adata = map_ensembl_to_symbols(adata)
    return adata


# ── Control Cell Extraction ───────────────────────────────────────────────────

# Known labels used for unperturbed/non-targeting control cells
# across common Perturb-seq dataset conventions.
_CONTROL_LABELS = {
    'control', 'non-targeting', 'ctrl', 'neg_ctrl',
    'CONTROL', 'non_targeting', 'NonTargeting', 'non-targeting_ctrl',
    'non_targeting', 'safe-targeting',
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

    # ── Housekeeping gene exclusion ───────────────────────────────────────────
    # Mitochondrial (MT-*), ribosomal (RPS*, RPL*), and a handful of
    # high-expression lncRNAs (MALAT1, NEAT1) dominate raw count variance
    # but are biologically uninformative for perturbation effects.
    #
    # When seurat_v3 selects HVGs from raw counts these go straight to the top
    # because they are highly expressed and variable — but they covary primarily
    # with library size and cell-cycle stress, not with gene knockouts. Leaving
    # them in causes NMF programs to all point in the same diffuse direction
    # (as seen in the K-sweep: Frobenius error flat at 0.56 across K=50–90,
    # cosine similarity between programs ~0.4, effective genes 1600–2000).
    #
    # This step removes them from the pool BEFORE HVG selection so they cannot
    # be nominated regardless of how variable they are.
    excl_prefixes = ('MT-', 'mt-', 'RPS', 'RPL', 'Rps', 'Rpl')
    excl_exact    = {'MALAT1', 'NEAT1', 'XIST', 'TSIX'}

    keep_mask = np.array([
        not (g.startswith(excl_prefixes) or g in excl_exact)
        for g in adata.var_names
    ])
    n_excluded = (~keep_mask).sum()
    if n_excluded > 0:
        adata = adata[:, keep_mask].copy()
        gc.collect()
        logger.info(
            f"Housekeeping gene exclusion: removed {n_excluded:,} genes "
            f"(MT-*, RPS*, RPL*, MALAT1, NEAT1, XIST). "
            f"{adata.n_vars:,} genes remain."
        )
    else:
        logger.info(
            "Housekeeping gene exclusion: no MT-/ribosomal genes found "
            "(var_names may already be filtered, or naming convention differs)."
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