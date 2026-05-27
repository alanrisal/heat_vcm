"""
scripts/02_compute_diffusion.py

Component 2 runner: Graph Heat Diffusion Perturbation Encoding.

Produces the P_g matrix — one K-dimensional vector per perturbation gene —
encoding how each gene knockout's regulatory signal propagates through the
STRING-DB network and maps onto the NMF gene programs.

Prerequisites
-------------
Component 1 must have completed successfully:
    outputs/programs/H_matrix.npy
    outputs/programs/checkpoint_gene_names.txt

Usage
-----
    # Full run (downloads STRING-DB on first call)
    python scripts/02_compute_diffusion.py

    # Custom β
    python scripts/02_compute_diffusion.py --beta 0.3

    # Run β sweep to choose β before committing
    python scripts/02_compute_diffusion.py --beta_sweep --sweep_gene TP53

    # Force CPU (skip GPU)
    python scripts/02_compute_diffusion.py --no_gpu

Outputs
-------
    outputs/diffusion/P_matrix.npy           (n_perts × K)
    outputs/diffusion/perturbation_genes.txt  one gene name per line
    outputs/diffusion/seed_labels.txt         direct/neighbour/missing per gene
    outputs/diffusion/H_diffusion.npy        (G × n_perts) raw profiles
    outputs/diffusion/coverage_stats.csv
    outputs/diffusion/figures/component2_verification.png
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional
from src.data.loader import (
    load_replogle_k562,
    extract_control_cells,
    log_ram_usage,
)
from src.diffusion.graph import build_regulatory_graph
from src.diffusion.heat import compute_pg_matrix
from src.diffusion.verify import (
    check_coverage,
    check_diffusion_concentration,
    check_program_signal,
    run_beta_sweep,
    plot_component2_summary,
    print_component2_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

_CONTROL_LABELS = {
    'control', 'non-targeting', 'ctrl', 'neg_ctrl',
    'CONTROL', 'non_targeting', 'NonTargeting',
}


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Component 2: Graph Heat Diffusion Perturbation Encoding'
    )
    p.add_argument('--beta', type=float, default=0.1,
        help='Heat diffusion β (default: 0.1). '
             'Larger = more spread; smaller = more local. '
             'Run --beta_sweep first to pick a good value.')
    p.add_argument('--min_score', type=int, default=700,
        help='STRING-DB minimum combined score (default: 700 = high confidence).')
    p.add_argument('--programs_dir', type=str, default='outputs/programs',
        help='Directory containing Component 1 outputs.')
    p.add_argument('--output_dir', type=str, default='outputs/diffusion',
        help='Directory to write Component 2 outputs.')
    p.add_argument('--string_cache', type=str, default='data/string_cache',
        help='Directory for STRING-DB cache files (~400 MB on first download).')
    p.add_argument('--data_path', type=str, default=None,
        help='Path to local .h5ad file. Leave unset to load via pertpy.')
    p.add_argument('--no_pertpy', action='store_true')
    p.add_argument('--no_gpu', action='store_true',
        help='Force CPU for all computation.')
    p.add_argument('--perturbation_col', type=str, default='gene',
        help='obs column name containing perturbation gene labels. '
             'Auto-detected if not specified.')
    p.add_argument('--force_perturbation_col', type=str, default=None,
        help='Force-use this obs column regardless of cardinality. '
             'Use when the column has guide-coordinate identifiers (>5000 '
             'unique values) that the auto-detector rejects.')
    p.add_argument('--beta_sweep', action='store_true',
        help='Run a β sweep for a sample gene before the main computation. '
             'Use this to pick β before committing to a full run.')
    p.add_argument('--sweep_gene', type=str, default='TP53',
        help='Gene to use for the β sweep visualisation (default: TP53).')
    p.add_argument('--save_diffusion', action='store_true',
        help='Save the full (G × n_perts) H_diffusion matrix. '
             'This can be large (~200 MB) — omit unless you need it.')
    return p.parse_args()


# ── Perturbation gene extraction ──────────────────────────────────────────────

def _detect_perturbation_column(adata, requested_key: str, force: bool = False) -> str:
    """
    Find the obs column containing perturbation labels.

    Always prints the full obs column table regardless of outcome.
    With force=True (--force_perturbation_col), the requested column
    is accepted unconditionally.
    """
    obs = adata.obs

    # ── Always print full diagnostics ────────────────────────────────────────
    logger.info("obs column diagnostics:")
    logger.info(f"  {'Column':<35} {'Dtype':<12} {'N unique':>9}  Example")
    logger.info(f"  {'─'*35} {'─'*12} {'─'*9}  {'─'*30}")
    for col in obs.columns:
        try:
            n_uniq  = obs[col].nunique()
            example = str(obs[col].iloc[0])[:45]
            note = '  ← low-card' if 2 <= n_uniq <= 5000 else ''
            logger.info(f"  {col:<35} {str(obs[col].dtype):<12} {n_uniq:>9}  {example}{note}")
        except Exception:
            pass

    # ── Force override ────────────────────────────────────────────────────────
    if force:
        if requested_key not in obs.columns:
            raise KeyError(
                f"--force_perturbation_col '{requested_key}' not in obs columns.\n"
                f"Available: {list(obs.columns)}"
            )
        logger.info(f"Force-using column '{requested_key}' (--force_perturbation_col).")
        return requested_key

    # ── Try standard candidates with low-cardinality check ───────────────────
    candidates = [requested_key, 'perturbation', 'gene', 'gene_name',
                  'target', 'target_gene', 'guide_target', 'perturbation_name',
                  'condition', 'sgRNA_target', 'gene_id']

    for col in candidates:
        if col not in obs.columns:
            continue
        n_uniq = obs[col].nunique()
        if 2 <= n_uniq <= 5000:
            logger.info(f"Auto-detected perturbation column: '{col}' ({n_uniq} unique values).")
            return col

    # ── If nothing found, print clear guidance ────────────────────────────────
    raise ValueError(
        "Could not auto-detect a perturbation gene column (expected 2–5,000 unique values).\n"
        "If your perturbation column contains guide-RNA coordinate identifiers\n"
        "(e.g. 'chr10.845_top_two_chr1.11183'), use one of:\n\n"
        "  (a) Run  python scripts/inspect_adata.py  for a full structure dump.\n"
        "  (b) If a low-cardinality gene-name column exists:\n"
        "        --perturbation_col COLUMN_NAME\n"
        "  (c) If the guide-coordinate column IS the only option and you want\n"
        "      to force-use it (values will be matched against STRING-DB):\n"
        "        --force_perturbation_col COLUMN_NAME\n"
    )


def _collapse_guides_to_genes(labels: list, var_names: set) -> tuple:
    """
    Attempt to collapse guide-coordinate identifiers to gene names.

    Two strategies tried in order:
      1. Exact match against var_names (dataset gene symbols).
      2. Fragment match: split on '_', check each fragment against var_names.

    Returns (gene_list, n_collapsed, n_unresolved).
    Labels that cannot be resolved are silently dropped.
    """
    resolved = {}
    n_unresolved = 0

    for label in labels:
        # Strategy 1: direct match
        if label in var_names:
            resolved[label] = label
            continue

        # Strategy 2: try splitting on '_' and common separators
        parts = label.replace('.', '_').split('_')
        matched = None
        for part in parts:
            if part in var_names:
                matched = part
                break
            if part.upper() in var_names:
                matched = part.upper()
                break

        if matched:
            resolved[label] = matched
        else:
            n_unresolved += 1

    unique_genes = sorted(set(resolved.values()))
    n_collapsed  = len(labels) - len(set(resolved.keys()))

    return unique_genes, len(resolved), n_unresolved


def extract_perturbation_genes(
    data_path: Optional[str],
    use_pertpy: bool,
    perturbation_key: str = 'gene',
    force: bool = False,
) -> list:
    """
    Load the dataset and return unique non-control perturbation gene names.

    If the perturbation column contains guide-RNA coordinate identifiers
    rather than gene symbols, attempts to collapse them to gene names using
    the dataset's own var_names as the reference gene set.
    """
    
    logger.info("Extracting perturbation gene list from dataset...")

    try:
        if use_pertpy:
            import pertpy as pt
            adata = pt.data.replogle_2022_k562_essential()
        else:
            import scanpy as sc
            adata = sc.read_h5ad(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    logger.info(
        f"Dataset loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes. "
        f"{len(adata.obs.columns)} obs columns."
    )

    detected_key = _detect_perturbation_column(adata, perturbation_key, force=force)

    all_labels = adata.obs[detected_key].astype(str).unique().tolist()
    var_names  = set(adata.var_names.astype(str))

    # Remove control labels
    raw_labels = [l for l in all_labels if l not in _CONTROL_LABELS]

    # Check whether values look like gene symbols or guide coordinates
    # (guide coordinates tend to start with 'chr' or contain '.' positional info)
    guide_like = sum(
        1 for l in raw_labels[:200]
        if l.startswith('chr') or ('.' in l and '_' in l)
    )
    is_guide_format = guide_like > len(raw_labels[:200]) * 0.5

    if is_guide_format:
        logger.warning(
            f"Column '{detected_key}' appears to contain guide-coordinate identifiers "
            f"(e.g. '{raw_labels[0][:60]}'). "
            f"Attempting to collapse to gene symbols using dataset var_names..."
        )
        gene_list, n_resolved, n_unresolved = _collapse_guides_to_genes(
            raw_labels, var_names
        )
        logger.info(
            f"Guide collapse: {len(raw_labels)} guide labels → "
            f"{len(gene_list)} unique gene symbols "
            f"({n_resolved} resolved, {n_unresolved} unresolved/dropped)."
        )
        if not gene_list:
            logger.error(
                "Could not resolve any guide identifiers to gene symbols. "
                "Run: python scripts/inspect_adata.py  for the full dataset structure."
            )
    else:
        gene_list = sorted(set(raw_labels))
        logger.info(
            f"Perturbation column contains gene symbols directly. "
            f"{len(gene_list)} unique targets."
        )

    del adata
    gc.collect()

    return gene_list


# ── Load Component 1 outputs ──────────────────────────────────────────────────

def load_component1_outputs(programs_dir: str) -> tuple:
    """Load H_matrix and gene_names saved by Component 1."""
    d = Path(programs_dir)

    H_path     = d / 'H_matrix.npy'
    genes_path = d / 'checkpoint_gene_names.txt'

    if not H_path.exists():
        raise FileNotFoundError(
            f"H_matrix.npy not found at '{H_path}'. "
            f"Run 01_extract_programs.py first."
        )
    if not genes_path.exists():
        # Fallback to gene_names.txt saved by save_nmf_results
        genes_path = d / 'gene_names.txt'
        if not genes_path.exists():
            raise FileNotFoundError(
                f"Gene names file not found in '{d}'. "
                f"Expected checkpoint_gene_names.txt or gene_names.txt."
            )

    H_nmf      = np.load(str(H_path))        # (K × G)
    gene_names = genes_path.read_text().strip().splitlines()

    logger.info(
        f"Loaded Component 1 outputs: "
        f"H_nmf={H_nmf.shape}, {len(gene_names)} HVG genes."
    )
    return H_nmf, gene_names


# ── Save Component 2 outputs ──────────────────────────────────────────────────

def save_component2_outputs(result: dict, output_dir: str, save_diffusion: bool):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # P_matrix — primary output for Component 3
    np.save(str(out / 'P_matrix.npy'), result['P_matrix'])

    # Perturbation gene names and seed labels (row metadata)
    (out / 'perturbation_genes.txt').write_text(
        '\n'.join(result['perturbation_genes'])
    )
    (out / 'seed_labels.txt').write_text(
        '\n'.join(result['seed_labels'])
    )

    # Coverage stats as CSV
    pd.DataFrame([result['coverage_stats']]).to_csv(
        out / 'coverage_stats.csv', index=False
    )

    # Optional: raw diffusion profiles (large)
    if save_diffusion:
        np.save(str(out / 'H_diffusion.npy'), result['H_diffusion'])
        logger.info(
            f"H_diffusion saved: {result['H_diffusion'].shape}  "
            f"({result['H_diffusion'].nbytes / 1e6:.0f} MB)"
        )

    logger.info(f"Component 2 outputs saved to '{out}/'")
    logger.info(f"  P_matrix shape: {result['P_matrix'].shape}")


# ── Pass / fail ───────────────────────────────────────────────────────────────

_CHECKS = [
    {
        'name':      'Coverage (fraction represented)',
        'get':       lambda cov, conc, prog: cov['frac_represented'],
        'threshold': 0.60,
        'direction': 'above',
        'fix':       'Lower --min_score (e.g. 500) to include more STRING-DB edges.',
    },
    {
        'name':      'Concentration (self in top-50)',
        'get':       lambda cov, conc, prog: conc.get('frac_self_in_top_50', 1.0),
        'threshold': 0.50,
        'direction': 'above',
        'fix':       'Reduce β (e.g. --beta 0.05) for more concentrated diffusion.',
    },
    {
        'name':      'Program signal (median CV)',
        'get':       lambda cov, conc, prog: prog['median_cv'],
        'threshold': 0.30,
        'direction': 'above',
        'fix':       'Increase β (e.g. --beta 0.3) to spread signal across programs.',
    },
]


def evaluate_pass_fail(coverage, concentration, program_signal):
    failures = []
    for chk in _CHECKS:
        val = chk['get'](coverage, concentration, program_signal)
        if np.isnan(val):
            continue
        passed = val >= chk['threshold'] if chk['direction'] == 'above' else val <= chk['threshold']
        if not passed:
            failures.append({**chk, 'value': val})
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    use_gpu  = not args.no_gpu
    out_dir  = Path(args.output_dir)
    fig_dir  = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info('  COMPONENT 2: Graph Heat Diffusion Encoding')
    logger.info(f'  β={args.beta}  min_score={args.min_score}')
    logger.info('=' * 60)

    # ── Step 1: Load Component 1 outputs ──────────────────────────────────────
    logger.info('STEP 1 — Loading Component 1 outputs')
    H_nmf, gene_names = load_component1_outputs(args.programs_dir)
    log_ram_usage('after loading H and gene names')

    # ── Step 2: Determine perturbation gene list ──────────────────────────────
    # Component 2 builds p_g vectors as a regulatory prior — one vector per
    # gene in the HVG set. We do NOT need the dataset at this stage.
    #
    # Why: every HVG gene is a *candidate* perturbation target. Component 3
    # handles the mapping from per-cell guide-coordinate identifiers back to
    # gene names (using the separate guide_map utility). Here we simply build
    # the p_g prior for every gene we could possibly need.
    #
    # Only the STRING-DB-connected subset (3,034 / 5,000 genes in practice)
    # will have non-zero p_g vectors. Isolated genes receive all-zero rows.
    logger.info('STEP 2 — Using HVG genes as perturbation targets')
    pert_genes = gene_names           # all 5,000 HVG genes from Component 1
    logger.info(
        f'  {len(pert_genes):,} genes → one p_g vector each. '
        f'Guide-to-gene mapping for the actual expression data is handled '
        f'separately in Component 3 (see scripts/build_guide_map.py).'
    )
    log_ram_usage(f'perturbation target list set')

    # ── Step 3: Build regulatory graph ────────────────────────────────────────
    logger.info(f'STEP 3 — Building regulatory graph  (min_score={args.min_score})')
    graph_data = build_regulatory_graph(
        gene_names=gene_names,
        cache_dir=args.string_cache,
        min_score=args.min_score,
    )
    log_ram_usage('after graph construction')

    # ── Step 4 (optional): β sweep ────────────────────────────────────────────
    beta_sweep_result = None
    if args.beta_sweep:
        logger.info(f'STEP 4 — Running β sweep for gene: {args.sweep_gene}')
        beta_sweep_result = run_beta_sweep(
            L=graph_data['L'],
            example_gene=args.sweep_gene,
            gene_to_idx=graph_data['gene_to_idx'],
            gene_names=gene_names,
            use_gpu=use_gpu,
        )

    # ── Step 5: Compute heat diffusion + P_g matrix ───────────────────────────
    logger.info(f'STEP 5 — Computing heat diffusion  (β={args.beta})')
    result = compute_pg_matrix(
        perturbation_genes=pert_genes,
        graph_data=graph_data,
        H_nmf=H_nmf,
        beta=args.beta,
        use_gpu=use_gpu,
    )
    log_ram_usage('after heat diffusion')

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    logger.info('STEP 6 — Saving outputs')
    save_component2_outputs(result, args.output_dir, args.save_diffusion)

    # ── Step 7: Verification ──────────────────────────────────────────────────
    logger.info('STEP 7 — Verification checks')

    coverage = check_coverage(result['coverage_stats'])

    concentration = check_diffusion_concentration(
        H_diffusion=result['H_diffusion'],
        perturbation_genes=result['perturbation_genes'],
        gene_to_idx=graph_data['gene_to_idx'],
        seed_labels=result['seed_labels'],
        top_k=50,
    )

    program_signal = check_program_signal(
        P_matrix=result['P_matrix'],
        seed_labels=result['seed_labels'],
    )

    # ── Step 8: Report and figure ─────────────────────────────────────────────
    logger.info('STEP 8 — Report and figure')

    print_component2_report(
        coverage, concentration, program_signal,
        beta=args.beta,
        graph_data=graph_data,
    )

    plot_component2_summary(
        coverage=coverage,
        concentration=concentration,
        program_signal=program_signal,
        beta_sweep=beta_sweep_result,
        P_matrix=result['P_matrix'],
        output_path=str(fig_dir / f'component2_beta{args.beta}.png'),
    )

    # ── Step 9: Pass / fail ───────────────────────────────────────────────────
    failures = evaluate_pass_fail(coverage, concentration, program_signal)

    if failures:
        logger.warning('')
        logger.warning('⚠  VERIFICATION ISSUES:')
        for f in failures:
            logger.warning(
                f"  FAIL  {f['name']:<42} "
                f"value={f['value']:.4f}  threshold={f['threshold']}"
            )
            logger.warning(f"        → Fix: {f['fix']}")
        logger.warning('')
        logger.warning(
            'Adjust β or min_score and re-run. '
            'Use --beta_sweep to guide β selection.'
        )
    else:
        logger.info('')
        logger.info('✓  ALL CHECKS PASSED.')
        logger.info(f'   P_matrix is ready for Component 3.')
        logger.info(f'   Shape: {result["P_matrix"].shape}')
        logger.info(f'   Saved to: {args.output_dir}/')
        logger.info('')

    return result


from typing import Optional

if __name__ == '__main__':
    main()