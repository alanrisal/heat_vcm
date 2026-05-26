"""
scripts/01_extract_programs.py

Component 1 runner: gene program extraction and full verification.

This script executes the complete Component 1 pipeline:
  1. Load Replogle K562 dataset
  2. Isolate and preprocess control cells
  3. Fit NMF to extract K gene programs
  4. Run all four verification checks
  5. Run pathway enrichment (optional — requires internet)
  6. Save results, plots, and a pass/fail decision report

Usage:
    # Default: K=50, loads via pertpy
    python scripts/01_extract_programs.py

    # Custom K, local data file
    python scripts/01_extract_programs.py --k 40 --data_path data/raw/k562.h5ad

    # Skip enrichment (faster, no internet needed)
    python scripts/01_extract_programs.py --skip_enrichment

    # Try multiple K values to find the best
    python scripts/01_extract_programs.py --k 30
    python scripts/01_extract_programs.py --k 50
    python scripts/01_extract_programs.py --k 70

Reading the output:
    The script ends with a PASS / ISSUES DETECTED verdict.
    If any check fails, it tells you exactly what to change (K, max_iter, etc.)
    before proceeding to Component 2 (graph heat diffusion).
    Do not proceed to Component 2 until this script exits cleanly.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the Python path regardless of where the script is run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import (
    load_replogle_k562,
    extract_control_cells,
    preprocess_control_cells,
)
from src.programs.nmf import (
    fit_nmf,
    build_program_dataframe,
    save_nmf_results,
)
from src.programs.verify import (
    compute_reconstruction_quality,
    compute_program_sparsity,
    compute_program_uniqueness,
    test_nmf_stability,
    run_pathway_enrichment,
    plot_verification_summary,
    print_verification_report,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Component 1: Extract and verify gene programs from K562 control cells.'
    )
    p.add_argument(
        '--k', type=int, default=50,
        help='Number of NMF programs to extract (default: 50). '
             'Start here and adjust based on verification output.'
    )
    p.add_argument(
        '--n_hvg', type=int, default=5000,
        help='Number of highly variable genes for NMF (default: 5000).'
    )
    p.add_argument(
        '--data_path', type=str, default=None,
        help='Path to local .h5ad file. Leave unset to load via pertpy.'
    )
    p.add_argument(
        '--no_pertpy', action='store_true',
        help='Disable pertpy loading (requires --data_path).'
    )
    p.add_argument(
        '--stability_runs', type=int, default=5,
        help='Number of NMF runs for the stability check (default: 5).'
    )
    p.add_argument(
        '--skip_enrichment', action='store_true',
        help='Skip pathway enrichment. Use this if you have no internet access.'
    )
    p.add_argument(
        '--alpha_h', type=float, default=0.1,
        help='L2 regularization on NMF gene loadings H (default: 0.1). '
             'Increase to encourage sparser programs.'
    )
    p.add_argument(
        '--output_dir', type=str, default='outputs/programs',
        help='Directory to save NMF results and figures (default: outputs/programs).'
    )
    return p.parse_args()


# ── Pass / Fail Criteria ──────────────────────────────────────────────────────

# These thresholds are the minimum bar to proceed to Component 2.
# Each check has a specific corrective action if it fails.
_CHECKS = [
    {
        'name': 'Reconstruction (median gene R²)',
        'get': lambda r, sp, u, st: r['median_gene_r2'],
        'threshold': 0.3,
        'direction': 'above',
        'fix': 'Increase K (more programs can explain more variance).',
    },
    {
        'name': 'Sparsity (mean Gini coefficient)',
        'get': lambda r, sp, u, st: sp['mean_gini'],
        'threshold': 0.6,
        'direction': 'above',
        'fix': 'Increase alpha_H (stronger regularization → sparser programs).',
    },
    {
        'name': 'Uniqueness (near-duplicate program pairs)',
        'get': lambda r, sp, u, st: u['n_near_duplicate_pairs'],
        'threshold': 3,
        'direction': 'below_or_equal',
        'fix': 'Reduce K (fewer programs removes redundancy).',
    },
    {
        'name': 'Stability (mean cosine similarity across runs)',
        'get': lambda r, sp, u, st: st['mean_stability_score'],
        'threshold': 0.75,
        'direction': 'above',
        'fix': 'Reduce K or increase max_iter.',
    },
]


def evaluate_pass_fail(
    recon: dict,
    sparsity: dict,
    uniqueness: dict,
    stability: dict,
) -> list:
    """
    Apply pass/fail criteria to verification metrics.
    Returns a list of failing check dicts (empty list = all pass).
    """
    failures = []
    for check in _CHECKS:
        value = check['get'](recon, sparsity, uniqueness, stability)
        if check['direction'] == 'above':
            passed = value >= check['threshold']
        else:  # below_or_equal
            passed = value <= check['threshold']
        if not passed:
            failures.append({**check, 'value': value})
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    args = parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f'  COMPONENT 1: Gene Program Extraction  (K={args.k})')
    logger.info('=' * 60)

    # ── Step 1: Load dataset ───────────────────────────────────────────────────
    logger.info('STEP 1 — Loading dataset')
    adata = load_replogle_k562(
        data_path=args.data_path,
        use_pertpy=not args.no_pertpy,
    )

    # ── Step 2: Isolate control cells ─────────────────────────────────────────
    logger.info('STEP 2 — Extracting control cells')
    ctrl_adata = extract_control_cells(adata)

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    logger.info(f'STEP 3 — Preprocessing  (n_hvg={args.n_hvg})')
    ctrl_processed = preprocess_control_cells(ctrl_adata, n_top_genes=args.n_hvg)

    gene_names = list(ctrl_processed.var_names)
    cell_barcodes = list(ctrl_processed.obs_names)

    if hasattr(ctrl_processed.X, 'toarray'):
        X = ctrl_processed.X.toarray().astype(np.float32)
    else:
        X = np.array(ctrl_processed.X, dtype=np.float32)

    logger.info(f'Expression matrix ready: {X.shape}  (cells × HVGs)')

    # ── Step 4: Fit NMF ───────────────────────────────────────────────────────
    logger.info(f'STEP 4 — Fitting NMF  (K={args.k}, alpha_H={args.alpha_h})')
    model, W, H = fit_nmf(
        ctrl_processed,
        n_programs=args.k,
        alpha_H=args.alpha_h,
    )

    top_genes_df = build_program_dataframe(H, gene_names, n_top=100)
    save_nmf_results(W, H, gene_names, cell_barcodes, str(out_dir))

    # ── Step 5: Verification checks ───────────────────────────────────────────
    logger.info('STEP 5 — Running verification checks')

    logger.info('  [1/4] Reconstruction quality...')
    recon = compute_reconstruction_quality(X, W, H)

    logger.info('  [2/4] Program sparsity...')
    sparsity = compute_program_sparsity(H)

    logger.info('  [3/4] Program uniqueness...')
    uniqueness = compute_program_uniqueness(H)

    logger.info(f'  [4/4] NMF stability ({args.stability_runs} runs)...')
    stability = test_nmf_stability(
        X, args.k, n_runs=args.stability_runs, max_iter=500
    )

    # ── Step 6: Pathway enrichment (optional) ─────────────────────────────────
    enrichment_df = None
    if not args.skip_enrichment:
        logger.info('STEP 6 — Pathway enrichment (requires internet)...')
        enrichment_df = run_pathway_enrichment(
            top_genes_df,
            n_top=100,
            output_dir=str(out_dir),
        )
    else:
        logger.info('STEP 6 — Pathway enrichment skipped (--skip_enrichment).')

    # ── Step 7: Report and figures ────────────────────────────────────────────
    logger.info('STEP 7 — Generating report and verification figure')

    print_verification_report(
        recon, sparsity, uniqueness, stability,
        enrichment_df=enrichment_df,
        n_programs=args.k,
    )

    plot_verification_summary(
        recon, sparsity, uniqueness, stability,
        output_path=str(fig_dir / f'verification_K{args.k}.png'),
    )

    # ── Step 8: Pass / fail decision ──────────────────────────────────────────
    logger.info('STEP 8 — Pass / fail evaluation')
    failures = evaluate_pass_fail(recon, sparsity, uniqueness, stability)

    if failures:
        logger.warning('')
        logger.warning('⚠  VERIFICATION ISSUES — do not proceed to Component 2 yet:')
        for f in failures:
            logger.warning(
                f"  FAIL  {f['name']:<45} "
                f"value={f['value']:.4f}  threshold={f['threshold']}"
            )
            logger.warning(f"        → Fix: {f['fix']}")
        logger.warning('')
        logger.warning(
            'Adjust K or alpha_H and re-run this script until all checks pass.'
        )
    else:
        logger.info('')
        logger.info('✓  ALL CHECKS PASSED.')
        logger.info(f'   K={args.k} gene programs are ready for Component 2.')
        logger.info(f'   Results saved to: {out_dir}/')
        logger.info('')

    return {
        'adata': ctrl_processed,
        'W': W,
        'H': H,
        'gene_names': gene_names,
        'recon': recon,
        'sparsity': sparsity,
        'uniqueness': uniqueness,
        'stability': stability,
        'enrichment_df': enrichment_df,
        'passed': len(failures) == 0,
    }


if __name__ == '__main__':
    main()