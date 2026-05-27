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
import gc
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
    log_ram_usage,
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
        '--tol', type=float, default=1e-3,
        help='NMF convergence tolerance (default: 1e-3).'
    )
    p.add_argument(
        '--no_gpu', action='store_true',
        help='Disable GPU acceleration (force CPU). By default GPU is used '
             'if CUDA is available.'
    )
    p.add_argument(
        '--from_checkpoint', action='store_true',
        help='Skip data loading and preprocessing entirely. '
             'Loads the preprocessed X matrix and gene names saved by a '
             'previous run from --output_dir. Use this when sweeping K values '
             'to avoid re-running the ~10 min load+preprocess stage each time.'
    )
    p.add_argument(
        '--fast', action='store_true',
        help='Fast sweep mode: reduces stability_runs to 3 and max_iter to 400. '
             'Use when exploring K values. Run without --fast for final verification '
             'on the chosen K. Cuts stability test from ~20 min to ~5 min.'
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
        # Frobenius error is the correct reconstruction check for sparse scRNA-seq.
        # Median per-gene R² was the original check but is misleading here:
        # thousands of near-zero-variance genes (dropouts) will always have R²
        # near 0 because there is nothing for NMF to explain in those genes.
        # Frobenius error < 0.5 means NMF captures > 50% of total matrix energy,
        # which is a meaningful signal for this data type.
        'name': 'Reconstruction (relative Frobenius error)',
        'get': lambda r, sp, u, st: r['relative_frobenius_error'],
        'threshold': 0.5,
        'direction': 'below_or_equal',
        'fix': 'Increase K (more programs capture more total variance).',
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

    # ── Apply --fast mode overrides before anything else ──────────────────────
    # Fast mode cuts the stability test from ~20 min to ~4-5 min.
    # Use it freely when sweeping K. Run without --fast for the final K choice.
    if args.fast:
        stability_runs = 3
        nmf_max_iter   = 400
        logger.info('FAST MODE — stability_runs=3, max_iter=400. '
                    'Good for K sweeps. Re-run without --fast for final verification.')
    else:
        stability_runs = args.stability_runs
        nmf_max_iter   = 1000

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Paths for the preprocessing checkpoint files
    checkpoint_X    = out_dir / 'checkpoint_X.npy'
    checkpoint_genes = out_dir / 'checkpoint_gene_names.txt'
    checkpoint_cells = out_dir / 'checkpoint_cell_barcodes.txt'

    logger.info('=' * 60)
    logger.info(f'  COMPONENT 1: Gene Program Extraction  (K={args.k})')
    if args.fast:
        logger.info('  [FAST SWEEP MODE]')
    logger.info('=' * 60)

    # ── Steps 1-3: Load + Preprocess (or load from checkpoint) ───────────────
    if args.from_checkpoint:
        # ── Checkpoint path: skip ~10 min of loading/preprocessing ──────────
        if not checkpoint_X.exists():
            raise FileNotFoundError(
                f"No checkpoint found at '{checkpoint_X}'.\n"
                f"Run without --from_checkpoint first to generate it."
            )
        logger.info('STEPS 1-3 — Loading from preprocessing checkpoint (skipping data load)...')
        X          = np.load(str(checkpoint_X))
        gene_names = checkpoint_genes.read_text().strip().splitlines()
        cell_barcodes = checkpoint_cells.read_text().strip().splitlines()
        log_ram_usage(f'checkpoint loaded — X={X.shape}')
        logger.info(f'Preprocessed matrix loaded: {X.shape}  (cells × HVGs)')

    else:
        # ── Full path: load raw data, preprocess, save checkpoint ────────────
        logger.info('STEP 1 — Loading dataset')
        adata = load_replogle_k562(
            data_path=args.data_path,
            use_pertpy=not args.no_pertpy,
        )
        log_ram_usage('after full dataset load')

        logger.info('STEP 2 — Extracting control cells')
        ctrl_adata = extract_control_cells(adata)

        # Free the full dataset immediately — the single largest RAM release.
        del adata
        gc.collect()
        log_ram_usage('after del full dataset')

        logger.info(f'STEP 3 — Preprocessing  (n_hvg={args.n_hvg})')
        ctrl_processed = preprocess_control_cells(ctrl_adata, n_top_genes=args.n_hvg)

        del ctrl_adata
        gc.collect()
        log_ram_usage('after del ctrl_adata')

        gene_names    = list(ctrl_processed.var_names)
        cell_barcodes = list(ctrl_processed.obs_names)

        if hasattr(ctrl_processed.X, 'toarray'):
            X = ctrl_processed.X.toarray().astype(np.float32)
        else:
            X = np.array(ctrl_processed.X, dtype=np.float32)

        del ctrl_processed
        gc.collect()
        log_ram_usage(f'after densify — X={X.shape}')

        # Save checkpoint so future K sweeps skip straight to NMF
        logger.info(f'Saving preprocessing checkpoint to {out_dir}/ ...')
        np.save(str(checkpoint_X), X)
        checkpoint_genes.write_text('\n'.join(gene_names))
        checkpoint_cells.write_text('\n'.join(cell_barcodes))
        logger.info('Checkpoint saved. Future runs can use --from_checkpoint to skip preprocessing.')

    logger.info(f'Expression matrix ready: {X.shape}  (cells × HVGs)')

    # ── Step 4: Fit NMF ───────────────────────────────────────────────────────
    logger.info(
        f'STEP 4 — Fitting NMF  '
        f'(K={args.k}, alpha_H={args.alpha_h}, tol={args.tol}, max_iter={nmf_max_iter})'
    )

    import anndata as ad
    adata_for_nmf = ad.AnnData(X=X)

    model, W, H = fit_nmf(
        adata_for_nmf,
        n_programs=args.k,
        alpha_H=args.alpha_h,
        tol=args.tol,
        max_iter=nmf_max_iter,
        use_gpu=not args.no_gpu,
    )

    del adata_for_nmf
    gc.collect()
    log_ram_usage('after NMF fit')

    top_genes_df = build_program_dataframe(H, gene_names, n_top=100)
    save_nmf_results(W, H, gene_names, cell_barcodes, str(out_dir))
    log_ram_usage('after save')

    # ── Step 5: Verification checks ───────────────────────────────────────────
    logger.info('STEP 5 — Running verification checks')

    logger.info('  [1/4] Reconstruction quality...')
    recon = compute_reconstruction_quality(X, W, H)

    logger.info('  [2/4] Program sparsity...')
    sparsity = compute_program_sparsity(H)

    logger.info('  [3/4] Program uniqueness...')
    uniqueness = compute_program_uniqueness(H)

    logger.info(f'  [4/4] NMF stability ({stability_runs} runs, max_iter={nmf_max_iter})...')
    stability = test_nmf_stability(
        X, args.k,
        n_runs=stability_runs,
        max_iter=nmf_max_iter,
        tol=args.tol,
        use_gpu=not args.no_gpu,
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
        if args.fast:
            logger.warning('Note: running in --fast mode. Re-run without --fast '
                           'before treating any result as final.')
        logger.warning('Adjust K or alpha_H and re-run with --from_checkpoint --fast '
                       'to skip preprocessing on future attempts.')
    else:
        logger.info('')
        logger.info('✓  ALL CHECKS PASSED.')
        if args.fast:
            logger.info('  Note: --fast mode was active. '
                        'Re-run without --fast to confirm with full stability settings.')
        logger.info(f'   K={args.k} gene programs are ready for Component 2.')
        logger.info(f'   Results saved to: {out_dir}/')
        logger.info('')

    return {
        'X': X,
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