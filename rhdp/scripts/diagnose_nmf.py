"""
scripts/diagnose_nmf.py

Two-part diagnostic:

  Part 1 — Data quality check
    Verifies the preprocessed matrix looks correct before NMF runs.
    Prints key statistics: sparsity, value range, per-cell and per-gene
    distributions. Flags anything that would cause NMF to behave poorly.

  Part 2 — K sweep
    Runs NMF at multiple K values and prints ALL raw metric values in a
    single comparison table — no pass/fail, just numbers.
    This shows exactly which metric is failing at each K and by how much,
    so you can make an informed decision rather than guessing K values one
    at a time.

Usage
-----
    # Data quality check only (fast — no NMF)
    python scripts/diagnose_nmf.py --check_only

    # Full K sweep (uses checkpoint so no data reload needed)
    python scripts/diagnose_nmf.py --k_values 50 60 70 75 80 85 90

    # Quick sweep with fast stability (3 runs instead of 5)
    python scripts/diagnose_nmf.py --k_values 60 70 80 90 --fast
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--k_values', type=int, nargs='+',
                   default=[50, 60, 70, 75, 80, 85, 90],
                   help='K values to sweep.')
    p.add_argument('--programs_dir', type=str, default='outputs/programs')
    p.add_argument('--check_only', action='store_true',
                   help='Only run data quality check, skip NMF sweep.')
    p.add_argument('--fast', action='store_true',
                   help='Use 3 stability runs instead of 5.')
    p.add_argument('--no_gpu', action='store_true')
    return p.parse_args()


# ── Part 1: Data quality check ────────────────────────────────────────────────

def check_data_quality(X: np.ndarray) -> dict:
    """
    Verify that the preprocessed matrix X is suitable for NMF.

    Returns a dict of diagnostics and a list of warnings.
    """
    n_cells, n_genes = X.shape
    warnings = []

    # Basic stats
    n_negative = (X < 0).sum()
    n_zero     = (X == 0).sum()
    sparsity   = n_zero / X.size
    val_min    = float(X.min())
    val_max    = float(X.max())
    val_mean   = float(X.mean())
    val_median = float(np.median(X))

    if n_negative > 0:
        warnings.append(
            f"CRITICAL: {n_negative:,} negative values found. "
            f"NMF requires non-negative input. "
            f"Check that log-normalization ran correctly."
        )

    # For log1p-normalized scRNA-seq: values should be in ~[0, 8]
    if val_max > 20:
        warnings.append(
            f"WARN: max value is {val_max:.2f} — unusually high for "
            f"log-normalized data (expected < 10). "
            f"Data may not be log-normalized, or outlier cells present."
        )
    if val_max < 0.5:
        warnings.append(
            f"CRITICAL: max value is {val_max:.4f} — near zero. "
            f"Data may be Z-scored (can be negative after centering) or "
            f"improperly normalized."
        )

    # Sparsity
    if sparsity < 0.3:
        warnings.append(
            f"WARN: sparsity is only {sparsity:.1%} — unusually dense for "
            f"scRNA-seq. Data may be pre-normalized or aggregated."
        )
    if sparsity > 0.99:
        warnings.append(
            f"WARN: sparsity is {sparsity:.1%} — extremely sparse. "
            f"HVG selection may not have filtered out near-zero genes."
        )

    # Per-cell coverage
    cells_per_gene = (X > 0).sum(axis=0)   # genes expressed across cells
    genes_per_cell = (X > 0).sum(axis=1)   # genes per cell
    n_zero_cells   = (genes_per_cell == 0).sum()
    n_zero_genes   = (cells_per_gene == 0).sum()

    if n_zero_cells > 0:
        warnings.append(
            f"WARN: {n_zero_cells:,} cells have zero expression across all "
            f"{n_genes:,} HVGs. These should have been filtered in QC."
        )
    if n_zero_genes > 0:
        warnings.append(
            f"WARN: {n_zero_genes:,} genes are zero in all cells. "
            f"These contribute nothing to NMF and inflate G."
        )

    # Variance check
    gene_var = X.var(axis=0)
    n_low_var = (gene_var < 1e-6).sum()
    if n_low_var > n_genes * 0.1:
        warnings.append(
            f"WARN: {n_low_var:,} / {n_genes:,} genes ({n_low_var/n_genes:.1%}) "
            f"have near-zero variance. NMF programs will be dominated by "
            f"the high-variance minority."
        )

    return {
        'n_cells':        n_cells,
        'n_genes':        n_genes,
        'n_negative':     int(n_negative),
        'sparsity':       float(sparsity),
        'val_min':        val_min,
        'val_max':        val_max,
        'val_mean':       val_mean,
        'val_median':     val_median,
        'median_genes_per_cell': float(np.median(genes_per_cell)),
        'median_cells_per_gene': float(np.median(cells_per_gene)),
        'n_zero_cells':   int(n_zero_cells),
        'n_zero_genes':   int(n_zero_genes),
        'n_low_var_genes': int(n_low_var),
        'warnings':       warnings,
    }


def print_data_quality(stats: dict):
    sep = '=' * 60
    print(f'\n{sep}')
    print('  DATA QUALITY REPORT')
    print(sep)
    print(f"  Matrix shape        : {stats['n_cells']:,} cells × {stats['n_genes']:,} genes")
    print(f"  Negative values     : {stats['n_negative']:,}  (should be 0)")
    print(f"  Sparsity            : {stats['sparsity']:.1%}  (typical scRNA-seq: 85–98%)")
    print(f"  Value range         : [{stats['val_min']:.4f}, {stats['val_max']:.2f}]")
    print(f"  Mean / Median       : {stats['val_mean']:.4f} / {stats['val_median']:.4f}")
    print(f"  Median genes/cell   : {stats['median_genes_per_cell']:.0f}")
    print(f"  Median cells/gene   : {stats['median_cells_per_gene']:.0f}")
    print(f"  Zero cells (all-0)  : {stats['n_zero_cells']:,}")
    print(f"  Zero genes (all-0)  : {stats['n_zero_genes']:,}")
    print(f"  Low-variance genes  : {stats['n_low_var_genes']:,}")

    if stats['warnings']:
        print(f'\n  ⚠  WARNINGS:')
        for w in stats['warnings']:
            print(f'    {w}')
    else:
        print('\n  ✓ No data quality issues detected.')
    print(f'{sep}\n')


# ── Part 2: K sweep ───────────────────────────────────────────────────────────

def run_k_sweep(
    X:          np.ndarray,
    k_values:   list,
    n_runs:     int  = 3,
    max_iter:   int  = 1000,
    tol:        float = 1e-3,
    use_gpu:    bool  = True,
) -> list:
    """
    Run NMF at each K and collect all verification metrics.
    Returns a list of dicts, one per K value.
    """
    from src.programs.nmf import fit_nmf
    from src.programs.verify import (
        compute_reconstruction_quality,
        compute_program_sparsity,
        compute_program_uniqueness,
        test_nmf_stability,
    )
    import anndata as ad

    results = []

    for K in k_values:
        logger.info(f'  Fitting NMF K={K}...')
        adata_tmp = ad.AnnData(X=X)
        model, W, H = fit_nmf(
            adata_tmp, n_programs=K, max_iter=max_iter,
            tol=tol, use_gpu=use_gpu,
        )

        recon     = compute_reconstruction_quality(X, W, H)
        sparsity  = compute_program_sparsity(H)
        unique    = compute_program_uniqueness(H)
        stability = test_nmf_stability(
            X, K, n_runs=n_runs, max_iter=max_iter,
            tol=tol, use_gpu=use_gpu,
        )

        row = {
            'K':              K,
            'fro_error':      round(recon['relative_frobenius_error'], 4),
            'median_r2':      round(recon['median_gene_r2'], 4),
            'mean_gini':      round(sparsity['mean_gini'], 4),
            'eff_genes':      round(sparsity['mean_effective_genes'], 1),
            'top10_share':    round(sparsity['top10_gene_fraction'], 3),
            'cosine_mean':    round(unique['mean_off_diag_cosine_sim'], 4),
            'near_dupes':     unique['n_near_duplicate_pairs'],
            'stability':      round(stability['mean_stability_score'], 4),
            'stab_label':     stability['interpretation'],
            'converged':      model.converged,
            'n_iter':         model.n_iter_,
        }
        results.append(row)

        # Print status after each K so user can monitor progress
        _print_sweep_row(row, header=(K == k_values[0]))

    return results


def _print_sweep_row(row: dict, header: bool = False):
    """Print one row of the sweep table, with optional header."""
    cols = [
        ('K',           5),
        ('fro_err',     8),
        ('gini',        7),
        ('eff_genes',   10),
        ('cosine',      8),
        ('dupes',       6),
        ('stability',   10),
        ('converged',   10),
        ('iters',       6),
    ]
    if header:
        hdr = '  ' + ''.join(f'{name:<{w}}' for name, w in cols)
        print(hdr)
        print('  ' + '-' * sum(w for _, w in cols))

    vals = [
        row['K'],
        row['fro_error'],
        row['mean_gini'],
        row['eff_genes'],
        row['cosine_mean'],
        row['near_dupes'],
        row['stability'],
        str(row['converged']),
        row['n_iter'],
    ]
    line = '  ' + ''.join(f'{str(v):<{w}}' for v, (_, w) in zip(vals, cols))

    # Flag rows that look promising
    promising = (
        row['fro_error'] < 0.5
        and row['mean_gini'] > 0.6
        and row['near_dupes'] <= 3
        and row['stability'] > 0.70
    )
    print(line + ('  ← candidate' if promising else ''))


def print_sweep_summary(results: list):
    """Print recommendations based on the sweep results."""
    sep = '=' * 60
    print(f'\n{sep}')
    print('  K SWEEP SUMMARY')
    print(sep)

    # Find all candidates (meeting relaxed criteria)
    candidates = [
        r for r in results
        if r['fro_error'] < 0.5
        and r['mean_gini'] > 0.6
        and r['near_dupes'] <= 3
        and r['stability'] > 0.70
    ]

    if candidates:
        best = max(candidates, key=lambda r: r['stability'])
        print(f'\n  ✓  Best candidate: K={best["K"]}')
        print(f'     fro_error={best["fro_error"]}  gini={best["mean_gini"]}  '
              f'near_dupes={best["near_dupes"]}  stability={best["stability"]}')
        print(f'\n  To use this K:')
        print(f'    python scripts/01_extract_programs.py '
              f'--from_checkpoint --k {best["K"]}')
    else:
        # Diagnose which metric is consistently failing
        print('\n  ✗  No K value passes all criteria. Metric breakdown:\n')

        avg_fro   = np.mean([r['fro_error'] for r in results])
        avg_gini  = np.mean([r['mean_gini'] for r in results])
        avg_dupes = np.mean([r['near_dupes'] for r in results])
        avg_stab  = np.mean([r['stability'] for r in results])

        checks = [
            ('Frobenius error < 0.5',       avg_fro,   avg_fro < 0.5,
             'Not enough K programs to capture variance. Try higher K.'),
            ('Gini > 0.6',                  avg_gini,  avg_gini > 0.6,
             'Programs too diffuse. Try higher alpha_H.'),
            ('Near-duplicate pairs ≤ 3',    avg_dupes, avg_dupes <= 3,
             'Too many redundant programs. Try lower K.'),
            ('Stability > 0.70',            avg_stab,  avg_stab > 0.70,
             'NMF solutions unstable. Data may have weak program structure.'),
        ]
        for name, val, passing, advice in checks:
            status = '✓' if passing else '✗'
            print(f'    {status}  {name:<35} avg={val:.3f}')
            if not passing:
                print(f'       → {advice}')

        # Offer the best available K even if not all criteria are met
        best_effort = max(
            results,
            key=lambda r: (
                (r['fro_error'] < 0.5) * 3
                + (r['mean_gini'] > 0.6) * 2
                + (r['near_dupes'] <= 3) * 2
                + r['stability']
            )
        )
        print(f'\n  Best available: K={best_effort["K"]} '
              f'(stability={best_effort["stability"]:.4f}, '
              f'fro_error={best_effort["fro_error"]:.4f})')
        print(f'  Consider using K={best_effort["K"]} and proceeding to Component 2.')
        print(f'  The strict thresholds were calibrated on a different dataset;')
        print(f'  raw count data typically has lower stability than pre-normalized data.')

    print(f'\n{sep}\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    use_gpu  = not args.no_gpu
    prog_dir = Path(args.programs_dir)
    n_runs   = 3 if args.fast else 5

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = prog_dir / 'checkpoint_X.npy'
    if not checkpoint.exists():
        logger.error(
            f"Checkpoint not found at '{checkpoint}'.\n"
            f"Run 01_extract_programs.py first to generate it."
        )
        sys.exit(1)

    logger.info(f"Loading preprocessed matrix from checkpoint...")
    X = np.load(str(checkpoint))
    logger.info(f"  X shape: {X.shape}")

    # ── Part 1: Data quality ──────────────────────────────────────────────────
    logger.info("Running data quality check...")
    stats = check_data_quality(X)
    print_data_quality(stats)

    if args.check_only:
        logger.info("--check_only set, skipping NMF sweep.")
        return

    # Exit early on critical data issues
    if stats['n_negative'] > 0:
        logger.error(
            "Critical data issue (negative values) — fix preprocessing before NMF."
        )
        sys.exit(1)

    if stats['val_max'] < 0.5:
        logger.error(
            "Critical data issue (near-zero max) — data may be incorrectly loaded."
        )
        sys.exit(1)

    # ── Part 2: K sweep ───────────────────────────────────────────────────────
    logger.info(
        f"Starting K sweep: {args.k_values}  "
        f"(stability_runs={n_runs}, fast={args.fast})"
    )
    print()

    results = run_k_sweep(
        X        = X,
        k_values = args.k_values,
        n_runs   = n_runs,
        use_gpu  = use_gpu,
    )

    print_sweep_summary(results)

    return results


if __name__ == '__main__':
    main()