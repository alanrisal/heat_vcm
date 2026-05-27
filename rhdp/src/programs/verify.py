"""
src/programs/verify.py

Four independent verification checks for NMF gene programs.

Each check targets a specific failure mode described in the architecture notes:

  [1] Reconstruction Quality — Did NMF actually capture the variance in the data?
      Failure mode: if programs can't reconstruct control cells, the program
      coordinate system is wrong and all downstream predictions will be noisy.

  [2] Program Sparsity — Are programs focused on specific genes or diffuse?
      Failure mode: diffuse programs spread weight across hundreds of genes
      uniformly, making biological interpretation impossible and downstream
      gene-level decoding inaccurate.

  [3] Program Uniqueness — Are programs distinct from each other?
      Failure mode: near-duplicate programs waste capacity on the same biology
      and signal that K is too large for this dataset.

  [4] NMF Stability — Does NMF converge to the same programs from different seeds?
      Failure mode: unstable NMF means the programs are not a reliable feature
      of the data — they are fitting noise, and will not generalize.

  [5] Pathway Enrichment — Do the top genes of each program correspond to known biology?
      This is the ultimate biological ground-truth check. Uses gseapy + Enrichr.
      Requires internet access.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ── [1] Reconstruction Quality ────────────────────────────────────────────────

def compute_reconstruction_quality(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
) -> dict:
    """
    Measure how faithfully NMF reconstructs the original expression matrix.

    Metrics:
        relative_frobenius_error:
            ||X - WH||_F / ||X||_F
            Fraction of total matrix magnitude not captured.
            < 0.5 is acceptable; < 0.3 is good for single-cell data.

        mean/median_gene_r2:
            Per-gene R² — fraction of each gene's variance across cells
            explained by the NMF reconstruction.
            Median > 0.4 is acceptable; > 0.6 is good.

        frac_genes_r2_above_0.5:
            Fraction of genes where at least half the variance is explained.
    """
    X_hat = W @ H
    fro_error = np.linalg.norm(X - X_hat, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)

    ss_res = np.sum((X - X_hat) ** 2, axis=0)
    ss_tot = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, np.nan)

    return {
        'relative_frobenius_error': float(fro_error),
        'mean_gene_r2': float(np.nanmean(r2)),
        'median_gene_r2': float(np.nanmedian(r2)),
        'frac_genes_r2_above_0.5': float(np.nanmean(r2 > 0.5)),
        'r2_per_gene': r2,
    }


# ── [2] Program Sparsity ──────────────────────────────────────────────────────

def compute_program_sparsity(H: np.ndarray) -> dict:
    """
    Assess how focused (sparse) each program is in gene space.

    A biologically meaningful program should have a small set of high-loading
    genes. A diffuse program that spreads weight uniformly across thousands of
    genes is not a coherent biological signal.

    Metrics:
        gini_per_program:
            Gini coefficient of loading weights per program (0=uniform, 1=all
            weight on one gene). Well-defined programs typically score > 0.7.

        effective_genes_per_program:
            Entropy-based effective number of genes: exp(H(p)) where p is the
            normalized loading distribution. Lower = more concentrated.
            Good programs typically have < 200 effective genes out of 5000.

        top10/top50_gene_fraction:
            Fraction of total loading weight held by the top 10 or 50 genes.
            Good programs: top-10 genes hold > 20% of weight.
    """
    K, G = H.shape
    gini_scores, eff_genes, top10_frac, top50_frac = [], [], [], []

    for k in range(K):
        h = H[k]
        total = h.sum()
        if total < 1e-12:
            gini_scores.append(np.nan)
            eff_genes.append(np.nan)
            top10_frac.append(np.nan)
            top50_frac.append(np.nan)
            continue

        h_norm = h / total

        # Gini coefficient (sorting-based formula)
        s = np.sort(h_norm)
        n = len(s)
        gini = (2 * np.dot(np.arange(1, n + 1), s) - (n + 1)) / n
        gini_scores.append(float(gini))

        # Effective number of genes (entropy-based)
        nz = h_norm[h_norm > 0]
        entropy = -np.sum(nz * np.log(nz + 1e-12))
        eff_genes.append(float(np.exp(entropy)))

        # Top-N weight fractions
        sorted_h = np.sort(h)[::-1]
        top10_frac.append(float(sorted_h[:10].sum() / total))
        top50_frac.append(float(sorted_h[:50].sum() / total))

    return {
        'gini_per_program': np.array(gini_scores),
        'mean_gini': float(np.nanmean(gini_scores)),
        'effective_genes_per_program': np.array(eff_genes),
        'mean_effective_genes': float(np.nanmean(eff_genes)),
        'top10_gene_fraction': float(np.nanmean(top10_frac)),
        'top50_gene_fraction': float(np.nanmean(top50_frac)),
    }


# ── [3] Program Uniqueness ────────────────────────────────────────────────────

def compute_program_uniqueness(H: np.ndarray) -> dict:
    """
    Check that programs are not duplicating each other's information.

    Two programs with cosine similarity > 0.8 are covering essentially the
    same gene set — one of them is wasted capacity.

    Metrics:
        cosine_sim_matrix:   Full (K, K) pairwise cosine similarity matrix.
        mean_off_diag:       Mean of off-diagonal elements. < 0.3 is good.
        max_off_diag:        Highest similarity between any two programs.
        n_near_duplicates:   Count of pairs with cosine sim > 0.8. Should be 0.
    """
    sim = cosine_similarity(H)  # (K, K)
    K = H.shape[0]

    mask = ~np.eye(K, dtype=bool)
    off_diag = sim[mask]

    # Count unique pairs (matrix is symmetric, don't double-count)
    upper_off_diag = sim[np.triu_indices(K, k=1)]
    n_dups = int(np.sum(upper_off_diag > 0.8))

    return {
        'cosine_sim_matrix': sim,
        'mean_off_diag_cosine_sim': float(off_diag.mean()),
        'max_off_diag_cosine_sim': float(off_diag.max()),
        'n_near_duplicate_pairs': n_dups,
    }


# ── [4] NMF Stability ─────────────────────────────────────────────────────────

def test_nmf_stability(
    X: np.ndarray,
    n_programs: int,
    n_runs: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-3,
    use_gpu: bool = True,
) -> dict:
    """
    Test whether NMF produces consistent programs across independent random seeds.

    Calls fit_nmf() directly so stability runs benefit from GPU acceleration.

    Strategy:
        Run NMF n_runs times with different seeds. For each pair of runs,
        greedily match programs by maximum cosine similarity. The mean matched
        similarity is the stability score.

    Thresholds:
        > 0.85  — stable.
        0.75–0.85 — acceptable.
        < 0.75  — unstable. Reduce K and retest.

    Args:
        X:          Expression matrix (n_cells, n_genes).
        n_programs: K to test.
        n_runs:     Number of independent NMF fits.
        max_iter:   Should match the main fit's max_iter.
        tol:        Should match the main fit's tol.
        use_gpu:    Passed through to fit_nmf.

    Returns:
        Dict with mean_stability_score, pairwise_scores, interpretation.
    """
    # Import here to avoid circular dependency at module level
    from src.programs.nmf import fit_nmf

    logger.info(f"Running NMF {n_runs}× with K={n_programs} to assess stability...")

    H_runs = []
    for i in range(n_runs):
        _, _, H_i = fit_nmf(
            # Wrap X in a minimal AnnData so fit_nmf's interface is unchanged
            __import__('anndata').AnnData(X=X),
            n_programs=n_programs,
            random_state=i * 137,
            max_iter=max_iter,
            tol=tol,
            use_gpu=use_gpu,
        )
        H_runs.append(H_i)

    pairwise_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            sim = cosine_similarity(H_runs[i], H_runs[j])  # (K, K)

            # Greedy bipartite matching: assign each program in run i
            # to its best available match in run j.
            matched = []
            available = list(range(sim.shape[1]))
            for k in range(sim.shape[0]):
                if not available:
                    break
                best_idx = max(available, key=lambda idx: sim[k, idx])
                matched.append(sim[k, best_idx])
                available.remove(best_idx)

            pairwise_scores.append(float(np.mean(matched)))

    mean_score = float(np.mean(pairwise_scores))
    interpretation = (
        'stable'
        if mean_score > 0.85 else
        'acceptable'
        if mean_score > 0.75 else
        'unstable — consider reducing K'
    )

    logger.info(
        f"Stability score: {mean_score:.3f} ({interpretation})"
    )

    return {
        'mean_stability_score': mean_score,
        'pairwise_scores': pairwise_scores,
        'interpretation': interpretation,
    }


# ── [5] Pathway Enrichment ────────────────────────────────────────────────────

def run_pathway_enrichment(
    top_genes_df: pd.DataFrame,
    n_top: int = 100,
    gene_sets: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query Enrichr for biological term enrichment of each program's top genes.

    Requires internet access and gseapy (pip install gseapy).

    A program whose top genes are enriched for a coherent pathway (e.g.,
    "Cell Cycle", "Ribosome Biogenesis", "Interferon Signaling") is
    biologically interpretable and trustworthy. A program with no significant
    enrichment is either noise or represents biology not captured by curated
    pathway databases.

    Args:
        top_genes_df: DataFrame [program_id, gene, loading_score].
        n_top:        Number of top genes per program to submit.
        gene_sets:    Enrichr gene set databases to query.
        output_dir:   If set, save results to CSV here.

    Returns:
        DataFrame of significant (adj.p < 0.05) enrichment results.
        Columns include: program_id, Term, Adjusted P-value, Overlap, Genes.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.error(
            "gseapy is required for pathway enrichment. "
            "Install it with: pip install gseapy"
        )
        return pd.DataFrame()

    if gene_sets is None:
        gene_sets = [
            'MSigDB_Hallmark_2020',
            'KEGG_2021_Human',
            'GO_Biological_Process_2023',
        ]

    all_results = []
    program_ids = sorted(top_genes_df['program_id'].unique())

    for prog_id in program_ids:
        gene_list = (
            top_genes_df[top_genes_df['program_id'] == prog_id]
            .nlargest(n_top, 'loading_score')['gene']
            .tolist()
        )

        if len(gene_list) < 5:
            logger.warning(f"Program {prog_id}: fewer than 5 genes, skipping enrichment.")
            continue

        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism='human',
                outdir=None,
                verbose=False,
            )
            results = enr.results.copy()
            results['program_id'] = prog_id
            sig = results[results['Adjusted P-value'] < 0.05].copy()
            if not sig.empty:
                all_results.append(sig)
        except Exception as e:
            logger.warning(f"Enrichment failed for program {prog_id}: {e}")

    if not all_results:
        logger.warning("Pathway enrichment: no significant results found across all programs.")
        return pd.DataFrame()

    enrichment_df = pd.concat(all_results, ignore_index=True)
    enrichment_df = enrichment_df.sort_values(['program_id', 'Adjusted P-value'])

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        enrichment_df.to_csv(out / 'pathway_enrichment.csv', index=False)
        logger.info(f"Enrichment results saved to '{out}/pathway_enrichment.csv'")

    return enrichment_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_verification_summary(
    recon_metrics: dict,
    sparsity_metrics: dict,
    uniqueness_metrics: dict,
    stability_metrics: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel diagnostic figure covering all verification checks.

    Panel 1 (top-left):  Per-gene R² distribution — reconstruction quality.
    Panel 2 (top-right): Gini coefficient distribution — program sparsity.
    Panel 3 (bot-left):  Program-program cosine similarity heatmap — uniqueness.
    Panel 4 (bot-right): Stability scores across run pairs.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Per-gene R² ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    r2 = recon_metrics['r2_per_gene']
    r2_clean = r2[~np.isnan(r2)]
    ax1.hist(r2_clean, bins=60, color='steelblue', edgecolor='none', alpha=0.85)
    med = float(np.median(r2_clean))
    ax1.axvline(med, color='crimson', lw=1.8, linestyle='--', label=f'Median = {med:.3f}')
    ax1.axvline(0.3, color='orange', lw=1.2, linestyle=':', label='Acceptable threshold (0.3)')
    ax1.set_xlabel('Per-gene R²', fontsize=12)
    ax1.set_ylabel('Number of genes', fontsize=12)
    ax1.set_title('Reconstruction Quality', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)

    # ── Panel 2: Gini coefficients ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gini = sparsity_metrics['gini_per_program']
    ax2.hist(gini[~np.isnan(gini)], bins=20, color='darkorange', edgecolor='none', alpha=0.85)
    mean_g = float(np.nanmean(gini))
    ax2.axvline(mean_g, color='crimson', lw=1.8, linestyle='--', label=f'Mean = {mean_g:.3f}')
    ax2.axvline(0.7, color='green', lw=1.2, linestyle=':', label='Good threshold (0.7)')
    ax2.set_xlabel('Gini coefficient (higher = more focused)', fontsize=12)
    ax2.set_ylabel('Number of programs', fontsize=12)
    ax2.set_title('Program Sparsity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)

    # ── Panel 3: Cosine similarity heatmap ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sim = uniqueness_metrics['cosine_sim_matrix']
    K = sim.shape[0]

    # Subsample if K is large to keep heatmap readable (≤ 25 programs shown)
    step = max(1, K // 25)
    idx = list(range(0, K, step))
    sub_sim = sim[np.ix_(idx, idx)]

    im = ax3.imshow(sub_sim, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax3, label='Cosine similarity', shrink=0.9)
    ax3.set_title(
        f'Program Uniqueness\n'
        f'mean off-diag={uniqueness_metrics["mean_off_diag_cosine_sim"]:.3f}  '
        f'| near-duplicates={uniqueness_metrics["n_near_duplicate_pairs"]}',
        fontsize=12, fontweight='bold'
    )
    ax3.set_xlabel('Program index (subsampled)', fontsize=11)
    ax3.set_ylabel('Program index (subsampled)', fontsize=11)

    # ── Panel 4: Stability scores ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    scores = stability_metrics['pairwise_scores']
    bars = ax4.bar(range(len(scores)), scores, color='mediumseagreen', edgecolor='none', alpha=0.85)
    ax4.axhline(0.85, color='green',  lw=1.5, linestyle='--', label='Stable threshold (0.85)')
    ax4.axhline(0.75, color='orange', lw=1.2, linestyle='--', label='Acceptable threshold (0.75)')
    mean_s = stability_metrics['mean_stability_score']
    ax4.axhline(mean_s, color='crimson', lw=1.8, linestyle='-', label=f'Mean = {mean_s:.3f}')
    ax4.set_xlabel('Run pair index', fontsize=12)
    ax4.set_ylabel('Mean matched cosine similarity', fontsize=12)
    ax4.set_ylim(0, 1.05)
    ax4.set_title(
        f'NMF Stability — {stability_metrics["interpretation"].upper()}',
        fontsize=12, fontweight='bold'
    )
    ax4.legend(fontsize=9)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Verification figure saved to '{output_path}'")

    return fig


# ── Report ────────────────────────────────────────────────────────────────────

def print_verification_report(
    recon_metrics: dict,
    sparsity_metrics: dict,
    uniqueness_metrics: dict,
    stability_metrics: dict,
    enrichment_df: Optional[pd.DataFrame] = None,
    n_programs: Optional[int] = None,
) -> None:
    """Print a structured, human-readable verification summary."""
    sep = '=' * 62
    print(f'\n{sep}')
    print('  GENE PROGRAM VERIFICATION REPORT')
    if n_programs:
        print(f'  K = {n_programs} programs')
    print(sep)

    # [1] Reconstruction
    print('\n[1] RECONSTRUCTION QUALITY')
    _row('Relative Frobenius error', recon_metrics['relative_frobenius_error'],
         fmt='.4f', note='< 0.5 acceptable')
    _row('Mean per-gene R²',         recon_metrics['mean_gene_r2'],          fmt='.4f')
    _row('Median per-gene R²',       recon_metrics['median_gene_r2'],        fmt='.4f', note='> 0.4 good')
    _row('Genes with R² > 0.5',      recon_metrics['frac_genes_r2_above_0.5'], fmt='.1%')

    # [2] Sparsity
    print('\n[2] PROGRAM SPARSITY')
    _row('Mean Gini coefficient',      sparsity_metrics['mean_gini'],            fmt='.4f', note='> 0.7 good')
    _row('Mean effective genes/prog',  sparsity_metrics['mean_effective_genes'], fmt='.1f', note='< 200 good')
    _row('Top-10 genes share',         sparsity_metrics['top10_gene_fraction'],  fmt='.1%')
    _row('Top-50 genes share',         sparsity_metrics['top50_gene_fraction'],  fmt='.1%')

    # [3] Uniqueness
    print('\n[3] PROGRAM UNIQUENESS')
    _row('Mean inter-program cosine sim', uniqueness_metrics['mean_off_diag_cosine_sim'], fmt='.4f', note='< 0.3 good')
    _row('Max inter-program cosine sim',  uniqueness_metrics['max_off_diag_cosine_sim'],  fmt='.4f')
    _row('Near-duplicate pairs (>0.8)',   uniqueness_metrics['n_near_duplicate_pairs'],   fmt='d',   note='0 is ideal')

    # [4] Stability
    print('\n[4] NMF STABILITY')
    _row('Mean stability score', stability_metrics['mean_stability_score'], fmt='.4f')
    print(f"  {'Assessment':<30}: {stability_metrics['interpretation'].upper()}")

    # [5] Enrichment
    print('\n[5] PATHWAY ENRICHMENT')
    if enrichment_df is not None and not enrichment_df.empty:
        print('  Top enriched term per program (first 15 programs):')
        for pid in sorted(enrichment_df['program_id'].unique())[:15]:
            row = enrichment_df[enrichment_df['program_id'] == pid].iloc[0]
            term = str(row['Term'])[:48]
            pval = float(row['Adjusted P-value'])
            print(f"    Prog {pid:2d}: {term:<48}  adj.p={pval:.1e}")
    else:
        print('  Not run or no significant results found.')

    print(f'\n{sep}\n')


def _row(label: str, value, fmt: str = '.4f', note: str = '') -> None:
    """Helper to print a single report row."""
    if fmt == 'd':
        val_str = str(int(value))
    elif fmt == '.1%':
        val_str = f'{value:.1%}'
    else:
        val_str = format(float(value), fmt)
    note_str = f'  ({note})' if note else ''
    print(f"  {label:<35}: {val_str}{note_str}")