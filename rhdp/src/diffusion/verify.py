"""
src/diffusion/verify.py

Four verification checks for the Component 2 heat diffusion outputs.

  [1] Coverage       — what fraction of perturbation genes have valid p_g?
  [2] Concentration  — does diffusion peak at the perturbed gene itself?
  [3] Beta sweep     — how sensitive are profiles to the β hyperparameter?
  [4] Program signal — do p_g vectors have meaningful spread across programs?
      (all-zero or all-uniform p_g means the diffusion found no program signal)
"""

import logging
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ── [1] Coverage ──────────────────────────────────────────────────────────────

def check_coverage(coverage_stats: dict) -> dict:
    """
    Report what fraction of perturbation genes are represented.

    'direct'    — gene is in HVG set: full one-hot diffusion.
    'neighbour' — gene is absent from HVGs but has HVG STRING-DB neighbours:
                  diffusion seeded at neighbourhood (weaker signal).
    'missing'   — gene absent from HVGs and from STRING-DB: zero p_g vector.
                  These perturbations will generalize poorly in Component 3.

    Pass threshold: >= 60% represented (direct + neighbour).
    """
    n    = coverage_stats['n_perturbations']
    direct = coverage_stats['n_direct']
    nbr  = coverage_stats['n_neighbour']
    miss = coverage_stats['n_missing']
    frac = coverage_stats['frac_represented']

    return {
        'n_perturbations': n,
        'n_direct':        direct,
        'n_neighbour':     nbr,
        'n_missing':       miss,
        'frac_represented': frac,
        'passes':          frac >= 0.60,
    }


# ── [2] Concentration ─────────────────────────────────────────────────────────

def check_diffusion_concentration(
    H_diffusion: np.ndarray,
    perturbation_genes: list,
    gene_to_idx: dict,
    seed_labels: list,
    top_k: int = 50,
) -> dict:
    """
    For each 'direct' perturbation gene, check that its diffusion profile
    assigns the highest score to itself (or near the top).

    A well-behaved heat diffusion should have the highest signal at the
    source gene, decaying with network distance. If the source gene is not
    in the top-k of its own diffusion profile, something is wrong with the
    graph or β is too large (diffusion over-spreads).

    Args:
        H_diffusion:        (G × n_perts) diffusion matrix.
        perturbation_genes: List of perturbed genes (columns of H_diffusion).
        gene_to_idx:        HVG gene → index mapping.
        seed_labels:        'direct'/'neighbour'/'missing' per perturbation.
        top_k:              How far down the ranked list to check.

    Returns:
        dict with fraction of direct genes that self-rank in top-k.
    """
    ranks = []
    for col, (gene, label) in enumerate(zip(perturbation_genes, seed_labels)):
        if label != 'direct':
            continue
        gene_idx = gene_to_idx[gene]
        scores   = H_diffusion[:, col]
        # rank of self (0 = highest)
        rank = int(np.sum(scores > scores[gene_idx]))
        ranks.append(rank)

    if not ranks:
        return {'n_checked': 0, 'frac_self_in_top_k': np.nan}

    frac_top_k = float(np.mean(np.array(ranks) < top_k))
    median_rank = float(np.median(ranks))

    logger.info(
        f"Concentration check: {len(ranks)} direct genes. "
        f"Median self-rank={median_rank:.0f}, "
        f"frac in top-{top_k}={frac_top_k:.2%}"
    )
    return {
        'n_checked':          len(ranks),
        'median_self_rank':   median_rank,
        f'frac_self_in_top_{top_k}': frac_top_k,
        'ranks':              ranks,
        'passes':             frac_top_k >= 0.50,
    }


# ── [3] Beta sweep ────────────────────────────────────────────────────────────

def run_beta_sweep(
    L,
    example_gene: str,
    gene_to_idx: dict,
    gene_names: list,
    beta_values: list = None,
    use_gpu: bool = True,
) -> dict:
    """
    Compute diffusion profiles for one gene at multiple β values.

    Produces a matrix (G × len(beta_values)) showing how concentration
    varies with β. Used to choose the right β before the full run.

    Expected behaviour:
        Small β (e.g. 0.01): very concentrated, only immediate neighbours lit up.
        Large β (e.g. 2.0):  very spread, most genes receive some signal.
        Good β (e.g. 0.1):   moderate spread — 1-3 hop neighbourhood.
    """
    if beta_values is None:
        beta_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]

    from src.diffusion.heat import compute_heat_diffusion

    if example_gene not in gene_to_idx:
        logger.warning(
            f"Beta sweep: '{example_gene}' not in HVG set. "
            f"Using first available gene."
        )
        example_gene = gene_names[0]

    idx = gene_to_idx[example_gene]
    e   = np.zeros((L.shape[0], 1), dtype=np.float32)
    e[idx, 0] = 1.0

    profiles = {}
    for beta in beta_values:
        h = compute_heat_diffusion(L, e, beta, use_gpu=use_gpu)
        profiles[beta] = h[:, 0]

    # Summarise: effective spread (number of genes receiving > 1% of max signal)
    sweep_summary = []
    for beta, h in profiles.items():
        threshold = 0.01 * h.max()
        n_active  = int((h > threshold).sum())
        sweep_summary.append({'beta': beta, 'n_active_genes': n_active,
                               'max_score': float(h.max()),
                               'entropy': float(-np.sum(p * np.log(p + 1e-12))
                                               if (p := h / (h.sum() + 1e-12)) is not None else 0)})

    summary_df = pd.DataFrame(sweep_summary)
    logger.info(f"\nBeta sweep for '{example_gene}':\n{summary_df.to_string(index=False)}")

    return {
        'gene':       example_gene,
        'profiles':   profiles,
        'summary_df': summary_df,
    }


# ── [4] Program signal ────────────────────────────────────────────────────────

def check_program_signal(P_matrix: np.ndarray, seed_labels: list) -> dict:
    """
    Check that p_g vectors have meaningful non-uniform signal across programs.

    A p_g vector of all zeros means the diffusion found no path to any program
    (isolated gene). A very uniform vector means the diffusion spread so far
    that all programs receive equal weight (β too large).

    We use the coefficient of variation (CV = std/mean) of each p_g row as
    a proxy for signal concentration in program space.

    Pass threshold: median CV > 0.3 across non-missing perturbations.
    """
    # Only check represented perturbations
    represented = [i for i, l in enumerate(seed_labels) if l != 'missing']
    if not represented:
        return {
            'n_checked':   0,
            'median_cv':   np.nan,
            'frac_zero':   np.nan,
            'cv_per_pert': np.array([]),
            'passes':      False,
        }

    P_rep = P_matrix[represented]
    row_means = P_rep.mean(axis=1)
    row_stds  = P_rep.std(axis=1)

    # Avoid division by zero for zero rows
    cv = np.where(row_means > 1e-10, row_stds / row_means, 0.0)

    frac_zero = float(np.mean(row_means < 1e-10))
    median_cv = float(np.median(cv))

    logger.info(
        f"Program signal: median CV={median_cv:.3f}, "
        f"frac zero-vector={frac_zero:.2%}"
    )

    return {
        'n_checked':   len(represented),
        'median_cv':   median_cv,
        'frac_zero':   frac_zero,
        'cv_per_pert': cv,
        'passes':      median_cv > 0.3 and frac_zero < 0.5,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_component2_summary(
    coverage:      dict,
    concentration: dict,
    program_signal: dict,
    beta_sweep:    Optional[dict] = None,
    P_matrix:      Optional[np.ndarray] = None,
    output_path:   Optional[str] = None,
) -> plt.Figure:
    """
    4-panel summary figure for Component 2 verification.

    Panel 1: Coverage breakdown (pie chart).
    Panel 2: Self-rank distribution for direct genes.
    Panel 3: β sweep — n_active_genes vs β.
    Panel 4: P_matrix program signal — distribution of CV values.
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: Coverage ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sizes  = [coverage['n_direct'], coverage['n_neighbour'], coverage['n_missing']]
    labels = ['Direct (in HVG)', 'Neighbour seed', 'Missing']
    colors = ['#2196F3', '#FF9800', '#F44336']
    wedges, texts, autotexts = ax1.pie(
        [max(s, 0.001) for s in sizes],
        labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
    )
    ax1.set_title(
        f'Coverage  (n={coverage["n_perturbations"]})\n'
        f'{100*coverage["frac_represented"]:.1f}% represented',
        fontsize=12, fontweight='bold'
    )

    # ── Panel 2: Self-rank distribution ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if concentration.get('n_checked', 0) > 0:
        ranks = concentration['ranks']
        ax2.hist(ranks, bins=40, color='steelblue', edgecolor='none', alpha=0.85)
        ax2.axvline(concentration['median_self_rank'], color='crimson',
                    lw=2, linestyle='--',
                    label=f"Median={concentration['median_self_rank']:.0f}")
        ax2.axvline(50, color='orange', lw=1.5, linestyle=':',
                    label='Top-50 threshold')
        ax2.set_xlabel('Self-rank in diffusion profile', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No direct genes to check',
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Diffusion Concentration\n(rank of source gene in its own profile)',
                  fontsize=12, fontweight='bold')

    # ── Panel 3: Beta sweep ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if beta_sweep is not None:
        df = beta_sweep['summary_df']
        ax3.plot(df['beta'], df['n_active_genes'], 'o-',
                 color='darkorange', lw=2, markersize=6)
        ax3.set_xscale('log')
        ax3.set_xlabel('β (diffusion scale)', fontsize=11)
        ax3.set_ylabel('Active genes (>1% of max signal)', fontsize=11)
        ax3.set_title(
            f'β Sweep — Gene: {beta_sweep["gene"]}\n'
            f'(choose β where curve bends = moderate spread)',
            fontsize=12, fontweight='bold'
        )
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Beta sweep not run\n(use --beta_sweep flag)',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('β Sweep', fontsize=12, fontweight='bold')

    # ── Panel 4: Program signal CV distribution ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if program_signal.get('n_checked', 0) > 0:
        cv = program_signal['cv_per_pert']
        ax4.hist(cv, bins=40, color='mediumseagreen', edgecolor='none', alpha=0.85)
        med = program_signal['median_cv']
        ax4.axvline(med, color='crimson', lw=2, linestyle='--',
                    label=f'Median CV={med:.3f}')
        ax4.axvline(0.3, color='orange', lw=1.5, linestyle=':',
                    label='Pass threshold (0.3)')
        ax4.set_xlabel('Coefficient of variation (p_g)', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.legend(fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax4.transAxes)
    ax4.set_title('Program Signal Quality\n(CV of p_g across K programs)',
                  fontsize=12, fontweight='bold')

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Component 2 verification figure saved: {output_path}")

    return fig


# ── Report ────────────────────────────────────────────────────────────────────

def print_component2_report(
    coverage:       dict,
    concentration:  dict,
    program_signal: dict,
    beta:           float,
    graph_data:     dict,
) -> None:
    sep = '=' * 62
    print(f'\n{sep}')
    print('  COMPONENT 2 VERIFICATION REPORT')
    print(f'  β = {beta}')
    print(sep)

    print('\n[1] COVERAGE')
    _row('Total perturbations',  coverage['n_perturbations'], fmt='d')
    _row('Direct (gene in HVG)', coverage['n_direct'],        fmt='d')
    _row('Neighbour seed',        coverage['n_neighbour'],     fmt='d')
    _row('Missing (zero p_g)',    coverage['n_missing'],       fmt='d')
    _row('Fraction represented',  coverage['frac_represented'], fmt='.1%',
         note='>= 60% to pass')
    _pass(coverage.get('passes', False))

    print('\n[2] DIFFUSION CONCENTRATION')
    if concentration.get('n_checked', 0) > 0:
        _row('Genes checked',   concentration['n_checked'],      fmt='d')
        _row('Median self-rank', concentration['median_self_rank'], fmt='.0f',
             note='lower is better')
        k = [k for k in concentration if k.startswith('frac_self')][0]
        _row(k, concentration[k], fmt='.1%', note='>= 50% to pass')
        _pass(concentration.get('passes', False))
    else:
        print('  No direct genes to evaluate.')

    print('\n[3] GRAPH STATISTICS')
    _row('HVG genes in graph',
         int(graph_data['connected_mask'].sum()), fmt='d')
    _row('Total HVG genes',
         len(graph_data['gene_names']), fmt='d')
    _row('Graph edges retained',
         len(graph_data['edges']), fmt='d')

    print('\n[4] PROGRAM SIGNAL')
    _row('Perturbations checked', program_signal['n_checked'],  fmt='d')
    _row('Median CV of p_g',      program_signal['median_cv'],  fmt='.4f',
         note='> 0.3 to pass')
    _row('Fraction zero-vector',  program_signal['frac_zero'],  fmt='.1%',
         note='< 50% to pass')
    _pass(program_signal.get('passes', False))

    print(f'\n{sep}\n')


def _row(label, value, fmt='', note=''):
    if fmt == 'd':
        val_str = str(int(value))
    elif fmt == '.1%':
        val_str = f'{float(value):.1%}'
    elif fmt == '.0f':
        val_str = f'{float(value):.0f}'
    else:
        val_str = format(float(value), fmt) if fmt else str(value)
    note_str = f'  ({note})' if note else ''
    print(f"  {label:<38}: {val_str}{note_str}")


def _pass(passed: bool):
    status = '✓  PASS' if passed else '✗  FAIL'
    print(f"  {'Status':<38}: {status}")