#!/usr/bin/env python
"""
plot_review_figures.py  — Figures generated in response to JAMES reviewer comments.

Figures produced:
  1. ffs_vs_direct_scatter.png  — log-log scatter k_FFS vs k_direct, 1:1 line,
                                   case-study ICs highlighted (Reviewer comment 1/3)
  2. rate_ratio_distribution.png — histogram of k_FFS/k_direct for all 98 ICs
                                   + speedup distribution (Reviewer comment 8)
  3. conditional_prob_boxplot.png — box/violin plot of P_i across all 98 ICs,
                                    shows interface spacing is well-conditioned
                                    (Reviewer comment 4)

Usage:
    python plot_review_figures.py \
        --ffs_csv /glade/derecho/scratch/schreck/FFS/results_mar18/ffs_statistics_all_ics.csv \
        --plot_dir /glade/derecho/scratch/schreck/FFS/results_mar18/plots/review_figures
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── colour palette matching existing paper figures ────────────────────────────
FFS_COLOR    = '#2166ac'   # blue
DIRECT_COLOR = '#762a83'   # purple
CASE_COLORS  = {'Earl':  '#d73027',   # red
                'Fiona': '#fc8d59',   # orange
                'Ian':   '#fee090'}   # yellow

IFACE_COLORS = ['#4393c3', '#2166ac', '#d6604d', '#b2182b']
IFACE_LABELS = [r'$P(\lambda_1|\lambda_0)$  initial org.',
                r'$P(\lambda_2|\lambda_1)$',
                r'$P(\lambda_3|\lambda_2)$',
                r'$P(\lambda_4|\lambda_3)$  final intens.']
P_COLS = ['lambda1_P_forward', 'lambda2_P_forward',
          'lambda3_P_forward', 'lambda4_P_forward']

CASE_ICS = {
    'Earl':  '2022-09-02 00:00:00',
    'Fiona': '2022-09-09 12:00:00',
    'Ian':   '2022-09-22 00:00:00',
}

plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8})


# ─────────────────────────────────────────────────────────────────────────────
# 1. k_FFS vs k_direct  (log-log scatter)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ffs_vs_direct(df: pd.DataFrame, plot_dir: Path) -> None:
    k_ffs    = df['ffs_rate_per_day'].values
    k_direct = df['direct_formation_rate_per_day'].values

    # Ratio and uncertainty for colouring
    ratio = k_ffs / np.where(k_direct > 0, k_direct, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # All ICs: colour-map by ratio
    sc = ax.scatter(k_direct, k_ffs,
                    c=ratio, cmap='RdYlGn', vmin=0.5, vmax=1.5,
                    s=40, alpha=0.75, linewidths=0.4,
                    edgecolors='#555', zorder=3, label='All ICs')

    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(r'$k_{\mathrm{FFS}}/k_{\mathrm{direct}}$', fontsize=10)
    cb.ax.axhline(1.0, color='k', lw=1, ls='--')

    # 1:1 reference line
    lims_lo = min(k_direct.min(), k_ffs.min()) * 0.7
    lims_hi = max(k_direct.max(), k_ffs.max()) * 1.4
    lims = [lims_lo, lims_hi]
    ax.plot(lims, lims, 'k--', lw=1.2, label='1:1', zorder=2)

    # ±50% bands
    ax.fill_between(lims, [l * 0.5 for l in lims], [l * 1.5 for l in lims],
                    color='grey', alpha=0.12, zorder=1)

    # Case-study ICs
    for label, ic_time in CASE_ICS.items():
        row = df[df['ic_time'] == ic_time]
        if row.empty:
            continue
        ax.scatter(row['direct_formation_rate_per_day'].values,
                   row['ffs_rate_per_day'].values,
                   color=CASE_COLORS[label], s=120, zorder=6,
                   edgecolors='k', linewidths=1.2, marker='*',
                   label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r'$k_{\mathrm{direct}}$  (day$^{-1}$)', fontsize=12)
    ax.set_ylabel(r'$k_{\mathrm{FFS}}$  (day$^{-1}$)', fontsize=12)
    ax.set_title('FFS vs. Direct-Sampling Genesis Rate\n'
                 'All 98 initial conditions, 2022 Atlantic season',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', alpha=0.3, lw=0.5)

    # Annotate mean ratio
    valid = ratio[np.isfinite(ratio)]
    ax.text(0.97, 0.05,
            f'Mean ratio: {valid.mean():.3f} ± {valid.std():.3f}\n'
            f'Median: {np.median(valid):.3f}\n'
            f'Range: [{valid.min():.2f}, {valid.max():.2f}]',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                  fc='white', ec='#aaa', alpha=0.9))

    plt.tight_layout()
    out = plot_dir / 'ffs_vs_direct_scatter.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rate-ratio histogram  +  speedup distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame, plot_dir: Path) -> None:
    k_ffs    = df['ffs_rate_per_day'].values
    k_direct = df['direct_formation_rate_per_day'].values
    speedup  = df['ffs_speedup'].values

    ratio = k_ffs / np.where(k_direct > 0, k_direct, np.nan)
    ratio = ratio[np.isfinite(ratio)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── panel A: ratio histogram ──────────────────────────────────────────────
    ax = axes[0]
    bins_r = np.linspace(0.4, 1.8, 28)
    ax.hist(ratio, bins=bins_r, color=FFS_COLOR, edgecolor='white',
            linewidth=0.5, alpha=0.85)
    ax.axvline(1.0,            color='k',       lw=1.5, ls='--', label='Ideal (1.0)')
    ax.axvline(ratio.mean(),   color='#d73027', lw=1.5, ls='-',
               label=f'Mean = {ratio.mean():.3f}')
    ax.axvline(np.median(ratio), color='#fc8d59', lw=1.5, ls='-.',
               label=f'Median = {np.median(ratio):.3f}')

    # shade ±50 % band
    ax.axvspan(0.5, 1.5, color='grey', alpha=0.12, zorder=0, label='±50% band')

    ax.set_xlabel(r'$k_{\mathrm{FFS}} / k_{\mathrm{direct}}$', fontsize=12)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Rate Ratio Distribution\n(all 98 ICs)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.95,
            f'N = {len(ratio)}\n'
            f'Within 50%: {(np.abs(ratio-1)<0.5).sum()}/{len(ratio)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa', alpha=0.9))

    # ── panel B: speedup distribution (log) ──────────────────────────────────
    ax = axes[1]
    log_sp = np.log10(speedup[np.isfinite(speedup) & (speedup > 0)])
    bins_s = np.linspace(0, 2.5, 26)
    ax.hist(log_sp, bins=bins_s, color='#4dac26', edgecolor='white',
            linewidth=0.5, alpha=0.85)

    gm = np.exp(np.mean(np.log(speedup[np.isfinite(speedup) & (speedup > 0)])))
    ax.axvline(np.log10(gm),           color='#d73027', lw=1.5, ls='-',
               label=f'Geom. mean = {gm:.0f}×')
    ax.axvline(np.log10(np.mean(speedup[np.isfinite(speedup) & (speedup > 0)])),
               color='#fc8d59', lw=1.5, ls='--',
               label=f'Arith. mean = {np.mean(speedup[np.isfinite(speedup)]):.0f}×')

    ax.set_xlabel(r'$\log_{10}$(Speedup)', fontsize=12)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Computational Enhancement Factor\n(all 98 ICs)', fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{10**x:.0f}×' if x >= 1 else f'{10**x:.1f}×'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    sp_valid = speedup[np.isfinite(speedup) & (speedup > 0)]
    ax.text(0.97, 0.95,
            f'Range: {sp_valid.min():.1f}–{sp_valid.max():.0f}×\n'
            f'Geom. mean: {gm:.0f}×\nMedian: {np.median(sp_valid):.0f}×',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaa', alpha=0.9))

    plt.suptitle(r'Internal Validation: $k_{\mathrm{FFS}}$ vs $k_{\mathrm{direct}}$',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = plot_dir / 'rate_ratio_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Conditional probability distributions across all 98 ICs (violin + box)
# ─────────────────────────────────────────────────────────────────────────────

def plot_conditional_prob_distributions(df: pd.DataFrame, plot_dir: Path) -> None:
    """
    Shows P_i distributions across all 98 ICs.
    Addresses reviewer comment on interface conditioning:
    well-conditioned probabilities are neither near 0 nor near 1.
    """
    data = [df[col].dropna().values for col in P_COLS]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    positions = np.arange(1, 5)

    # Violin
    vp = ax.violinplot(data, positions=positions,
                       showmedians=True, showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(IFACE_COLORS[i])
        body.set_alpha(0.55)
        body.set_edgecolor('#333')
        body.set_linewidth(0.8)
    vp['cmedians'].set_colors(['#111'] * 4)
    vp['cmedians'].set_linewidth(2)

    # Box overlay
    bp = ax.boxplot(data, positions=positions, widths=0.25,
                    patch_artist=False, showfliers=True,
                    medianprops=dict(color='none'),
                    whiskerprops=dict(color='#444', lw=1),
                    capprops=dict(color='#444', lw=1),
                    boxprops=dict(color='#444', lw=1.2),
                    flierprops=dict(marker='o', markersize=3,
                                    color='#999', alpha=0.4))

    # Optimal-conditioning reference band (1/e ≈ 0.37 ± sensible range)
    ax.axhspan(0.25, 0.75, color='#b8d4e8', alpha=0.25, zorder=0,
               label='Well-conditioned range (0.25–0.75)')
    ax.axhline(1/np.e, color='#2166ac', lw=1.2, ls='--', alpha=0.7,
               label=r'Optimal $P = 1/e \approx 0.37$ (Allen et al.\ 2009)')

    # Case study markers
    for label, ic_time in CASE_ICS.items():
        row = df[df['ic_time'] == ic_time]
        if row.empty:
            continue
        vals = [row[c].values[0] for c in P_COLS]
        ax.plot(positions, vals, 'o--', color=CASE_COLORS[label],
                lw=1.5, ms=9, markeredgecolor='k', markeredgewidth=0.8,
                label=label, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([r'$P_1$'+'\n'+r'$\lambda_0{\to}\lambda_1$',
                        r'$P_2$'+'\n'+r'$\lambda_1{\to}\lambda_2$',
                        r'$P_3$'+'\n'+r'$\lambda_2{\to}\lambda_3$',
                        r'$P_4$'+'\n'+r'$\lambda_3{\to}\lambda_B$'],
                       fontsize=10)
    ax.set_ylabel('Conditional crossing probability', fontsize=11)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title('Conditional Probability Distributions Across All 98 ICs\n'
                 'Violin = full distribution; box = quartiles; '
                 'markers = case studies', fontsize=10)
    ax.legend(fontsize=9, loc='lower center')
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate means
    for i, (pos, col) in enumerate(zip(positions, P_COLS)):
        m = df[col].mean()
        s = df[col].std()
        ax.text(pos, 0.97, f'{m:.2f}±{s:.2f}',
                ha='center', va='bottom', fontsize=10, color='#333')

    plt.tight_layout()
    out = plot_dir / 'conditional_prob_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Generate JAMES review-response figures from FFS CSV.')
    ap.add_argument('--ffs_csv', required=True,
                    help='ffs_statistics_all_ics.csv')
    ap.add_argument('--plot_dir', default='./plots/review_figures')
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ffs_csv)
    print(f'Loaded {len(df)} ICs from {args.ffs_csv}')

    plot_ffs_vs_direct(df, plot_dir)
    plot_distributions(df, plot_dir)
    plot_conditional_prob_distributions(df, plot_dir)

    print(f'\nAll review figures saved to: {plot_dir}')


if __name__ == '__main__':
    main()
