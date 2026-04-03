#!/usr/bin/env python
"""
Bottleneck interface heatmap — Figure 3.

The committor curve (Fig. 4 in the paper) shows that the 1000→988 hPa interval
is the rate-limiting step for the Earl case study.  This script extends that
analysis to ALL 18 ICs by showing, for each IC × interface pair, the conditional
transition probability P(λᵢ₊₁ | λᵢ) — and marks which interface is the
bottleneck (minimum probability) for each IC.

Key scientific insight: is the bottleneck always at the same intensification
stage, or does it shift with the environmental regime?

Layout
------
Main panel  : heatmap (ICs × interfaces), colour = P_forward (log scale optional)
Right strip : bar showing which interface is the bottleneck per IC
Bottom strip: mean P_forward at each interface (mean committor curve projection)

Usage:
    python plot_bottleneck_heatmap.py \\
        --ffs_csv results/ffs_statistics_all_ics.csv \\
        --plot_dir results/plots \\
        [--log_scale]      # colour axis on log scale (recommended)
        [--sort_by_bottleneck]   # group ICs by their bottleneck interface
        [--highlight_ic 2022-08-21T00Z]
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

# ── Interface definition — auto-detected from CSV ────────────────────────────

def _detect_n_transitions(df: pd.DataFrame) -> int:
    """Count lambda{i}_P_forward columns to determine number of interfaces."""
    return len([c for c in df.columns if c.startswith('lambda') and c.endswith('_P_forward')])


def load_transition_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the N_IC × N_TRANSITIONS matrix of forward conditional probabilities.

    The FFS CSV stores columns lambda{i}_P_forward for i = 1…N.
    These are P(λᵢ | λᵢ₋₁) — the probability of advancing given λᵢ₋₁ is crossed.

    Returns
    -------
    probs : ndarray of shape (n_ics, N_TRANSITIONS)
        probs[ic, j] = P(λⱼ₊₁ | λⱼ)  for j = 0…N-1
    """
    n = _detect_n_transitions(df)
    cols = [f'lambda{i}_P_forward' for i in range(1, n + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in FFS CSV: {missing}")
    return df[cols].values.astype(float)


def find_bottleneck(probs: np.ndarray) -> np.ndarray:
    """
    Return the index (0…4) of the transition with the minimum probability per IC.

    For NaN entries (interface not reached), set to NaN.
    """
    bn = np.full(probs.shape[0], np.nan)
    for i, row in enumerate(probs):
        valid = np.isfinite(row) & (row > 0)
        if valid.any():
            bn[i] = np.argmin(np.where(valid, row, np.inf))
    return bn


def make_bottleneck_figure(df_ffs, plot_dir, log_scale=False,
                            sort_by_bottleneck=False, highlight_ic=None, stride=1):
    """
    Build the two-panel bottleneck heatmap figure.
    """
    probs    = load_transition_probs(df_ffs)           # (n_ics, N_TRANSITIONS)
    n_transitions = probs.shape[1]
    iface_labels = [f'λ{i}→λ{i+1}' for i in range(n_transitions)]
    bottleneck_colors = plt.cm.tab10(np.linspace(0, 0.5, n_transitions))
    ic_times = pd.to_datetime(df_ffs['ic_time'])

    # ── Drop ICs without complete shooting data ───────────────────────────────
    # An IC is complete when all 5 P_forward values are finite and positive.
    complete = np.all(np.isfinite(probs) & (probs > 0), axis=1)
    probs    = probs[complete]
    ic_times = ic_times[complete].reset_index(drop=True)
    print(f"  {complete.sum()} ICs with complete shooting data (dropped {(~complete).sum()})")

    # ── Subsample by stride ───────────────────────────────────────────────────
    if stride > 1:
        probs    = probs[::stride]
        ic_times = ic_times.iloc[::stride].reset_index(drop=True)
        print(f"  Keeping every {stride}rd/th IC → {len(ic_times)} shown")

    n_ics    = probs.shape[0]

    bn    = find_bottleneck(probs)                  # bottleneck interface idx per IC
    labels   = ic_times.dt.strftime('%b %d %HZ').tolist()

    # ── Sorting ───────────────────────────────────────────────────────────────
    if sort_by_bottleneck:
        order = np.lexsort((ic_times.values, bn))
    else:
        order = np.arange(n_ics)

    probs_sorted  = probs[order]
    bn_sorted     = bn[order]
    labels_sorted = [labels[i] for i in order]
    dof_sorted    = ic_times.iloc[order].dt.day_of_year.values.astype(float)

    # ── Highlight Earl IC ─────────────────────────────────────────────────────
    earl_idx_original = None
    if highlight_ic is not None:
        try:
            earl_dt = pd.to_datetime(highlight_ic)
            earl_idx_original = (ic_times - earl_dt).abs().idxmin()
        except Exception:
            pass

    earl_idx_sorted = None
    if earl_idx_original is not None and earl_idx_original in order.tolist():
        earl_idx_sorted = order.tolist().index(earl_idx_original)

    # ── Colour scale ──────────────────────────────────────────────────────────
    vmin  = np.nanmin(probs_sorted[probs_sorted > 0])
    vmax  = 1.0
    if log_scale:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-4), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

    cmap = plt.cm.RdYlGn   # red (low, bad) → yellow → green (high, easy)

    # ── Figure layout ─────────────────────────────────────────────────────────
    row_h   = max(0.18 * n_ics, 5)      # heatmap height in inches
    fig_h   = row_h + 3.5               # + room for bar chart
    fig = plt.figure(figsize=(15, fig_h))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[11, 1.3],
        height_ratios=[n_ics, 5],
        hspace=0.12,
        wspace=0.04,
        left=0.11, right=0.96, top=0.95, bottom=0.04,
    )

    ax_heat = fig.add_subplot(gs[0, 0])   # main heatmap
    ax_bn   = fig.add_subplot(gs[0, 1])   # bottleneck strip
    ax_mean = fig.add_subplot(gs[1, 0])   # mean P_forward bar chart
    ax_leg  = fig.add_subplot(gs[1, 1])   # bottleneck legend (bottom-right)

    # ── Main heatmap ──────────────────────────────────────────────────────────
    im = ax_heat.imshow(
        probs_sorted,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='none',
    )

    # Grid lines
    for j in range(n_transitions + 1):
        ax_heat.axvline(j - 0.5, color='white', linewidth=0.5)
    for i in range(n_ics):
        ax_heat.axhline(i - 0.5, color='white', linewidth=0.3, alpha=0.5)

    # Mark bottleneck cell with bold border
    for i, j in enumerate(bn_sorted):
        if np.isfinite(j):
            j_int = int(j)
            rect  = plt.Rectangle(
                (j_int - 0.5, i - 0.5), 1, 1,
                linewidth=2.0, edgecolor='black', facecolor='none', zorder=3
            )
            ax_heat.add_patch(rect)

    # Mark Earl IC row
    if earl_idx_sorted is not None:
        ax_heat.axhline(earl_idx_sorted, color='gold', linewidth=2.5, alpha=0.9, zorder=4)
        ax_heat.text(-0.3, earl_idx_sorted, '★', ha='right', va='center',
                     fontsize=10, color='gold', fontweight='bold',
                     transform=ax_heat.get_yaxis_transform())

    # Annotate probability values in cells
    for i, row in enumerate(probs_sorted):
        for j, val in enumerate(row):
            if np.isfinite(val) and val > 0:
                txt = f'{val:.2f}' if val >= 0.01 else f'{val:.1e}'
                ax_heat.text(j, i, txt, ha='center', va='center',
                             fontsize=6.5, color='black' if val > 0.3 else 'white')

    ax_heat.set_xticks(range(n_transitions))
    ax_heat.set_xticklabels(iface_labels, fontsize=9, rotation=0, ha='center')
    ax_heat.set_yticks(range(n_ics))
    ax_heat.set_yticklabels(labels_sorted, fontsize=8.5)
    # ax_heat.set_title('Conditional Forward Transition Probabilities  '
    #                    r'$P(\lambda_{i+1} | \lambda_i)$  per IC × interface',
    #                    fontsize=11, fontweight='bold', pad=8)

    # ── Bottleneck strip ──────────────────────────────────────────────────────
    for i, bn_j in enumerate(bn_sorted):
        color = bottleneck_colors[int(bn_j)] if np.isfinite(bn_j) else '#dddddd'
        ax_bn.barh(i, 1, color=color, edgecolor='white', linewidth=0.4)

    ax_bn.set_xlim(0, 1)
    ax_bn.set_ylim(-0.5, n_ics - 0.5)
    ax_bn.set_xticks([])
    ax_bn.set_yticks([])
    ax_bn.set_title('BN', fontsize=8, pad=4)

    # Colorbar goes to the right of BOTH ax_heat and ax_bn — no collision
    cb = fig.colorbar(im, ax=[ax_heat, ax_bn], fraction=0.018, pad=0.02,
                      shrink=0.95, aspect=35)
    cb.set_label('P_forward', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── Mean P_forward bar chart ───────────────────────────────────────────────
    mean_p  = np.nanmean(probs_sorted, axis=0)
    std_p   = np.nanstd(probs_sorted,  axis=0)
    x_bars  = np.arange(n_transitions)
    ax_mean.bar(x_bars, mean_p, yerr=std_p, capsize=5,
                color=[bottleneck_colors[j] for j in range(n_transitions)],
                edgecolor='white', linewidth=0.5, alpha=0.85)
    ax_mean.set_ylim(0, min(1.05, (mean_p + std_p).max() * 1.25))
    ax_mean.set_xticks(x_bars)
    ax_mean.set_xticklabels(iface_labels, fontsize=9, rotation=30, ha='right')
    ax_mean.set_ylabel('Mean P_forward', fontsize=10)
    ax_mean.set_title('Season-mean conditional probabilities', fontsize=10, pad=6)
    ax_mean.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    # Mark global minimum (season bottleneck)
    global_bn = int(np.argmin(mean_p))
    ax_mean.axvline(global_bn, color='red', linestyle='--', linewidth=1.4, alpha=0.75,
                    label=f'Season bottleneck: {iface_labels[global_bn]}')
    ax_mean.legend(fontsize=9, loc='upper right')

    # ── Bottleneck legend (bottom-right cell) ─────────────────────────────────
    ax_leg.axis('off')
    legend_patches = [
        mpatches.Patch(color=bottleneck_colors[j], label=iface_labels[j])
        for j in range(n_transitions)
    ]
    ax_leg.legend(handles=legend_patches, loc='center', fontsize=9,
                  framealpha=0.85, title='Bottleneck\ninterface', title_fontsize=9,
                  borderpad=1.0)

    # ── Summary stats ─────────────────────────────────────────────────────────
    bn_counts = {j: int((bn_sorted == j).sum()) for j in range(n_transitions)}
    print("\n  Bottleneck interface distribution:")
    for j, cnt in bn_counts.items():
        print(f"    {iface_labels[j]:30s}  {cnt:2d} ICs  ({100*cnt/n_ics:.0f}%)")
    print(f"  Season-mean bottleneck: {iface_labels[global_bn]}")

    # ── Title and save ────────────────────────────────────────────────────────
    # fig.suptitle('Bottleneck Interface Analysis — All ICs\n'
    #              'Black box = rate-limiting step;  Gold line = highlighted IC',
    #              fontsize=12, fontweight='bold', y=0.99)

    out = plot_dir / 'bottleneck_interface_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bottleneck interface heatmap across all FFS ICs'
    )
    parser.add_argument('--ffs_csv',  required=True, help='FFS statistics CSV')
    parser.add_argument('--plot_dir', default='./plots', help='Output directory')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use log colour scale (recommended when P spans many decades)')
    parser.add_argument('--sort_by_bottleneck', action='store_true',
                        help='Sort rows by bottleneck interface (groups similar regimes)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Show every Nth IC (e.g. 2 = every other, default 1 = all)')
    parser.add_argument('--highlight_ic', default='2022-08-21T00Z',
                        help='IC time string to highlight as Earl case study')
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print('Loading FFS CSV...')
    df_ffs = pd.read_csv(args.ffs_csv)

    print('Generating bottleneck heatmap...')
    make_bottleneck_figure(
        df_ffs, plot_dir,
        log_scale=args.log_scale,
        sort_by_bottleneck=args.sort_by_bottleneck,
        highlight_ic=args.highlight_ic,
        stride=args.stride,
    )


if __name__ == '__main__':
    main()
