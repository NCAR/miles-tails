#!/usr/bin/env python
"""
Enhancement factor scaling curve — Figure 1.

The paper's central practical claim: FFS achieves ~45× speedup scaling as
E ∝ 1/p.  This script produces the log-log plot of computational enhancement
factor vs. genesis probability p, with the 18 actual ICs as points and the
theoretical 1/p line overlaid.

Enhancement factor definition
------------------------------
For a rare event with genesis probability p, direct Monte Carlo (IFS) requires
on average 1/p trajectories to observe one event.  FFS achieves comparable
accuracy with N_FFS trajectories (fixed algorithm parameters: n_flux + n_shoot).
The theoretical enhancement is therefore:

    E(p) = N_direct / N_FFS  where N_direct = 1/p
    → E ∝ 1/p

Per IC, p_i = p_B(λ₀) = ∏ P_forward(λⱼ→λⱼ₊₁)   (product of conditional probs)
Theoretical E_i = 1 / p_i  (scales with the rare-event probability)

IFS-measured E (when IFS has events at state B):
    E_measured = σ²_IFS / σ²_FFS  (variance ratio at equal compute)
    ≈ n_IFS_trajectories_needed / N_FFS

    When k_IFS > 0:
        CV²_IFS  = 1 / n_IFS_events     (Poisson counting)
        CV²_FFS  = (SE_FFS / k_FFS)²    # requires per-IC SE, not always available

    If --n_ffs and --n_ifs are provided (total trajectory counts), a rough
    measured E is shown alongside the theoretical curve.

Usage:
    python plot_enhancement_factor.py \\
        --ffs_csv results/ffs_statistics_all_ics.csv \\
        --ifs_csv results/IFS/ifs_rates_FFS.csv \\
        --plot_dir results/plots \\
        [--n_ffs 5000]   # total FFS trajectories per IC (flux + shoot)
        [--n_ifs 1000]   # total IFS ensemble members per IC
        [--highlight_ic 2022-08-21T00Z]   # highlight a specific IC
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Interface definition ──────────────────────────────────────────────────────

INTERFACE_PRESSURES = [1000, 988, 980, 975, 970, 965]   # λ0 … λ5 = state B
N_IFACES            = len(INTERFACE_PRESSURES)

# ── Committor ─────────────────────────────────────────────────────────────────

def compute_ffs_committor(df: pd.DataFrame) -> pd.DataFrame:
    """p_B(λ₀) = ∏ P_forward(λᵢ→λᵢ₊₁) — auto-detects number of interfaces."""
    df = df.copy()
    n = len([c for c in df.columns if c.startswith('lambda') and c.endswith('_P_forward')])
    p = 1.0
    for i in range(n, 0, -1):
        p = df[f'lambda{i}_P_forward'] * p
    df['p_B_lambda0'] = p
    return df


# ── Main plot ─────────────────────────────────────────────────────────────────

def make_enhancement_figure(df_ffs, df_ifs, plot_dir, n_ffs=None, n_ifs=None,
                             highlight_ic=None):
    """
    Log-log plot of enhancement factor E vs genesis probability p.

    Parameters
    ----------
    df_ffs       : FFS statistics DataFrame (with p_B_lambda0 computed)
    df_ifs       : IFS statistics DataFrame
    n_ffs        : Total FFS trajectories per IC (for absolute E estimate)
    n_ifs        : Total IFS ensemble size per IC (for absolute E estimate)
    highlight_ic : IC time string to highlight (e.g. '2022-08-21T00Z')
    """
    p_ffs = df_ffs['p_B_lambda0'].values.astype(float)
    ic_times = pd.to_datetime(df_ffs['ic_time'])

    # Guard against zeros / NaNs
    valid = np.isfinite(p_ffs) & (p_ffs > 0)

    # Theoretical enhancement: E = 1/p  (normalised so mean E matches paper value)
    # The raw 1/p gives enhancement in units of "IFS trajectories per FFS trajectory"
    # Reported ~45× mean: verify against 1/p_mean
    p_mean    = np.nanmean(p_ffs[valid])
    E_mean    = 1.0 / p_mean       # theoretical mean enhancement
    E_i       = 1.0 / p_ffs       # theoretical per-IC

    print(f"  Geometric mean p_B = {np.exp(np.nanmean(np.log(p_ffs[valid]))):.4e}")
    print(f"  Arithmetic mean p_B = {p_mean:.4e}")
    print(f"  Mean theoretical E  = {E_mean:.1f}×")
    print(f"  Range p_B: [{p_ffs[valid].min():.4e}, {p_ffs[valid].max():.4e}]")
    print(f"  Range E:   [{E_i[valid].min():.1f}, {E_i[valid].max():.1f}]")

    # ── Align IFS to FFS by nearest time ─────────────────────────────────────
    # IFS may have more rows than FFS (incomplete runs), so we match each FFS
    # IC to its nearest IFS row.  All resulting arrays are length len(p_ffs).
    t_ifs = pd.to_datetime(df_ifs['init_time'])
    k_ffs = df_ffs['ffs_rate_per_day'].values.astype(float)
    # Find highest lambda crossed column in IFS CSV
    ifs_last_lambda_col = None
    for candidate in sorted([c for c in df_ifs.columns if c.startswith('n_crossed_lambda')], reverse=True):
        ifs_last_lambda_col = candidate
        break
    has_n5_col = ifs_last_lambda_col is not None

    k_ifs  = np.full(len(p_ffs), np.nan)
    ifs_n5 = np.zeros(len(p_ffs))
    for i, t in enumerate(ic_times):
        j = (t_ifs - t).abs().idxmin()
        k_ifs[i]  = df_ifs.loc[j, 'rate_B_bf_per_day']
        if has_n5_col:
            ifs_n5[i] = df_ifs.loc[j, ifs_last_lambda_col]

    has_ifs_events = (ifs_n5 > 0)

    # Rough measured E when both FFS and IFS have data and n_ffs/n_ifs provided:
    #   CV²_IFS ≈ 1/n_ifs_events = 1/ifs_n5
    #   CV²_FFS ≈ (SE_FFS/k_FFS)² — we approximate SE_FFS/k_FFS ≈ sqrt(p/n_ffs)
    # → E_measured ≈ n_ffs / (p * n_ifs_events) when proportional compute is used
    if n_ffs is not None and n_ifs is not None:
        E_measured = np.where(
            has_ifs_events & valid,
            n_ffs / (p_ffs * ifs_n5),   # rough measured enhancement
            np.nan
        )
        show_measured = True
    else:
        E_measured = np.full(len(p_ffs), np.nan)
        show_measured = False

    # ── Colour points by calendar date ───────────────────────────────────────
    doy   = ic_times.dt.day_of_year.values.astype(float)
    norm  = plt.Normalize(doy[valid].min(), doy[valid].max())
    cmap  = plt.cm.plasma

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # Theoretical 1/p line (smooth curve)
    p_line = np.logspace(np.log10(p_ffs[valid].min() * 0.4),
                          np.log10(p_ffs[valid].max() * 2.0), 200)
    E_line = 1.0 / p_line
    ax.plot(p_line, E_line, '-', color='#333333', linewidth=2.0,
            zorder=1, label=r'Theoretical: $\mathcal{E} = 1/p$')

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(E_mean, color='steelblue', linestyle=':', linewidth=1.2, alpha=0.7,
               label=f'Mean $\\mathcal{{E}}$ = {E_mean:.0f}×')
    ax.axhline(1.0,   color='grey',      linestyle='--', linewidth=0.8, alpha=0.5,
               label='No enhancement ($\\mathcal{E}$ = 1)')

    # ── IC points (theoretical) ───────────────────────────────────────────────
    sc = ax.scatter(p_ffs[valid], E_i[valid],
                    c=doy[valid], cmap=cmap, norm=norm,
                    s=80, zorder=3, edgecolors='white', linewidths=0.6,
                    label='FFS IC (theoretical $\\mathcal{E} = 1/p_i$)')

    # ── Measured E overlay (if available) ─────────────────────────────────────
    meas_valid = np.isfinite(E_measured)
    if show_measured and meas_valid.any():
        ax.scatter(p_ffs[meas_valid], E_measured[meas_valid],
                   c=doy[meas_valid], cmap=cmap, norm=norm,
                   s=80, marker='D', zorder=4, edgecolors='black', linewidths=0.9,
                   label='Measured $\\mathcal{E}$ (FFS vs IFS variance ratio)')
        # Connect theoretical→measured with grey segments
        for idx in np.where(meas_valid)[0]:
            ax.plot([p_ffs[idx], p_ffs[idx]],
                    [E_i[idx],   E_measured[idx]],
                    color='grey', linewidth=0.6, alpha=0.5, zorder=2)

    # ── ICs with zero IFS events (E = ∞ → lower-bound arrows) ────────────────
    zero_ifs = (k_ifs <= 0) & valid
    if zero_ifs.any() and n_ifs is not None:
        # Lower bound: with n_ifs trajectories and zero events, E > n_ifs * p_B
        E_lb = n_ifs * p_ffs[zero_ifs]   # minimum IFS efficiency
        ax.scatter(p_ffs[zero_ifs], 1.0 / p_ffs[zero_ifs],
                   c=doy[zero_ifs], cmap=cmap, norm=norm,
                   s=80, marker='^', zorder=3, edgecolors='red', linewidths=0.9,
                   label='Zero IFS events (IFS cannot estimate rate)')

    # ── Highlight selected IC ─────────────────────────────────────────────────
    if highlight_ic is not None:
        try:
            hl_dt  = pd.to_datetime(highlight_ic)
            hl_row = (ic_times - hl_dt).abs().idxmin()
            ax.scatter(p_ffs[hl_row], E_i[hl_row],
                       s=200, marker='*', color='gold', edgecolors='black',
                       linewidths=1.2, zorder=5, label=f'Highlighted IC ({highlight_ic})')
            ax.annotate(highlight_ic, (p_ffs[hl_row], E_i[hl_row]),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=9, fontweight='bold')
        except Exception as exc:
            print(f"  Warning: could not highlight IC: {exc}")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Genesis probability  $p = p_B(\lambda_0)$', fontsize=13)
    ax.set_ylabel(r'Enhancement factor  $\mathcal{E}$', fontsize=13)
    ax.set_title('FFS Computational Enhancement Factor\n'
                 r'$\mathcal{E} \propto 1/p$ — log-log scaling across 2022 season ICs',
                 fontsize=13, fontweight='bold')

    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(mticker.LogFormatter())
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)

    # Colourbar — date
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Day of year  (2022)', fontsize=10)

    ax.legend(fontsize=9, loc='lower left', framealpha=0.85)

    # ── Annotation box ────────────────────────────────────────────────────────
    n_ic    = valid.sum()
    n_zero  = int(zero_ifs.sum())
    msg     = (f"N = {n_ic} ICs\n"
               f"Mean p_B = {p_mean:.3e}\n"
               f"Mean $\\mathcal{{E}}$ = {E_mean:.0f}×\n"
               f"ICS with k_IFS = 0: {n_zero}")
    ax.text(0.98, 0.97, msg,
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    plt.tight_layout()
    out = plot_dir / 'enhancement_factor_scaling.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Enhancement factor scaling curve (log-log) for FFS paper'
    )
    parser.add_argument('--ffs_csv',  required=True,  help='FFS statistics CSV')
    parser.add_argument('--ifs_csv',  required=True,  help='IFS brute-force CSV')
    parser.add_argument('--plot_dir', default='./plots',
                        help='Output directory')
    parser.add_argument('--n_ffs',    type=float, default=None,
                        help='Total FFS trajectories per IC (flux+shoot), e.g. 5000')
    parser.add_argument('--n_ifs',    type=float, default=None,
                        help='Total IFS ensemble size per IC, e.g. 1000')
    parser.add_argument('--highlight_ic', default=None,
                        help='IC datetime string to highlight (e.g. 2022-08-21T00Z)')
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print('Loading CSVs...')
    df_ffs = pd.read_csv(args.ffs_csv)
    df_ifs = pd.read_csv(args.ifs_csv)

    print('Computing p_B...')
    df_ffs = compute_ffs_committor(df_ffs)

    print('Generating enhancement factor figure...')
    make_enhancement_figure(
        df_ffs, df_ifs, plot_dir,
        n_ffs=args.n_ffs,
        n_ifs=args.n_ifs,
        highlight_ic=args.highlight_ic,
    )


if __name__ == '__main__':
    main()
