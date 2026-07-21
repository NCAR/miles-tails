#!/usr/bin/env python
"""
Standalone script to generate rates_plot.png and ffs_interface_probs.png
for the FFS hurricane genesis manuscript.

rates_plot.png      — FFS vs IFS genesis rates vs time (all ICs), log scale
ffs_interface_probs.png — λ0 flux rate + conditional P(λ_{i+1}|λ_i) vs time

Usage:
    python plot_rates_comparison.py \
        --ffs_config /glade/derecho/scratch/schreck/FFS/ffs.yml \
        --ffs_csv    /glade/derecho/scratch/schreck/FFS/results_mar6/ffs_statistics_all_ics.csv \
        --ifs_csv    /glade/derecho/scratch/schreck/FFS/results_mar6/IFS/ifs_rates_FFS.csv \
        --plot_dir   /glade/work/schreck/repos/miles-tails/plots
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import argparse
import re
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


FFS_COLOR   = '#2166ac'   # blue        (FFS λ0 flux rate)
IFS_COLOR   = '#d6604d'   # orange-red  (IFS λ0 crossing rate)
FFS2_COLOR  = '#4dac26'   # green       (full FFS genesis rate)
IFS2_COLOR  = '#b2182b'   # red         (IFS brute-force)
AINWP_COLOR = '#762a83'   # purple      (AI NWP brute-force direct rate)

plt.rcParams.update({'font.size': 11})


# ------------------------------------------------------------------ #
# rates_plot.png
# ------------------------------------------------------------------ #

def plot_rates(df_ffs: pd.DataFrame, df_ifs: pd.DataFrame,
               plot_dir: Path) -> None:
    """
    Five-series log-scale plot:
      Blue circles   — FFS λ0 flux rate  Φ_{A,0}
      Orange diamonds— IFS λ0 crossing rate
      Green squares  — Full FFS genesis rate  k_{A→B}
      Purple stars   — AI NWP direct BF genesis rate (±1σ Poisson)
      Red triangles  — IFS brute-force genesis rate  (↑ for upper bounds)
    """
    dfF = df_ffs.copy()
    dfI = df_ifs.copy()

    dfF['time'] = pd.to_datetime(dfF['ic_time'])
    dfI['time'] = pd.to_datetime(dfI['init_time'])

    # FFS λ0 flux rate
    dfF['ffs_lambda0'] = (
        dfF['flux_lambda0_crossings'] / dfF['flux_total_time_days']
    )

    # Full FFS rate: λ0 flux × ∏ P_forward
    dfF['ffs_rate'] = dfF['ffs_lambda0'].copy()
    for col in sorted(dfF.columns):
        if re.match(r'lambda\d+_P_forward', col):
            dfF['ffs_rate'] *= dfF[col]

    # AI NWP direct BF rate Poisson errors: sqrt(N)/T
    dfF['ainwp_err_lo'] = np.sqrt(dfF['flux_direct_B_formations']) / dfF['flux_total_time_days']
    dfF['ainwp_err_hi'] = (np.sqrt(dfF['flux_direct_B_formations']) + 1.0) / dfF['flux_total_time_days']

    # Merge on time
    df = pd.merge(
        dfF[['time', 'ffs_lambda0', 'ffs_rate', 'direct_formation_rate_per_day',
             'ainwp_err_lo', 'ainwp_err_hi']],
        dfI[['time', 'rate_lambda0_flux_per_day', 'rate_B_bf_per_day',
             'n_reached_B', 'total_sim_days']],
        on='time', how='outer'
    ).sort_values('time').reset_index(drop=True)

    # IFS BF Poisson 1-sigma: sqrt(N)/T; for N=0 upper bound 1.15/T
    n_ifs = df['n_reached_B'].fillna(0)
    T_ifs = df['total_sim_days'].fillna(750.0)
    ifs_err_lo = np.where(n_ifs > 0, np.sqrt(n_ifs) / T_ifs, 0.0)
    ifs_err_hi = np.where(n_ifs > 0, (np.sqrt(n_ifs) + 1.0) / T_ifs, 1.15 / T_ifs)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df['time'], df['ffs_lambda0'],
            'o-', color=FFS_COLOR, lw=2, ms=7, label=r'FFS $\lambda_0$ flux rate $\Phi_{A,0}$')

    ax.plot(df['time'], df['rate_lambda0_flux_per_day'],
            'D-', color=IFS_COLOR, lw=2, ms=7, label=r'IFS $\lambda_0$ crossing rate')

    ax.plot(df['time'], df['ffs_rate'],
            's-', color=FFS2_COLOR, lw=2, ms=7, label=r'FFS genesis rate $k_{A\to B}$')

    # AI NWP direct BF rate with Poisson error bars
    ax.errorbar(
        df['time'], df['direct_formation_rate_per_day'],
        yerr=[df['ainwp_err_lo'].fillna(0), df['ainwp_err_hi'].fillna(0)],
        fmt='*-', color=AINWP_COLOR, lw=2, ms=9, capsize=4, elinewidth=1.5,
        label=r'AI NWP direct BF rate (±1σ Poisson)'
    )

    # IFS brute-force with Poisson error bars
    mask_pos  = n_ifs > 0
    mask_zero = n_ifs == 0

    ax.errorbar(
        df.loc[mask_pos, 'time'],
        df.loc[mask_pos, 'rate_B_bf_per_day'],
        yerr=[ifs_err_lo[mask_pos], ifs_err_hi[mask_pos]],
        fmt='^-', color=IFS2_COLOR, lw=2, ms=7, capsize=4, elinewidth=1.5,
        label='IFS brute-force genesis rate (±1σ Poisson)'
    )

    if mask_zero.any():
        # Upper bound only (no lower): draw open marker + upward arrow
        ub_rate = 1.15 / T_ifs[mask_zero]
        ax.scatter(df.loc[mask_zero, 'time'], ub_rate,
                   marker='^', facecolors='none', edgecolors=IFS2_COLOR,
                   s=60, zorder=5, linewidths=1.5)
        for x, y in zip(df.loc[mask_zero, 'time'], ub_rate):
            ax.annotate('', xy=(x, y * 1.6), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color=IFS2_COLOR, lw=1.5))

    ax.set_yscale('log')
    ax.set_xlabel('Initial Condition Time')
    ax.set_ylabel('Rate (per day)')
    ax.set_title('Genesis Rate Estimates — FFS vs IFS Ensemble')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    out = plot_dir / 'rates_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ------------------------------------------------------------------ #
# ffs_interface_probs.png
# ------------------------------------------------------------------ #

def plot_interface_probs(df_ffs: pd.DataFrame, df_ifs: pd.DataFrame,
                         interfaces: list, plot_dir: Path) -> None:
    """
    One panel per quantity: λ0 flux rate then P(λ_{i+1}|λ_i) for each interface.
    Blue = FFS, Orange = IFS.
    """
    dfF = df_ffs.copy()
    dfI = df_ifs.copy()

    dfF['time'] = pd.to_datetime(dfF['ic_time'])
    dfI['time'] = pd.to_datetime(dfI['init_time'])

    # Detect which lambdas are present in both CSVs
    ffs_lambdas = sorted([
        int(re.search(r'lambda(\d+)_P_forward', c).group(1))
        for c in dfF.columns if re.match(r'lambda\d+_P_forward', c)
    ])
    shared_lambdas = [i for i in ffs_lambdas if f'prob_lambda{i}' in dfI.columns]

    n_plots = 1 + len(shared_lambdas)   # λ0 flux + one per interface
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    # λ0 flux rate panel
    ax = axes[0]
    ax.plot(dfF['time'], dfF['flux_rate_per_day'],
            'o-', color=FFS_COLOR, lw=2, ms=6, label='FFS')
    ax.plot(dfI['time'], dfI['rate_lambda0_flux_per_day'],
            's-', color=IFS_COLOR, lw=2, ms=6, label='IFS')
    ax.set_title(r'$\lambda_0$ Flux Rate', fontweight='bold')
    ax.set_ylabel('Rate (per day)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # One panel per λi → λ_{i+1}
    for panel, i in enumerate(shared_lambdas, start=1):
        ax = axes[panel]
        ax.plot(dfF['time'], dfF[f'lambda{i}_P_forward'],
                'o-', color=FFS_COLOR, lw=2, ms=6, label='FFS')
        ax.plot(dfI['time'], dfI[f'prob_lambda{i}'],
                's-', color=IFS_COLOR, lw=2, ms=6, label='IFS')
        ax.set_title(rf'$P(\lambda_{i+1}|\lambda_{i})$', fontweight='bold')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(r'$\lambda_0$ Flux Rate and Conditional Transition Probabilities',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = plot_dir / 'ffs_interface_probs.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Generate rates_plot.png and ffs_interface_probs.png'
    )
    parser.add_argument('--ffs_config', required=True,
                        help='FFS config YAML (ffs.yml)')
    parser.add_argument('--ffs_csv', required=True,
                        help='FFS statistics CSV (ffs_statistics_all_ics.csv)')
    parser.add_argument('--ifs_csv', required=True,
                        help='IFS brute-force CSV (ifs_rates_FFS.csv)')
    parser.add_argument('--plot_dir', default='./plots',
                        help='Output directory for PNG files (default: ./plots)')
    args = parser.parse_args()

    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)
    interfaces = ffs_config['interfaces']

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_ffs = pd.read_csv(args.ffs_csv)
    df_ifs = pd.read_csv(args.ifs_csv)

    print(f'FFS rows: {len(df_ffs)}   IFS rows: {len(df_ifs)}')
    print(f'Interfaces from config: {interfaces}')

    plot_rates(df_ffs, df_ifs, plot_dir)
    plot_interface_probs(df_ffs, df_ifs, interfaces, plot_dir)


if __name__ == '__main__':
    main()
