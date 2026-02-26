#!/usr/bin/env python
"""
Commitment curve (hockey stick) for FFS hurricane genesis.

p_B(λᵢ) = probability of reaching state_B given that interface λᵢ has
just been crossed.

FFS:  p_B = ∏ P_forward(λᵢ → λᵢ₊₁)           [low variance, product rule]
IFS:  p_B = n_crossed_λ_last / n_crossed_λᵢ    [brute-force count ratio]

Interfaces and state_B are read from ffs.yml — no hardcoded pressures.

Usage:
    python plot_commitment_curve.py \
        --ffs_config ffs.yml \
        --ffs_csv    results/ffs_statistics_all_ics.csv \
        --ifs_csv    results/IFS/ifs_rates_FFS.csv \
        --plot_dir   results/plots
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FFS_COLOR = '#2166ac'
IFS_COLOR = '#d6604d'


def ffs_committor(df: pd.DataFrame, interfaces: list) -> pd.DataFrame:
    """
    Compute FFS committor p_B(λᵢ) for each IC row.

    Uses columns lambda{i+1}_P_forward from the statistics CSV.
    p_B at the last interface (state_B) = 1 by definition.
    Working backward: p_B(λᵢ) = P_fwd(λᵢ→λᵢ₊₁) × p_B(λᵢ₊₁).
    """
    N  = len(interfaces)
    df = df.copy()
    df[f'p_B_{N-1}'] = 1.0
    for i in range(N - 2, -1, -1):
        col = f'lambda{i+1}_P_forward'
        df[f'p_B_{i}'] = df[col] * df[f'p_B_{i+1}']
    return df


def ifs_committor(df: pd.DataFrame, interfaces: list) -> pd.DataFrame:
    """
    Compute IFS committor p_B(λᵢ) from brute-force crossing counts.

    p_B(λᵢ) = n_crossed_λ_last / n_crossed_λᵢ.
    Falls back to rate ratio for λ₀ if counts are zero.
    """
    N    = len(interfaces)
    last = N - 1
    df   = df.copy()
    n_last = df[f'n_crossed_lambda{last}']
    for i in range(N - 1):
        ni = df[f'n_crossed_lambda{i}'].replace(0, np.nan)
        df[f'p_B_{i}'] = n_last / ni
    df[f'p_B_{last}'] = 1.0
    # λ₀ fallback: rate ratio when crossing counts are zero
    if df['p_B_0'].isna().all():
        df['p_B_0'] = (df['rate_B_bf_per_day'] /
                       df['rate_lambda0_flux_per_day'].replace(0, np.nan))
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Commitment curve p_B(λᵢ) — FFS vs IFS brute force'
    )
    parser.add_argument('--ffs_config', required=True,
                        help='FFS config file (ffs.yml) — source of interfaces and state_B')
    parser.add_argument('--ffs_csv',    required=True,
                        help='FFS statistics CSV (ffs_statistics_all_ics.csv)')
    parser.add_argument('--ifs_csv',    required=True,
                        help='IFS brute-force rates CSV (ifs_rates_FFS.csv)')
    parser.add_argument('--plot_dir',   default='.')
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)

    interfaces = ffs_config['interfaces'].copy()
    state_B    = float(ffs_config['state_B'])
    if interfaces[-1] != state_B:
        interfaces.append(state_B)
    N = len(interfaces)
    print(f'Interfaces: {interfaces}  (N={N})')

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute committors ───────────────────────────────────────────────────
    df_ffs = ffs_committor(pd.read_csv(args.ffs_csv), interfaces)
    df_ifs = ifs_committor(pd.read_csv(args.ifs_csv), interfaces)

    ffs_mu = np.array([df_ffs[f'p_B_{i}'].mean() for i in range(N)])
    ffs_sd = np.array([df_ffs[f'p_B_{i}'].std()  for i in range(N)])
    ifs_mu = np.array([df_ifs[f'p_B_{i}'].mean() for i in range(N)])
    ifs_sd = np.array([df_ifs[f'p_B_{i}'].std()  for i in range(N)])

    # Rate-limiting step = largest single drop in FFS curve
    drops = ffs_mu[:-1] - ffs_mu[1:]
    k     = int(np.argmax(drops))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(interfaces,
                    np.clip(ffs_mu - ffs_sd, 0, 1),
                    np.clip(ffs_mu + ffs_sd, 0, 1),
                    color=FFS_COLOR, alpha=0.15)
    ax.fill_between(interfaces,
                    np.clip(ifs_mu - ifs_sd, 0, 1),
                    np.clip(ifs_mu + ifs_sd, 0, 1),
                    color=IFS_COLOR, alpha=0.15)

    ax.plot(interfaces, ffs_mu, 'o-', color=FFS_COLOR, linewidth=2.5,
            markersize=8, label='FFS — AI model (1°, ∏P_fwd)')
    ax.plot(interfaces, ifs_mu, 's--', color=IFS_COLOR, linewidth=2.5,
            markersize=8, label='IFS — brute-force ensemble (0.25°→1.5°)')

    for p, v in zip(interfaces, ffs_mu):
        ax.annotate(f'{v:.3f}', (p, v),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, color=FFS_COLOR, fontweight='bold')

    ax.axvspan(interfaces[k+1], interfaces[k], alpha=0.12, color='grey',
               label=f'Rate-limiting step  ({interfaces[k]}→{interfaces[k+1]} hPa)')

    ax.invert_xaxis()
    ax.set_xlabel('Interface pressure (hPa)', fontsize=13)
    ax.set_ylabel(f'p_B  —  probability of reaching ≤{int(state_B)} hPa', fontsize=13)
    ax.set_title('Hurricane Genesis Commitment Curve\n'
                 'Atlantic 2022  |  FFS (AI) vs IFS (ensemble)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-0.02, 1.10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.25)

    out = plot_dir / 'commitment_curve.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ── Print table ──────────────────────────────────────────────────────────
    print(f'\n{"":4s}  {"hPa":>6}  {"FFS p_B":>10}  {"±sd":>8}  '
          f'{"IFS p_B":>10}  {"±sd":>8}')
    for i, p in enumerate(interfaces):
        print(f'  λ{i}  {int(p):6d}  {ffs_mu[i]:10.4f}  {ffs_sd[i]:8.4f}  '
              f'{ifs_mu[i]:10.4f}  {ifs_sd[i]:8.4f}')
    print(f'\n  Rate-limiting step: λ{k}→λ{k+1}  '
          f'({interfaces[k]}→{interfaces[k+1]} hPa)  Δp_B = {drops[k]:.4f}')


if __name__ == '__main__':
    main()
