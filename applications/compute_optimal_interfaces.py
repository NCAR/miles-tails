"""
compute_optimal_interfaces.py

Reads FFS JSONL trajectory logs and computes optimal interface placement
for the next production run, using empirical penetration CDFs.

Core idea (Kratzer, Arnold & Allen 2013):
  At each interface i, launch N trajectories and record min(MSLP).
  The optimal next interface λ_{i+1}* is the x where:
    P(min_MSLP < x | launched from λ_i) = p_target ≈ 1/e ≈ 0.368
  i.e. the p_target-th quantile of the min-MSLP distribution at that stage.

Usage:
    python compute_optimal_interfaces.py \
        --log_dir /path/to/ffs_output/logs \
        --state_b 970 975 \
        --lambda0 1000 \
        --p_target 0.368 \
        --output_dir ./interface_analysis
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def parse_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def _parse_file(path):
    """Parse a single JSONL file. Returns list of (lambda_label, min_mslp)."""
    results = []
    skipped = 0
    for e in parse_jsonl(path):
        traj = e.get('mslp_trajectory')
        if not traj:
            skipped += 1
            continue
        label = e.get('lambda_label')
        if label is None:
            if e.get('phase') == 'flux_generation':
                label = -1
            else:
                skipped += 1
                continue
        results.append((label, min(traj)))
    return results, skipped


def load_all_trajectories(log_dirs, n_jobs=-1):
    """
    Recursively read all JSONL logs under each directory in log_dirs.
    Files are parsed in parallel (n_jobs=-1 = all cores).
    Aggregates trajectories across all IC directories into one dict.
    Returns dict: lambda_label -> list of min(mslp_trajectory)

    lambda_label is the interface the trajectory was LAUNCHED FROM:
      -1 = launched from state A (flux phase)
       0 = launched from λ₀
       1 = launched from λ₁
       etc.
    """
    if isinstance(log_dirs, (str, Path)):
        log_dirs = [log_dirs]

    # Collect all files across all directories first
    all_files = []
    for log_dir in log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.exists():
            print(f"WARNING: {log_dir} does not exist, skipping.")
            continue
        files = sorted(log_dir.rglob('*.jsonl'))
        print(f"  {log_dir}: {len(files)} JSONL files")
        all_files.extend(files)

    if not all_files:
        return {}

    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Parsing {len(all_files)} files across {n_cores} cores...")

    # Parallel parse
    raw = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_parse_file)(path) for path in all_files
    )

    # Merge results
    by_stage = defaultdict(list)
    n_skipped = 0
    for results, skipped in raw:
        n_skipped += skipped
        for label, min_mslp in results:
            by_stage[label].append(min_mslp)

    print(f"Total: {len(all_files)} files, {sum(len(v) for v in by_stage.values())} trajectories")
    print(f"Trajectories per stage: { {k: len(v) for k, v in sorted(by_stage.items())} }")
    if n_skipped:
        print(f"  ({n_skipped} entries skipped — no mslp_trajectory field)")
    return dict(by_stage)


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def penetration_cdf(min_mslps, x_grid):
    """
    Empirical P(min_MSLP < x) for each x in x_grid.
    x_grid should be decreasing (high → low hPa).
    """
    arr = np.asarray(min_mslps)
    return np.array([np.mean(arr < x) for x in x_grid])


def optimal_next_interface(min_mslps, p_target):
    """
    The p_target-th quantile of the min-MSLP distribution.
    By definition: P(min < quantile_p) ≈ p_target.
    Returns the MSLP threshold for the next interface.
    """
    return float(np.percentile(min_mslps, p_target * 100))


def build_optimal_chain(by_stage, lambda0, state_b, p_target):
    """
    Iteratively build the optimal interface chain from lambda0 to state_b.

    At stage i (trajectories launched from λ_i), the optimal λ_{i+1} is the
    p_target quantile of the min-MSLP distribution — the threshold where
    exactly p_target fraction of trajectories would cross.

    Returns:
        chain      : list of hPa values [lambda0, λ1*, λ2*, ..., state_b]
        stage_stats: dict with per-stage diagnostics
    """
    sorted_labels = sorted(k for k in by_stage if k >= -1)
    chain = [lambda0]
    stage_stats = {}

    for label in sorted_labels:
        mins = np.asarray(by_stage[label])
        n = len(mins)

        # Optimal next interface from this stage
        opt_next = optimal_next_interface(mins, p_target)
        p_at_opt = float(np.mean(mins < opt_next))  # should ≈ p_target

        # P(reaching state_b directly from this stage)
        p_to_b = float(np.mean(mins < state_b))

        # Current empirical P (using whatever the EXISTING next interface was)
        # We don't know the existing next interface precisely, but
        # the success fraction in the log gives it directly if status is stored.
        # Approximate: fraction below the median of successess.
        # Just report p_to_b and p_at_opt; the user can compare.

        stage_name = "flux (state A)" if label == -1 else f"λ{label}"
        stage_stats[label] = {
            'name': stage_name,
            'n': n,
            'min_mslp_mean': float(np.mean(mins)),
            'min_mslp_min':  float(np.min(mins)),
            'opt_next': round(opt_next, 1),
            'p_at_opt': p_at_opt,
            'p_to_stateB': p_to_b,
        }

        # Add to chain only if opt_next > state_b (still making progress)
        if opt_next > state_b and opt_next < (chain[-1] - 0.5):
            chain.append(round(opt_next, 1))

    # Always end with state_b
    if chain[-1] != state_b:
        chain.append(state_b)

    return chain, stage_stats


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_penetration_cdfs(by_stage, state_b_list, lambda0, p_target, output_path):
    """
    One subplot per stage showing P(min_MSLP < x) vs x.
    Marks:
      - Red dashed line: p_target (= 1/e)
      - Red dot + vertical: optimal next interface
      - Green verticals: each state_B candidate
    """
    sorted_labels = sorted(k for k in by_stage if k >= -1)
    n_plots = len(sorted_labels)
    if n_plots == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, n_plots,
                              figsize=(4.5 * n_plots, 5),
                              sharey=True)
    if n_plots == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_plots))

    for ax, label, color in zip(axes, sorted_labels, colors):
        mins = np.asarray(by_stage[label])
        n = len(mins)
        stage_name = "flux\n(state A)" if label == -1 else f"λ{label}"

        # x grid: span the full range of min-MSLP values seen
        x_lo = max(min(mins) - 5, lambda0 - 80)
        x_hi = min(max(mins) + 3, lambda0 + 5)
        x_grid = np.linspace(x_hi, x_lo, 400)  # high → low hPa
        p_curve = penetration_cdf(mins, x_grid)

        ax.plot(x_grid, p_curve, color=color, lw=2.5, label='empirical CDF')

        # Target probability lines
        ax.axhline(p_target, color='red', ls='--', lw=1.5,
                   label=f'p* = {p_target:.3f}')
        ax.axhline(0.25, color='gray', ls=':', lw=1.0, alpha=0.7)
        ax.axhline(0.50, color='gray', ls=':', lw=1.0, alpha=0.7)

        # Optimal next interface
        opt_x = optimal_next_interface(mins, p_target)
        p_opt = float(np.mean(mins < opt_x))
        if x_lo < opt_x < x_hi:
            ax.axvline(opt_x, color='red', ls='-', lw=1.5, alpha=0.8,
                       label=f'λ_opt = {opt_x:.0f} hPa')
            ax.scatter([opt_x], [p_opt], color='red', s=60, zorder=5)

        # state_B candidates
        sb_colors = ['#2ca02c', '#17becf']
        for sb, sbc in zip(state_b_list, sb_colors):
            p_sb = float(np.mean(mins < sb))
            ax.axvline(sb, color=sbc, ls='-', lw=1.5, alpha=0.7,
                       label=f'state_B={sb} hPa\n(P={p_sb:.3f})')
            ax.scatter([sb], [p_sb], color=sbc, s=60, zorder=5)

        ax.set_xlabel('MSLP threshold (hPa)', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('P(min MSLP reached < threshold)', fontsize=11)
        ax.set_title(f'Stage: {stage_name}\n(N = {n})', fontsize=11)
        ax.set_xlim(x_hi + 2, x_lo - 2)   # inverted: high → low hPa
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.suptitle(
        f'FFS Penetration CDFs by Stage\n'
        f'Optimal next interface where curve crosses p* = {p_target:.3f} (= 1/e)',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_optimal_chains(chains_by_stateB, stage_stats_by_stateB,
                         lambda0, p_target, output_path):
    """
    Summary plot: bar chart of P per step for each proposed chain,
    overlaid with the p* target.
    """
    n_sb = len(chains_by_stateB)
    fig, axes = plt.subplots(1, n_sb, figsize=(6 * n_sb, 4))
    if n_sb == 1:
        axes = [axes]

    for ax, (state_b, chain) in zip(axes, chains_by_stateB.items()):
        stage_stats = stage_stats_by_stateB[state_b]

        # Get P per step from stage_stats
        # stage label i → P(opt_next) ≈ p_target (by construction)
        # but we also know P to state_B
        labels = sorted(k for k in stage_stats if k >= -1)
        step_labels = []
        step_p = []

        for i, label in enumerate(labels):
            s = stage_stats[label]
            step_labels.append(f"λ{label}→\nλ{label+1}")
            step_p.append(s['p_at_opt'])

        x = np.arange(len(step_labels))
        bars = ax.bar(x, step_p, color='steelblue', alpha=0.7, edgecolor='k', lw=0.8)
        ax.axhline(p_target, color='red', ls='--', lw=2, label=f'p* = {p_target:.3f}')
        ax.axhline(0.25, color='orange', ls=':', lw=1.5, label='p = 0.25')
        ax.axhline(0.50, color='orange', ls=':', lw=1.5, label='p = 0.50')

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel('P(crossing)', fontsize=11)
        ax.set_title(f'state_B = {state_b} hPa\nProposed chain: {chain}', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        for bar, p in zip(bars, step_p):
            ax.text(bar.get_x() + bar.get_width() / 2, p + 0.02,
                    f'{p:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Per-step crossing probabilities for proposed optimal chains\n'
                  f'p* = {p_target:.3f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute optimal FFS interfaces from trajectory log data.'
    )
    parser.add_argument('--log_dir', required=True, nargs='+',
                        help='One or more log directories (shell globs are fine: results/*/logs)')
    parser.add_argument('--state_b', type=float, nargs='+', default=[970.0, 975.0],
                        help='Candidate state_B values in hPa (default: 970 975)')
    parser.add_argument('--lambda0', type=float, default=1000.0,
                        help='λ₀ threshold in hPa (default: 1000)')
    parser.add_argument('--p_target', type=float, default=1.0 / np.e,
                        help=f'Target per-step crossing probability (default: 1/e = {1/np.e:.4f})')
    parser.add_argument('--output_dir', default='./interface_analysis',
                        help='Directory for plots and results')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel workers for file parsing (-1 = all cores, default: -1)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\np_target = {args.p_target:.4f}  (1/e = {1/np.e:.4f})")
    print(f"λ₀       = {args.lambda0} hPa")
    print(f"state_B  = {args.state_b} hPa")
    print(f"log_dirs = {args.log_dir} ({len(args.log_dir)} directories)\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    by_stage = load_all_trajectories(args.log_dir, n_jobs=args.n_jobs)

    if not by_stage:
        print("ERROR: No trajectory data found. Check --log_dir.")
        return

    # ── Per-stage CDF plot ────────────────────────────────────────────────────
    cdf_plot = output_dir / 'penetration_cdfs.png'
    plot_penetration_cdfs(by_stage, args.state_b, args.lambda0,
                           args.p_target, cdf_plot)

    # ── Build optimal chain for each state_B ─────────────────────────────────
    chains_by_stateB = {}
    stage_stats_by_stateB = {}

    for state_b in args.state_b:
        chain, stage_stats = build_optimal_chain(
            by_stage, args.lambda0, state_b, args.p_target
        )
        chains_by_stateB[state_b] = chain
        stage_stats_by_stateB[state_b] = stage_stats

    # ── Per-step P bar chart ──────────────────────────────────────────────────
    bar_plot = output_dir / 'optimal_chain_Ps.png'
    plot_optimal_chains(chains_by_stateB, stage_stats_by_stateB,
                         args.lambda0, args.p_target, bar_plot)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PER-STAGE DIAGNOSTICS")
    print("=" * 65)

    # Print once (same stages regardless of state_B)
    first_stats = next(iter(stage_stats_by_stateB.values()))
    header = f"{'Stage':<20} {'N':>6} {'mean_min':>9} {'abs_min':>9} {'λ_opt':>8} {'P@opt':>7}"
    print(header)
    print("-" * 65)
    for label in sorted(first_stats):
        s = first_stats[label]
        print(f"{s['name']:<20} {s['n']:>6} "
              f"{s['min_mslp_mean']:>9.1f} {s['min_mslp_min']:>9.1f} "
              f"{s['opt_next']:>8.1f} {s['p_at_opt']:>7.3f}")

    print("\n" + "=" * 65)
    print("OPTIMAL INTERFACE CHAINS")
    print("=" * 65)
    print(f"(p_target = {args.p_target:.4f},  gaps chosen so P(cross) ≈ p*)\n")

    for state_b in args.state_b:
        chain = chains_by_stateB[state_b]
        stats = stage_stats_by_stateB[state_b]

        # Compute P to this state_B at each stage for context
        p_to_b_per_stage = {
            label: stats[label]['p_to_stateB']
            for label in sorted(stats)
        }

        print(f"state_B = {state_b} hPa")
        print(f"  Proposed: interfaces = {chain}")
        print(f"  N steps  = {len(chain) - 1}")
        gaps = [round(chain[i] - chain[i+1], 1) for i in range(len(chain)-1)]
        print(f"  Gaps     = {gaps} hPa")

        # Estimate cumulative probability (product of P@opt across stages)
        p_vals = [first_stats[lbl]['p_at_opt'] for lbl in sorted(first_stats)]
        # Only use stages up to those needed for this state_B
        n_steps = len(chain) - 1
        p_prod = np.prod(p_vals[:n_steps]) if p_vals else 0.0
        print(f"  Est. cumulative P(λ₀ → state_B) ≈ {p_prod:.5f}")

        print("  P(reach state_B) from each stage:")
        for label, p_b in p_to_b_per_stage.items():
            name = first_stats[label]['name']
            print(f"    {name}: {p_b:.4f}")
        print()

    print("=" * 65)
    print("RECOMMENDATION")
    print("=" * 65)
    for state_b in args.state_b:
        chain = chains_by_stateB[state_b]
        # Drop lambda0 from the interfaces list (it's the flux threshold)
        ffs_interfaces = chain[1:-1]  # everything between lambda0 and state_B
        print(f"\nstate_B = {state_b} hPa:")
        print(f"  state_B   = {state_b}")
        print(f"  interfaces = {ffs_interfaces}   # intermediate stepping stones")
        print(f"  (λ₀ = {args.lambda0} is the flux threshold, passed separately)")

    print(f"\nPlots saved to: {output_dir}/")


if __name__ == '__main__':
    main()
