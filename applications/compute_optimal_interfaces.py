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


def subdivide_stage(min_mslps, p_target, state_b, min_n=20):
    """
    Extract as many optimal interfaces as the data supports from one stage.

    Algorithm:
      1. pool = all min-MSLPs from this stage
      2. opt = p_target quantile of pool  → candidate next interface
      3. If opt > state_b and pool has >= min_n entries: record opt, then
         filter pool to trajectories that passed through opt (min < opt),
         and repeat from step 2.

    This works because trajectories launched from λ_i that crossed some
    intermediate threshold y are unbiased samples of "what happens after y"
    (they carry the atmospheric state at the moment they crossed y).
    Sample size shrinks by ~p_target each iteration, so precision degrades —
    hence the min_n guard.

    Returns list of interface values (not including the stage launch point
    or state_b itself).
    """
    pool = np.asarray(min_mslps, dtype=float)
    interfaces = []

    while len(pool) >= min_n:
        opt = float(np.percentile(pool, p_target * 100))

        # Stop if we've reached or passed state_b
        if opt <= state_b:
            break

        interfaces.append(round(opt, 1))

        # Keep only trajectories that penetrated strictly past this interface.
        # This guarantees pool shrinks each iteration → no infinite loop.
        pool = pool[pool < opt]

    return interfaces


def build_optimal_chain(by_stage, lambda0, state_b, p_target, min_n=20, verbose=False):
    """
    Build the optimal interface chain from lambda0 to state_b.

    For each shooting stage (lambda_label >= 0), the p_target quantile of
    that stage's min-MSLP distribution gives the optimal next interface:
    exactly p_target fraction of trajectories launched from that stage will
    cross it.  One interface per stage.  Flux (label=-1) is skipped — its
    min-MSLP is the basin minimum over a full run, not a single storm track.

    If a stage's p_target quantile is already at or below state_b, that
    stage contributes no new interface (the chain is already at state_b).

    min_n: reserved for future sub-division support (currently unused).

    Returns:
        chain      : list of hPa values [lambda0, λ1*, λ2*, ..., state_b]
        stage_stats: dict with per-stage diagnostics
    """
    # Only use shooting stages (label >= 0), ordered shallowest → deepest
    shooting_labels = sorted(k for k in by_stage if k >= 0)
    stage_stats = {}
    chain = [lambda0]

    for label in shooting_labels:
        mins = np.asarray(by_stage[label])
        n = len(mins)
        stage_name = f"λ{label}"

        opt_next = float(np.percentile(mins, p_target * 100))
        p_at_opt = float(np.mean(mins < opt_next))
        p_to_b   = float(np.mean(mins < state_b))

        above_stateB = opt_next > state_b
        below_tip    = opt_next < chain[-1] - 0.5

        if verbose:
            if above_stateB and below_tip:
                reason = '→ added'
            elif not above_stateB:
                reason = f'→ skipped (λ_opt={opt_next:.1f} ≤ state_B={state_b})'
            else:
                reason = f'→ skipped (λ_opt={opt_next:.1f} too close to tip={chain[-1]:.1f})'
            print(f"  [{stage_name}] N={n}  λ_opt={opt_next:.1f}  P@opt={p_at_opt:.3f}  "
                  f"P→B={p_to_b:.3f}  {reason}")

        stage_stats[label] = {
            'name': stage_name,
            'n': n,
            'min_mslp_mean': float(np.mean(mins)),
            'min_mslp_min':  float(np.min(mins)),
            'opt_next': round(opt_next, 1),
            'p_at_opt': p_at_opt,
            'p_to_stateB': p_to_b,
            'n_sub_interfaces': int(above_stateB and below_tip),
            'sub_interfaces': [round(opt_next, 1)] if (above_stateB and below_tip) else [],
        }

        if above_stateB and below_tip:
            chain.append(round(opt_next, 1))

    if chain[-1] != state_b:
        chain.append(state_b)

    return chain, stage_stats


def build_chain_target_n(by_stage, lambda0, state_b, n_steps, min_n=20, verbose=False):
    """
    Build a chain with exactly n_steps transitions (lambda0 → i1 → … → state_b)
    using the λ0 min-MSLP distribution as the reference.

    Solves for the per-step crossing probability:
        p_step = P(min < state_b | λ0) ^ (1 / n_steps)
    Interface k is placed at the p_step^k quantile of λ0's distribution, giving:
        P(cross step k | crossed step k-1) ≈ p_step   (Markov property)

    This allows you to request more (or fewer) steps than the number of original
    stages.  With 273K λ0 trajectories, precision is good down to ~0.5% quantiles.

    Args:
        n_steps : total number of transitions  (chain length = n_steps + 1,
                  including lambda0 and state_b)
        min_n   : unused; kept for API compatibility with build_optimal_chain
    """
    label0 = min(k for k in by_stage if k >= 0)
    mins0  = np.asarray(by_stage[label0])
    n0     = len(mins0)

    p_to_b = float(np.mean(mins0 < state_b))
    if p_to_b == 0:
        print(f"WARNING: No λ0 trajectories reached state_B={state_b:.0f} hPa. Cannot build chain.")
        return [lambda0, state_b], {}

    p_step = p_to_b ** (1.0 / n_steps)

    if verbose:
        print(f"  λ0 (N={n0})  P(min<{state_b:.0f})={p_to_b:.5f}"
              f"  →  p_step = {p_step:.4f}  (1/e = {1/np.e:.4f})")

    chain      = [lambda0]
    step_stats = {}

    for k in range(1, n_steps):
        q     = p_step ** k
        iface = round(float(np.percentile(mins0, q * 100)), 1)

        # Conditional probability from the previous step
        p_prev = float(np.mean(mins0 < chain[-1])) if k > 1 else 1.0
        p_here = float(np.mean(mins0 < iface))
        p_cond = p_here / p_prev if p_prev > 0 else 0.0

        stop = iface <= state_b or iface >= chain[-1] - 0.5
        if verbose:
            note = (' → stop (at/past state_B)' if iface <= state_b
                    else ' → stop (too close to tip)' if iface >= chain[-1] - 0.5
                    else '')
            print(f"    k={k}: q={q*100:.3f}th pct → {iface:.1f} hPa  "
                  f"p_cond≈{p_cond:.3f}{note}")

        step_stats[k] = {
            'name':           f'step{k}',
            'n':              n0,
            'min_mslp_mean':  float(np.mean(mins0)),
            'min_mslp_min':   float(np.min(mins0)),
            'opt_next':       iface,
            'p_at_opt':       p_cond,
            'p_to_stateB':    p_to_b,
            'n_sub_interfaces': int(not stop),
            'sub_interfaces': [iface] if not stop else [],
        }

        if stop:
            break
        chain.append(iface)

    if chain[-1] != state_b:
        chain.append(state_b)

    return chain, step_stats


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
    parser.add_argument('--n_interfaces', type=int, default=None,
                        help='Request exactly this many steps from λ₀ to state_B '
                             '(e.g. --n_interfaces 5 gives chain of length 6 incl. endpoints). '
                             'Overrides the default one-interface-per-stage heuristic. '
                             'Uses the λ0 distribution to solve for the required p_step.')
    parser.add_argument('--min_n', type=int, default=20,
                        help='Min trajectories required for sub-interface subdivision (reserved for future use; '
                             'current build uses one interface per stage, default: 20)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-stage sub-interfaces and merge steps')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\np_target     = {args.p_target:.4f}  (1/e = {1/np.e:.4f})")
    print(f"λ₀           = {args.lambda0} hPa")
    print(f"state_B      = {args.state_b} hPa")
    if args.n_interfaces is not None:
        print(f"n_interfaces = {args.n_interfaces}  (fixed-step mode: p_step solved per state_B)")
    print(f"log_dirs     = {args.log_dir} ({len(args.log_dir)} directories)\n")

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
    chains_by_stateB     = {}
    stage_stats_by_stateB = {}
    pstep_by_stateB       = {}   # only populated in fixed-step mode

    for state_b in args.state_b:
        if args.verbose:
            print(f"\n--- state_B = {state_b} hPa ---")

        if args.n_interfaces is not None:
            chain, stage_stats = build_chain_target_n(
                by_stage, args.lambda0, state_b, args.n_interfaces,
                min_n=args.min_n, verbose=args.verbose
            )
            # Record the p_step that was actually used
            label0  = min(k for k in by_stage if k >= 0)
            mins0   = np.asarray(by_stage[label0])
            p_to_b  = float(np.mean(mins0 < state_b))
            p_step  = p_to_b ** (1.0 / args.n_interfaces) if p_to_b > 0 else float('nan')
            pstep_by_stateB[state_b] = p_step
        else:
            chain, stage_stats = build_optimal_chain(
                by_stage, args.lambda0, state_b, args.p_target,
                min_n=args.min_n, verbose=args.verbose
            )

        chains_by_stateB[state_b]      = chain
        stage_stats_by_stateB[state_b] = stage_stats

    # ── Per-step P bar chart ──────────────────────────────────────────────────
    bar_plot = output_dir / 'optimal_chain_Ps.png'
    plot_optimal_chains(chains_by_stateB, stage_stats_by_stateB,
                         args.lambda0, args.p_target, bar_plot)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PER-STAGE DIAGNOSTICS")
    print("=" * 65)

    # In fixed-step mode, each state_B gets its own step_stats; print per state_B.
    # In per-stage mode, stats are the same for all state_B, so print once.
    stats_to_print = (stage_stats_by_stateB if args.n_interfaces is not None
                      else {None: next(iter(stage_stats_by_stateB.values()))})

    for sb_key, first_stats in stats_to_print.items():
        if sb_key is not None:
            print(f"state_B = {sb_key} hPa:")
        header = (f"{'Step/Stage':<20} {'N':>6} {'mean_min':>9} {'abs_min':>9} "
                  f"{'λ_opt':>8} {'P@opt':>7}  sub_interfaces")
        print(header)
        print("-" * 85)
        for label in sorted(first_stats):
            s = first_stats[label]
            print(f"{s['name']:<20} {s['n']:>6} "
                  f"{s['min_mslp_mean']:>9.1f} {s['min_mslp_min']:>9.1f} "
                  f"{s['opt_next']:>8.1f} {s['p_at_opt']:>7.3f}  {s['sub_interfaces']}")
        print()

    print("\n" + "=" * 65)
    print("OPTIMAL INTERFACE CHAINS")
    print("=" * 65)
    if args.n_interfaces is not None:
        print(f"(fixed-step mode: n_interfaces={args.n_interfaces}, p_step solved per state_B)\n")
    else:
        print(f"(p_target = {args.p_target:.4f},  gaps chosen so P(cross) ≈ p*)\n")

    for state_b in args.state_b:
        chain = chains_by_stateB[state_b]
        stats = stage_stats_by_stateB[state_b]

        p_to_b_per_stage = {
            label: stats[label]['p_to_stateB']
            for label in sorted(stats)
        }

        print(f"state_B = {state_b} hPa")
        if args.n_interfaces is not None:
            p_step = pstep_by_stateB.get(state_b, float('nan'))
            print(f"  p_step   = {p_step:.4f}  (cf. 1/e = {1/np.e:.4f})")
        print(f"  Proposed: interfaces = {chain}")
        n_steps_actual = len(chain) - 1
        print(f"  N steps  = {n_steps_actual}")
        gaps = [round(chain[i] - chain[i+1], 1) for i in range(len(chain)-1)]
        print(f"  Gaps     = {gaps} hPa")

        # Cumulative crossing probability
        p_vals  = [stats[lbl]['p_at_opt'] for lbl in sorted(stats)]
        p_prod  = np.prod(p_vals) if p_vals else 0.0
        print(f"  Est. cumulative P(λ₀ → state_B) ≈ {p_prod:.5f}")

        print(f"  P(reach state_B) from reference stage:")
        for label in sorted(stats):
            s = stats[label]
            print(f"    {s['name']}: {s['p_to_stateB']:.4f}")
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
