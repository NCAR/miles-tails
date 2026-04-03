#!/usr/bin/env python
"""
plot_trajectory_physics.py — Physics maps along a single FFS reactive pathway.

Traces one State-B trajectory (λ0 → λ1 → … → State B) and plots storm-centred
physics for each step.  Fast: only one pkl per interface level.

Usage:
    python plot_trajectory_physics.py \
        --model_config model.yml \
        --ffs_config results_feb14/ffs.yml \
        --ic_time "2022-08-21 00:00:00" \
        [--pathway_idx 0]     # which State-B event (sorted by min MSLP, default: 0)
        [--list_pathways]     # just print available pathways and exit
        [--box_deg 12]
        [--plot_dir /path/to/plots]
        [--plot_vws]
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import matplotlib
matplotlib.use('Agg')

import argparse
import warnings
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

warnings.filterwarnings('ignore')

_here        = Path(__file__).resolve().parent
_credit_root = _here.parents[1] / 'miles-credit-main'
for _p in [str(_here), str(_credit_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-use shared physics infrastructure from the composites script
from plot_physics_composites import (
    FIELDS, FIELD_KEYS, STATIC_NC,
    _worker_init, _process_pkl,
    _box_axes, _plot_panel,
)
from analyze_ffs_logs import (
    load_all_logs,
    build_genealogy,
    trace_pathway,
    find_stateB_configs,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_ic(ic_time_str: str) -> str:
    from datetime import datetime
    return datetime.strptime(ic_time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%HZ')


def _find_pkl(ic_dir: Path, config_name: str) -> Path | None:
    """Locate the pkl for a config name using the naming convention."""
    if config_name.startswith('stateB_'):
        c = ic_dir / 'stateB' / f'{config_name}.pkl'
    elif config_name.startswith('lambda0_'):
        c = ic_dir / 'flux' / f'{config_name}.pkl'
    else:
        try:
            n = int(config_name.split('_')[0].replace('lambda', ''))
            c = ic_dir / str(n) / f'{config_name}.pkl'
        except (ValueError, IndexError):
            c = None

    if c and c.exists():
        return c

    # Fallback: recursive search (slow but safe)
    for p in ic_dir.rglob(f'{config_name}.pkl'):
        return p
    return None


# ── plotting ──────────────────────────────────────────────────────────────────

def _setup_atlantic_ax(ax):
    """Add basemap features to an existing axes (GeoAxes or plain)."""
    if HAS_CARTOPY:
        transform = ccrs.PlateCarree()
        ax.set_extent([-100, -10, 5, 65], crs=transform)
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e8e8e8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#d0e8f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.6, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.3, alpha=0.4, zorder=3)
        gl = ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.3,
                          linestyle='--', zorder=4)
    else:
        ax.set_xlim(-100, -10)
        ax.set_ylim(0, 70)
        ax.grid(True, alpha=0.3)


def _draw_track_panel(ax, all_steps, up_to_idx):
    """
    Draw an Atlantic basin map for one column.
    - Full future track shown as light grey dashed line for context.
    - Track from step 0 to up_to_idx drawn as solid colored line.
    - Current step (up_to_idx) highlighted with a large marker.
    - Past steps shown as smaller dots.
    """
    all_lats = [s['result']['storm_lat'] for s in all_steps]
    all_lons = [s['result']['storm_lon'] for s in all_steps]

    past_lats = all_lats[:up_to_idx + 1]
    past_lons = all_lons[:up_to_idx + 1]

    cmap   = plt.get_cmap('plasma_r', len(all_steps))
    colors = [cmap(i) for i in range(len(all_steps))]

    _setup_atlantic_ax(ax)

    kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}

    # Full track in grey dashed (context)
    ax.plot(all_lons, all_lats, color='#aaaaaa', linewidth=0.8,
            linestyle='--', alpha=0.5, zorder=5, **kw)

    # Track up to current step in solid black
    if len(past_lons) > 1:
        ax.plot(past_lons, past_lats, color='k', linewidth=1.4,
                alpha=0.7, zorder=6, **kw)

    # All positions — uniform dot size; current step outlined more boldly
    for i in range(up_to_idx + 1):
        is_cur = (i == up_to_idx)
        ax.scatter(all_lons[i], all_lats[i], color=colors[i], s=30,
                   edgecolors='k', linewidths=0.8 if is_cur else 0.4,
                   zorder=7, **kw)


def plot_trajectory(steps, box_deg, output_path, title, field_keys=None):
    """
    N_fields rows × N_steps cols.
    Each panel is a single config's storm-centred physics box (no averaging).
    """
    if field_keys is None:
        field_keys = [k for k in FIELD_KEYS if k != 'vws']

    n_fields = len(field_keys)
    n_steps  = len(steps)
    if n_steps == 0 or n_fields == 0:
        return

    dlat, dlon = _box_axes(box_deg)

    fig_w = max(3.2 * n_steps, 10)
    map_h = 3.2                            # map row same height as physics rows
    fig_h = map_h + 3.2 * n_fields

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: row 0 = per-column track maps, rows 1..n_fields = physics
    gs = GridSpec(
        n_fields + 1, n_steps,
        figure=fig,
        height_ratios=[map_h] + [3.2] * n_fields,
        hspace=0.35, wspace=0.15,
    )

    # ── Track map row — one small map per column ───────────────────────────────
    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(central_longitude=-60.0,
                                     central_latitude=35.0,
                                     standard_parallels=(30, 50))

    for col, step in enumerate(steps):
        if HAS_CARTOPY:
            map_ax = fig.add_subplot(gs[0, col], projection=proj)
        else:
            map_ax = fig.add_subplot(gs[0, col])
        _draw_track_panel(map_ax, steps, up_to_idx=col)
        lbl    = step['label']
        mslp   = step.get('mslp')
        mslp_s = f"{mslp:.1f} hPa" if isinstance(mslp, float) else str(mslp)
        lat    = step['result'].get('storm_lat', float('nan'))
        map_ax.set_title(f"{lbl}\n{mslp_s}\n{lat:.1f}°N",
                         fontsize=8, fontweight='bold')

    # ── Physics rows ───────────────────────────────────────────────────────────
    axes = np.empty((n_fields, n_steps), dtype=object)
    for col in range(n_steps):
        for row in range(n_fields):
            axes[row, col] = fig.add_subplot(gs[row + 1, col])

    row_pcm = [None] * n_fields

    for col, step in enumerate(steps):
        result = step['result']

        for row, key in enumerate(field_keys):
            ax   = axes[row, col]
            meta = FIELDS[key]
            data = result['boxes'].get(key) if result else None
            pcm  = _plot_panel(ax, data, meta, dlat, dlon, n_configs=1)

            if pcm is not None and row_pcm[row] is None:
                row_pcm[row] = pcm

            if col == 0:
                ax.set_ylabel(f"{meta['label']}\n({meta['unit']})", fontsize=8)
            else:
                ax.set_yticklabels([])

            if row == n_fields - 1:
                ax.set_xlabel('Δlon (°)', fontsize=8)
            else:
                ax.set_xticklabels([])

    fig.suptitle('', y=1.01)
    fig.subplots_adjust(right=0.88)

    for row, key in enumerate(field_keys):
        if row_pcm[row] is None:
            continue
        y0 = axes[row, -1].get_position().y0
        y1 = axes[row,  0].get_position().y1
        cax = fig.add_axes([0.895, y0, 0.012, y1 - y0])
        fig.colorbar(row_pcm[row], cax=cax, label=FIELDS[key]['unit'])
        cax.tick_params(labelsize=7)
        cax.yaxis.label.set_size(8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Physics maps along a single FFS reactive pathway'
    )
    parser.add_argument('--model_config',  required=True)
    parser.add_argument('--ffs_config',    required=True)
    parser.add_argument('--ic_time',       required=True,
                        help='IC time, e.g. "2022-08-21 00:00:00"')
    parser.add_argument('--pathway_idx',   type=int, default=0,
                        help='Which State-B event to trace, sorted by min MSLP (default: 0 = most intense)')
    parser.add_argument('--list_pathways', action='store_true',
                        help='Print all available State-B pathways and exit')
    parser.add_argument('--plot_dir',      default=None)
    parser.add_argument('--box_deg',       type=int, default=12)
    parser.add_argument('--plot_vws',      action='store_true',
                        help='Include VWS row (default: off)')
    args = parser.parse_args()

    field_keys = [k for k in FIELD_KEYS if k != 'vws' or args.plot_vws]

    with open(args.model_config) as fh:
        model_config = yaml.safe_load(fh)
    with open(args.ffs_config) as fh:
        ffs_config = yaml.safe_load(fh)

    model_config.setdefault('data', {})

    latlons_path = model_config['loss']['latitude_weights']
    ffs_out      = Path(ffs_config['output_dir'])
    state_B      = ffs_config['state_B']
    time_label   = _fmt_ic(args.ic_time)
    ic_dir       = ffs_out / time_label
    plot_dir     = Path(args.plot_dir) if args.plot_dir else ffs_out / 'physics' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not ic_dir.exists():
        print(f"IC directory not found: {ic_dir}")
        return

    logs_dir = ic_dir / 'logs'
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    print(f"Loading logs from {logs_dir} …")
    entries   = load_all_logs(logs_dir)
    genealogy = build_genealogy(entries)
    b_configs = find_stateB_configs(entries, state_B)

    if not b_configs:
        print("No State-B events found.")
        return

    # Sort most intense (lowest MSLP) first
    b_configs_sorted = sorted(
        [b for b in b_configs if b.get('mslp') is not None],
        key=lambda x: x['mslp'],
    )

    if args.list_pathways:
        print(f"\n{'idx':>4}  {'config':<40}  {'min MSLP':>9}")
        print('-' * 60)
        for i, b in enumerate(b_configs_sorted):
            print(f"{i:>4}  {b['config']:<40}  {b['mslp']:>9.2f}")
        return

    if args.pathway_idx >= len(b_configs_sorted):
        print(f"pathway_idx {args.pathway_idx} out of range "
              f"(0–{len(b_configs_sorted) - 1})")
        return

    target  = b_configs_sorted[args.pathway_idx]
    pathway = trace_pathway(genealogy, target['config'])

    print(f"\nSelected pathway {args.pathway_idx}  "
          f"(State-B MSLP = {target['mslp']:.2f} hPa)")
    print(f"{'Step':<6}  {'config':<45}  {'interface'}")
    print('-' * 65)
    for step in pathway:
        iface = step.get('interface_idx', step.get('interface', '?'))
        print(f"  {str(iface):<6}  {step.get('config', '?'):<45}")

    # Initialise worker globals (single process — no pool needed)
    _worker_init(latlons_path, STATIC_NC)

    steps_out = []
    for step in pathway:
        config_name = step.get('config')
        if config_name is None:
            continue

        # Label by the config's own interface level, not the edge crossed
        cname = step.get('config', '')
        if cname.startswith('stateB_'):
            lbl = "State B"
        else:
            try:
                lv = int(cname.split('_')[0].replace('lambda', ''))
                lbl = f"λ{lv}"
            except (ValueError, IndexError):
                lbl = "State B"

        pkl_path = _find_pkl(ic_dir, config_name)
        if pkl_path is None:
            print(f"  ✗ pkl not found for {config_name}, skipping")
            continue

        print(f"  Processing {lbl}: {pkl_path.name} …", end=' ', flush=True)
        result = _process_pkl((str(pkl_path), model_config, args.box_deg))

        if result is None or 'error' in result:
            err = (result.get('error', 'None returned')[:80]
                   if result else 'None returned')
            print(f"✗  {err}")
            continue

        print(f"✓  MSLP={result.get('mslp_value', '?'):.2f} hPa  "
              f"lat={result.get('storm_lat', float('nan')):.1f}°N")

        steps_out.append({
            'label':  lbl,
            'mslp':   result.get('mslp_value'),
            'result': result,
        })

    if not steps_out:
        print("No steps processed successfully.")
        return

    out_path = (plot_dir /
                f'trajectory_{time_label}_pathway{args.pathway_idx}.png')
    plot_trajectory(
        steps_out, args.box_deg, out_path,
        title=(f'Physics along reactive pathway — {time_label}  '
               f'(pathway {args.pathway_idx}, '
               f'State-B MSLP = {target["mslp"]:.2f} hPa)'),
        field_keys=field_keys,
    )
    print("\nDone.")


if __name__ == '__main__':
    main()
