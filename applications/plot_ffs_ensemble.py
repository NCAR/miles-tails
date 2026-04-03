#!/usr/bin/env python
"""
plot_ffs_ensemble.py — FFS ensemble figure: genesis candidates for one IC.

Uses the pre-computed committor_pts_cache.pkl (tiny — no 31MB pkl loading).
The cache stores (lat, lon, is_reactive) tuples for every interface crossing.

Shows WHERE in the Atlantic the FFS algorithm:
  (a) found storms crossing each interface threshold
  (b) identified genesis events (state B)
  (c) connected those genesis events back to λ₀ via reactive trajectories

This replaces the old ffs_tree when interface spacing is tight.

Usage
-----
    python plot_ffs_ensemble.py \\
        --ffs_config /path/to/ffs.yml \\
        --ic_dir     /path/to/results_mar18/2022-09-02T00Z \\
        --plot_dir   /path/to/plots \\
        [--max_per_level 300]   # cap scatter points per level (default 300)
        [--connect_reactive]    # draw lines between reactive crossing points
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json, pickle, argparse, warnings, yaml, sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ── Config ────────────────────────────────────────────────────────────────────
INTERFACE_PRESSURES: list = []
STATE_B: float = 975.0
N_IFACES: int = 0


def _init_config(ffs_config: dict):
    global INTERFACE_PRESSURES, STATE_B, N_IFACES
    ifaces = ffs_config['interfaces'].copy()
    STATE_B = float(ffs_config['state_B'])
    if ifaces[-1] != STATE_B:
        ifaces.append(STATE_B)
    INTERFACE_PRESSURES = ifaces
    N_IFACES = len(ifaces)


# ── map axes ──────────────────────────────────────────────────────────────────

def _make_ax(fig, lons=None, lats=None, pad=8.0):
    if lons is not None and lats is not None and len(lons) > 0:
        lon0 = max(float(np.nanmin(lons)) - pad, -110)
        lon1 = min(float(np.nanmax(lons)) + pad,   20)
        lat0 = max(float(np.nanmin(lats)) - pad,    0)
        lat1 = min(float(np.nanmax(lats)) + pad,   75)
    else:
        lon0, lon1, lat0, lat1 = -110, 20, 0, 75

    if HAS_CARTOPY:
        clat = (lat0 + lat1) / 2
        clon = (lon0 + lon1) / 2
        proj = ccrs.LambertConformal(
            central_longitude=clon,
            central_latitude=clat,
            standard_parallels=(clat - 8, clat + 8),
        )
        ax = fig.add_subplot(111, projection=proj)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e8e8e8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#d0e8f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.9, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.4, alpha=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4,
                          linestyle='--', zorder=4)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.grid(True, alpha=0.3)
    return ax


# ── main figure ───────────────────────────────────────────────────────────────

def make_ensemble_figure(ic_dir: Path, plot_dir: Path,
                         max_per_level: int = 300,
                         connect_reactive: bool = True):

    date_str = ic_dir.name
    print(f'\n=== FFS Ensemble: {date_str} ===')

    # ── 1. Load pre-computed committor cache ──────────────────────────────────
    cache_path = ic_dir / 'committor_pts_cache.pkl'
    if not cache_path.exists():
        print(f'  No cache: {cache_path}')
        return

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    # cache = {iface_idx (int): [(lat, lon, is_reactive), ...]}
    print(f'  Cache loaded. Interfaces: {sorted(cache.keys())}')
    for k in sorted(cache.keys()):
        pts = cache[k]
        n_react = sum(1 for _, _, r in pts if r)
        print(f'  λ{k}: {len(pts)} crossings  ({n_react} reactive)')

    # Map cache key → N_IFACES index
    # key 0 = λ₀ (1000 hPa), key N_IFACES-1 = last interface, key N_IFACES = state B?
    # From observed data: key 0 has most points (flux), key 4 has few (state B)
    # → keys are [0..N_IFACES]  inclusive (0=λ₀, N_IFACES=state B)
    cache_keys_sorted = sorted(cache.keys())

    # ── 2. Optional: load reactive trajectories for connection lines ──────────
    reactive_chains = []
    if connect_reactive:
        json_path = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            raw_trajs = data.get('reactive_trajectories', [])
            full_trajs = [t for t in raw_trajs
                          if t.get('pathway_length', 0) >= max(2, len(cache_keys_sorted) - 1)]
            print(f'  Full reactive trajectories: {len(full_trajs)}')

    # ── 3. Collect all positions for auto-zoom ────────────────────────────────
    all_lats, all_lons = [], []
    for k, pts in cache.items():
        for lat, lon, _ in pts:
            all_lats.append(lat)
            all_lons.append(lon)

    # ── 4. Colour scheme per interface ────────────────────────────────────────
    n_levels = len(cache_keys_sorted)
    base_cmap = plt.cm.plasma
    level_colors = [base_cmap(0.05 + 0.85 * i / max(n_levels - 1, 1))
                    for i in range(n_levels)]

    # ── 5. Draw figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    pc  = ccrs.PlateCarree() if HAS_CARTOPY else None
    ax  = _make_ax(fig,
                   lons=np.array(all_lons),
                   lats=np.array(all_lats),
                   pad=8.0)
    kw_base = dict(transform=pc) if HAS_CARTOPY else {}

    rng = np.random.default_rng(42)

    for level_idx, k in enumerate(cache_keys_sorted):
        pts = cache[k]
        is_last = (level_idx == len(cache_keys_sorted) - 1)
        color    = level_colors[level_idx]
        pressure = (INTERFACE_PRESSURES[k]
                    if k < len(INTERFACE_PRESSURES)
                    else STATE_B)

        # Split into reactive vs non-reactive
        non_react = [(lat, lon) for lat, lon, r in pts if not r]
        react     = [(lat, lon) for lat, lon, r in pts if r]

        # Sub-sample non-reactive if needed
        if max_per_level > 0 and len(non_react) > max_per_level:
            idxs = rng.choice(len(non_react), size=max_per_level, replace=False)
            non_react = [non_react[i] for i in idxs]

        # Non-reactive crossings
        if non_react:
            lats_nr = [p[0] for p in non_react]
            lons_nr = [p[1] for p in non_react]
            ax.scatter(lons_nr, lats_nr,
                       c=[color],
                       s=25 if not is_last else 50,
                       alpha=0.35 if not is_last else 0.6,
                       marker='o',
                       edgecolors='none',
                       zorder=5 + level_idx,
                       **kw_base)

        # Reactive crossings (highlighted)
        if react:
            lats_r = [p[0] for p in react]
            lons_r = [p[1] for p in react]
            ax.scatter(lons_r, lats_r,
                       c=[color],
                       s=120 if is_last else 60,
                       alpha=0.95 if is_last else 0.80,
                       marker='*' if is_last else 'D',
                       edgecolors='black' if is_last else 'none',
                       linewidths=0.5,
                       zorder=8 + level_idx,
                       **kw_base)

    # ── 6. Legend ─────────────────────────────────────────────────────────────
    handles = []
    for level_idx, k in enumerate(cache_keys_sorted):
        is_last = (level_idx == len(cache_keys_sorted) - 1)
        pressure = (INTERFACE_PRESSURES[k]
                    if k < len(INTERFACE_PRESSURES)
                    else STATE_B)
        color = level_colors[level_idx]
        n_total  = len(cache[k])
        n_react  = sum(1 for _, _, r in cache[k] if r)
        if is_last:
            label = (f'State B — genesis  ({pressure:.0f} hPa)'
                     f'  n={n_total}  (all reactive)')
        else:
            label = (f'λ{k}  {pressure:.0f} hPa  '
                     f'n={n_total}  ({n_react} reactive ◆)')
        handles.append(
            Line2D([0], [0],
                   marker='*' if is_last else 'o',
                   color='w',
                   markerfacecolor=color,
                   markersize=13 if is_last else 8,
                   markeredgecolor='black' if is_last else 'none',
                   markeredgewidth=0.5,
                   label=label)
        )

    ax.legend(handles=handles, fontsize=8.5, loc='lower left',
              framealpha=0.9,
              title='FFS interface crossings\n(solid = reactive, faded = unsuccessful)',
              title_fontsize=8.5)

    # Summary stats
    n_B = len(cache.get(max(cache_keys_sorted), []))
    n_l0 = len(cache.get(min(cache_keys_sorted), []))
    p_B  = n_B / max(n_l0, 1)
    iface_str = '→'.join(f'{p:.0f}' for p in INTERFACE_PRESSURES)

    ax.set_title(
        f'FFS Shooting Ensemble  —  IC {date_str}\n'
        f'Interfaces: {iface_str} hPa   |   '
        f'λ₀ crossings: {n_l0}   |   Genesis events: {n_B}   |   '
        f'p_B ≈ {p_B:.3f}',
        fontsize=11, fontweight='bold',
    )

    plt.tight_layout()
    out = plot_dir / f'ffs_ensemble_{date_str}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FFS ensemble — interface crossings from committor cache'
    )
    parser.add_argument('--ffs_config',    required=True)
    parser.add_argument('--ic_dir',        required=True)
    parser.add_argument('--plot_dir',      default='./plots')
    parser.add_argument('--max_per_level', type=int, default=300,
                        help='Max non-reactive points per level (0=all, default 300)')
    parser.add_argument('--no_connect',    action='store_true',
                        help='Do not draw reactive trajectory lines')
    args = parser.parse_args()

    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)
    _init_config(ffs_config)
    print(f'Interfaces: {INTERFACE_PRESSURES}  state_B={STATE_B}')

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    make_ensemble_figure(
        ic_dir        = Path(args.ic_dir),
        plot_dir      = plot_dir,
        max_per_level = args.max_per_level,
        connect_reactive = not args.no_connect,
    )


if __name__ == '__main__':
    main()
