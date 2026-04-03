#!/usr/bin/env python
"""
plot_committor_map.py — Spatial committor p_B(x | λᵢ) across the Atlantic.

For each FFS interface λᵢ, bins every config that crossed that interface by
lat/lon and estimates the local committor as:

    p_B(bin) = configs_with_B_descendants / configs_total

Aggregated across all ICs, this gives a spatial portrait of where storms are
most likely to intensify at each stage of the FFS cascade — the hurricane-
genesis analog of a binding-site probability map.

Notes
-----
- Uses ALL configs in each interface directory (not just reactive ones), so
  both successful and failed attempts contribute to the denominator.
- state_B panel (final interface) always has p_B ≈ 1 by definition — shown
  as a genesis location density map rather than a committor.
- Full atmospheric state extraction (shear, SST, etc.) is deferred; only
  lat/lon from feature_location is used here.

Usage
-----
    python plot_committor_map.py \\
        --ffs_config ffs.yml \\
        --ffs_csv    results/ffs_statistics_all_ics.csv \\
        --output_dir results \\
        --plot_dir   results/plots \\
        --workers    8 \\
        --bin_size   2.0 \\
        --min_samples 3
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import matplotlib
matplotlib.use('Agg')

import pickle
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from analyze_ffs_logs import (
    load_all_logs,
    build_genealogy,
    find_stateB_configs,
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — falling back to plain lat/lon axes')


# ── pkl helpers ───────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _iface_dir(ic_dir: Path, iface_idx: int, n_ifaces: int) -> Path:
    """Map interface index → directory name."""
    if iface_idx == 0:
        return ic_dir / 'flux'
    if iface_idx == n_ifaces - 1:
        return ic_dir / 'stateB'
    return ic_dir / str(iface_idx)


def _load_iface_pkls(ic_dir: Path, iface_idx: int, n_ifaces: int) -> list:
    """
    Load (config_name, lat, lon) for every pkl in an interface directory.
    Uses a ThreadPoolExecutor for parallel I/O.
    """
    pkl_dir = _iface_dir(ic_dir, iface_idx, n_ifaces)
    if not pkl_dir.exists():
        return []

    pkl_paths = list(pkl_dir.glob('*.pkl'))
    if not pkl_paths:
        return []

    def _load_one(p):
        cfg = _load_pkl(p)
        if cfg is None:
            return None
        loc = getattr(cfg, 'feature_location', None)
        if loc is None:
            return None
        try:
            return p.stem, float(loc[0]), float(loc[1])
        except (TypeError, IndexError):
            return None

    results = []
    with ThreadPoolExecutor(max_workers=min(32, len(pkl_paths))) as ex:
        for r in ex.map(_load_one, pkl_paths):
            if r is not None:
                results.append(r)
    return results


# ── genealogy helper ──────────────────────────────────────────────────────────

def _build_b_reachable_set(genealogy: dict, stateB_configs: list) -> set:
    """
    Backward BFS from every state-B config through the success-only genealogy.
    Returns the set of all config names that have at least one B-descendant.
    """
    reverse = {}
    for parent, children in genealogy.items():
        for info in children:
            if info.get('status') in ('success', 'reached_B', 'instant_success'):
                c = info.get('child')
                if c:
                    reverse[c] = parent

    b_reachable = set()
    queue = [s['config'] for s in stateB_configs]
    while queue:
        cur = queue.pop()
        if cur in b_reachable:
            continue
        b_reachable.add(cur)
        par = reverse.get(cur)
        if par and par not in b_reachable:
            queue.append(par)

    return b_reachable


# ── Cache helpers ─────────────────────────────────────────────────────────────

_CACHE_NAME = 'committor_pts_cache.pkl'

def _load_cache(ic_dir: Path):
    """Return cached {iface_idx: [(lat,lon,reached_B)]} or None."""
    p = ic_dir / _CACHE_NAME
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(ic_dir: Path, result: dict):
    try:
        with open(ic_dir / _CACHE_NAME, 'wb') as f:
            pickle.dump(result, f, protocol=4)
    except Exception:
        pass   # non-fatal — script still works without cache


# ── Per-IC pts computation (with cache) ──────────────────────────────────────

def _compute_ic_pts(ic_dir: Path, state_B: float,
                    n_ifaces: int, no_cache: bool) -> dict:
    """
    Return {iface_idx: [(lat, lon, reached_B), ...]} for one IC.
    Loads from cache if available; saves cache after first computation.
    """
    if not no_cache:
        cached = _load_cache(ic_dir)
        if cached is not None:
            return cached

    logs_dir = ic_dir / 'logs'
    if not logs_dir.exists():
        return {}

    try:
        entries     = load_all_logs(logs_dir)
        genealogy   = build_genealogy(entries)
        stateB_cfgs = find_stateB_configs(entries, state_B)
    except Exception:
        return {}

    if not stateB_cfgs:
        return {}

    b_reachable = _build_b_reachable_set(genealogy, stateB_cfgs)
    stateB_set  = {s['config'] for s in stateB_cfgs}

    result = {}
    for iface_idx in range(n_ifaces):
        configs = _load_iface_pkls(ic_dir, iface_idx, n_ifaces)
        if not configs:
            continue
        result[iface_idx] = [
            (lat, lon, (cname in b_reachable) or (cname in stateB_set))
            for cname, lat, lon in configs
        ]

    _save_cache(ic_dir, result)
    return result


# ── Combined per-IC worker: compute + plot (module-level for pickling) ────────

def _ic_plot_worker(args: tuple):
    """
    Compute committor pts for one IC and render its figure.
    Runs entirely inside a subprocess — matplotlib is process-safe.
    """
    ic_dir_str, state_B, n_ifaces, ifaces, plot_dir_str, bin_size, min_samples, no_cache = args
    ic_dir   = Path(ic_dir_str)
    plot_dir = Path(plot_dir_str)

    pts_by_iface = _compute_ic_pts(ic_dir, state_B, n_ifaces, no_cache)
    if not pts_by_iface:
        return

    _plot_ic(pts_by_iface, ifaces, ic_dir.name, plot_dir, bin_size, min_samples)


# ── Binning ───────────────────────────────────────────────────────────────────

def bin_committor(pts: list, bin_size: float, min_samples: int):
    """
    Bin (lat, lon, reached_B) onto a regular grid.

    Returns
    -------
    lon_edges, lat_edges : 1D arrays  (for pcolormesh)
    p_B                  : 2D masked array  (lat × lon) — masked where n < min_samples
    n_total              : 2D array  raw config counts per bin
    """
    lon_edges = np.arange(-102, -7  + bin_size, bin_size)
    lat_edges = np.arange(   3,  67 + bin_size, bin_size)

    if not pts:
        shape = (len(lat_edges) - 1, len(lon_edges) - 1)
        return lon_edges, lat_edges, np.ma.masked_all(shape), np.zeros(shape)

    lats = np.array([p[0] for p in pts])
    lons = np.array([p[1] for p in pts])
    reached = np.array([p[2] for p in pts], dtype=float)

    n_total, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    n_b,     _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges],
                                   weights=reached)

    with np.errstate(invalid='ignore', divide='ignore'):
        p_B = np.where(n_total >= min_samples, n_b / n_total, np.nan)

    p_B = np.ma.masked_invalid(p_B)
    return lon_edges, lat_edges, p_B, n_total


# ── Map axes ──────────────────────────────────────────────────────────────────

def _make_ax(fig, pos):
    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(
            central_longitude=-60, central_latitude=30,
            standard_parallels=(20, 45),
        )
        ax = fig.add_subplot(pos, projection=proj)
        ax.set_extent([-100, -10, 5, 65], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#d8d8d8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#cce5f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.6, zorder=3)
        ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.35,
                     linestyle='--', zorder=4)
    else:
        ax = fig.add_subplot(pos)
        ax.set_xlim(-100, -10)
        ax.set_ylim(5, 65)
        ax.grid(True, alpha=0.3)
    return ax


# ── Per-IC figure ─────────────────────────────────────────────────────────────

def _plot_ic(iface_data: dict, interfaces: list, ic_name: str,
             plot_dir: Path, bin_size: float, min_samples: int):
    n     = len(interfaces)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig      = plt.figure(figsize=(8 * ncols, 6 * nrows))
    cmap     = plt.cm.RdYlGn
    norm     = mcolors.Normalize(vmin=0, vmax=1)
    mappable = None

    for iface_idx, pressure in enumerate(interfaces):
        pts = iface_data.get(iface_idx, [])
        pos = nrows * 100 + ncols * 10 + (iface_idx + 1)
        ax  = _make_ax(fig, pos)

        n_total = len(pts)
        n_b     = sum(1 for _, _, rb in pts if rb)

        if iface_idx == 0:
            label = 'λ₀  (flux seed)'
        elif iface_idx == n - 1:
            label = 'State B  (genesis density)'
        else:
            label = f'λ{iface_idx}'

        if n_total == 0:
            ax.set_title(f'{label}\n{pressure} hPa  —  no data', fontsize=9)
            continue

        lon_edges, lat_edges, p_B, _ = bin_committor(pts, bin_size, min_samples)

        pkw = dict(cmap=cmap, norm=norm, alpha=0.88, zorder=5)
        if HAS_CARTOPY:
            pkw['transform'] = ccrs.PlateCarree()

        mappable = ax.pcolormesh(lon_edges, lat_edges, p_B, **pkw)

        frac = f'{n_b}/{n_total}  ({100*n_b/max(1,n_total):.0f}%) reach B'
        ax.set_title(
            f'{label}  |  {pressure} hPa\n'
            f'p_B  ·  {frac}  ·  {bin_size}° bins  ≥{min_samples} configs',
            fontsize=9, fontweight='bold',
        )

    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
        cb = fig.colorbar(mappable, cax=cbar_ax)
        cb.set_label('Committor  p_B', fontsize=11)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        f'Spatial committor  p_B(x | λᵢ)  —  IC {ic_name}\n'
        f'Fraction of configs at each location that eventually reached state B',
        fontsize=13, fontweight='bold', y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out = plot_dir / f'committor_map_{ic_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Per-IC 2D spatial committor p_B(x|λᵢ) maps from FFS output'
    )
    parser.add_argument('--ffs_config',  required=True,
                        help='FFS config file (ffs.yml)')
    parser.add_argument('--ffs_csv',     required=True,
                        help='FFS statistics CSV (for IC time_labels)')
    parser.add_argument('--output_dir',  required=True,
                        help='FFS output directory containing IC subdirectories')
    parser.add_argument('--plot_dir',    default='./plots')
    parser.add_argument('--workers',     type=int, default=min(8, os.cpu_count() or 1),
                        help='Parallel workers (default: min(8, ncpu))')
    parser.add_argument('--bin_size',    type=float, default=2.0,
                        help='Bin size in degrees (default: 2.0)')
    parser.add_argument('--min_samples', type=int, default=3,
                        help='Min configs per bin to display p_B (default: 3)')
    parser.add_argument('--no_cache',    action='store_true',
                        help='Ignore per-IC cache and recompute from pkls')
    args = parser.parse_args()

    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)

    ifaces  = ffs_config['interfaces'].copy()
    state_B = float(ffs_config['state_B'])
    if ifaces[-1] != state_B:
        ifaces.append(state_B)
    n_ifaces = len(ifaces)
    print(f'Interfaces: {ifaces}  state_B={state_B}')

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.ffs_csv)
    out_dir = Path(args.output_dir)
    ic_dirs = [out_dir / tl for tl in df['time_label'].tolist()
               if (out_dir / tl).exists()]
    print(f'Found {len(ic_dirs)} IC directories')

    worker_args = [
        (str(d), state_B, n_ifaces, ifaces, str(plot_dir),
         args.bin_size, args.min_samples, args.no_cache)
        for d in ic_dirs
    ]
    n_workers = min(args.workers, len(ic_dirs))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_ic_plot_worker, a): Path(a[0]) for a in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='ICs', unit='IC', dynamic_ncols=True):
            ic_dir_path = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f'  ERROR {ic_dir_path.name}: {e}')

    print(f'\nDone. Figures saved to {plot_dir}/')


if __name__ == '__main__':
    main()
