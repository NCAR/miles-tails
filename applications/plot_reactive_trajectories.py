#!/usr/bin/env python
"""
plot_reactive_trajectories.py — reactive trajectory tracks on the Atlantic map.

For every reactive trajectory (λ₀ → state_B), loads each pkl in the
pathway chain, extracts feature_location (lat, lon) + MSLP, and plots
the ensemble of tracks.

Parallelised at the IC level: one worker per IC directory.
Interfaces and state_B are read from ffs.yml — no hardcoded pressures.

Usage:
    python plot_reactive_trajectories.py \
        --ffs_config ffs.yml \
        --ffs_csv    results/ffs_statistics_all_ics.csv \
        --output_dir results \
        --plot_dir   results/plots \
        --workers    16
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import json
import pickle
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — falling back to plain lat/lon axes')

# ── Module-level config — overwritten from ffs.yml in main() ──────────────────
# These are placeholders; all code uses them after main() initialises them.
INTERFACE_PRESSURES: list      = []
N_IFACES:            int       = 0
IFACE_COLORS:        object    = None   # np.ndarray after init

# Track line style
TRACK_COLOR    = '#333333'
LW_MIN, LW_MAX = 0.4, 3.5
DT_HOURS       = 6.0


def _init_from_config(ffs_config: dict):
    """Set module-level interface constants from the loaded ffs.yml dict."""
    global INTERFACE_PRESSURES, N_IFACES, IFACE_COLORS
    ifaces  = ffs_config['interfaces'].copy()
    state_B = float(ffs_config['state_B'])
    if ifaces[-1] != state_B:
        ifaces.append(state_B)
    INTERFACE_PRESSURES = ifaces
    N_IFACES            = len(ifaces)
    IFACE_COLORS        = plt.cm.YlOrRd(np.linspace(0.25, 0.95, N_IFACES))


# ── pkl helpers ───────────────────────────────────────────────────────────────

def _pkl_path(ic_dir: Path, config_name: str) -> Path:
    if config_name.startswith('lambda0_'):
        return ic_dir / 'flux' / f'{config_name}.pkl'
    elif config_name.startswith('stateB_'):
        return ic_dir / 'stateB' / f'{config_name}.pkl'
    else:
        iface_num = config_name.split('_')[0].replace('lambda', '')
        return ic_dir / iface_num / f'{config_name}.pkl'


def _load_pkl(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _cal_time(cfg) -> datetime:
    return cfg.restart_datetime + timedelta(hours=cfg.forecast_step * DT_HOURS)


# ── Per-IC worker ─────────────────────────────────────────────────────────────

def load_ic_tracks(ic_dir: Path) -> list:
    """
    Load all reactive trajectories for one IC directory.

    Returns a list of track dicts:
        {
          'ic_time'  : str,
          'traj_id'  : int,
          'path_type': str,
          'cluster'  : int,
          'points'   : [{'lat', 'lon', 'mslp', 'iface_idx', 'cal_time'}, ...]
                        ordered λ₀ → state_B
        }
    Points with missing feature_location are skipped; tracks with fewer
    than 2 valid points are dropped.
    """
    json_path = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
    if not json_path.exists():
        return []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return []

    ic_time = data.get('ic_time', ic_dir.name)
    tracks  = []

    for traj in data.get('reactive_trajectories', []):
        pathway    = traj.get('pathway', [])
        path_len   = traj.get('pathway_length', len(pathway))
        final_mslp = traj.get('final_mslp', INTERFACE_PRESSURES[-1])
        cluster    = traj.get('cluster_size', 1)

        if path_len <= 2:
            path_type = 'direct_B'
        elif path_len == N_IFACES:
            path_type = 'full'
        else:
            path_type = 'partial'

        points = []
        for config_name in pathway:
            cfg = _load_pkl(_pkl_path(ic_dir, config_name))
            if cfg is None:
                continue

            loc = getattr(cfg, 'feature_location', None)
            if loc is None:
                continue

            try:
                lat, lon = float(loc[0]), float(loc[1])
            except (TypeError, IndexError):
                continue

            iface_idx = getattr(cfg, 'interface_idx', -1)
            if iface_idx == -1:
                iface_idx = N_IFACES - 1

            points.append({
                'lat'      : lat,
                'lon'      : lon,
                'mslp'     : float(getattr(cfg, 'mslp_value', final_mslp)),
                'iface_idx': int(iface_idx),
                'cal_time' : _cal_time(cfg),
            })

        if len(points) < 2:
            continue

        points.sort(key=lambda p: p['iface_idx'])

        tracks.append({
            'ic_time'  : ic_time,
            'traj_id'  : traj['trajectory_id'],
            'path_type': path_type,
            'cluster'  : cluster,
            'points'   : points,
        })

    return tracks


# ── Map helpers ───────────────────────────────────────────────────────────────

def make_atlantic_axes(fig, rect=111):
    """Return an axes covering the full Atlantic basin."""
    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(
            central_longitude=-60.0,
            central_latitude=35.0,
            standard_parallels=(30, 50),
        )
        ax = fig.add_subplot(rect, projection=proj)
        ax.set_extent([-100, -10, 5, 65], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e8e8e8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#d0e8f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=1.0, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.5, alpha=0.6, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.5,
                          linestyle='--', zorder=4)
        gl.top_labels   = False
        gl.right_labels = False
    else:
        ax = fig.add_subplot(rect)
        ax.set_xlim(-100, -10)
        ax.set_ylim(0, 85)
        ax.set_xlabel('Longitude (°)', fontsize=11)
        ax.set_ylabel('Latitude (°)',  fontsize=11)
        ax.grid(True, alpha=0.3)
    return ax


def plot_track(ax, points, color, alpha=0.35, lw=0.8, zorder=3):
    lons = [p['lon'] for p in points]
    lats = [p['lat'] for p in points]
    kw   = dict(color=color, alpha=alpha, linewidth=lw, zorder=zorder,
                solid_capstyle='round')
    if HAS_CARTOPY:
        ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kw)
    else:
        ax.plot(lons, lats, **kw)


def scatter_points(ax, points, colors, sizes, alpha=0.7, zorder=4):
    lons = [p['lon'] for p in points]
    lats = [p['lat'] for p in points]
    kw   = dict(c=colors, s=sizes, alpha=alpha, zorder=zorder,
                edgecolors='none')
    if HAS_CARTOPY:
        ax.scatter(lons, lats, transform=ccrs.PlateCarree(), **kw)
    else:
        ax.scatter(lons, lats, **kw)


# ── Heatmap + flow panel ──────────────────────────────────────────────────────

def _draw_heatmap_panel(ax, tracks):
    """
    Right panel: cluster-weighted 2D crossing-density heatmap +
    per-interface transition flow arrows (λᵢ → λᵢ₊₁, direction-normalised).

    Unlike a KDE approach, binned histograms preserve bimodal spatial
    structure (Gulf-bound vs. Atlantic-seaboard-bound populations appear
    as separate density peaks rather than one merged blob).

    Arrows show direction of movement between consecutive interfaces,
    coloured by the originating interface, alpha weighted by cluster_size.
    Arrow length is normalised to unit vectors — direction is the story.
    """
    if not tracks:
        return

    pc  = ccrs.PlateCarree() if HAS_CARTOPY else None
    pkw = dict(transform=pc) if HAS_CARTOPY else {}

    max_cluster = max(t['cluster'] for t in tracks)

    # ── Background: combined crossing-density heatmap (2° bins) ─────────────
    BIN      = 2.0
    lon_bins = np.arange(-102,  -8, BIN)
    lat_bins = np.arange(   3,  69, BIN)

    hist = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
    for track in tracks:
        for pt in track['points']:
            li = int((pt['lat'] - lat_bins[0]) / BIN)
            lo = int((pt['lon'] - lon_bins[0]) / BIN)
            if 0 <= li < hist.shape[0] and 0 <= lo < hist.shape[1]:
                hist[li, lo] += track['cluster']

    if hist.max() > 0:
        hist /= hist.max()

    hist_m = np.ma.masked_where(hist == 0, hist)
    ax.pcolormesh(lon_bins, lat_bins, hist_m,
                  cmap='Greys', vmin=0, vmax=1, alpha=0.40,
                  zorder=3, **pkw)

    # ── Per-interface scatter (dot size ∝ cluster_size) ──────────────────────
    for iface_idx in range(N_IFACES):
        lons_i, lats_i, szs = [], [], []
        for track in tracks:
            for pt in track['points']:
                if pt['iface_idx'] == iface_idx:
                    lons_i.append(pt['lon'])
                    lats_i.append(pt['lat'])
                    szs.append(10 + 30 * (track['cluster'] / max_cluster))
        if not lons_i:
            continue
        ax.scatter(lons_i, lats_i, s=szs, color=IFACE_COLORS[iface_idx],
                   edgecolors='k', linewidths=0.3, alpha=0.85,
                   zorder=6, **pkw)

    # ── Transition flow arrows: λᵢ → λᵢ₊₁, direction-normalised ────────────
    for iface_idx in range(N_IFACES - 1):
        lons_s, lats_s, us, vs, wts = [], [], [], [], []

        for track in tracks:
            iface_map = {p['iface_idx']: p for p in track['points']}
            p0 = iface_map.get(iface_idx)
            p1 = iface_map.get(iface_idx + 1)
            if p0 is None or p1 is None:
                continue
            dlon = float(p1['lon'] - p0['lon'])
            dlat = float(p1['lat'] - p0['lat'])
            mag  = np.hypot(dlon, dlat)
            if mag < 0.01:
                continue
            lons_s.append(p0['lon'])
            lats_s.append(p0['lat'])
            us.append(dlon / mag)
            vs.append(dlat / mag)
            wts.append(track['cluster'])

        if not lons_s:
            continue

        color = IFACE_COLORS[iface_idx]
        w_arr = np.array(wts, dtype=float)
        alpha = float(np.clip(0.25 + 0.50 * w_arr.mean() / max_cluster, 0.2, 0.8))

        qkw = dict(transform=pc, zorder=5) if HAS_CARTOPY else dict(zorder=5)
        ax.quiver(
            np.array(lons_s), np.array(lats_s),
            np.array(us),     np.array(vs),
            color=color, alpha=alpha,
            scale=35, width=0.0025,
            headwidth=4, headlength=5, headaxislength=4,
            **qkw
        )


# ── Per-day figure ────────────────────────────────────────────────────────────

def plot_day(ic_time: str, tracks: list, plot_dir: Path):
    """
    Side-by-side reactive trajectory plot for a single IC date.

    Left  — spaghetti tracks, linewidth ∝ cluster_size, dots colored by interface depth.
    Right — cluster-weighted 2D histogram + transition flow arrows.
    """
    from matplotlib.lines import Line2D

    date_str = str(ic_time).split(' ')[0][:10]
    n        = len(tracks)

    fig = plt.figure(figsize=(26, 10))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.06)
    ax1 = make_atlantic_axes(fig, gs[0])
    ax2 = make_atlantic_axes(fig, gs[1])

    # ── Linewidth scale: sqrt of cluster_size → [LW_MIN, LW_MAX] ────────────
    clusters = [t['cluster'] for t in tracks]
    c_min    = max(1, min(clusters))
    c_max    = max(clusters) if max(clusters) > c_min else c_min + 1

    def _lw(c):
        t = (np.sqrt(c) - np.sqrt(c_min)) / (np.sqrt(c_max) - np.sqrt(c_min))
        return LW_MIN + t * (LW_MAX - LW_MIN)

    # ── LEFT: spaghetti tracks ────────────────────────────────────────────────
    for track in tracks:
        lw    = _lw(track['cluster'])
        alpha = 0.30 + 0.40 * (lw - LW_MIN) / (LW_MAX - LW_MIN)
        plot_track(ax1, track['points'], color=TRACK_COLOR, alpha=alpha, lw=lw)

    for track in tracks:
        for pt in track['points']:
            idx = min(pt['iface_idx'], N_IFACES - 1)
            c   = IFACE_COLORS[idx]
            kw  = dict(s=18, alpha=0.85, zorder=5, edgecolors='k', linewidths=0.3)
            if HAS_CARTOPY:
                ax1.scatter(pt['lon'], pt['lat'], color=c,
                            transform=ccrs.PlateCarree(), **kw)
            else:
                ax1.scatter(pt['lon'], pt['lat'], color=c, **kw)

    c_mid = int(np.sqrt(c_min * c_max))
    seen, lw_handles = set(), []
    for c, label in [(c_min, 'few'), (c_mid, 'moderate'), (c_max, 'many')]:
        if c in seen:
            continue
        seen.add(c)
        lw_val = _lw(c)
        alpha  = 0.30 + 0.40 * (lw_val - LW_MIN) / (LW_MAX - LW_MIN)
        lw_handles.append(
            Line2D([0], [0], color=TRACK_COLOR, lw=lw_val, alpha=alpha,
                   label=f'{label}  (n={c})')
        )
    leg1 = ax1.legend(handles=lw_handles, fontsize=9, loc='lower left',
                      framealpha=0.9, title='Branching trajectories', title_fontsize=9)
    ax1.add_artist(leg1)

    ax1.set_title(f'Reactive trajectories — IC {date_str}\n'
                  f'{n} tracks  |  Atlantic 2022',
                  fontsize=12, fontweight='bold')

    # ── RIGHT: heatmap + flow arrows ─────────────────────────────────────────
    _draw_heatmap_panel(ax2, tracks)

    ax2.set_title(f'Crossing density + transition flow — IC {date_str}\n'
                  f'2° histogram (cluster-weighted)  |  arrows: λᵢ→λᵢ₊₁ direction',
                  fontsize=12, fontweight='bold')

    iface_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=IFACE_COLORS[i], markersize=8,
               markeredgecolor='k', markeredgewidth=0.4,
               label=f'λ{i}  {INTERFACE_PRESSURES[i]} hPa')
        for i in range(N_IFACES)
    ]
    iface_handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=6, markeredgecolor='k', markeredgewidth=0.4,
               label='dot size ∝ cluster count'),
        Line2D([0], [0], color='grey', lw=0, marker=(3, 0, 0), markersize=9,
               label='→ transition direction'),
    ]
    ax2.legend(handles=iface_handles, fontsize=8.5, loc='lower right',
               framealpha=0.9, title='Interface depth', title_fontsize=9)

    plt.tight_layout()
    out = plot_dir / f'reactive_trajectories_{date_str}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ── Combined parallel worker ──────────────────────────────────────────────────

def _load_and_plot(args: tuple) -> list:
    """
    Single worker: load tracks for one IC then immediately render the figure.
    Runs entirely inside the subprocess — matplotlib is process-safe.
    Returns the list of tracks for summary stats (or [] on failure/skip).
    """
    ic_dir, rt_dir, skip_plots = args
    tracks = load_ic_tracks(ic_dir)
    if not tracks:
        return []
    if not skip_plots:
        ic_time = tracks[0]['ic_time']
        plot_day(ic_time, tracks, rt_dir)
    return tracks


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_tracks):
    n = len(all_tracks)
    if n == 0:
        print('No tracks loaded.')
        return
    print(f'\n{"="*55}')
    print(f'REACTIVE TRAJECTORY SUMMARY  ({n} trajectories)')
    print(f'{"="*55}')
    for pt in ['full', 'partial', 'direct_B']:
        cnt = sum(1 for t in all_tracks if t['path_type'] == pt)
        print(f'  {pt:10s}: {cnt:4d}  ({100*cnt/n:.1f}%)')

    all_pts = [p for t in all_tracks for p in t['points']]
    lats = [p['lat'] for p in all_pts]
    lons = [p['lon'] for p in all_pts]
    print(f'\n  Lat range: {min(lats):.1f}°N – {max(lats):.1f}°N')
    print(f'  Lon range: {min(lons):.1f}°  – {max(lons):.1f}°')

    n_pts = [len(t['points']) for t in all_tracks]
    print(f'  Points/track: mean={np.mean(n_pts):.1f}  '
          f'min={np.min(n_pts)}  max={np.max(n_pts)}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Reactive trajectory tracks + density map from FFS output'
    )
    parser.add_argument('--ffs_config', required=True,
                        help='FFS config file (ffs.yml) — source of interfaces and state_B')
    parser.add_argument('--ffs_csv',    required=True,
                        help='FFS statistics CSV (used for IC time_labels)')
    parser.add_argument('--output_dir', required=True,
                        help='FFS output directory containing IC subdirectories')
    parser.add_argument('--plot_dir',   default='./plots')
    parser.add_argument('--workers',    type=int, default=min(8, cpu_count()))
    parser.add_argument('--no_plot',    action='store_true',
                        help='Skip figure creation (load tracks and print summary only)')
    args = parser.parse_args()

    # ── Load config and initialise interface constants ────────────────────────
    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)
    _init_from_config(ffs_config)
    print(f'Interfaces: {INTERFACE_PRESSURES}  (N={N_IFACES})')

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.ffs_csv)
    out_dir = Path(args.output_dir)
    ic_dirs = [out_dir / tl for tl in df['time_label'].tolist()
               if (out_dir / tl).exists()]
    print(f'Found {len(ic_dirs)} IC directories')

    rt_dir = plot_dir / 'reactive_trajectories'
    rt_dir.mkdir(exist_ok=True)

    n_workers   = min(args.workers, len(ic_dirs))
    skip_plots  = args.no_plot
    worker_args = [(ic_dir, rt_dir, skip_plots) for ic_dir in ic_dirs]

    verb = 'Loading tracks' if skip_plots else 'Loading + plotting'
    print(f'{verb} with {n_workers} workers...')

    all_tracks = []
    with Pool(processes=n_workers) as pool:
        for tracks in tqdm(
            pool.imap_unordered(_load_and_plot, worker_args),
            total=len(ic_dirs),
            desc='ICs',
            unit='IC',
            dynamic_ncols=True,
        ):
            all_tracks.extend(tracks)

    if skip_plots:
        print('\n(plots skipped — --no_plot flag set)')
    else:
        n_saved = sum(1 for t in all_tracks if t)   # non-empty ICs
        print(f'\n{len(ic_dirs)} ICs processed, plots saved to {rt_dir}/')
    print_summary(all_tracks)


if __name__ == '__main__':
    main()
