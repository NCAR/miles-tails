#!/usr/bin/env python
"""
Contrasting case study — Figure 4.

Shows the full FFS picture for TWO ICs side-by-side:
  (A) A high-genesis-rate case (default: Earl, 2022-08-21T00Z)
  (B) A suppressed-period case (default: first IC with zero IFS events, or
      user-specified)

For each IC the figure shows:
  Top    : Spaghetti track map (reactive trajectories λ₀→stateB)
  Bottom : Committor curve p_B(λᵢ) + MSLP funnel silhouette (optional)

The contrast between a tight coherent Earl ensemble and a diffuse / absent
suppressed-period ensemble demonstrates FFS dynamic range in a way no table can.

Usage:
    python plot_case_study_contrast.py \\
        --ffs_csv      results/ffs_statistics_all_ics.csv \\
        --ifs_csv      results/IFS/ifs_rates_FFS.csv \\
        --output_dir   results \\
        --plot_dir     results/plots \\
        [--highlight_ic 2022-08-21T00Z] \\
        [--suppressed_ic  2022-09-24T00Z] \\
        [--max_tracks  100]   # limit loaded tracks for speed

Structure of output_dir
-----------------------
Each IC has a subdirectory named by its ic_time (e.g. "2022-08-21T00Z") that
contains:
    reactive_trajectories/reactive_trajectories.json
    flux/lambda0_config_*.pkl
    {iface_num}/lambda{i}_config_*.pkl
    stateB/stateB_config_*.pkl
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — using plain lat/lon axes')

# ── Constants ─────────────────────────────────────────────────────────────────

INTERFACE_PRESSURES = [1000, 988, 980, 975, 970, 965]
N_IFACES            = len(INTERFACE_PRESSURES)
DT_HOURS            = 6.0

IFACE_COLORS = plt.cm.YlOrRd(np.linspace(0.25, 0.95, N_IFACES))
TRACK_COLOR  = '#333333'
LW_MIN, LW_MAX = 0.4, 3.5

FFS_COLOR = '#2166ac'
IFS_COLOR = '#d6604d'


def _naive(dt) -> pd.Timestamp:
    """Parse a datetime string/object and strip timezone to tz-naive."""
    ts = pd.to_datetime(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _naive_series(s: pd.Series) -> pd.Series:
    """Convert a Series of datetimes to tz-naive."""
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    return s


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


# ── Track loading ─────────────────────────────────────────────────────────────

def load_ic_tracks(ic_dir: Path, max_tracks: int = None) -> dict:
    """
    Load reactive trajectory tracks for one IC directory.

    Returns
    -------
    dict with keys:
        'ic_time'  : str
        'tracks'   : list of track dicts {'cluster', 'points'}
        'n_raw_tracks' : total tracks in JSON (before max_tracks limit)
    """
    json_path = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
    if not json_path.exists():
        print(f"  Warning: no reactive_trajectories.json in {ic_dir}")
        return {'ic_time': ic_dir.name, 'tracks': [], 'n_raw_tracks': 0}

    with open(json_path) as f:
        data = json.load(f)

    ic_time       = data.get('ic_time', ic_dir.name)
    raw_trajs     = data.get('reactive_trajectories', [])
    n_raw         = len(raw_trajs)

    # Optionally limit to max_tracks (sample uniformly)
    if max_tracks is not None and len(raw_trajs) > max_tracks:
        step     = max(1, len(raw_trajs) // max_tracks)
        raw_trajs = raw_trajs[::step][:max_tracks]

    tracks = []
    for traj in raw_trajs:
        pathway    = traj.get('pathway', [])
        cluster    = traj.get('cluster_size', 1)

        points = []
        for config_name in pathway:
            pkl_path = _pkl_path(ic_dir, config_name)
            cfg      = _load_pkl(pkl_path)
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
                'mslp'     : float(getattr(cfg, 'mslp_value',
                                           INTERFACE_PRESSURES[min(iface_idx, N_IFACES-1)])),
                'iface_idx': int(iface_idx),
            })

        if len(points) < 2:
            continue
        points.sort(key=lambda p: p['iface_idx'])
        tracks.append({'cluster': cluster, 'points': points})

    return {'ic_time': ic_time, 'tracks': tracks, 'n_raw_tracks': n_raw}


# ── Committor per IC ──────────────────────────────────────────────────────────

def get_committor_row(df_ffs: pd.DataFrame, ic_time_str: str) -> dict:
    """Extract p_B(λᵢ) values from the FFS CSV for a specific IC."""
    df = df_ffs.copy()
    df['_dt'] = _naive_series(df['ic_time'])
    target    = _naive(ic_time_str)
    idx       = (df['_dt'] - target).abs().idxmin()
    row       = df.loc[idx]

    # Reconstruct committor
    pB = {'lambda5': 1.0}
    pB['lambda4'] = float(row.get('lambda5_P_forward', np.nan))
    pB['lambda3'] = pB['lambda4'] * float(row.get('lambda4_P_forward', np.nan))
    pB['lambda2'] = pB['lambda3'] * float(row.get('lambda3_P_forward', np.nan))
    pB['lambda1'] = pB['lambda2'] * float(row.get('lambda2_P_forward', np.nan))
    pB['lambda0'] = pB['lambda1'] * float(row.get('lambda1_P_forward', np.nan))

    k_ffs       = float(row.get('ffs_rate_per_day', np.nan))
    k_flux      = float(row.get('flux_rate_per_day', np.nan))
    return {'p_B': pB, 'k_ffs': k_ffs, 'k_flux': k_flux,
            'ic_time': str(row['ic_time'])}


def get_ifs_row(df_ifs: pd.DataFrame, ic_time_str: str) -> dict:
    """Get IFS rate for a specific IC."""
    df = df_ifs.copy()
    df['_dt'] = _naive_series(df['init_time'])
    target    = _naive(ic_time_str)
    idx       = (df['_dt'] - target).abs().idxmin()
    row       = df.loc[idx]
    return {
        'k_ifs'    : float(row.get('rate_B_bf_per_day', 0.0)),
        'n_events' : float(row.get('n_crossed_lambda5', 0)),
    }


# ── Map helpers ───────────────────────────────────────────────────────────────

def make_atlantic_ax(fig, rect):
    """Atlantic basin map with LambertConformal projection."""
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
                       linewidth=0.9, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.4, alpha=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4,
                          linestyle='--', zorder=4)
        gl.top_labels   = False
        gl.right_labels = False
    else:
        ax = fig.add_subplot(rect)
        ax.set_xlim(-100, -10)
        ax.set_ylim(3, 65)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)
    return ax


def draw_tracks_on_ax(ax, tracks, title, subtitle=''):
    """
    Draw reactive trajectory spaghetti on ax.

    Track line width and alpha scale with cluster_size (as in plot_genesis_cone.py).
    Interface dots use YlOrRd colourmap.
    """
    if not tracks:
        if HAS_CARTOPY:
            ax.text(0.5, 0.5, 'No reactive trajectories\n(IFS cannot estimate this rate)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=13, color='#cc0000', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
        else:
            ax.text(-55, 30, 'No reactive trajectories', ha='center', va='center',
                    fontsize=13, color='#cc0000', fontweight='bold')
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')
        return

    max_cluster = max(t['cluster'] for t in tracks) or 1
    pc          = ccrs.PlateCarree() if HAS_CARTOPY else None
    pkw         = dict(transform=pc) if HAS_CARTOPY else {}

    for track in tracks:
        cluster = track['cluster']
        lw      = LW_MIN + (LW_MAX - LW_MIN) * np.sqrt(cluster / max_cluster)
        alpha   = 0.30 + 0.40 * np.sqrt(cluster / max_cluster)

        lons = [p['lon'] for p in track['points']]
        lats = [p['lat'] for p in track['points']]

        if HAS_CARTOPY:
            ax.plot(lons, lats, color=TRACK_COLOR, linewidth=lw,
                    alpha=alpha, zorder=3, solid_capstyle='round',
                    transform=pc)
        else:
            ax.plot(lons, lats, color=TRACK_COLOR, linewidth=lw,
                    alpha=alpha, zorder=3, solid_capstyle='round')

        # Interface dots
        for pt in track['points']:
            ii  = min(pt['iface_idx'], N_IFACES - 1)
            sz  = 15 + 20 * (cluster / max_cluster)
            col = IFACE_COLORS[ii]
            if HAS_CARTOPY:
                ax.scatter(pt['lon'], pt['lat'], s=sz, color=col,
                           edgecolors='none', alpha=0.80, zorder=4,
                           transform=pc)
            else:
                ax.scatter(pt['lon'], pt['lat'], s=sz, color=col,
                           edgecolors='none', alpha=0.80, zorder=4)

    ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')


def draw_committor_ax(ax, pB_dict_A, pB_dict_B, label_A, label_B,
                       k_ffs_A, k_ffs_B, k_ifs_A, k_ifs_B):
    """Bottom panel: committor curves for both ICs on the same axes."""
    pressures = INTERFACE_PRESSURES

    def extract_pB(pB_dict):
        return [pB_dict.get(f'lambda{i}', np.nan) for i in range(N_IFACES)]

    pB_A = extract_pB(pB_dict_A)
    pB_B = extract_pB(pB_dict_B)

    ax.plot(pressures, pB_A, 'o-', color='#2166ac', linewidth=2.2, markersize=8,
            label=f'{label_A}  k_FFS={k_ffs_A:.2e} day⁻¹')
    ax.plot(pressures, pB_B, 's--', color='#d6604d', linewidth=2.2, markersize=8,
            label=f'{label_B}  k_FFS={k_ffs_B:.2e} day⁻¹')

    # Shade rate-limiting step (steepest drop) for each IC
    for pB, color in [(pB_A, '#2166ac'), (pB_B, '#d6604d')]:
        valid = [v for v in pB if np.isfinite(v) and v > 0]
        if len(valid) < 2:
            continue
        drops = [pB[i] - pB[i+1] for i in range(N_IFACES-1)
                 if np.isfinite(pB[i]) and np.isfinite(pB[i+1])]
        if drops:
            k = int(np.argmax(drops))
            ax.axvspan(pressures[k+1], pressures[k],
                       alpha=0.08, color=color)

    ax.invert_xaxis()
    ax.set_xlabel('Interface pressure (hPa)', fontsize=11)
    ax.set_ylabel('p_B(λᵢ)', fontsize=11)
    ax.set_title('Commitment curve  p_B(λᵢ)  —  both ICs overlaid',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    # Annotate k_ifs
    ax.text(0.98, 0.15, f'k_IFS  {label_A}: {k_ifs_A:.2e}\nk_IFS  {label_B}: {k_ifs_B:.2e}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            color='grey', style='italic')


# ── Main figure ───────────────────────────────────────────────────────────────

def find_ic_dir(output_dir: Path, ic_time_str: str) -> Path:
    """
    Locate the IC directory.  Tries exact match and common time-string formats.
    """
    candidates = [
        output_dir / ic_time_str,
        output_dir / ic_time_str.replace(':', ''),
        output_dir / ic_time_str.replace('T', '_'),
    ]
    # Also search for dirs whose name matches the date portion
    date_str = ic_time_str[:10]
    for d in output_dir.iterdir():
        if d.is_dir() and date_str in d.name:
            candidates.append(d)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Cannot find IC directory for '{ic_time_str}' under {output_dir}.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def make_contrast_figure(df_ffs, df_ifs, output_dir: Path, plot_dir: Path,
                          highlight_ic: str, suppressed_ic: str,
                          max_tracks: int = None):

    # ── Auto-detect suppressed IC if not given ────────────────────────────────
    if suppressed_ic is None:
        # Find IC with lowest FFS rate (and ideally zero IFS events)
        ifs_times = _naive_series(df_ifs['init_time'])
        ffs_times = _naive_series(df_ffs['ic_time'])

        zero_ifs  = df_ifs[df_ifs['rate_B_bf_per_day'] <= 0]
        if not zero_ifs.empty:
            # Among zero-IFS ICs, pick the one FFS also found lowest
            for _, row in zero_ifs.iterrows():
                t_ifs = _naive(row['init_time'])
                # Match to nearest FFS IC
                ffs_idx = (ffs_times - t_ifs).abs().idxmin()
                candidate = str(df_ffs.loc[ffs_idx, 'ic_time'])
                if candidate != highlight_ic:
                    suppressed_ic = candidate
                    break

        if suppressed_ic is None:
            # Fallback: lowest FFS rate IC that isn't the highlighted one
            ffs_rates = df_ffs['ffs_rate_per_day'].values.astype(float)
            hl_idx    = (ffs_times - _naive(highlight_ic)).abs().idxmin()
            mask      = np.ones(len(ffs_rates), dtype=bool)
            mask[hl_idx] = False
            min_idx   = np.argmin(np.where(mask, ffs_rates, np.inf))
            suppressed_ic = str(df_ffs.loc[min_idx, 'ic_time'])

        print(f"  Auto-selected suppressed IC: {suppressed_ic}")

    # ── Load data for both ICs ────────────────────────────────────────────────
    print(f"  Loading active IC ({highlight_ic})...")
    earl_dir   = find_ic_dir(output_dir, highlight_ic)
    earl_data  = load_ic_tracks(earl_dir, max_tracks=max_tracks)

    print(f"  Loading suppressed IC ({suppressed_ic})...")
    supp_dir   = find_ic_dir(output_dir, suppressed_ic)
    supp_data  = load_ic_tracks(supp_dir, max_tracks=max_tracks)

    # ── Get committor curves ──────────────────────────────────────────────────
    earl_cmt = get_committor_row(df_ffs, highlight_ic)
    supp_cmt = get_committor_row(df_ffs, suppressed_ic)

    earl_ifs = get_ifs_row(df_ifs, highlight_ic)
    supp_ifs = get_ifs_row(df_ifs, suppressed_ic)

    n_earl  = len(earl_data['tracks'])
    n_supp  = len(supp_data['tracks'])

    print(f"  Earl:      {n_earl} reactive trajectories  "
          f"  k_FFS={earl_cmt['k_ffs']:.3e}  k_IFS={earl_ifs['k_ifs']:.3e}")
    print(f"  Suppressed: {n_supp} reactive trajectories  "
          f"  k_FFS={supp_cmt['k_ffs']:.3e}  k_IFS={supp_ifs['k_ifs']:.3e}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Top row: two maps; Bottom row: committor curve (spanning both columns)
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[2.5, 1],
        hspace=0.10,
        wspace=0.05,
    )

    ax_map_earl = make_atlantic_ax(fig, gs[0, 0])
    ax_map_supp = make_atlantic_ax(fig, gs[0, 1])
    ax_cmt      = fig.add_subplot(gs[1, :])   # committor spans both columns

    # ── Track panels ──────────────────────────────────────────────────────────
    earl_label = (f"Active  ({highlight_ic})\n"
                  f"{n_earl} reactive trajs  |  "
                  f"k_FFS = {earl_cmt['k_ffs']:.2e} day⁻¹")

    supp_label_ifs = (f"k_IFS = {supp_ifs['k_ifs']:.2e} day⁻¹"
                       if supp_ifs['k_ifs'] > 0 else "k_IFS = 0  (no direct events)")
    supp_label = (f"Suppressed  ({suppressed_ic})\n"
                  f"{n_supp} reactive trajs  |  "
                  f"k_FFS = {supp_cmt['k_ffs']:.2e} day⁻¹  |  {supp_label_ifs}")

    draw_tracks_on_ax(ax_map_earl, earl_data['tracks'],
                       title='(A) Active period — Earl case study',
                       subtitle=earl_label)
    draw_tracks_on_ax(ax_map_supp, supp_data['tracks'],
                       title='(B) Suppressed period',
                       subtitle=supp_label)

    # ── Colourbar for interface dots (shared) ──────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=mcolors.Normalize(
                                    INTERFACE_PRESSURES[0],
                                    INTERFACE_PRESSURES[-1]))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.38, 0.012, 0.40])
    cb      = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label('Interface pressure (hPa)', fontsize=9)
    cb.set_ticks(INTERFACE_PRESSURES)

    # ── Committor panel ───────────────────────────────────────────────────────
    draw_committor_ax(
        ax_cmt,
        earl_cmt['p_B'], supp_cmt['p_B'],
        label_A=f'Active ({highlight_ic})',
        label_B=f'Suppressed ({suppressed_ic})',
        k_ffs_A=earl_cmt['k_ffs'], k_ffs_B=supp_cmt['k_ffs'],
        k_ifs_A=earl_ifs['k_ifs'], k_ifs_B=supp_ifs['k_ifs'],
    )

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        'FFS Case Study Contrast: Active vs Suppressed Period\n'
        'Reactive trajectory spaghetti + commitment curves',
        fontsize=14, fontweight='bold', y=1.005,
    )

    out = plot_dir / 'case_study_contrast.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Contrasting case study: Earl vs suppressed-period IC'
    )
    parser.add_argument('--ffs_csv',       required=True,
                        help='FFS statistics CSV')
    parser.add_argument('--ifs_csv',       required=True,
                        help='IFS brute-force CSV')
    parser.add_argument('--output_dir',    required=True,
                        help='Root directory containing IC subdirectories')
    parser.add_argument('--plot_dir',      default='./plots',
                        help='Output directory for the figure')
    parser.add_argument('--highlight_ic',  default='2022-08-21T00Z',
                        help='IC time for the active/highlighted case (YYYY-MM-DDTHH Z)')
    parser.add_argument('--suppressed_ic', default=None,
                        help='IC time for the suppressed case (auto-detected if omitted)')
    parser.add_argument('--max_tracks',    type=int, default=None,
                        help='Max reactive trajectories to load per IC (None = all)')
    args = parser.parse_args()

    plot_dir   = Path(args.plot_dir)
    output_dir = Path(args.output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print('Loading CSVs...')
    df_ffs = pd.read_csv(args.ffs_csv)
    df_ifs = pd.read_csv(args.ifs_csv)

    print('Generating contrast figure...')
    make_contrast_figure(
        df_ffs, df_ifs, output_dir, plot_dir,
        highlight_ic  = args.highlight_ic,
        suppressed_ic = args.suppressed_ic,
        max_tracks    = args.max_tracks,
    )


if __name__ == '__main__':
    main()
