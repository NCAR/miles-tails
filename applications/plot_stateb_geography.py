#!/usr/bin/env python
"""
plot_stateb_geography.py — Geographic distribution of FFS state-B genesis events.

Loads every stateB/*.pkl file across all IC directories, extracts the
feature_location (lat, lon) of each arrival, and produces a single Atlantic-
basin map showing:

  - Scatter points for every state-B event, coloured by IC initialization date
    (light Aug → dark Oct sequential colourmap).
  - A 2° density heatmap contour in the background showing spatial clustering.
  - The Atlantic basin domain box [10-40°N, 100-20°W].
  - Text annotations for the approximate genesis locations of Earl, Fiona, Ian.

Usage
-----
    python plot_stateb_geography.py \\
        --results_dir /path/to/results \\
        --plot_dir    /path/to/plots \\
        --workers     8
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
# Ensure tails package is importable for pkl loading
sys.path.insert(0, '/glade/work/schreck/repos/miles-tails')

import matplotlib
matplotlib.use('Agg')

import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — falling back to plain lat/lon axes')


# ── IC date parsing ───────────────────────────────────────────────────────────

def _parse_ic_date(ic_name: str) -> datetime | None:
    """
    Parse an IC directory name like '2022-08-21T00Z' into a datetime.
    Returns None if parsing fails.
    """
    # Normalise: '2022-08-21T00Z' -> '2022-08-21T00:00'
    s = ic_name.strip().replace('T', ' ').replace('Z', '')
    # Handle both 'YYYY-MM-DD HH' and 'YYYY-MM-DD HH:MM'
    for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d %H', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ── pkl loading ───────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    """Load a single pkl and immediately extract feature_location; discard object to save memory."""
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        loc = getattr(obj, 'feature_location', None)
        del obj  # immediately free the large model-state arrays
        return loc  # (lat, lon) tuple or None
    except Exception:
        return None


def _load_stateB_for_ic(ic_dir: Path):
    """
    Load all stateB pkl files in ic_dir/stateB/*.pkl.

    Returns a list of (lat, lon) tuples extracted from obj.feature_location,
    or an empty list if the directory does not exist or no valid files are found.
    """
    stateb_dir = ic_dir / 'stateB'
    if not stateb_dir.exists():
        return []

    pkl_paths = list(stateb_dir.glob('*.pkl'))
    if not pkl_paths:
        return []

    def _extract_one(p: Path):
        loc = _load_pkl(p)  # now returns (lat, lon) or None directly
        if loc is None:
            return None
        try:
            lat, lon = float(loc[0]), float(loc[1])
        except (TypeError, IndexError):
            return None
        # Filter spurious Gulf-of-Mexico anomaly (extratropical, lon < -90, lat < 20)
        if lon < -90.0 and lat < 20.0:
            return None
        return lat, lon

    results = []
    # Limit concurrency to avoid holding too many large pkl objects in memory
    with ThreadPoolExecutor(max_workers=min(4, len(pkl_paths))) as ex:
        for r in ex.map(_extract_one, pkl_paths):
            if r is not None:
                results.append(r)
    return results


# ── Density heatmap ───────────────────────────────────────────────────────────

def _compute_density(lats, lons, bin_size=2.0):
    """
    Bin (lat, lon) points onto a regular grid and return the grid edges
    and count array for contour plotting.
    """
    lon_edges = np.arange(-110,  -10 + bin_size, bin_size)
    lat_edges = np.arange(   5,   50 + bin_size, bin_size)

    counts, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])
    # Smooth slightly with a simple uniform filter for nicer contours
    from scipy.ndimage import uniform_filter
    counts_smooth = uniform_filter(counts.astype(float), size=2)

    lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return lon_centres, lat_centres, counts_smooth


# ── Map axes ──────────────────────────────────────────────────────────────────

def _make_atlantic_ax(fig, rect=111):
    """Create a cartopy (or plain) Atlantic axes."""
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(rect, projection=proj)
        ax.set_extent([-110, -10, 5, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e8e4d8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#c8dff0', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.8, edgecolor='#444444', zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linewidth=0.4, edgecolor='#777777', zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.3, edgecolor='#999999', alpha=0.7, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.45,
                          linestyle='--', zorder=4,
                          crs=ccrs.PlateCarree())
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
    else:
        ax = fig.add_subplot(rect)
        ax.set_xlim(-110, -10)
        ax.set_ylim(5, 50)
        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)',  fontsize=10)
        ax.set_facecolor('#c8dff0')
        ax.grid(True, alpha=0.35, linestyle='--')
    return ax


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(all_lats, all_lons, all_dates,
                date_min: datetime, date_max: datetime,
                plot_path: Path):
    """
    Render and save the state-B geography figure.

    Parameters
    ----------
    all_lats, all_lons : list[float]
        Coordinates of every state-B event.
    all_dates : list[datetime]
        IC initialisation date for each event (same length as all_lats).
    date_min, date_max : datetime
        Extremes of the IC date range (for colourmap normalisation).
    plot_path : Path
        Output file path.
    """
    fig = plt.figure(figsize=(14, 8))
    ax  = _make_atlantic_ax(fig, 111)

    pc = ccrs.PlateCarree() if HAS_CARTOPY else None

    # ── 2° density heatmap contour (background) ───────────────────────────────
    if len(all_lats) >= 4:
        try:
            lon_c, lat_c, density = _compute_density(
                np.array(all_lats), np.array(all_lons), bin_size=2.0
            )
            LON_G, LAT_G = np.meshgrid(lon_c, lat_c)
            levels = np.linspace(0.5, density.max(), 9)
            pkw = dict(transform=pc, zorder=3) if HAS_CARTOPY else dict(zorder=3)
            ax.contourf(LON_G, LAT_G, density,
                        levels=levels, cmap='YlOrRd',
                        alpha=0.38, **pkw)
            ax.contour(LON_G, LAT_G, density,
                       levels=levels, colors='#8B2500',
                       linewidths=0.4, alpha=0.45, **pkw)
        except Exception as exc:
            print(f'  Warning: density heatmap failed ({exc})')

    # ── Scatter points coloured by IC date ───────────────────────────────────
    norm = mcolors.Normalize(
        vmin=date_min.timestamp(),
        vmax=date_max.timestamp(),
    )
    cmap = plt.cm.Blues_r  # light (early Aug) → dark (late Oct)
    # Reverse: we want early dates light and late dates dark.
    # Blues_r goes dark→light, so we want Blues (light→dark) for
    # early→late ordering:
    cmap = plt.cm.Blues

    date_nums = np.array([d.timestamp() for d in all_dates])
    colours   = cmap(norm(date_nums))

    skw = dict(zorder=6)
    if HAS_CARTOPY:
        skw['transform'] = pc

    sc = ax.scatter(
        all_lons, all_lats,
        c=date_nums,
        cmap=cmap, norm=norm,
        s=22, alpha=0.80,
        edgecolors='#222222', linewidths=0.25,
        **skw,
    )

    # ── Atlantic basin domain box [10-40°N, 100-20°W] ────────────────────────
    box_lons = [-100, -20, -20, -100, -100]
    box_lats = [  10,  10,  40,   40,   10]
    bkw = dict(zorder=7, linewidth=1.8, linestyle='--',
               edgecolor='#1a1a1a', facecolor='none')
    if HAS_CARTOPY:
        from matplotlib.patches import Polygon as MplPolygon
        import shapely.geometry as sgeom
        box_patch = mpatches.Rectangle(
            xy=(-100, 10), width=80, height=30,
            linewidth=1.8, linestyle='--',
            edgecolor='#1a1a1a', facecolor='none',
            zorder=7, transform=pc,
        )
        ax.add_patch(box_patch)
    else:
        ax.plot(box_lons, box_lats, **{k: v for k, v in bkw.items()
                                        if k != 'facecolor'}, color='#1a1a1a')

    # ── Storm genesis annotations ─────────────────────────────────────────────
    # NHC best-track genesis locations
    annotations = [
        ('Earl\n~Sep 2',   17.9, -58.6),   # 1800 UTC 2 Sep, 17.9N 58.6W
        ('Fiona\n~Sep 14', 16.0, -47.9),   # 0600 UTC 14 Sep, 16.0N 47.9W
        ('Ian\n~Sep 23',   13.7, -68.1),   # 0600 UTC 23 Sep, 13.7N 68.1W
    ]
    # offsets: (dlon, dlat, ha)
    ann_offsets = {
        'Earl\n~Sep 2':   (-3.5,  3.0, 'left'),   # upper-left, arrow points right-down
        'Fiona\n~Sep 14': ( 2.5,  2.5, 'left'),   # upper-right
        'Ian\n~Sep 23':   (-9.0,  4.0, 'right'),  # far left, arrow points right-down; clears Earl
    }
    akw = dict(zorder=10)
    if HAS_CARTOPY:
        akw['transform'] = pc

    for label, alat, alon in annotations:
        dlon, dlat, ha = ann_offsets[label]
        ax.plot(alon, alat, marker='*', markersize=13,
                color='#cc0000', markeredgecolor='white',
                markeredgewidth=0.7, zorder=9,
                **({'transform': pc} if HAS_CARTOPY else {}))
        ax.annotate(
            label,
            xy=(alon, alat),
            xytext=(alon + dlon, alat + dlat),
            fontsize=12, fontweight='bold', color='#cc0000',
            ha=ha, va='center',
            arrowprops=dict(arrowstyle='->', color='#cc0000',
                            lw=0.9, shrinkA=0, shrinkB=3),
            zorder=10,
            **({'xycoords': pc._as_mpl_transform(ax),
                'textcoords': pc._as_mpl_transform(ax)}
               if HAS_CARTOPY else {}),
        )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.03, shrink=0.75)
    cbar.set_label('IC initialization date (Aug 21 – Oct 8, 2022)',
                   fontsize=10)

    # Tick at the 1st of each month in the date range
    tick_dates = []
    for month in range(date_min.month, date_max.month + 2):
        yr = date_min.year + (month - 1) // 12
        mo = ((month - 1) % 12) + 1
        try:
            td = datetime(yr, mo, 1)
        except ValueError:
            continue
        if date_min <= td <= date_max:
            tick_dates.append(td)

    if tick_dates:
        tick_vals   = [d.timestamp() for d in tick_dates]
        tick_labels = [d.strftime('%b 1') for d in tick_dates]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(tick_labels, fontsize=9)

    # ── Labels and title ──────────────────────────────────────────────────────
    ax.set_title(
        'State-B arrival locations (MSLP ≤ 975 hPa) across all 98 ICs',
        fontsize=13, fontweight='bold', pad=10,
    )

    # Domain box legend entry
    box_legend = mpatches.Patch(
        linestyle='--', linewidth=1.8,
        edgecolor='#1a1a1a', facecolor='none',
        label='Atlantic basin domain [10-40°N, 100-20°W]',
    )
    storm_legend = plt.Line2D(
        [0], [0], marker='*', color='w',
        markerfacecolor='#cc0000', markeredgecolor='white',
        markeredgewidth=0.7, markersize=11,
        label='Observed storm genesis (Earl / Fiona / Ian)',
    )
    ax.legend(handles=[box_legend, storm_legend],
              loc='upper right', fontsize=12, framealpha=0.88)

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plot_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot geographic distribution of FFS state-B genesis events'
    )
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing IC subdirectories '
                             '(e.g. 2022-08-21T00Z, …)')
    parser.add_argument('--plot_dir', default='./plots',
                        help='Output directory for figures (default: ./plots)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel threads for pkl loading (default: 8)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plot_dir    = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover IC directories ───────────────────────────────────────────────
    ic_dirs = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and _parse_ic_date(p.name) is not None
    )
    if not ic_dirs:
        print(f'No IC directories found in {results_dir}. '
              'Expected names like 2022-08-21T00Z.')
        return

    print(f'Found {len(ic_dirs)} IC directories in {results_dir}')

    # ── Load state-B locations in parallel (IC-level threads) ────────────────
    all_lats:  list[float]    = []
    all_lons:  list[float]    = []
    all_dates: list[datetime] = []

    n_workers = min(args.workers, len(ic_dirs))

    def _worker(ic_dir: Path):
        ic_date = _parse_ic_date(ic_dir.name)
        try:
            pts = _load_stateB_for_ic(ic_dir)
        except Exception as exc:
            print(f'  ERROR {ic_dir.name}: {exc}')
            pts = []
        return ic_date, pts

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, d): d for d in ic_dirs}
        for fut in as_completed(futures):
            ic_dir = futures[fut]
            try:
                ic_date, pts = fut.result()
            except Exception as exc:
                print(f'  ERROR {ic_dir.name}: {exc}')
                continue
            if not pts:
                continue
            for lat, lon in pts:
                all_lats.append(lat)
                all_lons.append(lon)
                all_dates.append(ic_date)

    print(f'Loaded {len(all_lats)} state-B events across all ICs')

    if not all_lats:
        print('No state-B locations found — nothing to plot.')
        return

    # ── Date range for colourmap ──────────────────────────────────────────────
    date_min = min(all_dates)
    date_max = max(all_dates)
    print(f'IC date range: {date_min.strftime("%Y-%m-%d")} – '
          f'{date_max.strftime("%Y-%m-%d")}')

    # ── Render figure ─────────────────────────────────────────────────────────
    out_path = plot_dir / 'stateb_geography.png'
    make_figure(all_lats, all_lons, all_dates, date_min, date_max, out_path)


if __name__ == '__main__':
    main()
