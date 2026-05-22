#!/usr/bin/env python
"""
plot_ian_gulf_composites.py

3-row composite figure for Ian (2022-09-22T00Z) FFS simulations split by
terminal track:
  - West group:  final lon < -88W  (Bay of Campeche / Mexico coast)
  - North group: -88W <= final lon <= -75W  (eastern Gulf / Florida)

Atmospheric state is composited at the lambda2 interface — the early-time
checkpoint at which trajectories are committed to one track or the other.
Stars mark the eventual stateB terminal location (hurricane-strength crossing).

Row 1: 500 hPa geopotential height (filled) + 500 hPa wind vectors
Row 2: 850 hPa wind speed (filled) + 850 hPa wind vectors  [steering flow]
Row 3: Z500 anomaly (North − West), single panel spanning both columns

Usage:
    python plot_ian_gulf_composites.py
"""

import json, os, pickle, warnings
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from pathlib import Path

warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── paths ─────────────────────────────────────────────────────────────────────

RESULTS_BASE = Path('/glade/derecho/scratch/schreck/FFS/results_mar18')
IAN_DIR      = RESULTS_BASE / '2022-09-22T00Z'
STATEB_DIR   = IAN_DIR / 'stateB'
LAM4_DIR     = IAN_DIR / '4'
LAM2_DIR     = IAN_DIR / '2'
RT_PATH      = IAN_DIR / 'reactive_trajectories' / 'reactive_trajectories.json'
PLOT_DIR     = RESULTS_BASE / 'plots' / 'physics_composites'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MEAN_NC = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/mean_std/mean_6h_1979_2018_cesm.nc'
STD_NC  = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/mean_std/std_residual_6h_1979_2018_cesm.nc'

# Channel layout: U[0..15], V[16..31], T[32..47], Q[48..63],
#                 SP[64], t2m[65], V500[66], U500[67], T500[68], Z500[69], Q500[70]
# Level list: [10,30,40,50,60,70,80,90,95,100,105,110,120,130,136,137]
#   index 4  = level 60  ≈ 200 hPa
#   index 12 = level 120 ≈ 850 hPa
CH_U850 = 12;  CH_V850 = 28
CH_U500 = 67;  CH_V500 = 66;  CH_Z500 = 69

# Grid (192 lat × 288 lon)
LATS = np.linspace(90, -90, 192)
LONS = np.linspace(0, 360 - 360 / 288, 288)

GULF_EXTENT = [-105, -52, 7, 43]


# ── normalization ─────────────────────────────────────────────────────────────

def load_norm():
    ds_mean = xr.open_dataset(MEAN_NC)
    ds_std  = xr.open_dataset(STD_NC)
    return {
        'Z500': (float(ds_mean['Z500']), float(ds_std['Z500'])),
        'U500': (float(ds_mean['U500']), float(ds_std['U500'])),
        'V500': (float(ds_mean['V500']), float(ds_std['V500'])),
        'U850': (float(ds_mean['U'].values[12]), float(ds_std['U'].values[12])),
        'V850': (float(ds_mean['V'].values[12]), float(ds_std['V'].values[12])),
    }


def lon360_to_signed(lon):
    return lon if lon <= 180 else lon - 360


def load_atmos_state(pkl_path, norm):
    """Load atmospheric fields from any FFS pkl (stateB, lambda2, etc.)."""
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    s = obj.input_state.numpy()  # [1, 71, 1, 192, 288]

    def dn(ch, key):
        return s[0, ch, 0] * norm[key][1] + norm[key][0]

    z500 = dn(CH_Z500, 'Z500') / 9.80665
    u500 = dn(CH_U500, 'U500')
    v500 = dn(CH_V500, 'V500')
    u850 = dn(CH_U850, 'U850')
    v850 = dn(CH_V850, 'V850')
    spd850 = np.sqrt(u850 ** 2 + v850 ** 2)

    return dict(z=z500, u5=u500, v5=v500, u8=u850, v8=v850, s8=spd850)


def classify(lon):
    if lon < -88:
        return 'west'
    elif lon <= -75:
        return 'north'
    return None


def is_artifact(lat, lon):
    """Filter tracker artifacts: Pacific-coast Mexico (storm tracked across Yucatan)."""
    return lat < 18.0 and lon < -94.0


# ── compositing ───────────────────────────────────────────────────────────────

def build_composites(norm):
    # Step 1: build lambda4 → lambda2 mapping from full reactive pathways
    with open(RT_PATH) as f:
        rt_raw = json.load(f)
    rt_entries = rt_raw['reactive_trajectories']

    lam4_to_lam2 = {}
    for entry in rt_entries:
        pathway = entry.get('pathway', [])
        if len(pathway) >= 5:
            lam4_to_lam2[pathway[4]] = pathway[2]

    print(f'Full pathways (lambda4 terminals): {len(lam4_to_lam2)}')

    # Step 2: load restart_datetime and feature_location from lambda4 pkls
    # Multiple lambda4 configs can share the same restart_datetime (all are
    # perturbations from the same lambda4 threshold-crossing event).
    lam4_dt_map = {}      # datetime → [lam2_names]
    dt_to_lam4_locs = {}  # datetime → [(lat, lon)] for λ4 feature locations
    n_loaded = 0
    for lam4_name, lam2_name in lam4_to_lam2.items():
        p = LAM4_DIR / f'{lam4_name}.pkl'
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        dt = obj.restart_datetime
        lam4_dt_map.setdefault(dt, []).append(lam2_name)
        lam4_lat = float(obj.feature_location[0])
        lam4_lon = lon360_to_signed(float(obj.feature_location[1]))
        dt_to_lam4_locs.setdefault(dt, []).append((lam4_lat, lam4_lon))
        n_loaded += 1

    print(f'Lambda4 pkls loaded: {n_loaded} ({len(lam4_dt_map)} unique datetimes)')

    # Step 3: match stateB configs to lambda4 via restart_datetime → lambda2
    KEYS = ('z', 'u5', 'v5', 'u8', 'v8', 's8')
    west  = {k: [] for k in KEYS}; west['pos']  = []; west['lam4_pos']  = []
    north = {k: [] for k in KEYS}; north['pos'] = []; north['lam4_pos'] = []
    n_matched = 0
    seen_west  = set()   # deduplicate stateB terminal positions per group
    seen_north = set()

    pkls = sorted(STATEB_DIR.glob('stateB_config_*.pkl'))
    print(f'Loading {len(pkls)} stateB configs ...')
    for p in pkls:
        with open(p, 'rb') as f:
            sb = pickle.load(f)

        loc = sb.feature_location
        lat = float(loc[0])
        lon = lon360_to_signed(float(loc[1]))

        if is_artifact(lat, lon):
            print(f'  ARTIFACT filtered: lat={lat:.1f} lon={lon:.1f}')
            continue

        grp = classify(lon)
        if grp is None:
            continue

        lam2_names = lam4_dt_map.get(sb.restart_datetime, [])
        if not lam2_names:
            continue

        # Record unique stateB terminal position and corresponding λ4 locations
        pos_key = (round(lat, 2), round(lon, 2))
        seen = seen_west if grp == 'west' else seen_north
        d = west if grp == 'west' else north
        if pos_key not in seen:
            seen.add(pos_key)
            d['pos'].append((lat, lon))
            for lam4_lat, lam4_lon in dt_to_lam4_locs.get(sb.restart_datetime, []):
                d['lam4_pos'].append((lam4_lat, lam4_lon))

        for lam2_name in lam2_names:
            lam2_path = LAM2_DIR / f'{lam2_name}.pkl'
            if not lam2_path.exists():
                print(f'  MISS lambda2 pkl: {lam2_name}')
                continue
            try:
                fields = load_atmos_state(lam2_path, norm)
            except Exception as e:
                print(f'  SKIP {lam2_name}: {e}')
                continue
            for k in KEYS:
                d[k].append(fields[k])
            n_matched += 1

    print(f'Matched: {n_matched}  (west={len(west["z"])}, north={len(north["z"])})')
    print(f'Unique stateB pos: west={len(west["pos"])}, north={len(north["pos"])}')
    print(f'Lambda4 pos: west={len(west["lam4_pos"])}, north={len(north["lam4_pos"])}')

    def avg(d):
        return {k: np.mean(d[k], axis=0) for k in KEYS}

    return avg(west), west['pos'], west['lam4_pos'], avg(north), north['pos'], north['lam4_pos']


# ── plotting helpers ──────────────────────────────────────────────────────────

def _add_basemap(ax):
    ax.set_extent(GULF_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#c9dff0', zorder=0)

def _add_boundaries(ax, zorder=7):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='#333', zorder=zorder)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.5, edgecolor='#666', zorder=zorder)
    ax.add_feature(cfeature.STATES.with_scale('50m'),    linewidth=0.25, edgecolor='#999', zorder=zorder)

def _add_gridlines(ax):
    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--', zorder=10)
    gl.xlocator = mticker.FixedLocator([-100, -90, -80, -70, -60])
    gl.ylocator = mticker.FixedLocator([10, 20, 30, 40])
    return gl


def _add_axis_labels(ax, proj, left_labels=True, bottom_labels=True):
    """Place lat/lon labels via projection transforms (call after fig.canvas.draw())."""
    pc = ccrs.PlateCarree()
    if left_labels:
        for lat in [10, 20, 30, 40]:
            try:
                x_lc, y_lc = proj.transform_point(GULF_EXTENT[0], lat, pc)
                x_d, y_d = ax.transData.transform((x_lc, y_lc))
                _, y_ax = ax.transAxes.inverted().transform((x_d, y_d))
                if 0.0 <= y_ax <= 1.0:
                    ax.text(-0.01, y_ax, f'{lat}°N',
                            transform=ax.transAxes, ha='right', va='center',
                            fontsize=7, clip_on=False, zorder=20)
            except Exception:
                pass
    if bottom_labels:
        for lon in [-100, -90, -80, -70, -60]:
            try:
                x_lc, y_lc = proj.transform_point(lon, GULF_EXTENT[2], pc)
                x_d, y_d = ax.transData.transform((x_lc, y_lc))
                x_ax, _ = ax.transAxes.inverted().transform((x_d, y_d))
                if 0.0 <= x_ax <= 1.0:
                    ax.text(x_ax, -0.01, f'{abs(lon)}°W',
                            transform=ax.transAxes, ha='center', va='top',
                            fontsize=7, clip_on=False, zorder=20)
            except Exception:
                pass


def _add_quiver(ax, x, y, u, v, *, scale, color, zorder=6):
    q = ax.quiver(
        x, y, u, v,
        transform=ccrs.PlateCarree(),
        scale=scale,
        width=0.0032,
        headwidth=4.4,
        headlength=5.8,
        headaxislength=5.0,
        pivot='mid',
        color=color,
        alpha=0.95,
        zorder=zorder,
    )
    return q


def _mask_strongest_vector(u, v):
    """Hide one obvious outlier vector without changing the rest of the field."""
    mag = np.hypot(u, v)
    if not np.isfinite(mag).any():
        return u, v
    u = np.array(u, copy=True)
    v = np.array(v, copy=True)
    iy, ix = np.unravel_index(np.nanargmax(mag), mag.shape)
    u[iy, ix] = np.nan
    v[iy, ix] = np.nan
    return u, v


def _ridge_center(field, lat_mask, lon_mask):
    """Return the lat/lon of the local Z500 maximum within the plotted Gulf domain."""
    sub = field[np.ix_(lat_mask, lon_mask)]
    iy, ix = np.unravel_index(np.nanargmax(sub), sub.shape)
    return float(LATS[lat_mask][iy]), float(LONS[lon_mask][ix] - 360.0)


# ── main plotting ─────────────────────────────────────────────────────────────

def plot_composites(west_avg, west_pos, west_lam4_pos, north_avg, north_pos, north_lam4_pos):
    # Tight mask — used only for computing shared colour-scale limits
    lat_mask = (LATS >= GULF_EXTENT[2]) & (LATS <= GULF_EXTENT[3])
    lon_mask = (LONS >= GULF_EXTENT[0] + 360) & (LONS <= GULF_EXTENT[1] + 360)

    # Padded mask — used for plotting so the fill covers the full map extent
    PAD = 12
    lat_plot = (LATS >= GULF_EXTENT[2] - PAD) & (LATS <= GULF_EXTENT[3] + PAD)
    lon_plot = (LONS >= GULF_EXTENT[0] - PAD + 360) & (LONS <= GULF_EXTENT[1] + PAD + 360)

    sub_lats = LATS[lat_plot]
    sub_lons = LONS[lon_plot] - 360
    lons2d, lats2d = np.meshgrid(sub_lons, sub_lats)

    proj = ccrs.LambertConformal(central_longitude=-80, central_latitude=25,
                                  standard_parallels=(15, 40))
    pc = ccrs.PlateCarree()

    # 2 rows × 3 cols: west | north | difference
    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             subplot_kw=dict(projection=proj))
    # hspace/bottom leave room for horizontal colorbars below each row
    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.12,
                        hspace=0.10, wspace=0.10)

    def gulf_vals(key, *avgs):
        return np.concatenate([a[key][np.ix_(lat_mask, lon_mask)].ravel() for a in avgs])

    # ── shared colour scales for absolute panels ──────────────────────────────
    z_lo,  z_hi  = np.percentile(gulf_vals('z',  west_avg, north_avg), [2, 98])
    s8_lo, s8_hi = np.percentile(gulf_vals('s8', west_avg, north_avg), [2, 98])
    z_fill  = np.linspace(z_lo,  z_hi,  21)
    z_cont  = np.linspace(np.floor(z_lo  / 20) * 20, np.ceil(z_hi  / 20) * 20, 11)
    s8_fill = np.linspace(s8_lo, s8_hi, 21)
    s8_cont = np.linspace(np.floor(s8_lo / 2)  * 2,  np.ceil(s8_hi / 2)  * 2,  11)

    sk = 5
    pcm_z = pcm_s8 = None

    groups = [
        (0, 'West-track composite\nterminal positions in Bay of Campeche / Mexico coast', west_avg, west_pos),
        (1, 'North-track composite\nterminal positions in eastern Gulf / Florida',         north_avg, north_pos),
    ]

    for col, title, avg, pos in groups:
        z_sub  = avg['z'] [np.ix_(lat_plot, lon_plot)]
        u5_sub = avg['u5'][np.ix_(lat_plot, lon_plot)]
        v5_sub = avg['v5'][np.ix_(lat_plot, lon_plot)]
        s8_sub = avg['s8'][np.ix_(lat_plot, lon_plot)]
        u8_sub = avg['u8'][np.ix_(lat_plot, lon_plot)]
        v8_sub = avg['v8'][np.ix_(lat_plot, lon_plot)]
        ll = (col == 0)   # left-labels only on col 0

        # row 0: Z500
        a = axes[0, col]
        _add_basemap(a)
        pcm_z = a.contourf(lons2d, lats2d, z_sub, levels=z_fill,
                           cmap='plasma', transform=pc, zorder=1)
        a.contour(lons2d, lats2d, z_sub, levels=z_cont,
                  colors='white', linewidths=0.7, alpha=0.65, transform=pc, zorder=2)
        _add_quiver(a, lons2d[::sk, ::sk], lats2d[::sk, ::sk],
                    u5_sub[::sk, ::sk], v5_sub[::sk, ::sk],
                    scale=220, color='white', zorder=6)
        _add_boundaries(a, zorder=7)
        _add_gridlines(a)
        ridge_lat, ridge_lon = _ridge_center(avg['z'], lat_mask, lon_mask)
        a.text(ridge_lon, ridge_lat, 'H', transform=pc, ha='center', va='center',
               fontsize=16, fontweight='bold', color='#0d2a63', zorder=8,
               bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                         edgecolor='#0d2a63', linewidth=0.8, alpha=0.9))
        for lat, lon in pos:
            a.scatter(lon, lat, color='red', s=80, marker='*',
                      edgecolors='k', linewidths=0.5, transform=pc, zorder=9)
        a.set_title(title, fontsize=11, fontweight='bold', pad=5)

        # row 1: 850 hPa wind speed
        a = axes[1, col]
        _add_basemap(a)
        pcm_s8 = a.contourf(lons2d, lats2d, s8_sub, levels=s8_fill,
                             cmap='YlOrRd', transform=pc, zorder=1)
        a.contour(lons2d, lats2d, s8_sub, levels=s8_cont,
                  colors='k', linewidths=0.4, alpha=0.4, transform=pc, zorder=2)
        _add_quiver(a, lons2d[::sk, ::sk], lats2d[::sk, ::sk],
                    u8_sub[::sk, ::sk], v8_sub[::sk, ::sk],
                    scale=90, color='#2f2f2f', zorder=6)
        _add_boundaries(a, zorder=7)
        _add_gridlines(a)
        for lat, lon in pos:
            a.scatter(lon, lat, color='white', s=80, marker='*',
                      edgecolors='k', linewidths=0.5, transform=pc, zorder=9)

    # ── difference panels (col 2) ─────────────────────────────────────────────
    diffs = [
        ('z',  'u5', 'v5', 80,  'ΔZ500: North − West  (m)',
         0, False),
        ('s8', 'u8', 'v8', 30,  'Δ850 hPa wind speed: North − West  (m s⁻¹)',
         1, True),
    ]
    pcm_dz = pcm_ds8 = None
    for fkey, ukey, vkey, qscale, title, row, bot_labels in diffs:
        diff_field = (north_avg[fkey] - west_avg[fkey])[np.ix_(lat_plot, lon_plot)]
        du = (north_avg[ukey] - west_avg[ukey])[np.ix_(lat_plot, lon_plot)]
        dv = (north_avg[vkey] - west_avg[vkey])[np.ix_(lat_plot, lon_plot)]

        # colour scale based on tight Gulf region
        gulf_diff = (north_avg[fkey] - west_avg[fkey])[np.ix_(lat_mask, lon_mask)]
        absmax = np.percentile(np.abs(gulf_diff), 98)
        levels = np.linspace(-absmax, absmax, 21)

        a = axes[row, 2]
        _add_basemap(a)
        pcm = a.contourf(lons2d, lats2d, diff_field, levels=levels,
                         cmap='RdBu_r', transform=pc, zorder=1)
        a.contour(lons2d, lats2d, diff_field, levels=[0],
                  colors='k', linewidths=1.0, transform=pc, zorder=2)
        du_q = du[::sk, ::sk]
        dv_q = dv[::sk, ::sk]
        if fkey == 's8':
            du_q, dv_q = _mask_strongest_vector(du_q, dv_q)
        _add_quiver(a, lons2d[::sk, ::sk], lats2d[::sk, ::sk],
                    du_q, dv_q,
                    scale=qscale * 0.85, color='#222', zorder=6)
        _add_boundaries(a, zorder=7)
        _add_gridlines(a)
        for lat, lon in west_pos:
            a.scatter(lon, lat, color='blue', s=60, marker='*',
                      edgecolors='k', linewidths=0.4, transform=pc, zorder=9)
        for lat, lon in north_pos:
            a.scatter(lon, lat, color='red', s=60, marker='*',
                      edgecolors='k', linewidths=0.4, transform=pc, zorder=9)
        a.set_title(title, fontsize=10, fontweight='bold', pad=5)
        if row == 0:
            pcm_dz  = pcm
        else:
            pcm_ds8 = pcm

    # ── axis labels (post-draw so transforms are resolved) ────────────────────
    fig.canvas.draw()
    # row 0: no bottom labels (row 1 sits below); row 1: bottom labels on all cols
    # left labels only on col 0
    for col in range(2):
        _add_axis_labels(axes[0, col], proj, left_labels=(col == 0), bottom_labels=False)
        _add_axis_labels(axes[1, col], proj, left_labels=(col == 0), bottom_labels=True)
    _add_axis_labels(axes[0, 2], proj, left_labels=False, bottom_labels=False)
    _add_axis_labels(axes[1, 2], proj, left_labels=False, bottom_labels=True)

    # ── colorbars ─────────────────────────────────────────────────────────────
    # Horizontal colorbars below each row.
    # Left+center share one cb; right panel gets its own.
    fig.canvas.draw()

    cb_h   = 0.016   # colorbar height in figure fraction
    cb_gap = [0.008, 0.030]   # row 1 needs extra room for the °W labels

    for row, (pcm_abs, pcm_diff, abs_label, diff_label) in enumerate([
        (pcm_z,  pcm_dz,  'Z500 (m)',              'ΔZ500 (m)'),
        (pcm_s8, pcm_ds8, '850 hPa wind (m s⁻¹)', 'Δ850 wind (m s⁻¹)'),
    ]):
        p0 = axes[row, 0].get_position()
        p1 = axes[row, 1].get_position()
        p2 = axes[row, 2].get_position()
        cb_y = p0.y0 - cb_gap[row] - cb_h

        # shared abs cb spans cols 0–1
        cax_a = fig.add_axes([p0.x0, cb_y, p1.x1 - p0.x0, cb_h])
        cb_a  = fig.colorbar(pcm_abs, cax=cax_a, orientation='horizontal')
        cb_a.set_label(abs_label, fontsize=9)
        cb_a.ax.tick_params(labelsize=8)

        # diff cb spans col 2
        cax_d = fig.add_axes([p2.x0, cb_y, p2.x1 - p2.x0, cb_h])
        cb_d  = fig.colorbar(pcm_diff, cax=cax_d, orientation='horizontal')
        cb_d.set_label(diff_label, fontsize=9)
        cb_d.ax.tick_params(labelsize=8)

    for ax, letter in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left', color='white',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#222',
                          alpha=0.65, edgecolor='none'),
                zorder=15)

    out = PLOT_DIR / '2022-09-22T00Z_ian_gulf_composites_lam2.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')
    return out


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading normalization ...')
    norm = load_norm()
    west_avg, west_pos, west_lam4_pos, north_avg, north_pos, north_lam4_pos = build_composites(norm)
    out = plot_composites(west_avg, west_pos, west_lam4_pos, north_avg, north_pos, north_lam4_pos)
    print(f'Done → {out}')
