#!/usr/bin/env python
"""
plot_state_ab_slides.py

Generate two clean presentation figures:
  - State A: lambda0 crossing (Earl IC, seed lambda0_config_0501_UR)
  - State B: stateB crossing (Earl IC, stateB_config_0024_FI)

Outputs:
  plots/state_A_slide.png
  plots/state_B_slide.png
"""

import os, sys, pickle, warnings
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path('/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-02T00Z')
MEAN_NC = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/mean_std/mean_6h_1979_2018_cesm.nc'
STD_NC  = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_mlevel_cesm_stage1/mean_std/std_residual_6h_1979_2018_cesm.nc'

STATE_A_PKL = RESULTS_DIR / 'flux'   / 'lambda0_config_0501_UR.pkl'
STATE_B_PKL = RESULTS_DIR / 'stateB' / 'stateB_config_0024_FI.pkl'

PLOT_DIR = Path('/glade/work/schreck/repos/miles-tails/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Grid
LATS = np.linspace(90, -90, 192)
LONS = np.linspace(0, 360 - 360 / 288, 288)  # 0-360

# Channel: SP = 64
CH_SP = 64
# Channel: T at lowest model level (level index 15 = level 137) → channel 32+15=47
CH_T_SFC = 47


# ── normalization ─────────────────────────────────────────────────────────────

def load_norm():
    ds_mean = xr.open_dataset(MEAN_NC)
    ds_std  = xr.open_dataset(STD_NC)
    return {
        'SP':  (float(ds_mean['SP']),          float(ds_std['SP'])),
        'T':   (float(ds_mean['T'].values[15]), float(ds_std['T'].values[15])),
    }


def load_mslp(pkl_path, norm):
    """Load pkl, denormalize SP → hPa, apply light smoothing."""
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    s = obj.input_state.numpy()   # [1, 71, 1, 192, 288]

    sp_pa   = s[0, CH_SP,   0] * norm['SP'][1] + norm['SP'][0]   # Pa
    mslp_hpa = sp_pa / 100.0                                       # hPa (ocean: SP ≈ MSLP)

    # Periodic pad + smooth
    n_pad = 20
    padded = np.concatenate([mslp_hpa[:, -n_pad:], mslp_hpa, mslp_hpa[:, :n_pad]], axis=1)
    smoothed = gaussian_filter(padded, sigma=2.5)
    mslp_hpa = smoothed[:, n_pad:-n_pad]

    loc  = obj.feature_location
    lat  = float(loc[0])
    lon  = float(loc[1])
    if lon > 180:
        lon -= 360
    mslp_val = float(obj.mslp_value)
    dt_str   = obj.restart_datetime.strftime('%Y-%m-%d %H:%M UTC')

    return mslp_hpa, lat, lon, mslp_val, dt_str


# ── plotting ──────────────────────────────────────────────────────────────────

EXTENT = [-100, -10, 0, 65]

PROJ = ccrs.LambertConformal(
    central_longitude=-55.0,
    central_latitude=30.0,
    standard_parallels=(20, 45),
)
PC = ccrs.PlateCarree()


def make_figure(mslp_hpa, center_lat, center_lon, mslp_val, dt_str,
                title, subtitle, out_path, star_color='gold'):

    lons_signed = np.where(LONS > 180, LONS - 360, LONS)
    lons2d, lats2d = np.meshgrid(lons_signed, LATS)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5),
                           subplot_kw=dict(projection=PROJ))

    ax.set_extent(EXTENT, crs=PC)

    # Filled MSLP
    levels = np.arange(960, 1032, 2)
    cmap   = plt.cm.RdBu_r
    norm   = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    pcm = ax.pcolormesh(lons2d, lats2d, mslp_hpa,
                        norm=norm, cmap=cmap,
                        transform=PC, shading='auto', zorder=1)

    # Contour lines
    cs = ax.contour(lons2d, lats2d, mslp_hpa,
                    levels=np.arange(960, 1028, 4),
                    colors='k', linewidths=0.5, alpha=0.45,
                    transform=PC, zorder=2)
    ax.clabel(cs, fmt='%d', fontsize=7, inline=True)

    # Basemap
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),   facecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'),    facecolor='#e8e8e8', zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.4, alpha=0.6, zorder=4)
    ax.add_feature(cfeature.STATES.with_scale('50m'),    linewidth=0.3, alpha=0.4, zorder=4)

    # Gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.4, color='gray',
                      alpha=0.5, linestyle='--', zorder=5)
    gl.xlocator = mticker.FixedLocator(range(-100, 0, 10))
    gl.ylocator = mticker.FixedLocator(range(0, 70, 10))

    # Manual axis labels
    fig.canvas.draw()
    for lat_tick in [10, 20, 30, 40, 50, 60]:
        try:
            x_lc, y_lc = PROJ.transform_point(-100, lat_tick, PC)
            xd, yd = ax.transData.transform((x_lc, y_lc))
            _, y_ax = ax.transAxes.inverted().transform((xd, yd))
            if 0.01 <= y_ax <= 0.99:
                ax.text(-0.01, y_ax, f'{lat_tick}°N',
                        transform=ax.transAxes, ha='right', va='center',
                        fontsize=8, clip_on=False)
        except Exception:
            pass
    for lon_tick in range(-100, 0, 10):
        try:
            x_lc, y_lc = PROJ.transform_point(lon_tick, 0, PC)
            xd, yd = ax.transData.transform((x_lc, y_lc))
            x_ax, _ = ax.transAxes.inverted().transform((xd, yd))
            if 0.01 <= x_ax <= 0.99:
                ax.text(x_ax, -0.02, f'{abs(lon_tick)}°W',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=8, clip_on=False)
        except Exception:
            pass

    # Storm marker
    ax.scatter(center_lon, center_lat,
               s=220, marker='*', color=star_color,
               edgecolors='k', linewidths=0.8,
               transform=PC, zorder=10)

    # Annotation box near star
    ax.annotate(
        f'{mslp_val:.0f} hPa',
        xy=(center_lon, center_lat), xycoords=PC._as_mpl_transform(ax),
        xytext=(8, 8), textcoords='offset points',
        fontsize=10, fontweight='bold', color='#111',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                  edgecolor='#444', alpha=0.85),
        zorder=11,
    )

    # Colorbar
    fig.subplots_adjust(bottom=0.12, left=0.06, right=0.96, top=0.90)
    cax = fig.add_axes([0.10, 0.06, 0.80, 0.022])
    cb  = fig.colorbar(pcm, cax=cax, orientation='horizontal',
                       ticks=np.arange(960, 1033, 8))
    cb.set_label('MSLP (hPa)', fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # Title
    ax.set_title(f'{title}\n{subtitle}   ·   {dt_str}',
                 fontsize=13, fontweight='bold', pad=8)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading normalization ...')
    norm = load_norm()

    print('Loading State A pkl ...')
    mslp_a, lat_a, lon_a, val_a, dt_a = load_mslp(STATE_A_PKL, norm)
    make_figure(
        mslp_a, lat_a, lon_a, val_a, dt_a,
        title='State A  (λ₀)',
        subtitle='Unorganized tropical disturbance  |  MSLP ≥ 989 hPa',
        out_path=PLOT_DIR / 'state_A_slide.png',
        star_color='gold',
    )

    print('Loading State B pkl ...')
    mslp_b, lat_b, lon_b, val_b, dt_b = load_mslp(STATE_B_PKL, norm)
    make_figure(
        mslp_b, lat_b, lon_b, val_b, dt_b,
        title='State B',
        subtitle='Organized tropical cyclone  |  MSLP ≤ 975 hPa',
        out_path=PLOT_DIR / 'state_B_slide.png',
        star_color='cyan',
    )

    print('Done.')
