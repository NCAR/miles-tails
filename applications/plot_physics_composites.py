#!/usr/bin/env python
"""
plot_physics_composites.py — Storm-centred physics composite maps at FFS interfaces.
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
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm, Normalize
from pathlib import Path
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter as _gf
from tqdm import tqdm

warnings.filterwarnings('ignore')

_here = Path(__file__).resolve().parent
_credit_root = _here.parents[1] / 'miles-credit-main'
for _p in [str(_here), str(_credit_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_ffs_logs import (
    load_all_logs,
    build_genealogy,
    trace_pathway,
    find_stateB_configs,
)

STATIC_NC = '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc'

PRES_LEVELS = np.array([163.0, 500.0, 700.0, 850.0])  # 163 hPa = ERA5 level 70, nearest native level to 200 hPa

EARTH_RADIUS = 6.371e6
DEG2RAD      = np.pi / 180.0

FIELDS = {
    'vws': dict(
        label="VWS′ ~163–850 hPa", unit='m s⁻¹',
        cmap='RdBu_r', vmin=-15, vmax=15, sym=True,
        diff_cmap='RdBu_r', diff_lim=6,
    ),
    'vort850': dict(
        label='ζ₈₅₀',           unit='10⁻⁵ s⁻¹',
        cmap='RdBu_r', vmin=-5,  vmax=20,  sym=True,
        diff_cmap='RdBu_r', diff_lim=2.5,
    ),
    'rh700': dict(
        label='RH₇₀₀',           unit='%',
        cmap='BrBG',   vmin=0,   vmax=100, sym=False,
        diff_cmap='BrBG', diff_lim=20,
    ),
    't500_anom': dict(
        label="T′₅₀₀ (trop)",    unit='K',
        cmap='RdBu_r', vmin=-4,  vmax=4,   sym=True,
        diff_cmap='RdBu_r', diff_lim=2,
    ),
}
FIELD_KEYS = list(FIELDS.keys())


_lats              = None
_lons_360          = None
_lons_180          = None
_surface_geopotential = None


def _worker_init(latlons_path: str, static_nc_path: str):
    global _lats, _lons_360, _lons_180, _surface_geopotential
    import xarray as xr
    latlons = xr.open_dataset(latlons_path).load()
    _lats     = latlons.latitude.values
    _lons_360 = latlons.longitude.values
    _lons_180 = np.where(_lons_360 > 180, _lons_360 - 360, _lons_360)
    with xr.open_dataset(static_nc_path) as ds:
        _surface_geopotential = ds['Z_GDS4_SFC'].values


def _compute_vws(pres_interp):
    # Use 163 hPa (ERA5 level 70, b≈0.01, nearly pure pressure level) as the upper reference.
    # 200 hPa falls in a ~97 hPa gap between model levels 70 (163 hPa) and 80 (260 hPa);
    # requesting 163 hPa recovers the native level-70 data with negligible interpolation error.
    U_upper = pres_interp['U_PRES'].sel(pressure=163., method='nearest').values.squeeze()
    U850    = pres_interp['U_PRES'].sel(pressure=850., method='nearest').values.squeeze()
    V_upper = pres_interp['V_PRES'].sel(pressure=163., method='nearest').values.squeeze()
    V850    = pres_interp['V_PRES'].sel(pressure=850., method='nearest').values.squeeze()
    return np.sqrt((U_upper - U850)**2 + (V_upper - V850)**2)


def _compute_vorticity_850(pres_interp, lats):
    U = pres_interp['U_PRES'].sel(pressure=850., method='nearest').values.squeeze()
    V = pres_interp['V_PRES'].sel(pressure=850., method='nearest').values.squeeze()
    cos_lat = np.cos(lats * DEG2RAD)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)
    dV   = np.gradient(V, axis=1)
    dvdx = dV / (EARTH_RADIUS * cos_lat[:, np.newaxis] * DEG2RAD)
    dU   = np.gradient(U, axis=0)
    dudy = -dU / (EARTH_RADIUS * DEG2RAD)
    return (dvdx - dudy) * 1e5


def _compute_rh700(pres_interp):
    T   = pres_interp['T_PRES'].sel(pressure=700., method='nearest').values.squeeze()
    Q   = pres_interp['Q_PRES'].sel(pressure=700., method='nearest').values.squeeze()
    T_c = T - 273.15
    e_s = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))
    e   = Q * 700.0 / (0.622 + 0.378 * Q)
    return np.clip(100.0 * e / e_s, 0.0, 100.0)


def _compute_t500_anom(pres_interp):
    T = pres_interp['T_PRES'].sel(pressure=500., method='nearest').values.squeeze()
    if _lats is not None:
        band = (_lats >= 0) & (_lats <= 30)
        T_ref = np.nanmean(T[band, :]) if band.any() else np.nanmean(T)
    else:
        T_ref = np.nanmean(T)
    return T - T_ref


def _extract_box(field2d, center_lat, center_lon, lats, lons_180, box_deg):
    N        = box_deg
    box_size = 2 * N + 1
    box      = np.full((box_size, box_size), np.nan)
    ci = int(np.argmin(np.abs(lats - center_lat)))
    cj = int(np.argmin(np.abs(lons_180 - center_lon)))
    for row in range(box_size):
        gi = ci + (row - N)
        for col in range(box_size):
            gj = (cj + (col - N)) % len(lons_180)
            if 0 <= gi < len(lats):
                box[row, col] = field2d[gi, gj]
    return box


def _box_axes(box_deg):
    N = box_deg
    dlat = np.arange(N, -N - 1, -1, dtype=float)
    dlon = np.arange(-N, N + 1,       dtype=float)
    return dlat, dlon


def _center_mean(field2d, ci, cj, r=3):
    patch = field2d[max(0, ci - r):ci + r + 1, max(0, cj - r):cj + r + 1]
    return float(np.nanmean(patch))


def _smooth(data, n, sigma_base=6.0, sigma_min=0.25):
    if data is None or n is None or n < 2:
        return data
    sigma = sigma_base / np.sqrt(max(n, 1))
    if sigma < sigma_min:
        return data
    nan_mask = np.isnan(data)
    fill_val = np.nanmean(data) if not nan_mask.all() else 0.0
    filled   = np.where(nan_mask, fill_val, data)
    smoothed = _gf(filled, sigma=sigma)
    smoothed[nan_mask] = np.nan
    return smoothed


def _process_pkl(args):
    pkl_path, model_config, box_deg = args
    try:
        import xarray as xr
        from credit.output import make_xarray
        from credit.interp import full_state_pressure_interpolation

        with open(pkl_path, 'rb') as fh:
            cfg = pickle.load(fh)

        if not hasattr(cfg, '_y_phys') or cfg.feature_location is None:
            return None

        center_lat, center_lon = cfg.feature_location
        y_phys = cfg._y_phys

        dt_str = (cfg.restart_datetime.strftime('%Y-%m-%d %H:%M:%S')
                  if hasattr(cfg, 'restart_datetime')
                  else '2000-01-01 00:00:00')

        n_ch    = int(y_phys.shape[1])
        n_upper = len(model_config['data']['variables']) * model_config['model']['levels']
        n_surf  = len(model_config['data']['surface_variables'])
        n_diag  = max(n_ch - n_upper - n_surf, 0)

        _conf = dict(model_config)
        _conf['data'] = dict(model_config['data'])
        _conf['data']['diagnostic_variables'] = [f'_diag{i}' for i in range(n_diag)]

        darray_upper, darray_single = make_xarray(
            y_phys, dt_str, _lats, _lons_360, _conf
        )
        ds_merged = xr.merge([
            darray_upper.to_dataset(dim='vars'),
            darray_single.to_dataset(dim='vars'),
        ])

        interp_kwargs = model_config.get('predict', {}).get('interp_pressure', {}).copy()
        interp_kwargs['pressure_levels'] = PRES_LEVELS

        pres_interp = full_state_pressure_interpolation(
            ds_merged,
            _surface_geopotential,
            **interp_kwargs,
        )

        vws_abs = _compute_vws(pres_interp)
        vws_domain_mean = float(np.nanmean(vws_abs))

        raw_fields = {
            'vws':       vws_abs - vws_domain_mean,  # anomaly: removes background shear gradient
            'vort850':   _compute_vorticity_850(pres_interp, _lats),
            'rh700':     _compute_rh700(pres_interp),
            't500_anom': _compute_t500_anom(pres_interp),
        }

        Q700    = pres_interp['Q_PRES'].sel(pressure=700., method='nearest').values.squeeze()
        T700    = pres_interp['T_PRES'].sel(pressure=700., method='nearest').values.squeeze()
        U_upper = pres_interp['U_PRES'].sel(pressure=163., method='nearest').values.squeeze()
        U850    = pres_interp['U_PRES'].sel(pressure=850., method='nearest').values.squeeze()
        V_upper = pres_interp['V_PRES'].sel(pressure=163., method='nearest').values.squeeze()
        V850    = pres_interp['V_PRES'].sel(pressure=850., method='nearest').values.squeeze()
        q700_median_gkg  = float(np.nanmedian(Q700)) * 1e3
        t700_mean_k      = float(np.nanmean(T700))
        rh700_mean       = float(np.nanmean(raw_fields['rh700']))
        u200_mean        = float(np.nanmean(U_upper))
        u850_mean        = float(np.nanmean(U850))
        du_mean          = float(np.nanmean(U_upper - U850))
        dv_mean          = float(np.nanmean(V_upper - V850))

        ci = int(np.argmin(np.abs(_lats     - center_lat)))
        cj = int(np.argmin(np.abs(_lons_180 - center_lon)))

        boxes   = {k: _extract_box(v, center_lat, center_lon,
                                    _lats, _lons_180, box_deg)
                   for k, v in raw_fields.items()}
        scalars = {k: _center_mean(v, ci, cj) for k, v in raw_fields.items()}
        vws_abs_center = float(_center_mean(vws_abs, ci, cj))

        return {
            'config_name':   cfg.config_name,
            'interface_idx': cfg.interface_idx,
            'mslp_value':    cfg.mslp_value,
            'storm_lat':     center_lat,
            'storm_lon':     center_lon,
            'boxes':         boxes,
            'q700_median_gkg': q700_median_gkg,
            't700_mean_k':     t700_mean_k,
            'rh700_mean':      rh700_mean,
            'u200_mean':       u200_mean,
            'u850_mean':       u850_mean,
            'du_mean':         du_mean,
            'dv_mean':         dv_mean,
            **{f'{k}_center': scalars[k] for k in FIELD_KEYS},
            'vws_abs_center': vws_abs_center,
        }

    except Exception as exc:
        import traceback
        return {'error': str(exc), 'traceback': traceback.format_exc(),
                'pkl': str(pkl_path)}


def _mean_composite(stack):
    if not stack:
        return None
    return np.nanmean(np.stack(stack, axis=0), axis=0)


def _build_composites(results, field_keys=FIELD_KEYS):
    stacks = {}
    for r in results:
        idx = r['interface_idx']
        if idx not in stacks:
            stacks[idx] = {k: [] for k in field_keys}
        for k in field_keys:
            if k in r.get('boxes', {}):
                stacks[idx][k].append(r['boxes'][k])

    return {
        idx: {k: _mean_composite(stacks[idx][k]) for k in field_keys}
        for idx in stacks
    }


def _make_norm(meta, diff=False):
    if diff:
        lim = meta['diff_lim']
        return TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), meta['diff_cmap']
    if meta['sym']:
        return TwoSlopeNorm(vmin=meta['vmin'], vcenter=0, vmax=meta['vmax']), meta['cmap']
    return Normalize(vmin=meta['vmin'], vmax=meta['vmax']), meta['cmap']


def _plot_panel(ax, data, meta, dlat, dlon, diff=False, n_configs=None):
    if data is None or np.all(np.isnan(data)):
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='grey')
        ax.set_axis_off()
        return None

    data = _smooth(data, n_configs)
    norm, cmap = _make_norm(meta, diff=diff)
    pcm = ax.pcolormesh(dlon, dlat, data, cmap=cmap, norm=norm, shading='auto')
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.plot(0, 0, 'k*', markersize=8, zorder=5)
    ax.set_aspect('equal')
    return pcm


def plot_composites(composites, iface_list, box_deg, output_path, title,
                    field_keys=None):
    """
    N-row (fields) × M-col (interfaces) composite figure.
    Colorbars are placed in a dedicated right-margin strip so that no data
    panel is resized (fixes the shrunken last-column problem).
    """
    if field_keys is None:
        field_keys = FIELD_KEYS
    n_fields = len(field_keys)
    n_ifaces = len(iface_list)
    if n_ifaces == 0 or n_fields == 0:
        return

    dlat, dlon = _box_axes(box_deg)

    # Reserve right margin for colorbars via gridspec_kw
    fig, axes = plt.subplots(
        n_fields, n_ifaces,
        figsize=(3.2 * n_ifaces, 3.2 * n_fields),
        squeeze=False,
    )

    # First pass: fill all panels, collect one pcm per row for the colorbar
    row_pcm = [None] * n_fields

    for col, (idx, thresh, n_cfg) in enumerate(iface_list):
        lbl = f"λ{idx}" if idx >= 0 else "State B"
        axes[0, col].set_title(f"{lbl}\n{thresh} hPa\nn={n_cfg}",
                               fontsize=8, fontweight='bold')
        for row, key in enumerate(field_keys):
            ax   = axes[row, col]
            meta = FIELDS[key]
            data = composites[col].get(key)
            pcm  = _plot_panel(ax, data, meta, dlat, dlon, n_configs=n_cfg)

            # Save first valid pcm per row — used for the shared colorbar
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

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)

    # tight_layout first so all subplot positions are finalised
    plt.tight_layout()

    # Shrink the right edge of the subplot grid to open a colorbar strip,
    # then place one thin cax per row aligned to that row's bounding box.
    fig.subplots_adjust(right=0.88)

    for row, key in enumerate(field_keys):
        if row_pcm[row] is None:
            continue
        # Bounding box of this row in figure-fraction coordinates
        y0 = axes[row, -1].get_position().y0
        y1 = axes[row,  0].get_position().y1
        cax = fig.add_axes([0.895, y0, 0.012, y1 - y0])
        fig.colorbar(row_pcm[row], cax=cax, label=FIELDS[key]['unit'])
        cax.tick_params(labelsize=7)
        cax.yaxis.label.set_size(8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_differences(composites, iface_list, box_deg, output_path, title,
                     field_keys=None):
    """
    Interface-to-interface difference composites.
    Same colorbar fix applied: dedicated right-margin cax per row.
    """
    if field_keys is None:
        field_keys = FIELD_KEYS
    n_fields = len(field_keys)
    n_transitions = len(composites) - 1
    if n_transitions <= 0 or n_fields == 0:
        return

    dlat, dlon = _box_axes(box_deg)
    fig, axes = plt.subplots(
        n_fields, n_transitions,
        figsize=(3.2 * n_transitions, 3.2 * n_fields),
        squeeze=False,
    )

    row_pcm = [None] * n_fields

    for col in range(n_transitions):
        idx_a, thresh_a, n_a = iface_list[col]
        idx_b, thresh_b, n_b = iface_list[col + 1]
        lbl_a = f"λ{idx_a}" if idx_a >= 0 else "B"
        lbl_b = f"λ{idx_b}" if idx_b >= 0 else "B"
        axes[0, col].set_title(f"{lbl_a}→{lbl_b}\n{thresh_a}→{thresh_b} hPa",
                               fontsize=8, fontweight='bold')
        n_smooth = min(n_a, n_b) if (n_a and n_b) else None
        for row, key in enumerate(field_keys):
            ax   = axes[row, col]
            meta = FIELDS[key]
            d_a  = composites[col].get(key)
            d_b  = composites[col + 1].get(key)
            diff = (d_b - d_a) if (d_a is not None and d_b is not None) else None
            pcm  = _plot_panel(ax, diff, meta, dlat, dlon, diff=True, n_configs=n_smooth)

            if pcm is not None and row_pcm[row] is None:
                row_pcm[row] = pcm

            if col == 0:
                ax.set_ylabel(f"Δ {meta['label']}\n({meta['unit']})", fontsize=8)
            else:
                ax.set_yticklabels([])

            if row == n_fields - 1:
                ax.set_xlabel('Δlon (°)', fontsize=8)
            else:
                ax.set_xticklabels([])

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()

    fig.subplots_adjust(right=0.88)

    for row, key in enumerate(field_keys):
        if row_pcm[row] is None:
            continue
        y0 = axes[row, -1].get_position().y0
        y1 = axes[row,  0].get_position().y1
        cax = fig.add_axes([0.895, y0, 0.012, y1 - y0])
        fig.colorbar(row_pcm[row], cax=cax, label=f"Δ{FIELDS[key]['unit']}")
        cax.tick_params(labelsize=7)
        cax.yaxis.label.set_size(8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_vws_scatter(df, output_path, title):
    """Scatter of absolute VWS at storm centre vs. storm latitude, coloured by interface."""
    import matplotlib.cm as cm

    idxs   = sorted(df['interface_idx'].unique())
    colors = cm.get_cmap('viridis', len(idxs))

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, idx in enumerate(idxs):
        sub = df[df['interface_idx'] == idx].dropna(subset=['storm_lat', 'vws_abs_center'])
        lbl = f'λ{idx}' if idx >= 0 else 'State B'
        ax.scatter(sub['storm_lat'], sub['vws_abs_center'],
                   color=colors(k), label=lbl, alpha=0.55, s=14, zorder=3)

    # Overall linear trend
    valid = df.dropna(subset=['storm_lat', 'vws_abs_center'])
    if len(valid) > 2:
        m, b = np.polyfit(valid['storm_lat'], valid['vws_abs_center'], 1)
        xlim = np.array([valid['storm_lat'].min(), valid['storm_lat'].max()])
        ax.plot(xlim, m * xlim + b, 'k--', lw=1.2,
                label=f'trend: {m:.1f} m s⁻¹ / °N')

    ax.axhline(10, color='grey', ls=':', lw=0.9, label='10 m s⁻¹')
    ax.set_xlabel('Storm latitude (°N)', fontsize=10)
    ax.set_ylabel('VWS at storm centre (m s⁻¹)', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_committor_split(split_composites, iface_thresh, box_deg, plot_dir,
                         title_prefix, field_keys=None, min_n=5):
    """
    One figure per interface: reactive | non-reactive | (reactive − non-reactive).
    Only produced for interfaces where both groups have ≥ min_n samples.
    Two colorbars per row: shared main scale (cols 0–1) + diff scale (col 2).
    """
    if field_keys is None:
        field_keys = FIELD_KEYS
    if not field_keys:
        return

    n_fields = len(field_keys)
    dlat, dlon = _box_axes(box_deg)

    for idx in sorted(split_composites.keys()):
        data  = split_composites[idx]
        n_r   = data['n_reactive']
        n_nr  = data['n_nonreactive']
        n_tot = n_r + n_nr

        lbl = f"λ{idx}" if idx >= 0 else "State B"
        if n_r < min_n or n_nr < min_n:
            print(f"  [committor] {lbl}: skipping "
                  f"(reactive={n_r}, non-reactive={n_nr}, need ≥{min_n} each)")
            continue

        p_commit = n_r / n_tot
        thresh   = iface_thresh.get(idx, '?')

        fig, axes = plt.subplots(
            n_fields, 3,
            figsize=(10.0, 3.2 * n_fields),
            squeeze=False,
        )

        col_titles = [
            f'Reactive (genesis)\nn={n_r}',
            f'Non-reactive (fail)\nn={n_nr}',
            'Reactive − Non-reactive',
        ]
        for col, ctitle in enumerate(col_titles):
            axes[0, col].set_title(ctitle, fontsize=8, fontweight='bold')

        row_pcm_main = [None] * n_fields
        row_pcm_diff = [None] * n_fields

        for row, key in enumerate(field_keys):
            meta = FIELDS[key]
            d_r  = data['reactive'].get(key)
            d_nr = data['nonreactive'].get(key)
            diff = (d_r - d_nr) if (d_r is not None and d_nr is not None) else None

            pcm_r  = _plot_panel(axes[row, 0], d_r,  meta, dlat, dlon, n_configs=n_r)
            pcm_nr = _plot_panel(axes[row, 1], d_nr, meta, dlat, dlon, n_configs=n_nr)
            pcm_d  = _plot_panel(axes[row, 2], diff, meta, dlat, dlon, diff=True,
                                 n_configs=min(n_r, n_nr))

            if pcm_r is not None and row_pcm_main[row] is None:
                row_pcm_main[row] = pcm_r
            if pcm_d is not None and row_pcm_diff[row] is None:
                row_pcm_diff[row] = pcm_d

            axes[row, 0].set_ylabel(f"{meta['label']}\n({meta['unit']})", fontsize=8)
            axes[row, 1].set_yticklabels([])
            axes[row, 2].set_yticklabels([])

            if row == n_fields - 1:
                for col in range(3):
                    axes[row, col].set_xlabel('Δlon (°)', fontsize=8)
            else:
                for col in range(3):
                    axes[row, col].set_xticklabels([])

        fig.suptitle(
            f'{title_prefix} — {lbl} ({thresh} hPa)\n'
            f'p_commit = {p_commit:.2f}   '
            f'(reactive={n_r}, non-reactive={n_nr})',
            fontsize=10, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
        fig.subplots_adjust(right=0.80)

        # Two colorbars per row: main (shared for cols 0-1) and diff (col 2)
        for row, key in enumerate(field_keys):
            y0 = axes[row, -1].get_position().y0
            y1 = axes[row,  0].get_position().y1
            if row_pcm_main[row] is not None:
                cax = fig.add_axes([0.815, y0, 0.010, y1 - y0])
                fig.colorbar(row_pcm_main[row], cax=cax, label=FIELDS[key]['unit'])
                cax.tick_params(labelsize=6)
                cax.yaxis.label.set_size(7)
            if row_pcm_diff[row] is not None:
                cax2 = fig.add_axes([0.845, y0, 0.010, y1 - y0])
                fig.colorbar(row_pcm_diff[row], cax=cax2,
                             label=f"Δ{FIELDS[key]['unit']}")
                cax2.tick_params(labelsize=6)
                cax2.yaxis.label.set_size(7)

        fname = (f'committor_split_lambda{idx}.png' if idx >= 0
                 else 'committor_split_stateB.png')
        out_path = plot_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def _collect_pkls(ic_dir: Path, n_interfaces: int):
    out = {}
    flux_pkls = sorted((ic_dir / 'flux').glob('lambda0_config_*.pkl')) \
                if (ic_dir / 'flux').exists() else []
    if flux_pkls:
        out[0] = flux_pkls

    for i in range(1, n_interfaces):
        d    = ic_dir / str(i)
        pkls = sorted(d.glob(f'lambda{i}_config_*.pkl')) if d.exists() else []
        if pkls:
            out[i] = pkls

    sb_pkls = sorted((ic_dir / 'stateB').glob('stateB_config_*.pkl')) \
              if (ic_dir / 'stateB').exists() else []
    if sb_pkls:
        out[-1] = sb_pkls

    return out


def _fmt_ic(ic_time_str: str) -> str:
    from datetime import datetime
    return datetime.strptime(ic_time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%HZ')


def main():
    parser = argparse.ArgumentParser(
        description='Storm-centred physics composite maps at FFS interfaces'
    )
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--ffs_config',   required=True)
    parser.add_argument('--output_dir',   default=None)
    parser.add_argument('--plot_dir',     default=None)
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--box_deg',      type=int, default=12)
    parser.add_argument('--force',        action='store_true')
    parser.add_argument('--reactive_only', action='store_true')
    parser.add_argument('--plot_vws',     action='store_true',
                        help='Include VWS row in composite/difference plots (default: off)')
    parser.add_argument('--min_committor_n', type=int, default=5,
                        help='Min samples in each group to produce committor split plot (default: 5)')
    args = parser.parse_args()

    plot_field_keys = [k for k in FIELD_KEYS if k != 'vws' or args.plot_vws]

    with open(args.model_config) as fh:
        model_config = yaml.safe_load(fh)
    with open(args.ffs_config) as fh:
        ffs_config = yaml.safe_load(fh)

    model_config.setdefault('data', {})

    latlons_path   = model_config['loss']['latitude_weights']
    static_nc_path = STATIC_NC

    ffs_out    = Path(ffs_config['output_dir'])
    output_dir = Path(args.output_dir) if args.output_dir else ffs_out / 'physics'
    plot_dir   = Path(args.plot_dir)   if args.plot_dir   else output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    interfaces   = sorted(ffs_config['interfaces'], reverse=True)
    state_B      = ffs_config['state_B']
    n_interfaces = len(interfaces)

    iface_thresh = {i: interfaces[i] for i in range(n_interfaces)}
    iface_thresh[-1] = state_B

    all_records        = []
    csv_path           = output_dir / 'physics_scalars.csv'
    global_stacks      = {}
    global_split_stacks = {}  # {idx: {'r': {k: [boxes]}, 'nr': {k: [boxes]}}}

    for ic_time in ffs_config['forecast_start_times']:
        time_label = _fmt_ic(ic_time)
        ic_dir     = ffs_out / time_label

        if not ic_dir.exists():
            continue

        _suffix    = '_reactive' if args.reactive_only else ''
        cache_path = output_dir / f'{time_label}_physics{_suffix}.pkl'

        if cache_path.exists() and not args.force:
            print(f"[{time_label}] Loading cache ({cache_path.name}) …")
            with open(cache_path, 'rb') as fh:
                ic_results = pickle.load(fh)
        else:
            pkl_map  = _collect_pkls(ic_dir, n_interfaces)
            all_pkls = [(pkl, idx) for idx, pkls in pkl_map.items() for pkl in pkls]
            if not all_pkls:
                print(f"[{time_label}] No pkls found, skipping.")
                continue

            if args.reactive_only:
                logs_dir = ic_dir / 'logs'
                entries  = load_all_logs(logs_dir)
                genealogy = build_genealogy(entries)
                b_configs = find_stateB_configs(entries, state_B)

                reactive_names: set[str] = set()
                for b_info in b_configs:
                    for step in trace_pathway(genealogy, b_info['config']):
                        if step['config'] is not None:
                            reactive_names.add(step['config'])

                n_before = len(all_pkls)
                all_pkls = [(pkl, idx) for pkl, idx in all_pkls
                            if pkl.stem in reactive_names]
                print(f"[{time_label}] reactive_only: {len(b_configs)} B-state events, "
                      f"{len(all_pkls)}/{n_before} configs are ancestors")

                if not all_pkls:
                    print(f"  No reactive configs found (no B-state crossings yet?), skipping.")
                    continue

            print(f"[{time_label}] {len(all_pkls)} configs → {args.workers} workers …")
            worker_args = [(str(pkl), model_config, args.box_deg)
                           for pkl, _ in all_pkls]

            print(f"  Preflight check …", end=' ', flush=True)
            _worker_init(latlons_path, static_nc_path)
            test = _process_pkl(worker_args[0])
            if test is not None and 'error' in test:
                print(f"FAILED\n  ✗ {test['error']}\n  Skipping {time_label}.")
                continue
            print("OK")
            print(f"    [diag] Q700 median   = {test.get('q700_median_gkg', float('nan')):.4e} g/kg")
            print(f"    [diag] T700 mean     = {test.get('t700_mean_k', float('nan')):.2f} K")
            print(f"    [diag] RH700 mean    = {test.get('rh700_mean', float('nan')):.2f} %")
            print(f"    [diag] U163 mean     = {test.get('u200_mean', float('nan')):.4f} m/s  (ERA5 level 70, ≈163 hPa)")
            print(f"    [diag] U850 mean     = {test.get('u850_mean', float('nan')):.4f} m/s")
            print(f"    [diag] ΔU(163-850)   = {test.get('du_mean', float('nan')):.4f} m/s  ← expect nonzero for VWS")
            print(f"    [diag] ΔV(163-850)   = {test.get('dv_mean', float('nan')):.4f} m/s")

            with Pool(
                processes=args.workers,
                initializer=_worker_init,
                initargs=(latlons_path, static_nc_path),
            ) as pool:
                raw = list(tqdm(
                    pool.imap(_process_pkl, worker_args),
                    total=len(worker_args),
                    desc=time_label,
                    dynamic_ncols=True,
                ))

            ic_results = [r for r in raw if r is not None and 'error' not in r]
            errors     = [r for r in raw if r is not None and 'error' in r]
            if errors:
                print(f"  ⚠ {len(errors)} errors — first: {errors[0]['error'][:120]}")

            with open(cache_path, 'wb') as fh:
                pickle.dump(ic_results, fh)
            print(f"  Cached {len(ic_results)} results → {cache_path.name}")

        if not ic_results:
            continue

        # ── Committor tagging ──────────────────────────────────────────────
        # Always build reactive_names from logs so we can tag each result,
        # even when loading from cache (fast, log parsing only).
        reactive_names_ic: set[str] = set()
        logs_dir = ic_dir / 'logs'
        if logs_dir.exists():
            try:
                _entries   = load_all_logs(logs_dir)
                _genealogy = build_genealogy(_entries)
                _b_configs = find_stateB_configs(_entries, state_B)
                for b_info in _b_configs:
                    for step in trace_pathway(_genealogy, b_info['config']):
                        if step.get('config') is not None:
                            reactive_names_ic.add(step['config'])
                n_react = sum(1 for r in ic_results
                              if r['config_name'] in reactive_names_ic)
                print(f"  [committor] {len(_b_configs)} State-B events → "
                      f"{len(reactive_names_ic)} reactive ancestors "
                      f"({n_react}/{len(ic_results)} configs in results)")
            except Exception as _e:
                print(f"  [committor] Could not build reactive_names: {_e}")

        for r in ic_results:
            r['is_reactive'] = r['config_name'] in reactive_names_ic
        # ──────────────────────────────────────────────────────────────────

        _diag_cols = ['q700_median_gkg', 't700_mean_k', 'rh700_mean',
                      'u200_mean', 'u850_mean', 'du_mean', 'dv_mean']
        for r in ic_results:
            all_records.append({
                'ic_time':       ic_time,
                'time_label':    time_label,
                'config_name':   r['config_name'],
                'interface_idx': r['interface_idx'],
                'mslp_value':    r['mslp_value'],
                'storm_lat':     r['storm_lat'],
                'storm_lon':     r['storm_lon'],
                **{f'{k}_center': r.get(f'{k}_center', np.nan) for k in FIELD_KEYS},
                'vws_abs_center': r.get('vws_abs_center', np.nan),
                **{col: r.get(col, np.nan) for col in _diag_cols},
            })
        pd.DataFrame(all_records).to_csv(csv_path, index=False)
        print(f"  CSV: {len(all_records)} rows → {csv_path.name}")

        ic_composites = _build_composites(ic_results)

        sorted_idxs  = sorted([i for i in ic_composites if i >= 0]) + \
                       ([-1] if -1 in ic_composites else [])
        n_per_iface  = {idx: len([r for r in ic_results if r['interface_idx'] == idx])
                        for idx in sorted_idxs}

        composites_list = [ic_composites[idx] for idx in sorted_idxs]
        iface_list      = [(idx, iface_thresh.get(idx, '?'), n_per_iface[idx])
                           for idx in sorted_idxs]

        plot_composites(
            composites_list, iface_list, args.box_deg,
            plot_dir / f'{time_label}_composites.png',
            title=f'Storm-Centred Physics Composites — {time_label}',
            field_keys=plot_field_keys,
        )

        for idx, field_dict in ic_composites.items():
            if idx not in global_stacks:
                global_stacks[idx] = {k: [] for k in FIELD_KEYS}
            for k in FIELD_KEYS:
                src = [r['boxes'].get(k) for r in ic_results
                       if r['interface_idx'] == idx and k in r.get('boxes', {})]
                global_stacks[idx][k].extend(src)

        # Accumulate split stacks (reactive vs non-reactive) for committor plots
        for r in ic_results:
            idx   = r['interface_idx']
            group = 'r' if r.get('is_reactive', False) else 'nr'
            if idx not in global_split_stacks:
                global_split_stacks[idx] = {
                    'r':  {k: [] for k in FIELD_KEYS},
                    'nr': {k: [] for k in FIELD_KEYS},
                }
            for k in FIELD_KEYS:
                box = r.get('boxes', {}).get(k)
                if box is not None:
                    global_split_stacks[idx][group][k].append(box)

    if global_stacks:
        sorted_idxs = sorted([i for i in global_stacks if i >= 0]) + \
                      ([-1] if -1 in global_stacks else [])

        global_composites = {
            idx: {k: _mean_composite(global_stacks[idx][k]) for k in FIELD_KEYS}
            for idx in sorted_idxs
        }

        _count_key = 'vort850'  # stable field always present regardless of --plot_vws
        n_per_iface = {idx: len(global_stacks[idx].get(_count_key, []))
                       for idx in sorted_idxs}

        composites_list = [global_composites[idx] for idx in sorted_idxs]
        iface_list      = [(idx, iface_thresh.get(idx, '?'), n_per_iface[idx])
                           for idx in sorted_idxs]

        plot_composites(
            composites_list, iface_list, args.box_deg,
            plot_dir / 'all_ics_composites.png',
            title='Storm-Centred Physics Composites — All ICs',
            field_keys=plot_field_keys,
        )
        plot_differences(
            composites_list, iface_list, args.box_deg,
            plot_dir / 'all_ics_differences.png',
            title='Interface-to-Interface Physics Changes — All ICs',
            field_keys=plot_field_keys,
        )

        if all_records:
            plot_vws_scatter(
                pd.DataFrame(all_records),
                plot_dir / 'vws_lat_scatter.png',
                title='VWS at storm centre vs. storm latitude — All ICs',
            )

        print("\n── Global mean scalars by interface ──")
        if all_records:
            df = pd.DataFrame(all_records)
            scalar_cols = [f'{k}_center' for k in FIELD_KEYS]
            tbl = df.groupby('interface_idx')[scalar_cols].agg(['mean', 'std'])
            print(tbl.to_string())
            print("\n── Storm latitude by interface (mean ± std) ──")
            lat_tbl = df.groupby('interface_idx')['storm_lat'].agg(['mean', 'std', 'min', 'max'])
            print(lat_tbl.to_string())

        # ── Committor split plots ──────────────────────────────────────────
        if global_split_stacks:
            count_key = 'vort850'
            print("\n── Committor probabilities by interface ──")
            print(f"  {'Interface':10s}  {'reactive':>10s}  {'non-reactive':>12s}  "
                  f"{'total':>7s}  {'p_commit':>9s}")
            commit_composites = {}
            for idx in sorted(global_split_stacks.keys()):
                v    = global_split_stacks[idx]
                n_r  = len(v['r'].get(count_key, []))
                n_nr = len(v['nr'].get(count_key, []))
                n_tot = n_r + n_nr
                p    = n_r / n_tot if n_tot > 0 else float('nan')
                lbl  = f"λ{idx}" if idx >= 0 else "State B"
                print(f"  {lbl:10s}  {n_r:10d}  {n_nr:12d}  "
                      f"{n_tot:7d}  {p:9.3f}")
                commit_composites[idx] = {
                    'reactive':      {k: _mean_composite(v['r'][k])  for k in FIELD_KEYS},
                    'nonreactive':   {k: _mean_composite(v['nr'][k]) for k in FIELD_KEYS},
                    'n_reactive':    n_r,
                    'n_nonreactive': n_nr,
                }

            plot_committor_split(
                commit_composites, iface_thresh, args.box_deg,
                plot_dir,
                title_prefix='Committor Split — All ICs',
                field_keys=plot_field_keys,
                min_n=args.min_committor_n,
            )
        # ──────────────────────────────────────────────────────────────────

    if all_records:
        print(f"\nFinal CSV: {csv_path}  ({len(all_records)} rows total)")

    print("\nDone.")


if __name__ == '__main__':
    main()