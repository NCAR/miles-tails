#!/usr/bin/env python
"""
plot_committor_fields.py — Atmospheric-variable committor p_B(ξ | λᵢ) curves.

One figure per IC.  For each FFS interface λᵢ and each atmospheric variable ξ
extracted at the storm center from the full model state stored in each pkl,
estimates:

    p_B(ξ ∈ bin) = configs_with_B_descendants_in_bin / configs_in_bin

ALL channels from cfg._y_phys  [shape: 1 × n_ch × 1 × H × W] are extracted at
the storm center and plotted.  The variable list is built dynamically from the
actual channel count so any extra pressure-interpolated diagnostics beyond the
base 72 channels are automatically included.

Known channel layout (n_ch may be > 72 if the model has additional diagnostics):
  ch   0–15 : U  at IFS model levels [10,30,40,50,60,70,80,90,95,100,105,110,120,130,136,137]
  ch  16–31 : V  at same levels
  ch  32–47 : T  at same levels
  ch  48–63 : Q  at same levels
  ch  64    : SP (Pa)
  ch  65    : t2m (K)
  ch  66    : V500 (m/s)     ← pressure-interpolated surface diagnostics
  ch  67    : U500 (m/s)
  ch  68    : T500 (K)
  ch  69    : Z500 (m²/s²)
  ch  70    : Q500 (kg/kg)
  ch  71    : MSLP (Pa)
  ch  72+   : additional pressure-interpolated vars (names labelled ch72, ch73, …)

'mslp' is also extracted from cfg.mslp_value attr (same quantity as ch71 but
already in hPa and always present even without _y_phys).

Channel layout (matches model.yml):
  variables      = ['U','V','T','Q']
  level_ids      = [10,30,40,50,60,70,80,90,95,100,105,110,120,130,136,137]
                   ^^^^ IFS hybrid-sigma MODEL LEVELS, not pressure levels ^^^^
  surface_vars   = ['SP','t2m','V500','U500','T500','Z500','Q500']
  MSLP appended last  (channel 71, Pa)

Per-IC results are cached in  <ic_dir>/committor_fields_cache.pkl  so re-runs
with different plot settings (--n_bins, --min_samples) are fast.

Notes
-----
- If a pkl lacks _y_phys (e.g. older files), MSLP is still reported.
- Field extraction adds significant I/O on first run — expect ~1 min per IC
  if _y_phys stores the full global tensor.  Use --workers for parallelism.
- Shear proxy uses level_ids[1]=30 as "upper" level — not standard 200 hPa
  but captures qualitative upper-troposphere vs mid-troposphere shear signal.

Usage
-----
    python plot_committor_fields.py \\
        --ffs_config ffs.yml \\
        --ffs_csv    results/ffs_statistics_all_ics.csv \\
        --output_dir results \\
        --plot_dir   results/plots \\
        --workers    8 \\
        --min_samples 5 \\
        --n_bins     15
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


# ── Channel layout ─────────────────────────────────────────────────────────────
# _y_phys: [batch=1, channels=72, time=1, H, W]
# Access:  _y_phys[0, ch, 0, lat_i, lon_j]
#
# Upper-air (each variable occupies 16 channels in level_ids order):
#   level_ids = [10,30,40,50,60,70,80,90,95,100,105,110,120,130,136,137]
#   These are IFS hybrid-sigma MODEL LEVELS, not pressure levels.
#   Approximate pressure values at standard surface pressure (Ps ≈ 1013 hPa):
#     index  0  = level_id  10  (≈ stratosphere, very high)
#     index  1  = level_id  30  (≈ lower stratosphere, ~50 hPa)  ← NOT upper-trop!
#     index  3  = level_id  50  (≈ 200 hPa proxy, upper troposphere)
#     index  6  = level_id  80  (≈ 500 hPa proxy)
#     index  9  = level_id 100  (≈ 700 hPa proxy)
#     index 12  = level_id 120  (≈ 850 hPa proxy)
#     index 13  = level_id 130  (≈ 925 hPa proxy)
#     index 15  = level_id 137  (lowest model level, ~surface)
#   Pressure is approximate and varies with surface pressure — label accordingly.
#
#   U: channels  0–15
#   V: channels 16–31
#   T: channels 32–47
#   Q: channels 48–63
#
# Surface diagnostics (model interpolated to fixed pressure levels):
#   SP=64, t2m=65, V500=66, U500=67, T500=68, Z500=69, Q500=70
# MSLP (Pa): channel 71

# ── 3D field channels (model level index → channel) ──────────────────────────
# Upper (≈200 hPa proxy):  level_id=50, index 3
CH_U_200   =  3    # U at level_id=50  (≈200 hPa proxy)
CH_V_200   = 19    # V at level_id=50
# Mid-level (≈500 hPa proxy): level_id=80, index 6
# Note: prefer surface diagnostic U500/V500/T500/Z500/Q500 for 500 hPa (more accurate)
# Low-level (≈850 hPa proxy): level_id=120, index 12
CH_U_850   = 12    # U at level_id=120 (≈850 hPa proxy)
CH_V_850   = 28    # V at level_id=120
CH_T_850   = 44    # T at level_id=120 (K)
CH_Q_700   = 57    # Q at level_id=100 (≈700 hPa proxy, index 9)
CH_Q_850   = 60    # Q at level_id=120 (≈850 hPa proxy, index 12)

# ── Surface diagnostic channels ───────────────────────────────────────────────
CH_T2M     = 65
CH_V500    = 66    # V-wind at 500 hPa (surface diagnostic)
CH_U500    = 67    # U-wind at 500 hPa
CH_T500    = 68    # Temperature at 500 hPa
CH_Z500    = 69    # Geopotential at 500 hPa
CH_Q500    = 70    # Specific humidity at 500 hPa
CH_MSLP    = 71    # MSLP (Pa)

# Grid: 1° global, ERA5 / CREDIT convention
#   lats: 90→-90 (north-to-south), shape 181
#   lons: 0→359  (west-to-east),   shape 360  (stored 0-360, not -180/180)
LAT_NORTH = 90.0


def _lat_idx(lat_deg: float) -> int:
    """Storm lat (°N) → row index in N→S 1° grid."""
    return int(round((LAT_NORTH - lat_deg)))


def _lon_idx(lon_deg: float) -> int:
    """Storm lon (°, any convention) → col index in 0→359 grid."""
    return int(round(lon_deg % 360.0)) % 360


# ── Variable list — built dynamically from tensor channel count ────────────────
#
# Known layout (72 channels, channels 72+ are additional pressure-interp vars):
#   U(0-15), V(16-31), T(32-47), Q(48-63)  ← IFS model levels
#   SP(64), t2m(65), V500(66), U500(67), T500(68), Z500(69), Q500(70), MSLP(71)
#   ch72+ : additional pressure-interpolated diagnostics (names unknown a priori)
#
_LEVEL_IDS = [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]
_N_LEVELS  = len(_LEVEL_IDS)  # 16

# Known surface diagnostics in channel order starting at 64
_SFC_KNOWN = [
    ('SP',   'Pa',    '#a05d56'),
    ('t2m',  'K',     '#ff7f0e'),
    ('V500', 'm/s',   '#2ca02c'),
    ('U500', 'm/s',   '#1f77b4'),
    ('T500', 'K',     '#d62728'),
    ('Z500', 'm²/s²', '#9467bd'),
    ('Q500', 'kg/kg', '#17becf'),
    ('MSLP', 'Pa',    '#333333'),
]

# Colour ramps for the 4 upper-air groups
_GROUP_CMAPS = ['Blues', 'Oranges', 'Reds', 'Greens']
_GROUP_META  = [
    ('U', 'm/s',   ),
    ('V', 'm/s',   ),
    ('T', 'K',     ),
    ('Q', 'kg/kg', ),
]


def _build_variable_list(n_channels: int) -> list:
    """
    Build the full variable list for n_channels tensor channels + the mslp attr.

    Returns list of dict(key, label, unit, color).
    The 'mslp' key is always first (sourced from cfg.mslp_value, not the tensor).
    Remaining keys are 'ch0' … 'ch{n_channels-1}'.
    """
    import matplotlib.pyplot as _plt

    variables = [
        dict(key='mslp', label='MSLP attr (hPa)', unit='hPa', color='#1f77b4'),
    ]

    ch = 0
    # ── 3D upper-air: U, V, T, Q × 16 model levels ───────────────────────────
    for gi, ((vname, unit), cmap_name) in enumerate(zip(_GROUP_META, _GROUP_CMAPS)):
        cmap = _plt.cm.get_cmap(cmap_name)
        for k, lv in enumerate(_LEVEL_IDS):
            if ch < n_channels:
                # Use darker end of the ramp so colours are legible
                c = cmap(0.35 + 0.55 * k / max(1, _N_LEVELS - 1))
                variables.append(dict(
                    key=f'ch{ch}',
                    label=f'{vname} lv{lv}',
                    unit=unit,
                    color=c,
                ))
                ch += 1

    # ── Surface diagnostics (known names) ─────────────────────────────────────
    for name, unit, color in _SFC_KNOWN:
        if ch < n_channels:
            variables.append(dict(key=f'ch{ch}', label=name, unit=unit, color=color))
            ch += 1

    # ── Additional pressure-interpolated vars (names unknown) ─────────────────
    while ch < n_channels:
        variables.append(dict(
            key=f'ch{ch}', label=f'ch{ch}', unit='?', color='#888888',
        ))
        ch += 1

    return variables


def _get_n_channels(records: dict) -> int:
    """Peek at first record to find tensor channel count."""
    for entries in records.values():
        for entry in entries:
            n = entry[3].get('_n_channels')
            if n is not None:
                return n
    return 72  # fallback to known layout size


# ── Curated variable selection ─────────────────────────────────────────────────
# Derived from TVD separability ranking across all ICs (run --top_k_channels).
# One representative per physical cluster — redundant correlated channels dropped.
#
#   ch44  T lv120   (≈850 hPa proxy)   TVD 0.373  lower-trop warm anomaly
#   ch65  t2m                           TVD 0.322  SST / surface temperature
#   ch36  T lv60    (≈300 hPa proxy)   TVD 0.271  upper-trop warm core
#   ch63  Q lv137   (lowest model lv)  TVD 0.292  surface moisture  ← strongest Q signal
#   ch60  Q lv120   (≈850 hPa proxy)   TVD 0.197  low-level moisture
#   ch50  Q lv40    (≈100 hPa proxy)   TVD 0.232  upper-trop moisture
#   ch3   U lv50    (≈200 hPa proxy)   TVD 0.246  upper-level wind (shear / outflow)
#   ch69  Z500      (surface diag)     TVD 0.202  mid-level geopotential
#   mslp  attr                         TVD 0.368  FFS order parameter (sanity check)
#
# SP (ch64) ranked #1 but is redundant with MSLP over ocean — excluded.
# T lv130/136/137/110 are highly correlated with T lv120 — excluded.
_CURATED_KEYS = {'mslp', 'ch44', 'ch65', 'ch36', 'ch63', 'ch60', 'ch50', 'ch3', 'ch69'}


def _get_curated_variables(all_variables: list) -> list:
    """Filter a full variable list down to the curated set, preserving order."""
    return [v for v in all_variables if v['key'] in _CURATED_KEYS]


# ── pkl helpers ────────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _iface_dir(ic_dir: Path, iface_idx: int, n_ifaces: int) -> Path:
    if iface_idx == 0:
        return ic_dir / 'flux'
    if iface_idx == n_ifaces - 1:
        return ic_dir / 'stateB'
    return ic_dir / str(iface_idx)


def _extract_fields(cfg) -> dict:
    """
    Extract ALL tensor channels from cfg._y_phys at the storm center.

    Returns dict with:
      'mslp'        : float  — from cfg.mslp_value attr (hPa)
      '_n_channels' : int    — number of channels in the tensor
      'ch0' … 'chN' : float  — raw value of each tensor channel at storm center

    Returns {} if feature_location is absent.
    Channel keys are absent (only 'mslp' present) if _y_phys is missing.
    """
    loc = getattr(cfg, 'feature_location', None)
    if loc is None:
        return {}
    try:
        lat, lon = float(loc[0]), float(loc[1])
    except (TypeError, IndexError):
        return {}

    result = {'mslp': float(getattr(cfg, 'mslp_value', np.nan))}

    y_phys = getattr(cfg, '_y_phys', None)
    if y_phys is None:
        return result

    try:
        # Support both torch.Tensor (.numpy()) and np.ndarray
        try:
            field = y_phys[0, :, 0].numpy()   # shape [n_ch, H, W]
        except AttributeError:
            field = np.asarray(y_phys[0, :, 0])

        n_ch, H, W = field.shape
        li = max(0, min(_lat_idx(lat), H - 1))
        lj = max(0, min(_lon_idx(lon), W - 1))

        result['_n_channels'] = n_ch
        for ch in range(n_ch):
            result[f'ch{ch}'] = float(field[ch, li, lj])

    except Exception:
        pass   # mslp still returned

    return result


def _load_iface_fields(ic_dir: Path, iface_idx: int, n_ifaces: int) -> list:
    """
    Load (config_name, lat, lon, fields_dict) for every pkl in an interface dir.
    Uses ThreadPoolExecutor for parallel I/O.
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
            lat, lon = float(loc[0]), float(loc[1])
        except (TypeError, IndexError):
            return None
        fields = _extract_fields(cfg)
        if not fields:
            return None
        return p.stem, lat, lon, fields

    results = []
    with ThreadPoolExecutor(max_workers=min(32, len(pkl_paths))) as ex:
        for r in ex.map(_load_one, pkl_paths):
            if r is not None:
                results.append(r)
    return results


# ── Genealogy helper ───────────────────────────────────────────────────────────

def _build_b_reachable_set(genealogy: dict, stateB_configs: list) -> set:
    """Backward BFS from state-B configs to find all ancestor configs."""
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


# ── Cache helpers ──────────────────────────────────────────────────────────────

_CACHE_NAME = 'committor_fields_cache.pkl'


def _cache_is_valid(cached: dict) -> bool:
    """
    Check that the cached records use the current ch{i} format.
    Old caches have named keys (q500, shear, …); new caches have ch0, ch1, …
    Returns False for stale caches so they are transparently recomputed.
    """
    for entries in cached.values():
        for entry in entries:
            fields = entry[3] if len(entry) > 3 else {}
            # New format must have _n_channels; old format never does
            return '_n_channels' in fields
    return True   # empty cache — let it through, will just produce no data


def _load_cache(ic_dir: Path):
    p = ic_dir / _CACHE_NAME
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            cached = pickle.load(f)
        if not _cache_is_valid(cached):
            return None   # stale format → recompute
        return cached
    except Exception:
        return None


def _save_cache(ic_dir: Path, result: dict):
    try:
        with open(ic_dir / _CACHE_NAME, 'wb') as f:
            pickle.dump(result, f, protocol=4)
    except Exception:
        pass


# ── Per-IC computation ─────────────────────────────────────────────────────────

def _compute_ic_records(ic_dir: Path, state_B: float,
                        n_ifaces: int, no_cache: bool) -> dict:
    """
    Return {iface_idx: [(lat, lon, reached_B, fields_dict), ...]} for one IC.
    Loads from cache on subsequent runs; saves cache after first computation.
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
        configs = _load_iface_fields(ic_dir, iface_idx, n_ifaces)
        if not configs:
            continue
        result[iface_idx] = [
            (lat, lon,
             (cname in b_reachable) or (cname in stateB_set),
             fields)
            for cname, lat, lon, fields in configs
        ]

    _save_cache(ic_dir, result)
    return result


# ── Module-level worker (picklable for ProcessPoolExecutor) ───────────────────

def _ic_fields_worker(args: tuple):
    """
    Compute committor field records for one IC, render its per-IC figure,
    and return the records dict for optional aggregation in main().
    """
    ic_dir_str, state_B, n_ifaces, ifaces, plot_dir_str, \
        min_samples, n_bins, no_cache, use_curated = args
    ic_dir   = Path(ic_dir_str)
    plot_dir = Path(plot_dir_str)

    records = _compute_ic_records(ic_dir, state_B, n_ifaces, no_cache)
    if not records:
        return {}

    try:
        variables = None
        if use_curated:
            all_vars  = _build_variable_list(_get_n_channels(records))
            variables = _get_curated_variables(all_vars)
        _plot_ic(records, ifaces, ic_dir.name, plot_dir, min_samples, n_bins,
                 variables=variables)
    except Exception as e:
        print(f'  WARNING: plot failed for {ic_dir.name}: {e}')

    return records


# ── Committor curve computation ────────────────────────────────────────────────

def _committor_curve(vals_b: list, vals_nb: list, n_bins: int = 15):
    """
    Bin values and compute p_B per bin.

    Returns
    -------
    centers  : 1D array  bin centres
    p_B      : 1D array  committor estimate (nan where n < 1)
    n_all    : 1D array  total configs per bin
    edges    : 1D array  bin edges (length n_bins+1)
    """
    all_vals = [v for v in vals_b + vals_nb if np.isfinite(v)]
    if len(all_vals) < 4:
        empty = np.array([])
        return empty, empty, empty, empty

    lo = np.percentile(all_vals, 2)
    hi = np.percentile(all_vals, 98)
    if lo >= hi:
        empty = np.array([])
        return empty, empty, empty, empty

    edges   = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_b     = np.zeros(n_bins)
    n_all   = np.zeros(n_bins)

    def _bin_idx(v):
        if not np.isfinite(v):
            return -1
        idx = int((v - lo) / (hi - lo) * n_bins)
        return max(0, min(idx, n_bins - 1))

    for v in vals_b:
        i = _bin_idx(v)
        if i >= 0:
            n_b[i]   += 1
            n_all[i] += 1
    for v in vals_nb:
        i = _bin_idx(v)
        if i >= 0:
            n_all[i] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        p_B = np.where(n_all > 0, n_b / n_all, np.nan)

    return centers, p_B, n_all, edges


# ── Channel separability ranking ──────────────────────────────────────────────

def _channel_tvd(vals_b: list, vals_nb: list, n_bins: int = 25) -> float:
    """
    Total Variation Distance between the reach-B and fail distributions.

    TVD ∈ [0, 1]:  0 = identical distributions,  1 = no overlap.
    Returns 0.0 when there are too few samples to estimate reliably.
    """
    all_vals = [v for v in vals_b + vals_nb if np.isfinite(v)]
    if len(all_vals) < 10 or not vals_b or not vals_nb:
        return 0.0
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    if lo >= hi:
        return 0.0
    edges   = np.linspace(lo, hi, n_bins + 1)
    h_b,  _ = np.histogram(vals_b,  bins=edges, density=True)
    h_nb, _ = np.histogram(vals_nb, bins=edges, density=True)
    bw = edges[1] - edges[0]
    return 0.5 * float(np.sum(np.abs(h_b * bw - h_nb * bw)))


def _rank_channels(agg_records: dict, variables: list,
                   min_samples: int = 5) -> tuple:
    """
    Compute per-interface TVD for each variable, then rank by maximum.

    Returns
    -------
    scored      : list of (max_tvd, best_iface_idx, variable_dict) sorted descending
    tvd_matrix  : ndarray shape (n_vars, n_non_B_ifaces) — TVD for every cell
    iface_idxs  : list of int — interface indices used (State-B excluded)

    The State-B column (last interface) is excluded — p_B is trivially 1 there.
    """
    iface_idxs = sorted(agg_records.keys())
    if iface_idxs:
        iface_idxs = iface_idxs[:-1]   # drop State-B column

    n_vars     = len(variables)
    n_ifaces   = len(iface_idxs)
    tvd_matrix = np.full((n_vars, n_ifaces), np.nan)

    for vi, vinfo in enumerate(variables):
        key = vinfo['key']
        for ji, ii in enumerate(iface_idxs):
            recs    = agg_records.get(ii, [])
            vals_b  = [r[3][key] for r in recs
                       if r[2] and key in r[3] and np.isfinite(r[3][key])]
            vals_nb = [r[3][key] for r in recs
                       if not r[2] and key in r[3] and np.isfinite(r[3][key])]
            if len(vals_b) >= min_samples and len(vals_nb) >= min_samples:
                tvd_matrix[vi, ji] = _channel_tvd(vals_b, vals_nb)

    # Rank by max TVD across interfaces
    scored = []
    for vi, vinfo in enumerate(variables):
        row    = tvd_matrix[vi]
        finite = np.isfinite(row)
        if not finite.any():
            scored.append((0.0, -1, vinfo))
        else:
            best_ji  = int(np.nanargmax(row))
            best_tvd = float(row[best_ji])
            best_ii  = iface_idxs[best_ji]
            scored.append((best_tvd, best_ii, vinfo))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored, tvd_matrix, iface_idxs


def _rank_channels_late(tvd_matrix: np.ndarray, iface_idxs: list,
                        variables: list, late_col: int = None) -> list:
    """
    Rank variables by *late TVD gain*: max(TVD[late_col:]) − max(TVD[:late_col]).

    A positive gain means the variable becomes MORE discriminating as the storm
    develops.  Ranking by absolute late TVD fails because early-dominant variables
    (T850, t2m, …) also have the highest absolute TVD late — they just drop less.
    Gain isolates variables that genuinely *emerge* late.

    Parameters
    ----------
    late_col : int, optional
        Column index in tvd_matrix to split early vs late.
        Defaults to n_ifaces // 2.

    Returns list of (late_gain, best_late_iface_idx, variable_dict) sorted descending.
    """
    n_ifaces = tvd_matrix.shape[1]
    if late_col is None:
        late_col = max(1, n_ifaces // 2)

    scored = []
    for vi, vinfo in enumerate(variables):
        row   = tvd_matrix[vi]
        early = row[:late_col]
        late  = row[late_col:]

        max_early = float(np.nanmax(early)) if np.any(np.isfinite(early)) else 0.0

        finite_late = np.isfinite(late)
        if not finite_late.any():
            scored.append((0.0, -1, vinfo))
            continue

        best_ji_rel   = int(np.nanargmax(late))
        best_tvd_late = float(late[best_ji_rel])
        best_ii       = iface_idxs[late_col + best_ji_rel]
        gain          = best_tvd_late - max_early
        scored.append((gain, best_ii, vinfo))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Plotting ───────────────────────────────────────────────────────────────────

# Interface colour ramp: blue (λ₀) → green (middle) → red (state B)
_IFACE_COLORS_CACHE = {}


def _iface_color(idx: int, n: int):
    key = (idx, n)
    if key not in _IFACE_COLORS_CACHE:
        cmap = plt.cm.RdYlBu_r
        _IFACE_COLORS_CACHE[key] = cmap(idx / max(1, n - 1))
    return _IFACE_COLORS_CACHE[key]


def _plot_ic(records: dict, interfaces: list, ic_name: str,
             plot_dir: Path, min_samples: int, n_bins: int,
             variables: list = None):
    """
    One figure per IC.
      rows = variables (full channel list, or curated/caller-supplied list)
      cols = FFS interfaces

    Each panel: green/red histograms only (per-IC samples too sparse for p_B).

    Parameters
    ----------
    variables : list, optional
        Pre-built / pre-filtered variable list.  If None, builds from channel count.
    """
    n_ifaces  = len(interfaces)
    if variables is None:
        variables = _build_variable_list(_get_n_channels(records))
    n_vars    = len(variables)

    # Scale panel height: smaller when there are many variables
    ph = max(1.8, min(3.8, 200.0 / max(n_vars, 1)))
    fig, axes = plt.subplots(
        n_vars, n_ifaces,
        figsize=(4.5 * n_ifaces, ph * n_vars),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    for vi, vinfo in enumerate(variables):
        key = vinfo['key']

        for ii, pressure in enumerate(interfaces):
            ax   = axes[vi][ii]
            recs = records.get(ii, [])

            vals_b  = [r[3][key] for r in recs
                       if r[2] and key in r[3] and np.isfinite(r[3][key])]
            vals_nb = [r[3][key] for r in recs
                       if not r[2] and key in r[3] and np.isfinite(r[3][key])]

            n_b_total  = len(vals_b)
            n_nb_total = len(vals_nb)
            n_total    = n_b_total + n_nb_total

            if n_total < 2 * min_samples:
                ax.text(0.5, 0.5,
                        f'insufficient data\n({n_total} configs)',
                        transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
            else:
                all_finite = [v for v in vals_b + vals_nb if np.isfinite(v)]
                lo  = np.percentile(all_finite, 1)
                hi  = np.percentile(all_finite, 99)
                bins = np.linspace(lo, hi, 25)

                if vals_b:
                    ax.hist(vals_b,  bins=bins, density=True, alpha=0.50,
                            color='#2ca02c', label=f'reach B  n={n_b_total}')
                if vals_nb:
                    ax.hist(vals_nb, bins=bins, density=True, alpha=0.50,
                            color='#d62728', label=f'fail      n={n_nb_total}')

                ax.legend(fontsize=6, loc='upper left',
                          framealpha=0.6, handlelength=1)

            if vi == 0:
                if ii == 0:
                    ttl = 'λ₀  (flux seed)'
                elif ii == n_ifaces - 1:
                    ttl = 'State B'
                else:
                    ttl = f'λ{ii}'
                ax.set_title(f'{ttl}\n{pressure} hPa',
                             fontsize=9, fontweight='bold')

            if ii == 0:
                ax.set_ylabel(vinfo['label'], fontsize=8)

            ax.tick_params(labelsize=7)
            ax.set_xlabel(vinfo['unit'], fontsize=7)

    fig.suptitle(
        f'Atmospheric-variable distributions  —  IC {ic_name}\n'
        'Green = reaches genesis  |  Red = fails',
        fontsize=11, fontweight='bold', y=1.005,
    )
    plt.tight_layout()

    out = plot_dir / f'committor_fields_{ic_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved  {out}')


def _plot_aggregated(agg_records: dict, interfaces: list,
                     plot_dir: Path, min_samples: int, n_bins: int,
                     variables: list = None, suffix: str = ''):
    """
    One aggregated figure across ALL ICs.
      rows = variables (full channel list or caller-supplied filtered list)
      cols = FFS interfaces

    Each panel: histogram (green=reach B, red=fail) PLUS p_B curve on right
    axis — valid here because the sample sizes are large enough.

    Parameters
    ----------
    variables : list, optional
        Pre-built / pre-filtered variable list.  If None, builds from channel count.
    suffix : str, optional
        Appended to output filename before '.png' (e.g. '_top20').
    """
    n_ifaces  = len(interfaces)
    if variables is None:
        variables = _build_variable_list(_get_n_channels(agg_records))
    n_vars    = len(variables)

    ph = max(1.8, min(3.8, 200.0 / max(n_vars, 1)))
    fig, axes = plt.subplots(
        n_vars, n_ifaces,
        figsize=(4.5 * n_ifaces, ph * n_vars),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    for vi, vinfo in enumerate(variables):
        key   = vinfo['key']
        color = vinfo['color']

        for ii, pressure in enumerate(interfaces):
            ax   = axes[vi][ii]
            recs = agg_records.get(ii, [])

            vals_b  = [r[3][key] for r in recs
                       if r[2] and key in r[3] and np.isfinite(r[3][key])]
            vals_nb = [r[3][key] for r in recs
                       if not r[2] and key in r[3] and np.isfinite(r[3][key])]

            n_b_total  = len(vals_b)
            n_nb_total = len(vals_nb)
            n_total    = n_b_total + n_nb_total

            if n_total < 2 * min_samples:
                ax.text(0.5, 0.5,
                        f'insufficient data\n({n_total} configs)',
                        transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
            else:
                all_finite = [v for v in vals_b + vals_nb if np.isfinite(v)]
                lo   = np.percentile(all_finite, 1)
                hi   = np.percentile(all_finite, 99)
                bins = np.linspace(lo, hi, 25)

                if vals_b:
                    ax.hist(vals_b,  bins=bins, density=True, alpha=0.50,
                            color='#2ca02c', label=f'reach B  n={n_b_total}')
                if vals_nb:
                    ax.hist(vals_nb, bins=bins, density=True, alpha=0.50,
                            color='#d62728', label=f'fail      n={n_nb_total}')

                # p_B curve on right axis (only shown for aggregated figure)
                centers, p_B, n_all, _ = _committor_curve(
                    vals_b, vals_nb, n_bins=n_bins
                )
                if len(centers) > 0:
                    mask = n_all >= min_samples
                    if mask.any():
                        ax2 = ax.twinx()
                        ax2.plot(centers[mask], p_B[mask],
                                 color=color, linewidth=2.0,
                                 marker='o', markersize=4, zorder=5,
                                 label='p_B')
                        ax2.axhline(np.nanmean(p_B[mask]),
                                    color=color, linewidth=1.0,
                                    linestyle='--', alpha=0.6)
                        ax2.set_ylim(-0.05, 1.15)
                        ax2.set_ylabel('p_B', fontsize=7, color=color)
                        ax2.tick_params(labelsize=6, colors=color)

                ax.legend(fontsize=6, loc='upper left',
                          framealpha=0.6, handlelength=1)

            if vi == 0:
                if ii == 0:
                    ttl = 'λ₀  (flux seed)'
                elif ii == n_ifaces - 1:
                    ttl = 'State B'
                else:
                    ttl = f'λ{ii}'
                ax.set_title(f'{ttl}\n{pressure} hPa',
                             fontsize=9, fontweight='bold')

            if ii == 0:
                ax.set_ylabel(vinfo['label'], fontsize=8)

            ax.tick_params(labelsize=7)
            ax.set_xlabel(vinfo['unit'], fontsize=7)

    fig.suptitle(
        'Atmospheric-variable committor p_B(ξ | λᵢ)  —  All ICs aggregated\n'
        'Green = reaches genesis  |  Red = fails  |  Coloured line = p_B',
        fontsize=11, fontweight='bold', y=1.005,
    )
    plt.tight_layout()

    out = plot_dir / f'committor_fields_aggregated{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved  {out}')


def _plot_tvd_heatmap(tvd_matrix: np.ndarray, all_variables: list,
                      iface_idxs: list, plot_variables: list,
                      interfaces: list, plot_dir: Path,
                      suffix: str = '', title: str = None):
    """
    Compact heatmap — rows = variables, cols = FFS interfaces (λ₀ … λN).

    Each cell is coloured and annotated with the TVD value so the reader can
    immediately see *which* variables discriminate and *when* (at which
    interface) they are most informative.

    Parameters
    ----------
    tvd_matrix    : shape (len(all_variables), len(iface_idxs))
    all_variables : full variable list whose row order matches tvd_matrix
    iface_idxs    : list of interface indices (columns of tvd_matrix)
    plot_variables: subset of variables to display as rows (curated or top-K)
    interfaces    : full interface list (for pressure labels)
    """
    # Locate each plot variable's row in the full matrix
    key_to_idx = {v['key']: i for i, v in enumerate(all_variables)}
    plot_rows  = [key_to_idx[v['key']] for v in plot_variables
                  if v['key'] in key_to_idx]
    plot_vars  = [plot_variables[k] for k in range(len(plot_variables))
                  if plot_variables[k]['key'] in key_to_idx]
    if not plot_rows:
        return

    sub_matrix = tvd_matrix[np.array(plot_rows, dtype=int), :]
    n_rows, n_cols = sub_matrix.shape

    row_labels = [v['label'] for v in plot_vars]
    col_labels = []
    for ii in iface_idxs:
        p = interfaces[ii] if ii < len(interfaces) else '?'
        lbl = f'λ₀\n({p} hPa)' if ii == 0 else f'λ{ii}\n({p} hPa)'
        col_labels.append(lbl)

    fig_w = max(5.0, 1.5 * n_cols + 1.5)
    fig_h = max(3.5, 0.48 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')

    vmax = float(np.nanmax(sub_matrix)) if np.any(np.isfinite(sub_matrix)) else 0.5
    vmax = max(vmax, 0.05)
    im = ax.imshow(sub_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0.0, vmax=vmax, interpolation='nearest')

    for ri in range(n_rows):
        for ci in range(n_cols):
            val = sub_matrix[ri, ci]
            if np.isfinite(val):
                txt_color = 'white' if val > 0.60 * vmax else 'black'
                ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=txt_color, fontweight='bold')
            else:
                ax.text(ci, ri, '—', ha='center', va='center',
                        fontsize=8, color='#aaaaaa')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('FFS Interface', fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.70, pad=0.03)
    cbar.set_label('TVD  (0 = no separation,  1 = perfect)', fontsize=8)

    if title is None:
        title = ('Variable Discriminability (TVD) by FFS Interface\n'
                 'Higher = stronger separation between reach-B and fail trajectories')
    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.tight_layout()

    out = plot_dir / f'committor_fields_tvd_heatmap{suffix}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved  {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Atmospheric-variable committor p_B(ξ|λᵢ) curves from FFS pkl files'
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
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Min configs per bin to plot p_B curve (default: 5)')
    parser.add_argument('--n_bins',      type=int, default=15,
                        help='Number of bins for p_B curves (default: 15)')
    parser.add_argument('--no_cache',    action='store_true',
                        help='Ignore per-IC cache and recompute from pkls')
    parser.add_argument('--aggregate',      action='store_true',
                        help='Also produce one aggregated figure (all ICs combined) '
                             'with p_B curves — saved as committor_fields_aggregated.png')
    parser.add_argument('--top_k_channels', type=int, default=0,
                        help='If > 0, also save a second aggregated figure showing only '
                             'the top-K channels ranked by TVD separability. '
                             'Implies --aggregate. Ranking is always printed to stdout.')
    parser.add_argument('--curated',        action='store_true',
                        help='Use the hardcoded curated 9-variable list (derived from '
                             'TVD ranking) for both per-IC and aggregated figures. '
                             'Produces clean publication-ready output.')
    args = parser.parse_args()

    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)

    ifaces  = ffs_config['interfaces'].copy()
    state_B = float(ffs_config['state_B'])
    if ifaces[-1] != state_B:
        ifaces.append(state_B)
    n_ifaces = len(ifaces)
    print(f'Interfaces: {ifaces}  state_B={state_B}')

    plot_dir = Path(args.plot_dir) / 'committor_fields'
    plot_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.ffs_csv)
    out_dir = Path(args.output_dir)
    ic_dirs = [out_dir / tl for tl in df['time_label'].tolist()
               if (out_dir / tl).exists()]
    print(f'Found {len(ic_dirs)} IC directories')

    worker_args = [
        (str(d), state_B, n_ifaces, ifaces, str(plot_dir),
         args.min_samples, args.n_bins, args.no_cache, args.curated)
        for d in ic_dirs
    ]
    n_workers = min(args.workers, max(1, len(ic_dirs)))

    all_ic_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_ic_fields_worker, a): Path(a[0])
                   for a in worker_args}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='ICs', unit='IC', dynamic_ncols=True):
            ic_dir_path = futures[fut]
            try:
                records = fut.result()
                if records:
                    all_ic_records.append(records)
            except Exception as e:
                print(f'  ERROR {ic_dir_path.name}: {e}')

    # ── Aggregated figure + channel ranking (optional) ────────────────────────
    do_aggregate = args.aggregate or (args.top_k_channels > 0)
    if do_aggregate:
        if not all_ic_records:
            print('\n  WARNING: no records collected — skipping aggregated figure.')
        else:
            agg = defaultdict(list)
            for recs in all_ic_records:
                for iface_idx, entries in recs.items():
                    agg[iface_idx].extend(entries)
            agg_dict        = dict(agg)
            n_total_configs = sum(len(v) for v in agg_dict.values())
            print(f'\nAggregating {len(all_ic_records)} ICs '
                  f'({n_total_configs} config records) …')

            all_variables = _build_variable_list(_get_n_channels(agg_dict))

            # ── Channel ranking + full TVD matrix ────────────────────────────
            ranked, tvd_matrix, tvd_iface_idxs = _rank_channels(
                agg_dict, all_variables, args.min_samples)
            print('\n── Channel separability ranking (TVD, excluding State-B) ──')
            print(f'{"Rank":>4}  {"TVD":>6}  {"Best λ":>6}  Label')
            print('─' * 50)
            for rank, (tvd, best_ii, vinfo) in enumerate(ranked, 1):
                best_lbl = f'λ{best_ii}' if best_ii >= 0 else '—'
                print(f'{rank:>4}  {tvd:>6.3f}  {best_lbl:>6}  {vinfo["label"]}')

            # ── Save ranking CSV (includes per-interface TVD columns) ─────────
            csv_path   = plot_dir / 'committor_fields_ranking.csv'
            key_to_row = {v['key']: i for i, v in enumerate(all_variables)}
            with open(csv_path, 'w') as fcsv:
                iface_hdrs = ','.join(f'tvd_lambda{ii}' for ii in tvd_iface_idxs)
                fcsv.write(f'rank,tvd_max,best_lambda,channel_key,label,{iface_hdrs}\n')
                for rank, (tvd, best_ii, vinfo) in enumerate(ranked, 1):
                    best_lbl = f'lambda{best_ii}' if best_ii >= 0 else 'none'
                    vi = key_to_row.get(vinfo['key'], -1)
                    if vi >= 0:
                        per_iface = ','.join(
                            f'{tvd_matrix[vi, ji]:.6f}'
                            if np.isfinite(tvd_matrix[vi, ji]) else 'nan'
                            for ji in range(len(tvd_iface_idxs))
                        )
                    else:
                        per_iface = ','.join('nan' for _ in tvd_iface_idxs)
                    fcsv.write(f'{rank},{tvd:.6f},{best_lbl},'
                               f'{vinfo["key"]},{vinfo["label"]},{per_iface}\n')
            print(f'Saved  {csv_path}')

            # ── Choose variable list for aggregated figures ───────────────────
            agg_vars = (_get_curated_variables(all_variables)
                        if args.curated else all_variables)

            # ── Full / curated aggregated figure ──────────────────────────────
            if args.aggregate or args.curated:
                suffix = '_curated' if args.curated else ''
                _plot_aggregated(agg_dict, ifaces, plot_dir,
                                 args.min_samples, args.n_bins,
                                 variables=agg_vars, suffix=suffix)

            # ── Top-K filtered figure ─────────────────────────────────────────
            if args.top_k_channels > 0:
                k          = args.top_k_channels
                top_vars   = [vinfo for _, _, vinfo in ranked[:k]]
                suffix     = f'_top{k}'
                print(f'\nPlotting top-{k} channels → '
                      f'committor_fields_aggregated{suffix}.png')
                _plot_aggregated(agg_dict, ifaces, plot_dir,
                                 args.min_samples, args.n_bins,
                                 variables=top_vars, suffix=suffix)

            # ── TVD heatmap (early discriminators) ───────────────────────────
            # Use curated set if requested, otherwise top-20 by global max TVD.
            if args.curated:
                hm_vars   = _get_curated_variables(all_variables)
                hm_suffix = '_curated'
            else:
                k_hm      = min(20, len(ranked))
                hm_vars   = [vinfo for _, _, vinfo in ranked[:k_hm]]
                hm_suffix = f'_top{k_hm}'
            _plot_tvd_heatmap(
                tvd_matrix, all_variables, tvd_iface_idxs,
                hm_vars, ifaces, plot_dir, suffix=hm_suffix,
                title=('Variable Discriminability (TVD) by FFS Interface\n'
                       'Ranked by max TVD across all interfaces  '
                       '— early-stage discriminators dominate'),
            )

            # ── TVD heatmap (late-stage discriminators) ───────────────────────
            # Rank by max TVD restricted to the second half of interfaces (λN/2+).
            # Surfaces variables that matter *late* even if uninformative at λ0.
            late_col  = max(1, len(tvd_iface_idxs) // 2)
            late_lbl  = f'λ{tvd_iface_idxs[late_col]}'
            late_ranked = _rank_channels_late(
                tvd_matrix, tvd_iface_idxs, all_variables, late_col=late_col)
            k_late    = min(20, len(late_ranked))
            late_vars = [vinfo for _, _, vinfo in late_ranked[:k_late]]

            print(f'\n── Late-stage ranking (TVD gain: max({late_lbl}+) − max(early)) ──')
            print(f'{"Rank":>4}  {"Gain":>6}  {"Peak λ":>6}  Label')
            print('─' * 55)
            for rank, (gain, best_ii, vinfo) in enumerate(late_ranked[:k_late], 1):
                best_lbl_p = f'λ{best_ii}' if best_ii >= 0 else '—'
                sign = '+' if gain >= 0 else ''
                print(f'{rank:>4}  {sign}{gain:>5.3f}  {best_lbl_p:>6}  {vinfo["label"]}')

            _plot_tvd_heatmap(
                tvd_matrix, all_variables, tvd_iface_idxs,
                late_vars, ifaces, plot_dir,
                suffix=f'_late_top{k_late}',
                title=(f'Late-Emerging Variables  (gain = max TVD at {late_lbl}+ minus early peak)\n'
                       'Positive gain → variable becomes MORE discriminating as storm develops'),
            )

    print(f'\nDone. Figures saved to {plot_dir}/')


if __name__ == '__main__':
    main()
