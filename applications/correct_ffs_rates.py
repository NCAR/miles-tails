#!/usr/bin/env python
"""
correct_ffs_rates.py — Post-processing correction for FFS rate estimates.

Two corrections are applied to bring the FFS direct (BF) rate and the
FFS interface-product rate into closer agreement:

1. **Flux stateB basin filter**: stateB events during flux generation are
   filtered to those whose storm center (feature_location in the pkl) lies
   within the Atlantic basin.  Storms that tracked out of the basin and
   deepened to state_B pressure outside the tropics are spurious.

2. **Shooting outside_geographic_bounds exclusion**: trajectories rejected
   as outside_geographic_bounds are excluded from the P_forward denominator.
   These are storms that exited the domain before they could cross the next
   interface; in flux mode the equivalent storm would simply be lost (no
   lambda0 credit), so it is inconsistent to count them as P-forward failures.

Usage:
    python correct_ffs_rates.py ffs.yml \
        [--ffs_csv  results_mar6/ffs_statistics_all_ics.csv] \
        [--out_csv  results_mar6/ffs_statistics_corrected.csv]

Requirements: the results_mar6 directory with per-IC stateB/ pkls and
logs/<iface>/*.jsonl shooting logs must exist.
"""

import argparse
import json
import pickle
import glob
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# Fast pkl loader — stubs out torch tensors so we skip large arrays
# ---------------------------------------------------------------------------

class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if ('torch' in module or
                name in ('Tensor', 'storage', 'LongStorage',
                         '_rebuild_tensor_v2', 'FloatStorage')):
            return lambda *args, **kwargs: None
        return super().find_class(module, name)


def _load_feature_location(pkl_path: str):
    try:
        with open(pkl_path, 'rb') as f:
            obj = _StubUnpickler(f).load()
        return getattr(obj, 'feature_location', None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-IC corrections
# ---------------------------------------------------------------------------

def correct_ic(ic_dir: Path, basin: dict, n_interfaces: int):
    """
    Returns a dict of corrections for one IC:
      direct_B_in_basin    : stateB count after filtering to basin
      direct_B_out_of_basin: filtered-out count
      geo_excl[i]          : outside_geographic_bounds count at interface i
                             (1-indexed, matches lambdaN columns)
    """
    result = {
        'direct_B_in_basin': 0,
        'direct_B_out_of_basin': 0,
        'geo_excl': {i: 0 for i in range(1, n_interfaces + 1)},
    }

    # ---- 1. stateB basin filter ----------------------------------------
    stateB_dir = ic_dir / 'stateB'
    if stateB_dir.exists():
        pkls = list(stateB_dir.glob('stateB_*.pkl'))
        locs = [_load_feature_location(str(p)) for p in pkls]
        for loc in locs:
            if loc is None:
                result['direct_B_in_basin'] += 1  # can't verify → keep
                continue
            lat, lon = loc
            if (basin['lat_min'] <= lat <= basin['lat_max'] and
                    basin['lon_min'] <= lon <= basin['lon_max']):
                result['direct_B_in_basin'] += 1
            else:
                result['direct_B_out_of_basin'] += 1

    # ---- 2. shooting outside_geographic_bounds per interface -----------
    logs_base = ic_dir / 'logs'
    for iface_idx in range(1, n_interfaces + 1):
        iface_log_dir = logs_base / str(iface_idx)
        if not iface_log_dir.exists():
            continue
        count = 0
        for logf in iface_log_dir.glob('*.jsonl'):
            with open(logf) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (entry.get('phase') == 'shooting' and
                            entry.get('status') == 'failure' and
                            entry.get('failure_reason') == 'outside_geographic_bounds'):
                        count += 1
        result['geo_excl'][iface_idx] = count

    return result


def _correct_ic_worker(args):
    ic_dir, basin, n_interfaces = args
    return correct_ic(ic_dir, basin, n_interfaces)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_file', help='Path to ffs.yml')
    parser.add_argument('--ffs_csv',  default=None,
                        help='Input CSV (default: <output_dir>/ffs_statistics_all_ics.csv)')
    parser.add_argument('--out_csv',  default=None,
                        help='Output corrected CSV (default: <output_dir>/ffs_statistics_corrected.csv)')
    parser.add_argument('--workers',  type=int, default=8,
                        help='Parallel workers for stateB pkl loading (default: 8)')
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg['output_dir'])
    basin = cfg.get('basin', {'lat_min': 10.0, 'lat_max': 45.0,
                               'lon_min': -98.0, 'lon_max': -20.0})
    interfaces = cfg['interfaces'].copy()
    state_B = cfg['state_B']
    if interfaces[-1] != state_B:
        interfaces.append(state_B)
    n_interfaces = len(interfaces) - 1   # number of shooting interfaces

    ffs_csv = Path(args.ffs_csv) if args.ffs_csv else output_dir / 'ffs_statistics_all_ics.csv'
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / 'ffs_statistics_corrected.csv'

    df = pd.read_csv(ffs_csv)
    print(f"Loaded {len(df)} ICs from {ffs_csv}")

    # Build list of IC dirs in CSV order
    ic_dirs = []
    for _, row in df.iterrows():
        tl = row['time_label']
        ic_dirs.append(output_dir / tl)

    # Process ICs in parallel (stateB pkl loading is the bottleneck)
    print(f"Computing corrections with {args.workers} workers …")
    worker_args = [(ic_dir, basin, n_interfaces) for ic_dir in ic_dirs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        corrections = list(ex.map(_correct_ic_worker, worker_args))

    # Apply corrections to dataframe
    corrected_rows = []
    for idx, (row, corr) in enumerate(zip(df.itertuples(index=False), corrections)):
        d = row._asdict()

        # --- direct rate correction ---
        total_B_orig = d.get('flux_direct_B_formations', 0)
        in_basin     = corr['direct_B_in_basin']
        out_basin    = corr['direct_B_out_of_basin']
        total_time   = d.get('flux_total_time_days', 0)

        d['flux_direct_B_formations_orig']    = total_B_orig
        d['flux_direct_B_out_of_basin']       = out_basin
        d['flux_direct_B_formations']         = in_basin
        d['direct_formation_rate_per_day']    = (in_basin / total_time
                                                  if total_time > 0 else 0.0)

        # --- P_forward corrections ---
        flux_rate = d.get('flux_rate_per_day', 0.0)
        corrected_probs = []
        for i in range(1, n_interfaces + 1):
            col_s   = f'lambda{i}_successes'
            col_f   = f'lambda{i}_failures'
            col_et  = f'lambda{i}_extratropical'
            col_at  = f'lambda{i}_attempts'
            col_pfw = f'lambda{i}_P_forward'

            if col_s not in d:
                break

            successes   = d.get(col_s, 0) or 0
            failures    = d.get(col_f, 0) or 0
            extratrop   = d.get(col_et, 0) or 0
            geo_excl    = corr['geo_excl'].get(i, 0)

            # original denominator includes geo-excluded; remove them
            denom_orig = successes + failures + extratrop
            denom_corr = denom_orig - geo_excl
            p_corr = successes / denom_corr if denom_corr > 0 else 0.0

            d[f'lambda{i}_geo_excluded']      = geo_excl
            d[f'lambda{i}_P_forward_orig']    = d.get(col_pfw, 0)
            d[f'lambda{i}_P_forward']         = p_corr
            if col_at in d:
                d[col_at] = denom_corr
            corrected_probs.append(p_corr)

        # recompute FFS rate
        if corrected_probs and flux_rate > 0:
            d['ffs_rate_per_day_orig'] = d.get('ffs_rate_per_day', 0)
            d['ffs_rate_per_day']      = flux_rate * float(np.prod(corrected_probs))
            if d['direct_formation_rate_per_day'] > 0:
                d['ffs_to_direct_ratio'] = (d['ffs_rate_per_day'] /
                                             d['direct_formation_rate_per_day'])

        corrected_rows.append(d)

    out_df = pd.DataFrame(corrected_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Corrected CSV written to {out_csv}")

    # Summary statistics
    if 'ffs_to_direct_ratio' in out_df.columns:
        ratios = out_df['ffs_to_direct_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"\nFFS / direct rate ratio after correction:")
        print(f"  median = {ratios.median():.3f}   mean = {ratios.mean():.3f}")
        print(f"  ICs within 2x: {(ratios.between(0.5, 2.0)).sum()} / {len(ratios)}")
        print(f"  ICs within 3x: {(ratios.between(1/3, 3.0)).sum()} / {len(ratios)}")

    total_geo = sum(sum(c['geo_excl'].values()) for c in corrections)
    total_out  = sum(c['direct_B_out_of_basin'] for c in corrections)
    total_b    = sum(c['direct_B_in_basin'] + c['direct_B_out_of_basin'] for c in corrections)
    print(f"\nTotal stateB events: {total_b}  out-of-basin removed: {total_out}"
          f"  ({100*total_out/max(total_b,1):.1f}%)")
    print(f"Total geo-excluded shooting trajectories: {total_geo:,}")


if __name__ == '__main__':
    main()
