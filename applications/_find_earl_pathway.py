"""Find pathway indices for Earl IC that trace back to the target seed 0501_UR (18.38N, -55W)."""
import json, pickle, sys
from pathlib import Path

RT_JSON = Path('/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-02T00Z/reactive_trajectories/reactive_trajectories.json')
FLUX_DIR = Path('/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-02T00Z/flux')

TARGET_SEED = 'lambda0_config_0501_UR'

# Also define a bounding box for seeds near Earl genesis
LAT_MIN, LAT_MAX = 15.0, 22.0
LON_MIN, LON_MAX = -65.0, -48.0

with open(RT_JSON) as f:
    data = json.load(f)

trajectories = data if isinstance(data, list) else data.get('trajectories', data.get('reactive_trajectories', []))
print(f"Total reactive trajectories: {len(trajectories)}")

# Load seed locations
seed_locs = {}
try:
    import numpy as np
    for pkl in FLUX_DIR.glob('lambda0_config_*.pkl'):
        try:
            with open(pkl, 'rb') as fh:
                obj = pickle.load(fh)
            loc = obj.feature_location
            seed_locs[pkl.stem] = (float(loc[0]), float(loc[1]))
        except Exception:
            pass
    print(f"Loaded {len(seed_locs)} seed locations")
except Exception as e:
    print(f"Warning: couldn't load seeds with numpy: {e}")

# Find trajectories whose first pathway element is the target seed (or in the zone)
hits_exact = []
hits_zone = []

for idx, traj in enumerate(trajectories):
    pathway = traj.get('pathway', [])
    if not pathway:
        continue
    first = pathway[0]
    # Normalize: strip 'lambda0_config_' prefix if needed
    if first == TARGET_SEED or first == TARGET_SEED.replace('lambda0_config_', ''):
        hits_exact.append((idx, traj.get('final_mslp'), traj.get('pathway_length'), first))
    elif first in seed_locs:
        lat, lon = seed_locs[first]
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            hits_zone.append((idx, traj.get('final_mslp'), traj.get('pathway_length'), first, lat, lon))

print(f"\n=== Exact match to {TARGET_SEED} ===")
for idx, mslp, plen, seed in hits_exact:
    print(f"  pathway_idx={idx:4d}  mslp={mslp}  len={plen}  seed={seed}")

print(f"\n=== Zone match [{LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}W] ===")
hits_zone.sort(key=lambda x: x[1] if x[1] else 9999)
for idx, mslp, plen, seed, lat, lon in hits_zone[:20]:
    print(f"  pathway_idx={idx:4d}  mslp={mslp:.1f}  len={plen}  seed={seed}  {lat:.2f}N {lon:.2f}W")

# Also show first pathway element structure for first few trajectories
print("\n=== First few trajectory pathway[0] values ===")
for idx, traj in enumerate(trajectories[:5]):
    pw = traj.get('pathway', [])
    print(f"  idx={idx}: pathway[0]={pw[0] if pw else 'EMPTY'}  len={len(pw)}")
