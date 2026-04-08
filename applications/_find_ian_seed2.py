"""
Find the best Ian IC seed: Caribbean location, plain-float feature_location,
max lambda4 descendants via full BFS through the FFS chain.
"""
import pickle, os, glob, json, collections, sys
from pathlib import Path

# ── safe unpickler ──────────────────────────────────────────────────────────
class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _SafeObj

# ── paths ───────────────────────────────────────────────────────────────────
ic_dir  = Path('/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-22T00Z')
log_dir = ic_dir / 'logs'
flux_dir = ic_dir / 'flux'

# ── load all JSONL shooting entries → parent: set(children) ─────────────────
print('Loading logs...', flush=True)
children = collections.defaultdict(set)   # parent_name -> set of child_names
total = 0
for logfile in sorted(log_dir.rglob('*.jsonl')):
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('phase') != 'shooting':
                continue
            parent = d.get('parent_config', '')
            child  = d.get('child_config', '')
            status = d.get('status', '')
            if parent and child and status == 'success':
                children[parent].add(child)
                total += 1

print(f'Loaded {total} successful shooting links, {len(children)} unique parents', flush=True)

# ── BFS: count lambda4 descendants of each lambda0 seed ─────────────────────
print('Counting lambda4 descendants per lambda0 seed...', flush=True)

lambda0_seeds = [k for k in children if k.startswith('lambda0_')]
print(f'  {len(lambda0_seeds)} lambda0 seeds have at least one child', flush=True)

def count_lambda4_descendants(seed):
    """BFS from seed, count how many unique lambda4_* nodes are reachable."""
    visited = set()
    queue = collections.deque([seed])
    lambda4_count = 0
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for child in children.get(node, set()):
            if child.startswith('lambda4_'):
                lambda4_count += 1
            if child not in visited:
                queue.append(child)
    return lambda4_count

# ── load lambda0 pkl files: get location, check if plain float ───────────────
print('Loading lambda0 pkl locations...', flush=True)

def load_seed_info(pkl_path):
    """Return (lat, lon, is_plain_float) or None."""
    try:
        with open(pkl_path, 'rb') as fh:
            obj = pickle.load(fh)
    except Exception:
        try:
            with open(pkl_path, 'rb') as fh:
                obj = _SafeUnpickler(fh).load()
        except Exception:
            return None
    try:
        loc = obj.feature_location
        lat, lon = loc[0], loc[1]
        # Check if these are plain floats/ints
        is_plain = isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        return float(lat), float(lon), is_plain
    except Exception:
        return None

results = []
pkl_files = sorted(flux_dir.glob('lambda0_config_*.pkl'))
print(f'  Found {len(pkl_files)} lambda0 pkl files', flush=True)

for p in pkl_files:
    name = p.stem  # e.g. lambda0_config_2000_QN
    info = load_seed_info(p)
    if info is None:
        continue
    lat, lon, is_plain = info
    # Caribbean filter: 10-25N, 100-70W
    if not (10 <= lat <= 25 and -100 <= lon <= -70):
        continue
    n_desc = count_lambda4_descendants(name)
    results.append((n_desc, is_plain, name, lat, lon))

results.sort(reverse=True)

print(f'\nTop Caribbean lambda0 seeds (sorted by lambda4 descendants):')
print(f'{"name":<35} {"lat":>6} {"lon":>8} {"λ4_desc":>8} {"plain_float":>12}')
print('-' * 75)
for n, is_plain, name, lat, lon in results[:20]:
    print(f'{name:<35} {lat:6.2f}N {lon:8.2f}W {n:8d} {str(is_plain):>12}')
