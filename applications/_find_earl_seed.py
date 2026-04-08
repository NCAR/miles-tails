"""
Find the best Earl IC seed: location near 17-19N, 55-68W (east of Leeward Islands),
plain-float feature_location, max lambda4 descendants via full BFS.
Earl genesis: 17.9N, 58.6W at 2 Sep 1800 UTC.
FFS IC: 2022-09-02T00Z (18 hrs before genesis).
"""
import pickle, os, glob, json, collections
from pathlib import Path

class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _SafeObj

ic_dir   = Path('/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-02T00Z')
log_dir  = ic_dir / 'logs'
flux_dir = ic_dir / 'flux'

# ── load all successful shooting links ──────────────────────────────────────
print('Loading logs...', flush=True)
children = collections.defaultdict(set)
total = 0
for logfile in sorted(log_dir.rglob('*.jsonl')):
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('phase') != 'shooting': continue
            parent = d.get('parent_config', '')
            child  = d.get('child_config', '')
            if parent and child and d.get('status') == 'success':
                children[parent].add(child)
                total += 1

print(f'Loaded {total} successful links, {len(children)} unique parents', flush=True)

def count_lambda4(seed):
    visited = set()
    queue = collections.deque([seed])
    n = 0
    while queue:
        node = queue.popleft()
        if node in visited: continue
        visited.add(node)
        for child in children.get(node, set()):
            if child.startswith('lambda4_'):
                n += 1
            if child not in visited:
                queue.append(child)
    return n

# ── scan ALL lambda0 seeds (wider area to see what's there) ─────────────────
print('Loading lambda0 pkl locations...', flush=True)
results = []
pkl_files = sorted(flux_dir.glob('lambda0_config_*.pkl'))
print(f'  Found {len(pkl_files)} lambda0 pkl files', flush=True)

for p in pkl_files:
    name = p.stem
    try:
        with open(p, 'rb') as fh:
            obj = _SafeUnpickler(fh).load()
        loc = obj.feature_location
        lat, lon = float(loc[0]), float(loc[1])
        is_plain = isinstance(loc[0], (int, float)) and isinstance(loc[1], (int, float))
    except Exception:
        continue
    # Show all Atlantic/Caribbean seeds 10-30N, 90W-40W
    if 10 <= lat <= 30 and -90 <= lon <= -40:
        n = count_lambda4(name)
        results.append((n, is_plain, name, lat, lon))

results.sort(reverse=True)

print(f'\nAll Atlantic/Caribbean lambda0 seeds (10-30N, 90-40W), sorted by lambda4 descendants:')
print(f'{"name":<35} {"lat":>6} {"lon":>8} {"λ4_desc":>8} {"plain":>6}')
print('-' * 70)
for n, is_plain, name, lat, lon in results:
    marker = ' <-- EARL ZONE' if (16 <= lat <= 20 and -68 <= lon <= -54) else ''
    print(f'{name:<35} {lat:6.2f}N {lon:8.2f}W {n:8d} {str(is_plain):>6}{marker}')
