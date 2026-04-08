"""Scan 09-23 and 09-24 ICs for Ian seeds in Gulf/Caribbean with spread."""
import pickle, json, collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name): return _SafeObj

BASE = Path('/glade/derecho/scratch/schreck/FFS/results_mar18')

def load_children(log_dir):
    ch = collections.defaultdict(set)
    for logfile in sorted(log_dir.rglob('*.jsonl')):
        try:
            with open(logfile) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    d = json.loads(line)
                    if d.get('phase') != 'shooting': continue
                    p = d.get('parent_config',''); c = d.get('child_config','')
                    if p and c and d.get('status') == 'success':
                        ch[p].add(c)
        except Exception: pass
    return ch

def count_lambda4(seed, ch):
    visited = set(); queue = collections.deque([seed]); n = 0
    while queue:
        node = queue.popleft()
        if node in visited: continue
        visited.add(node)
        for c in ch.get(node, set()):
            if c.startswith('lambda4_'): n += 1
            if c not in visited: queue.append(c)
    return n

def load_pkl_loc(p):
    try:
        with open(p, 'rb') as fh:
            obj = _SafeUnpickler(fh).load()
        loc = obj.feature_location
        lat, lon = float(loc[0]), float(loc[1])
        if not (isinstance(loc[0], (int,float)) and isinstance(loc[1], (int,float))): return None
        return p.stem, lat, lon
    except Exception: return None

for ic_label in ['2022-09-23T00Z', '2022-09-24T00Z', '2022-09-23T12Z']:
    ic_dir = BASE / ic_label
    if not (ic_dir / 'flux').exists():
        print(f'{ic_label}: missing', flush=True); continue
    print(f'\nLoading {ic_label}...', flush=True)
    ch = load_children(ic_dir / 'logs')
    pkls = list((ic_dir / 'flux').glob('lambda0_config_*.pkl'))
    with ThreadPoolExecutor(max_workers=8) as ex:
        locs = list(ex.map(load_pkl_loc, pkls))
    results = []
    for item in locs:
        if item is None: continue
        name, lat, lon = item
        if 12 <= lat <= 28 and -98 <= lon <= -70:
            n = count_lambda4(name, ch)
            if n >= 5:
                results.append((n, name, lat, lon))
    results.sort(reverse=True)
    print(f'=== IC={ic_label}  zone=[12-28N, 98-70W]  λ4≥5 ===', flush=True)
    for n, name, lat, lon in results[:12]:
        print(f'  {name:<35} {lat:.2f}N {lon:.2f}W  λ4={n}', flush=True)
    if not results: print('  (none)', flush=True)
