"""
Ian wave at earlier ICs when it was still a pre-depression (λ0 eligible).
Ian genesis: 13.7N, 68.1W on 23 Sep 0600 UTC.
Working backward at ~10-12 kts WNW:
  09-22T00Z: ~13N, 70W (but below 1000 hPa already in model — no seeds found)
  09-21T00Z: ~13N, 73W
  09-20T00Z: ~13N, 77W
  09-19T00Z: ~13N, 81W
  09-18T00Z: ~13N, 85W
  09-17T00Z: ~13N, 89W
Scan zone: 10-18N, 95-60W (wide — let model tell us where seeds are)
"""
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

for ic_label in ['2022-09-20T00Z', '2022-09-19T00Z', '2022-09-18T00Z', '2022-09-17T00Z']:
    ic_dir   = BASE / ic_label
    flux_dir = ic_dir / 'flux'
    log_dir  = ic_dir / 'logs'
    if not flux_dir.exists():
        print(f'{ic_label}: missing', flush=True); continue

    print(f'\nLoading {ic_label}...', flush=True)
    ch = load_children(log_dir)

    pkls = list(flux_dir.glob('lambda0_config_*.pkl'))
    with ThreadPoolExecutor(max_workers=8) as ex:
        locs = list(ex.map(load_pkl_loc, pkls))

    results = []
    for item in locs:
        if item is None: continue
        name, lat, lon = item
        if 10 <= lat <= 18 and -95 <= lon <= -60:
            n = count_lambda4(name, ch)
            results.append((n, name, lat, lon))
    results.sort(reverse=True)

    print(f'=== IAN  IC={ic_label}  zone=[10-18N, 95-60W] top 10 ===', flush=True)
    for n, name, lat, lon in results[:10]:
        print(f'  {name:<35} {lat:.2f}N {lon:.2f}W  λ4={n}', flush=True)
    if not results:
        print('  (none)', flush=True)
