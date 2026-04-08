"""
Targeted fast scan for Fiona and Ian seeds.

Fiona genesis: 16.0N, 47.9W on 14 Sep 0600 UTC
  Scan ICs: 09-12T00Z, 09-13T00Z, 09-14T00Z
  Zone: 14-18N, 53-41W

Ian genesis: 13.7N, 68.1W on 23 Sep 0600 UTC
  Scan IC: 09-22T00Z  (already done but zone was wrong — missed 62-70W)
  Zone: 11-17N, 73-60W (broader)

Speedup: use ThreadPoolExecutor for pkl loading; only rglob JSONL once per IC.
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

def count_lambda4(seed, children):
    visited = set(); queue = collections.deque([seed]); n = 0
    while queue:
        node = queue.popleft()
        if node in visited: continue
        visited.add(node)
        for ch in children.get(node, set()):
            if ch.startswith('lambda4_'): n += 1
            if ch not in visited: queue.append(ch)
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

SCANS = [
    ('FIONA', '2022-09-12T00Z', 14, 18, -53, -41),
    ('FIONA', '2022-09-13T00Z', 14, 18, -53, -41),
    ('FIONA', '2022-09-14T00Z', 14, 18, -53, -41),
    ('IAN',   '2022-09-22T00Z', 11, 17, -73, -60),
    ('IAN',   '2022-09-21T12Z', 11, 17, -73, -60),
]

for storm, ic_label, lat0, lat1, lon0, lon1 in SCANS:
    ic_dir   = BASE / ic_label
    flux_dir = ic_dir / 'flux'
    log_dir  = ic_dir / 'logs'
    if not flux_dir.exists():
        print(f'  {ic_label}: missing, skip', flush=True)
        continue

    print(f'\nLoading logs for {ic_label}...', flush=True)
    children = load_children(log_dir)

    print(f'  Loading pkls with 8 threads...', flush=True)
    pkls = list(flux_dir.glob('lambda0_config_*.pkl'))
    with ThreadPoolExecutor(max_workers=8) as ex:
        locs = list(ex.map(load_pkl_loc, pkls))

    results = []
    for item in locs:
        if item is None: continue
        name, lat, lon = item
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
            n = count_lambda4(name, children)
            results.append((n, name, lat, lon))

    results.sort(reverse=True)
    print(f'\n=== {storm}  IC={ic_label}  zone=[{lat0}-{lat1}N, {lon0}-{lon1}W] ===', flush=True)
    if not results:
        print('  (no seeds found in zone)', flush=True)
    for n, name, lat, lon in results[:10]:
        print(f'  {name:<35} {lat:.2f}N {lon:.2f}W  λ4={n}', flush=True)
