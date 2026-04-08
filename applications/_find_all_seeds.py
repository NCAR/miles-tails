"""
Find best seeds for Earl, Fiona, Ian across relevant IC dates.

Earl genesis:  17.9N,  58.6W  on 02 Sep 1800 UTC
  → at IC 09-02T00Z precursor ~17-19N, 54-62W
  → at IC 09-01T12Z precursor ~17-19N, 57-64W

Fiona genesis: 16.0N,  47.9W  on 14 Sep 0600 UTC
  → at IC 09-12T00Z precursor ~15-17N, 43-50W  (2 days before)
  → at IC 09-13T00Z precursor ~15-17N, 46-51W  (1 day before)
  → at IC 09-14T00Z precursor ~15-17N, 47-52W  (genesis day)

Ian genesis:   13.7N,  68.1W  on 23 Sep 0600 UTC
  → at IC 09-22T00Z precursor ~13-15N, 64-71W  (30 hrs before)
  → at IC 09-21T12Z precursor ~13-15N, 62-68W  (42 hrs before)
"""
import pickle, json, collections
from pathlib import Path

class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name): return _SafeObj

BASE = Path('/glade/derecho/scratch/schreck/FFS/results_mar18')

TARGETS = [
    # (label, ic_label, lat_min, lat_max, lon_min, lon_max)
    ('EARL',  '2022-09-02T00Z', 16, 20, -63, -52),
    ('EARL',  '2022-09-01T12Z', 16, 20, -65, -55),
    ('FIONA', '2022-09-12T00Z', 14, 18, -53, -42),
    ('FIONA', '2022-09-13T00Z', 14, 18, -52, -43),
    ('FIONA', '2022-09-14T00Z', 14, 18, -52, -44),
    ('IAN',   '2022-09-22T00Z', 11, 17, -73, -62),
    ('IAN',   '2022-09-21T12Z', 11, 17, -71, -60),
    ('IAN',   '2022-09-21T00Z', 11, 17, -70, -59),
]

for storm, ic_label, lat0, lat1, lon0, lon1 in TARGETS:
    ic_dir   = BASE / ic_label
    log_dir  = ic_dir / 'logs'
    flux_dir = ic_dir / 'flux'
    if not flux_dir.exists():
        print(f'  {ic_label}: no flux dir, skipping')
        continue

    children = collections.defaultdict(set)
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
                        children[p].add(c)
        except Exception: pass

    def count_lambda4(seed):
        visited = set(); queue = collections.deque([seed]); n = 0
        while queue:
            node = queue.popleft()
            if node in visited: continue
            visited.add(node)
            for ch in children.get(node, set()):
                if ch.startswith('lambda4_'): n += 1
                if ch not in visited: queue.append(ch)
        return n

    results = []
    for p in sorted(flux_dir.glob('lambda0_config_*.pkl')):
        name = p.stem
        try:
            with open(p, 'rb') as fh:
                obj = _SafeUnpickler(fh).load()
            loc = obj.feature_location
            lat, lon = float(loc[0]), float(loc[1])
            if not (isinstance(loc[0], (int,float)) and isinstance(loc[1], (int,float))): continue
        except Exception: continue
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
            n = count_lambda4(name)
            results.append((n, name, lat, lon))

    results.sort(reverse=True)
    print(f'\n=== {storm}  IC={ic_label}  zone=[{lat0}-{lat1}N, {lon0}-{lon1}W] ===', flush=True)
    if not results:
        print('  (no seeds found in zone)', flush=True)
    for n, name, lat, lon in results[:8]:
        print(f'  {name:<35} {lat:.2f}N {lon:.2f}W  λ4={n}', flush=True)
