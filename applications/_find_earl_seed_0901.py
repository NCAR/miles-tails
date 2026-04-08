"""Scan 2022-09-01T12Z IC for Earl zone seeds (17-20N, 54-70W)."""
import pickle, json, collections
from pathlib import Path

class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name): return _SafeObj

for ic_label in ['2022-09-01T00Z', '2022-09-01T12Z', '2022-09-02T12Z']:
    ic_dir   = Path(f'/glade/derecho/scratch/schreck/FFS/results_mar18/{ic_label}')
    log_dir  = ic_dir / 'logs'
    flux_dir = ic_dir / 'flux'

    children = collections.defaultdict(set)
    for logfile in sorted(log_dir.rglob('*.jsonl')):
        with open(logfile) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                try:
                    d = json.loads(line)
                except Exception: continue
                if d.get('phase') != 'shooting': continue
                p = d.get('parent_config',''); c = d.get('child_config','')
                if p and c and d.get('status') == 'success':
                    children[p].add(c)

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
        # Earl zone: 16-21N, 70-40W (broader)
        if 16 <= lat <= 21 and -70 <= lon <= -40:
            n = count_lambda4(name)
            results.append((n, name, lat, lon))

    results.sort(reverse=True)
    print(f'\n=== {ic_label} ===')
    for n, name, lat, lon in results[:10]:
        print(f'  {name:<35} {lat:.2f}N {lon:.2f}W  λ4={n}')
