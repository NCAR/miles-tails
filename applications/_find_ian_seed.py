import pickle, os, glob, json, collections, sys

class _SafeObj:
    def __init__(self, *a, **kw): pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _SafeObj

log_dir = '/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-22T00Z/logs'
children = collections.defaultdict(set)
for logfile in glob.glob(log_dir + '/**/*.jsonl', recursive=True):
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                if d.get('status') == 'success':
                    parent = d.get('parent_config','')
                    child  = d.get('child_config','')
                    if parent.startswith('lambda0_') and child.startswith('lambda4_'):
                        children[parent].add(child)
            except: pass

print(f'Found {len(children)} lambda0 seeds with lambda4 descendants', flush=True)

flux_dir = '/glade/derecho/scratch/schreck/FFS/results_mar18/2022-09-22T00Z/flux'
results = []
for p in glob.glob(flux_dir + '/lambda0_config_*.pkl'):
    name = os.path.basename(p).replace('.pkl','')
    try:
        with open(p, 'rb') as fh:
            obj = _SafeUnpickler(fh).load()
        loc = obj.feature_location
        lat, lon = float(loc[0]), float(loc[1])
        if 10 <= lat <= 25 and -100 <= lon <= -60:
            n_desc = len(children.get(name, set()))
            results.append((n_desc, name, lat, lon))
    except: pass

results.sort(reverse=True)
for n, name, lat, lon in results[:10]:
    print(f'{name}: {lat:.2f}N, {lon:.2f}W  lambda4_descendants={n}', flush=True)
