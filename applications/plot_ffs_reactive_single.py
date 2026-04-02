#!/usr/bin/env python
"""Plot a single reactive trajectory — the one from the highest-scoring lambda0 seed."""

import os, sys, json, pickle, yaml, argparse, warnings
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

sys.path.insert(0, '/glade/work/schreck/repos/miles-tails/applications')
from analyze_ffs_logs import load_all_logs, build_genealogy, find_stateB_configs

try:
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def pkl_path(ic_dir, cname):
    if cname.startswith('lambda0_'): return ic_dir/'flux'/f'{cname}.pkl'
    if cname.startswith('stateB_'):  return ic_dir/'stateB'/f'{cname}.pkl'
    num = cname.split('_')[0].replace('lambda','')
    return ic_dir/num/f'{cname}.pkl'

def get_latlon(ic_dir, cname):
    try:
        with open(pkl_path(ic_dir, cname),'rb') as f:
            obj = pickle.load(f)
        loc = getattr(obj,'feature_location',None)
        if loc: return float(loc[0]), float(loc[1])
    except: pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffs_config', required=True)
    ap.add_argument('--ic_dir',     required=True)
    ap.add_argument('--plot_dir',   default='./plots')
    args = ap.parse_args()

    ic_dir   = Path(args.ic_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    date_str = ic_dir.name

    with open(args.ffs_config) as f:
        cfg = yaml.safe_load(f)
    ifaces = cfg['interfaces'].copy()
    state_B = float(cfg['state_B'])
    if ifaces[-1] != state_B: ifaces.append(state_B)

    # 1. Score all lambda0 seeds from genealogy
    print('Loading genealogy...')
    entries   = load_all_logs(ic_dir/'logs')
    genealogy = build_genealogy(entries)
    stateB_cfgs = find_stateB_configs(entries, state_B)
    print(f'  {len(genealogy)} parents, {len(stateB_cfgs)} state-B configs')

    rev = {}
    for parent, kids in genealogy.items():
        for info in kids:
            if info.get('status') in ('success','reached_B','instant_success'):
                c = info.get('child')
                if c: rev[c] = parent

    scores = defaultdict(set)
    for b in stateB_cfgs:
        cur = b['config']
        while cur in rev:
            par = rev[cur]
            if par is None: break
            if par.startswith('lambda0_'):
                scores[par].add(b['config']); break
            cur = par
    print(f'  Scored {len(scores)} lambda0 seeds')
    if scores:
        best_l0, cnt = max(scores.items(), key=lambda x: len(x[1]))
        print(f'  Best: {best_l0}  ({cnt} B-descendants)')

    # 2. Find a full reactive path starting from best_l0
    json_path = ic_dir/'reactive_trajectories'/'reactive_trajectories.json'
    with open(json_path) as f:
        rdata = json.load(f)
    raw = rdata.get('reactive_trajectories', [])
    full = [t for t in raw if t.get('pathway_length',0) >= 4]

    # Prefer path from best_l0, else from runner-up
    ranked = sorted(scores.items(), key=lambda x: len(x[1]), reverse=True)
    chosen = None
    for l0_name, _ in ranked:
        matches = [t for t in full if t.get('pathway') and t['pathway'][0] == l0_name]
        if matches:
            chosen = matches[0]
            print(f'  Using path from {l0_name}  (pathway length {len(chosen["pathway"])})')
            break

    if chosen is None:
        # Just use any full path
        chosen = full[0]
        print(f'  Fallback: first full path (id {chosen["trajectory_id"]})')

    pathway = chosen['pathway']
    print(f'  Pathway: {pathway}')

    # 3. Load 5 pkls in parallel
    print('Loading pkl positions...')
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda c: (c, get_latlon(ic_dir, c)), pathway))
    points = [(ll, cname) for cname, ll in results if ll is not None]
    print(f'  Got {len(points)} positions')
    if len(points) < 2:
        print('ERROR: not enough positions. Exiting.')
        return

    # Sort by interface level
    def iface(cname):
        if cname.startswith('lambda0_'): return 0
        if cname.startswith('stateB_'):  return len(ifaces)
        try: return int(cname.split('_')[0].replace('lambda',''))
        except: return 0
    points.sort(key=lambda x: iface(x[1]))

    lats = [p[0][0] for p in points]
    lons = [p[0][1] for p in points]

    # 4. Plot
    cmap  = plt.cm.YlOrRd
    n     = len(ifaces) + 1
    colors= [cmap(0.2 + 0.75*i/max(n-1,1)) for i in range(n)]

    fig = plt.figure(figsize=(12,9))
    pad = 8.0
    lon0 = max(min(lons)-pad, -110); lon1 = min(max(lons)+pad, 10)
    lat0 = max(min(lats)-pad, 0);    lat1 = min(max(lats)+pad, 75)

    if HAS_CARTOPY:
        pc   = ccrs.PlateCarree()
        clat = (lat0+lat1)/2; clon = (lon0+lon1)/2
        proj = ccrs.LambertConformal(central_longitude=clon, central_latitude=clat,
                                      standard_parallels=(clat-8, clat+8))
        ax = fig.add_subplot(111, projection=proj)
        ax.set_extent([lon0,lon1,lat0,lat1], crs=pc)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#e8e8e8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),facecolor='#d0e8f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.0, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, alpha=0.5, zorder=3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4, linestyle='--', zorder=4)
        gl.top_labels = False; gl.right_labels = False
        tkw = dict(transform=pc)
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(lon0,lon1); ax.set_ylim(lat0,lat1)
        ax.grid(True, alpha=0.3)
        tkw = {}

    # Draw path segments coloured by source interface
    for i in range(len(points)-1):
        lat0s,lon0s = points[i][0]
        lat1s,lon1s = points[i+1][0]
        col = colors[min(i, len(colors)-1)]
        ax.plot([lon0s,lon1s],[lat0s,lat1s],'-', color=col, linewidth=2.5,
                alpha=0.85, zorder=5, **tkw)

    # Draw nodes
    for i, (ll, cname) in enumerate(points):
        lat,lon = ll
        is_l0 = cname.startswith('lambda0_')
        is_B  = cname.startswith('stateB_') or cname.startswith('lambda4_')
        col   = colors[min(iface(cname), len(colors)-1)]
        ax.scatter(lon, lat, s=350 if is_l0 else (200 if is_B else 80),
                   c=[col], marker='*' if is_l0 else ('*' if is_B else 'o'),
                   edgecolors='black', linewidths=1.0, zorder=8, **tkw)
        label_str = f'λ₀  {ifaces[0]:.0f} hPa' if is_l0 else \
                    f'State B  {state_B:.0f} hPa' if is_B else \
                    f'λ{iface(cname)}  {ifaces[iface(cname)]:.0f} hPa'
        ax.annotate(label_str, (lon, lat), textcoords='offset points',
                    xytext=(8,4), fontsize=8.5, zorder=9,
                    **({'transform': pc} if HAS_CARTOPY else {}))

    iface_str = '→'.join(f'{p:.0f}' for p in ifaces)
    ax.set_title(f'Reactive Trajectory  —  IC {date_str}\n'
                 f'Interfaces: {iface_str} hPa\n'
                 f'λ₀ seed: {pathway[0]}',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = plot_dir / f'ffs_reactive_single_{date_str}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
