#!/usr/bin/env python
"""
plot_ffs_tree.py — FFS shooting tree from a single λ₀ seed (geographic map).

Ranks λ₀ seeds by number of λ₄ descendants (lambda4_config_* only — stateB_
configs are excluded because they lack reliable geographic positions).  Plots
the top --top_n seeds as geographic branching trees on a cartopy map.

Usage
-----
    python plot_ffs_tree.py \\
        --ffs_config ffs.yml \\
        --ic_dir     results/2022-09-02T00Z \\
        --plot_dir   results/plots \\
        --top_n      5
"""

import os, sys, pickle, argparse, warnings, yaml
os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ffs_logs import load_all_logs, build_genealogy, find_stateB_configs

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ── helpers ──────────────────────────────────────────────────────────────────

def build_reverse(genealogy):
    rev = {}
    for parent, kids in genealogy.items():
        for info in kids:
            if info.get('status') in ('success', 'reached_B', 'instant_success'):
                c = info.get('child')
                if c:
                    rev[c] = parent
    return rev


def iface_level(cname):
    if cname.startswith('lambda0_'):
        return 0
    try:
        return int(cname.split('_')[0].replace('lambda', ''))
    except:
        return 0


def pkl_path(ic_dir, cname):
    if cname.startswith('lambda0_'):
        return ic_dir / 'flux' / f'{cname}.pkl'
    try:
        level = cname.split('_')[0].replace('lambda', '')
        return ic_dir / level / f'{cname}.pkl'
    except:
        return None


def load_loc(args):
    ic_dir, cname = args
    p = pkl_path(ic_dir, cname)
    if p is None or not p.exists():
        return cname, None
    try:
        obj = pickle.load(open(p, 'rb'))
        loc = obj.feature_location
        return cname, (float(loc[0]), float(loc[1]))
    except Exception:
        return cname, None


def score_lambda0s_lambda4only(genealogy):
    """Score λ₀ seeds by number of unique lambda4 descendants only."""
    rev = build_reverse(genealogy)
    scores = defaultdict(set)
    for cname in genealogy:
        pass
    # Walk every lambda4 config back to its lambda0 ancestor
    all_configs = set(rev.keys()) | set(genealogy.keys())
    for cname in all_configs:
        if not cname.startswith('lambda4_'):
            continue
        cur = cname
        while cur in rev:
            par = rev[cur]
            if par is None:
                break
            if par.startswith('lambda0_'):
                scores[par].add(cname)
                break
            cur = par
    return {k: len(v) for k, v in scores.items()}


def collect_tree_nodes(genealogy, l0_name, max_level=4):
    """Return all lambda configs reachable from l0_name up to max_level."""
    nodes = {l0_name}
    edges = []
    queue = [l0_name]
    while queue:
        parent = queue.pop()
        for info in genealogy.get(parent, []):
            if info.get('status') not in ('success', 'reached_B', 'instant_success'):
                continue
            child = info.get('child')
            if child is None:
                continue
            if child.startswith('stateB_'):
                continue  # skip stateB configs entirely
            lv = iface_level(child)
            if lv > max_level:
                continue
            if child not in nodes:
                nodes.add(child)
                queue.append(child)
            edges.append((parent, child))
    return nodes, edges


def plot_tree(l0_name, rank, nodes, edges, loc_map, ifaces, n_ifaces,
              date_str, n_lambda4, plot_dir):
    lats = [loc_map[n][0] for n in nodes if n in loc_map]
    lons = [loc_map[n][1] for n in nodes if n in loc_map]
    if not lats:
        print(f'  No locations for {l0_name}, skipping')
        return

    pad = 4
    lat_range = max(lats) - min(lats)
    lon_range = max(lons) - min(lons)
    # Enforce a minimum map size of 10° so single-point seeds still show context
    if lat_range < 10:
        clat = (max(lats) + min(lats)) / 2
        lats_ext = [clat - 5, clat + 5]
    else:
        lats_ext = lats
    if lon_range < 15:
        clon = (max(lons) + min(lons)) / 2
        lons_ext = [clon - 8, clon + 8]
    else:
        lons_ext = lons

    extent = [min(lons_ext) - pad, max(lons_ext) + pad,
              max(min(lats_ext) - pad, -5), min(max(lats_ext) + pad, 70)]

    cmap   = plt.cm.YlOrRd
    colors = [cmap(0.15 + 0.8 * i / max(n_ifaces - 1, 1)) for i in range(n_ifaces)]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=proj))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor='#e8e4d9', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#c9dff0', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#555', zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor='#888', zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                 alpha=0.5, linestyle='--', zorder=1)

    # Count how many times each node appears as a parent (branching weight)
    child_count = defaultdict(int)
    for pa, ch in edges:
        child_count[pa] += 1

    # Draw edges
    drawn = set()
    for pa, ch in edges:
        key = (pa, ch)
        if key in drawn:
            continue
        drawn.add(key)
        if pa not in loc_map or ch not in loc_map:
            continue
        lat0, lon0 = loc_map[pa]
        lat1, lon1 = loc_map[ch]
        lv  = iface_level(ch)
        col = colors[min(lv, len(colors) - 1)]
        ax.plot([lon0, lon1], [lat0, lat1], '-',
                color=col, alpha=0.6, linewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=3)

    # Draw nodes grouped by level for correct z-order
    node_counts = defaultdict(int)
    for pa, ch in edges:
        node_counts[ch] += 1
    node_counts[l0_name] = 1

    for lv in range(n_ifaces):
        level_nodes = [n for n in nodes if iface_level(n) == lv and n in loc_map]
        if not level_nodes:
            continue
        col = colors[min(lv, len(colors) - 1)]
        is_l0 = (lv == 0)
        lats_lv = [loc_map[n][0] for n in level_nodes]
        lons_lv = [loc_map[n][1] for n in level_nodes]
        sizes   = [300 if is_l0 else max(20, 15 * node_counts.get(n, 1))
                   for n in level_nodes]
        ax.scatter(lons_lv, lats_lv,
                   s=sizes, c=[col] * len(level_nodes),
                   marker='*' if is_l0 else 'o',
                   edgecolors='black' if is_l0 else 'none',
                   linewidths=0.8 if is_l0 else 0,
                   transform=ccrs.PlateCarree(),
                   zorder=6 if is_l0 else 4 + lv,
                   label=f'λ{lv}  {ifaces[lv]:.0f} hPa  (n={len(level_nodes)})')

    ax.legend(fontsize=9, loc='lower left', framealpha=0.9)
    ax.set_title(
        f'FFS shooting tree  —  IC {date_str}\n'
        f'λ₀ seed → {child_count[l0_name]} λ₁ branches → {n_lambda4} total λ₄ descendants',
        fontsize=11, fontweight='bold')

    out = plot_dir / f'ffs_tree_rank{rank:02d}_{date_str}_{l0_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffs_config',  required=True)
    ap.add_argument('--ic_dir',      required=True)
    ap.add_argument('--plot_dir',    default='./plots')
    ap.add_argument('--top_n',       type=int,   default=5)
    ap.add_argument('--scan_n',      type=int,   default=30,
                    help='Candidates to scan before applying spread filter')
    ap.add_argument('--min_spread',  type=float, default=1.5,
                    help='Min std-dev (degrees) of λ4 lat+lon to accept a seed')
    ap.add_argument('--workers',     type=int,   default=16)
    ap.add_argument('--seed_lat_min', type=float, default=None,
                    help='Minimum latitude of λ0 seed to consider')
    ap.add_argument('--seed_lat_max', type=float, default=None,
                    help='Maximum latitude of λ0 seed to consider')
    ap.add_argument('--seed_lon_min', type=float, default=None,
                    help='Minimum longitude of λ0 seed to consider (degrees, negative=W)')
    ap.add_argument('--seed_lon_max', type=float, default=None,
                    help='Maximum longitude of λ0 seed to consider (degrees, negative=W)')
    args = ap.parse_args()

    ic_dir   = Path(args.ic_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    date_str = ic_dir.name

    with open(args.ffs_config) as f:
        cfg = yaml.safe_load(f)
    ifaces  = cfg['interfaces'].copy()
    state_B = float(cfg['state_B'])
    ifaces.append(state_B)   # ifaces = [1000.0, 987.2, 984.7, 981.2, 975.0]
    n_ifaces = len(ifaces)   # = 5, so range(5) → levels 0..4

    print(f'Loading genealogy for {date_str}...')
    entries   = load_all_logs(ic_dir / 'logs')
    genealogy = build_genealogy(entries)
    print(f'  {len(genealogy)} parent configs')

    scores = score_lambda0s_lambda4only(genealogy)

    # ── Load locations for ALL scored seeds upfront so we can compute
    #    the geographic spread of λ4 descendants and sort by spread × count.
    #    Pure-count ranking rewards "squashed" seeds where a super-favorable
    #    spot lets trajectories cross all interfaces without moving; spread ×
    #    count rewards seeds whose descendants fan out into an organic cascade.
    all_seed_nodes = {}
    for l0_name in scores:
        nodes, _ = collect_tree_nodes(genealogy, l0_name)
        all_seed_nodes[l0_name] = nodes

    # Load all λ4 locations for every scored seed
    all_lv4 = set()
    for nodes in all_seed_nodes.values():
        all_lv4.update(n for n in nodes if n.startswith('lambda4_'))
    # Also load λ0 seed locations (needed for track-distance scoring)
    all_lv4.update(scores.keys())

    with ThreadPoolExecutor(max_workers=args.workers) as ex_pre:
        pre_locs = dict(ex_pre.map(load_loc, [(ic_dir, n) for n in all_lv4]))

    def _spread_score(l0_name):
        nodes = all_seed_nodes.get(l0_name, set())
        lv4 = [n for n in nodes if n.startswith('lambda4_') and n in pre_locs]
        if len(lv4) < 2:
            return 0.0
        lats = [pre_locs[n][0] for n in lv4]
        lons = [pre_locs[n][1] for n in lv4]
        spread = float(np.std(lats) + np.std(lons))
        # Also require the centroid of λ4 to be displaced from the seed
        # (filters squashed trees where everything stays near the origin)
        seed_loc = pre_locs.get(l0_name)
        if seed_loc is not None:
            centroid_lat = float(np.mean(lats))
            centroid_lon = float(np.mean(lons))
            track_dist = np.sqrt((centroid_lat - seed_loc[0])**2 +
                                 (centroid_lon - seed_loc[1])**2)
        else:
            track_dist = 0.0
        return spread * np.log1p(track_dist)  # favour spread + displacement

    composite_scores = {
        name: _spread_score(name) * np.log1p(count)
        for name, count in scores.items()
    }
    ranked = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    # Keep the original λ4 count accessible for printing/title
    count_map = scores
    print(f'  {len(ranked)} λ₀ seeds scored by spread × cascade distance')

    # ── Seed-location filter: load λ0 locations first if a bounding box is given
    seed_box = (args.seed_lat_min, args.seed_lat_max,
                args.seed_lon_min, args.seed_lon_max)
    if any(v is not None for v in seed_box):
        print(f'  Applying seed box filter: '
              f'lat=[{args.seed_lat_min},{args.seed_lat_max}] '
              f'lon=[{args.seed_lon_min},{args.seed_lon_max}]')
        seed_names = [n for n, _ in ranked]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            seed_locs = dict(ex.map(load_loc, [(ic_dir, n) for n in seed_names]))
        def in_box(name):
            loc = seed_locs.get(name)
            if loc is None:
                return False
            lat, lon = loc
            if args.seed_lat_min is not None and lat < args.seed_lat_min:
                return False
            if args.seed_lat_max is not None and lat > args.seed_lat_max:
                return False
            if args.seed_lon_min is not None and lon < args.seed_lon_min:
                return False
            if args.seed_lon_max is not None and lon > args.seed_lon_max:
                return False
            return True
        n_before = len(ranked)
        ranked = [(n, s) for n, s in ranked if in_box(n)]
        print(f'  {len(ranked)}/{n_before} seeds remain after box filter')

    # Scan up to scan_n candidates; collect their nodes for bulk location loading
    candidates = ranked[:args.scan_n]
    all_needed = set()
    candidate_trees = []
    for l0_name, n_lambda4 in candidates:
        nodes, edges = collect_tree_nodes(genealogy, l0_name)
        candidate_trees.append((l0_name, n_lambda4, nodes, edges))
        all_needed.update(nodes)

    print(f'  Loading {len(all_needed)} node locations (scanning {len(candidates)} candidates)...')
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(load_loc, [(ic_dir, c) for c in all_needed]))
    loc_map = {c: loc for c, loc in results if loc is not None}
    print(f'  {len(loc_map)} locations loaded')

    # Apply geographic-spread filter: require λ4 nodes to span > min_spread degrees
    def lambda4_spread(nodes):
        lv4 = [n for n in nodes if n.startswith('lambda4_') and n in loc_map]
        if len(lv4) < 2:
            return 0.0
        lats = [loc_map[n][0] for n in lv4]
        lons = [loc_map[n][1] for n in lv4]
        return float(np.std(lats) + np.std(lons))

    trees = []
    rank = 1
    for l0_name, n_lambda4, nodes, edges in candidate_trees:
        sp = lambda4_spread(nodes)
        status = 'OK' if sp >= args.min_spread else f'SKIP (spread={sp:.2f}°)'
        print(f'  {l0_name}  λ₄={n_lambda4}  spread={sp:.2f}°  → {status}')
        if sp < args.min_spread:
            continue
        trees.append((rank, l0_name, n_lambda4, nodes, edges))
        rank += 1
        if len(trees) >= args.top_n:
            break

    if not trees:
        print('  WARNING: no seeds passed the spread filter — lowering threshold to 0')
        for i, (l0_name, n_lambda4, nodes, edges) in enumerate(candidate_trees[:args.top_n], 1):
            trees.append((i, l0_name, n_lambda4, nodes, edges))

    print(f'\n  Plotting {len(trees)} seeds:')
    for rank, l0_name, n_lambda4, nodes, edges in trees:
        print(f'  rank{rank:02d}: {l0_name}  ({n_lambda4} λ₄ descendants)')

    for rank, l0_name, n_lambda4, nodes, edges in trees:
        print(f'Plotting rank{rank:02d}: {l0_name}')
        plot_tree(l0_name, rank, nodes, edges, loc_map, ifaces, n_ifaces,
                  date_str, n_lambda4, plot_dir)


if __name__ == '__main__':
    main()
