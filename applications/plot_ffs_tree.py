#!/usr/bin/env python
"""
plot_ffs_tree.py — FFS branching tree figure (single panel).

Uses the same JSONL log files as reactive_pathways.py to reconstruct
the full shooting genealogy.  For the λ₀ seed with the most state-B
descendants, traces the complete forward tree and draws it on a map.

Algorithm
---------
1. For each IC load ic_dir/logs/ → build genealogy + find stateB configs.
2. Trace every B-state config backward to its λ₀ ancestor; count per λ₀.
3. Pick the IC / λ₀ with the highest B-descendant count.
4. BFS forward through the genealogy from that λ₀ — collect all nodes.
5. Load each node's .pkl to get (lat, lon).
6. Draw tree: ★ = λ₀ seed, coloured circles = λ₁…λₙ / stateB nodes.

Usage
-----
    # Single IC
    python plot_ffs_tree.py \\
        --ic_dir   results_feb14/2022-08-21T00Z \\
        --state_B  960 \\
        --plot_dir results_feb14/plots

    # Scan all ICs in a CSV
    python plot_ffs_tree.py \\
        --ffs_csv    results_feb14/ffs_statistics_all_ics.csv \\
        --output_dir results_feb14 \\
        --state_B    960 \\
        --plot_dir   results_feb14/plots
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import matplotlib
matplotlib.use('Agg')

import pickle
import argparse
import warnings
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── allow importing sibling scripts (analyze_ffs_logs etc.) ───────────────────
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from analyze_ffs_logs import (
    load_all_logs,
    build_genealogy,
    find_stateB_configs,
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — falling back to plain lat/lon axes')

# ── Module-level config — overwritten from ffs.yml in main() ──────────────────
INTERFACE_PRESSURES: list   = []
N_IFACES:            int    = 0
IFACE_COLORS:        object = None   # np.ndarray after init


def _init_from_config(ffs_config: dict):
    """Set module-level interface constants from the loaded ffs.yml dict."""
    global INTERFACE_PRESSURES, N_IFACES, IFACE_COLORS
    ifaces  = ffs_config['interfaces'].copy()
    state_B = float(ffs_config['state_B'])
    if ifaces[-1] != state_B:
        ifaces.append(state_B)
    INTERFACE_PRESSURES = ifaces
    N_IFACES            = len(ifaces)
    IFACE_COLORS        = plt.cm.YlOrRd(np.linspace(0.25, 0.95, N_IFACES))


# ── pkl helpers ────────────────────────────────────────────────────────────────

def _pkl_path(ic_dir: Path, config_name: str) -> Path:
    if config_name.startswith('lambda0_'):
        return ic_dir / 'flux' / f'{config_name}.pkl'
    elif config_name.startswith('stateB_'):
        return ic_dir / 'stateB' / f'{config_name}.pkl'
    else:
        iface_num = config_name.split('_')[0].replace('lambda', '')
        return ic_dir / iface_num / f'{config_name}.pkl'


def _load_pkl(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _get_latlon(ic_dir: Path, config_name: str):
    """Return (lat, lon) from the pkl's feature_location, or None."""
    cfg = _load_pkl(_pkl_path(ic_dir, config_name))
    if cfg is None:
        return None
    loc = getattr(cfg, 'feature_location', None)
    if loc is None:
        return None
    try:
        return float(loc[0]), float(loc[1])
    except (TypeError, IndexError):
        return None


def _iface_idx_from_name(config_name: str) -> int:
    if config_name.startswith('lambda0_'):
        return 0
    if config_name.startswith('stateB_'):
        return N_IFACES - 1
    try:
        return int(config_name.split('_')[0].replace('lambda', ''))
    except ValueError:
        return 0


# ── genealogy helpers ──────────────────────────────────────────────────────────

def _make_children_map(genealogy: dict) -> dict:
    """
    genealogy = {parent: [{child, status, ...}, ...]}
    Returns   {parent: [child_name, ...]}  — successful shoots only.
    """
    children = defaultdict(list)
    for parent, child_list in genealogy.items():
        for info in child_list:
            if info.get('status') in ('success', 'reached_B', 'instant_success'):
                c = info.get('child')
                if c:
                    children[parent].append(c)
    return dict(children)


def _make_reverse_map(genealogy: dict) -> dict:
    """Returns {child_name: parent_name} for successful shoots only."""
    reverse = {}
    for parent, child_list in genealogy.items():
        for info in child_list:
            if info.get('status') in ('success', 'reached_B', 'instant_success'):
                c = info.get('child')
                if c:
                    reverse[c] = parent
    return reverse


def _score_lambda0s(genealogy: dict, stateB_configs: list) -> dict:
    """
    For each λ₀ config, count the number of distinct state-B descendants
    by tracing each B-state backward through the genealogy to its λ₀ root.

    Returns {lambda0_name: n_B_descendants}.
    """
    reverse = _make_reverse_map(genealogy)

    scores = defaultdict(set)   # lambda0_name → set of stateB config names
    for b_info in stateB_configs:
        b = b_info['config']
        cur = b
        while cur in reverse:
            par = reverse[cur]
            if par is None:
                break
            if par.startswith('lambda0_'):
                scores[par].add(b)
                break
            cur = par

    return {k: len(v) for k, v in scores.items()}


def _bfs_forward(children_map: dict, root: str) -> set:
    """All config names reachable from root via the children_map (BFS)."""
    visited, queue = set(), [root]
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        queue.extend(children_map.get(n, []))
    return visited


# ── IC / root selection ────────────────────────────────────────────────────────

def _load_genealogy_for_ic(ic_dir: Path, state_B: float):
    """
    Load logs for one IC and return (genealogy, stateB_configs).
    Returns (None, None) if logs directory is missing or empty.
    """
    logs_dir = ic_dir / 'logs'
    if not logs_dir.exists():
        return None, None
    try:
        entries   = load_all_logs(logs_dir)
        genealogy = build_genealogy(entries)
        stateB    = find_stateB_configs(entries, state_B)
        return genealogy, stateB
    except Exception as e:
        print(f'  ERROR loading logs for {ic_dir.name}: {e}')
        return None, None


def rank_lambda0s_for_ic(ic_dir: Path, state_B: float, top_n: int = 20):
    """
    Print a ranked table of all λ₀ roots for one IC by B-descendant count.
    Returns the full scores dict {lambda0_name: n_descendants}.
    """
    genealogy, stateB = _load_genealogy_for_ic(ic_dir, state_B)
    if genealogy is None:
        print(f'  No logs found in {ic_dir}')
        return {}

    scores = _score_lambda0s(genealogy, stateB)
    print(f'\n  IC: {ic_dir.name}')
    print(f'  Total B-state configs : {len(stateB)}')
    print(f'  Total λ₀ roots scored : {len(scores)}')
    print(f'\n  {"Rank":>4}  {"λ₀ config name":<40}  {"B-descendants":>13}')
    print('  ' + '-'*62)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, sc) in enumerate(ranked[:top_n], 1):
        marker = '  ← winner' if rank == 1 else ''
        print(f'  {rank:>4}  {name:<40}  {sc:>13}{marker}')
    if len(ranked) > top_n:
        print(f'  … and {len(ranked) - top_n} more')
    print()
    return scores


def find_best_example(ic_dirs: list, state_B: float):
    """
    Scan logs for all ICs and return
        (ic_dir, root_config_name, n_B_descendants)
    for the λ₀ with the highest number of state-B descendants.
    """
    best_score = 0
    best_ic    = None
    best_root  = None

    for ic_dir in ic_dirs:
        print(f'  Scanning {ic_dir.name} …', end=' ', flush=True)
        genealogy, stateB = _load_genealogy_for_ic(ic_dir, state_B)
        if genealogy is None:
            print('no logs')
            continue

        scores = _score_lambda0s(genealogy, stateB)
        if scores:
            top = max(scores, key=scores.get)
            print(f'{len(stateB)} B-configs, best λ₀ has {scores[top]} descendants')
        else:
            print(f'{len(stateB)} B-configs, no λ₀ scored')
            continue

        for l0, sc in scores.items():
            if sc > best_score:
                best_score = sc
                best_ic    = ic_dir
                best_root  = l0

    return best_ic, best_root, best_score


# ── full tree builder ──────────────────────────────────────────────────────────

def build_tree(ic_dir: Path, root_config: str, state_B: float):
    """
    Load logs for this IC, BFS-forward from root_config through all
    successful children, load lat/lon from each pkl.

    Returns
    -------
    nodes : dict
        {config_name: {lat, lon, iface_idx, parent}}
    root_config : str
    """
    genealogy, _ = _load_genealogy_for_ic(ic_dir, state_B)
    if genealogy is None:
        return None, None

    children_map = _make_children_map(genealogy)
    reverse_map  = _make_reverse_map(genealogy)

    reachable = _bfs_forward(children_map, root_config)
    print(f'  {len(reachable)} configs reachable from {root_config}')

    nodes = {}
    n_missing = 0
    for cname in reachable:
        ll = _get_latlon(ic_dir, cname)
        if ll is None:
            n_missing += 1
            continue
        lat, lon = ll
        nodes[cname] = {
            'lat':       lat,
            'lon':       lon,
            'iface_idx': min(_iface_idx_from_name(cname), N_IFACES - 1),
            'parent':    reverse_map.get(cname),      # None for the root
        }

    if n_missing:
        print(f'  Warning: {n_missing} nodes skipped (pkl missing or no feature_location)')

    n_per = [sum(1 for n in nodes.values() if n['iface_idx'] == i)
             for i in range(N_IFACES)]
    print(f'  Nodes per interface: {n_per}')

    return nodes, root_config


# ── map axes ──────────────────────────────────────────────────────────────────

def make_atlantic_axes(fig, nodes: dict = None, pad: float = 6.0):
    """
    If nodes is provided, auto-zoom to their bounding box + pad degrees.
    Falls back to full-basin extent if nodes is empty/None.
    """
    if nodes:
        lons = [n['lon'] for n in nodes.values()]
        lats = [n['lat'] for n in nodes.values()]
        lon0 = max(min(lons) - pad, -100)
        lon1 = min(max(lons) + pad,  -10)
        lat0 = max(min(lats) - pad,    5)
        lat1 = min(max(lats) + pad,   65)
    else:
        lon0, lon1, lat0, lat1 = -100, -10, 5, 65

    if HAS_CARTOPY:
        clat = (lat0 + lat1) / 2
        clon = (lon0 + lon1) / 2
        proj = ccrs.LambertConformal(
            central_longitude=clon,
            central_latitude=clat,
            standard_parallels=(clat - 5, clat + 5),
        )
        ax = fig.add_subplot(111, projection=proj)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale('50m'),
                       facecolor='#e8e8e8', zorder=2)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                       facecolor='#d0e8f5', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=0.9, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale('50m'),
                       linewidth=0.4, alpha=0.5, zorder=3)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4,
                     linestyle='--', zorder=4)
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
        ax.grid(True, alpha=0.3)
    return ax


# ── drawing ───────────────────────────────────────────────────────────────────

def _jitter_positions(nodes: dict, seed: int = 42, scale: float = 0.25) -> dict:
    """
    Return a {config_name: (jittered_lat, jittered_lon)} map.
    Nodes snap to a 1° grid so without jitter hundreds of lines overlap exactly.
    Scale ≈ 0.25° keeps points visually near their true grid cell.
    The seed node (λ₀) is never jittered.
    """
    rng = np.random.default_rng(seed)
    pos = {}
    for cname, node in nodes.items():
        if node['parent'] is None:          # root — no jitter
            pos[cname] = (node['lat'], node['lon'])
        else:
            pos[cname] = (
                node['lat'] + rng.uniform(-scale, scale),
                node['lon'] + rng.uniform(-scale, scale),
            )
    return pos


def _draw_tree(ax, nodes: dict, root: str):
    pc  = ccrs.PlateCarree() if HAS_CARTOPY else None
    pos = _jitter_positions(nodes)

    # Pass 1 — edges (parent → child lines)
    for cname, node in nodes.items():
        pname = node['parent']
        if pname not in nodes:
            continue
        iface = node['iface_idx']
        color = IFACE_COLORS[iface]
        lw    = max(0.5, 1.8 - 0.20 * iface)
        alpha = max(0.25, 0.75 - 0.08 * iface)
        plat, plon = pos[pname]
        clat, clon = pos[cname]
        kw = dict(color=color, alpha=alpha, linewidth=lw,
                  solid_capstyle='round', zorder=5)
        if HAS_CARTOPY:
            ax.plot([plon, clon], [plat, clat], transform=pc, **kw)
        else:
            ax.plot([plon, clon], [plat, clat], **kw)

    # Pass 2 — nodes
    for cname, node in nodes.items():
        iface   = node['iface_idx']
        color   = IFACE_COLORS[iface]
        is_seed = (cname == root)
        lat, lon = pos[cname]
        kw = dict(
            s          = 350 if is_seed else 35,
            color      = color,
            marker     = '*' if is_seed else 'o',
            edgecolors = 'k',
            linewidths = 1.0 if is_seed else 0.3,
            alpha      = 0.95,
            zorder     = 8 if is_seed else 6,
        )
        if HAS_CARTOPY:
            ax.scatter(lon, lat, transform=pc, **kw)
        else:
            ax.scatter(lon, lat, **kw)


# ── main figure ───────────────────────────────────────────────────────────────

def plot_tree(ic_dir: Path, plot_dir: Path, state_B: float,
              root_config: str = None):
    date_str = ic_dir.name[:10]

    # ── select root if not specified ──────────────────────────────────────────
    if root_config is None:
        genealogy, stateB = _load_genealogy_for_ic(ic_dir, state_B)
        if genealogy is None:
            print(f'  No logs found in {ic_dir}')
            return None
        scores = _score_lambda0s(genealogy, stateB)
        if not scores:
            print(f'  No λ₀ roots scored in {ic_dir}')
            return None
        root_config = max(scores, key=scores.get)
        print(f'  Best root: {root_config}  ({scores[root_config]} B-descendants)')

    # ── build tree ────────────────────────────────────────────────────────────
    print(f'Building tree: {ic_dir.name} / {root_config}')
    nodes, root = build_tree(ic_dir, root_config, state_B)

    if not nodes:
        print('  Tree build failed — no nodes with lat/lon.')
        return None

    n_per = [sum(1 for n in nodes.values() if n['iface_idx'] == i)
             for i in range(N_IFACES)]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax  = make_atlantic_axes(fig, nodes=nodes, pad=6.0)
    _draw_tree(ax, nodes, root)

    # legend
    handles = [
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=IFACE_COLORS[0], markersize=16,
               markeredgecolor='k', markeredgewidth=0.8,
               label=f'λ₀  {INTERFACE_PRESSURES[0]} hPa  — seed  (n={n_per[0]})'),
    ]
    for i in range(1, N_IFACES):
        if n_per[i] > 0:
            handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=IFACE_COLORS[i], markersize=9,
                       markeredgecolor='k', markeredgewidth=0.5,
                       label=f'λ{i}  {INTERFACE_PRESSURES[i]} hPa  (n={n_per[i]})')
            )
    ax.legend(handles=handles, fontsize=9, loc='lower left',
              framealpha=0.92, title='Interface level', title_fontsize=10)

    n_l1 = n_per[1] if len(n_per) > 1 else 0
    ax.set_title(
        f'FFS shooting tree — IC {date_str}\n'
        f'λ₀ seed → {n_l1} λ₁ branches → {len(nodes) - 1} total descendants',
        fontsize=12, fontweight='bold',
    )

    out = plot_dir / f'ffs_tree_{date_str}_{root}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FFS shooting tree — full genealogical tree from a single λ₀ seed'
    )
    parser.add_argument('--ffs_config', required=True,
                        help='FFS config file (ffs.yml) — source of interfaces and state_B')
    parser.add_argument('--ic_dir',     default=None,
                        help='Single IC directory (e.g. results/2022-08-21T00Z)')
    parser.add_argument('--ffs_csv',    default=None,
                        help='CSV with time_label column (scan multiple ICs)')
    parser.add_argument('--output_dir', default=None,
                        help='Parent directory of all IC directories')
    parser.add_argument('--plot_dir',   default='./plots')
    parser.add_argument('--root',       default=None,
                        help='Manually specify λ₀ config name to use as tree root')
    parser.add_argument('--rank',       action='store_true',
                        help='Print ranked table of all λ₀ roots by B-descendants, then exit')
    args = parser.parse_args()

    # ── Load config and initialise interface constants ────────────────────────
    with open(args.ffs_config) as f:
        ffs_config = yaml.safe_load(f)
    _init_from_config(ffs_config)
    state_B = float(ffs_config['state_B'])
    print(f'Interfaces: {INTERFACE_PRESSURES}  state_B={state_B}')

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── single IC mode ───────────────────────────────────────────────────────
    if args.ic_dir:
        ic_dir = Path(args.ic_dir)
        if args.rank:
            rank_lambda0s_for_ic(ic_dir, state_B)
            return
        plot_tree(ic_dir, plot_dir, state_B, root_config=args.root)
        return

    # ── multi-IC mode ────────────────────────────────────────────────────────
    if not (args.ffs_csv and args.output_dir):
        parser.error('Provide --ic_dir  OR  both --ffs_csv and --output_dir')

    import pandas as pd
    df      = pd.read_csv(args.ffs_csv)
    out_dir = Path(args.output_dir)
    ic_dirs = [out_dir / tl for tl in df['time_label'].tolist()
               if (out_dir / tl).exists()]

    if not ic_dirs:
        print('No IC directories found.')
        return

    print(f'Scanning {len(ic_dirs)} ICs …')
    ic_dir, root, score = find_best_example(ic_dirs, state_B)

    if ic_dir is None:
        print('No usable IC found.')
        return

    print(f'\nBest: {ic_dir.name}  root={root}  B-descendants={score}')
    plot_tree(ic_dir, plot_dir, state_B, root_config=root)


if __name__ == '__main__':
    main()
