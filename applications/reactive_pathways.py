#!/usr/bin/env python
"""
plot_ffs_tree.py — FFS branching tree figure (single panel).

The reactive_trajectories.json records one representative path per λ₀ root
with cluster_size encoding multiplicity — it does NOT contain the full
branching tree.  The actual tree (all shots fired from each node, including
non-reactive branches) lives in the pkl directories:

    ic_dir/flux/   — all λ₀ crossings
    ic_dir/1/      — all λ₁ configs generated during shooting
    ic_dir/2/      — all λ₂ configs
    …

This script:
  1. Scans reactive_trajectories.json across all ICs to find the λ₀ root
     with the highest total cluster_size (most reactive-trajectory weight).
  2. Scans the pkl directories for that IC to build the FULL shooting tree,
     auto-discovering which pkl attribute stores the parent config name.
  3. Draws the tree on a single Atlantic map panel.

Usage:
    python plot_ffs_tree.py \\
        --ffs_csv    results_feb14/ffs_statistics_all_ics.csv \\
        --output_dir results_feb14 \\
        --plot_dir   results_feb14/plots

    python plot_ffs_tree.py \\
        --ic_dir   results_feb14/2022-08-21T00Z \\
        --plot_dir results_feb14/plots
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import json
import pickle
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('cartopy not found — falling back to plain lat/lon axes')

# ── Constants (match plot_genesis_cone.py) ─────────────────────────────────────

INTERFACE_PRESSURES = [1000, 988, 980, 975, 970, 965]
N_IFACES            = len(INTERFACE_PRESSURES)
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


def _node_from_pkl(cfg, parent_name: str, iface_num: int) -> dict | None:
    """Extract position and metadata from a loaded pkl config object."""
    loc = getattr(cfg, 'feature_location', None)
    if loc is None:
        return None
    try:
        lat, lon = float(loc[0]), float(loc[1])
    except (TypeError, IndexError):
        return None
    iface_idx = getattr(cfg, 'interface_idx', iface_num)
    if iface_idx == -1:
        iface_idx = N_IFACES - 1
    return {
        'lat':           lat,
        'lon':           lon,
        'iface_idx':     int(iface_idx),
        'parent':        parent_name,
        'children':      set(),
        'cluster_total': 0,
    }


# ── Parent attribute discovery ─────────────────────────────────────────────────

def _find_parent_attr(cfg) -> str | None:
    """
    Inspect a pkl config object to find the attribute that holds the
    parent config name (a string starting with 'lambda' or 'stateB').

    Tries common names first; falls back to scanning all string attributes.
    """
    candidates = (
        'parent_config', 'restart_config', 'parent_config_name',
        'seed_config', 'parent_name', 'source_config', 'parent',
        'lambda0_config', 'origin_config',
    )
    for attr in candidates:
        val = getattr(cfg, attr, None)
        if isinstance(val, str) and ('lambda' in val or 'stateB' in val):
            return attr

    # Fallback: scan all string attributes
    try:
        for attr, val in vars(cfg).items():
            if (isinstance(val, str)
                    and ('lambda' in val or 'stateB' in val)
                    and '_config_' in val):
                return attr
    except TypeError:
        pass

    return None


# ── IC/root selection (JSON-based, no pkl I/O) ─────────────────────────────────

def find_best_example(ic_dirs: list):
    """
    Scan reactive_trajectories.json for every IC and return the (ic_dir,
    root_config_name) with the highest total cluster_size — the λ₀ that
    contributed the most reactive-trajectory weight.

    Returns (ic_dir, root_config, total_cluster_weight).
    """
    best_score = 0
    best_ic    = None
    best_root  = None

    for ic_dir in ic_dirs:
        jp = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
        if not jp.exists():
            continue
        try:
            with open(jp) as f:
                data = json.load(f)
        except Exception:
            continue

        l0_weight: dict[str, int] = {}
        for traj in data.get('reactive_trajectories', []):
            pw      = traj.get('pathway', [])
            cluster = int(traj.get('cluster_size', 1))
            if pw:
                l0_weight[pw[0]] = l0_weight.get(pw[0], 0) + cluster

        for l0, w in l0_weight.items():
            if w > best_score:
                best_score = w
                best_ic    = ic_dir
                best_root  = l0

    return best_ic, best_root, best_score


def best_root_in_ic(ic_dir: Path) -> tuple[str, int]:
    """Return (root_config_name, cluster_weight) for the best λ₀ in one IC."""
    jp = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
    if not jp.exists():
        return None, 0
    try:
        with open(jp) as f:
            data = json.load(f)
    except Exception:
        return None, 0

    l0_weight: dict[str, int] = {}
    for traj in data.get('reactive_trajectories', []):
        pw      = traj.get('pathway', [])
        cluster = int(traj.get('cluster_size', 1))
        if pw:
            l0_weight[pw[0]] = l0_weight.get(pw[0], 0) + cluster

    if not l0_weight:
        return None, 0
    best = max(l0_weight, key=l0_weight.get)
    return best, l0_weight[best]


# ── Full shooting tree from pkl directories ────────────────────────────────────

def build_shooting_tree(ic_dir: Path, root_config: str):
    """
    Build the FULL branching tree by scanning pkl directories.

    For each interface level, loads every pkl in ic_dir/{N}/ and checks
    whether its parent config attribute matches a known node at the
    previous level.  Auto-discovers the parent attribute name from the
    first readable pkl.

    Returns (nodes dict, root_config) or (None, None) on failure.
    """
    # ── Load root ────────────────────────────────────────────────────────────
    root_cfg = _load_pkl(_pkl_path(ic_dir, root_config))
    if root_cfg is None:
        print(f'  Could not load root pkl: {root_config}')
        return None, None

    loc = getattr(root_cfg, 'feature_location', None)
    if loc is None:
        print(f'  Root pkl has no feature_location')
        return None, None

    nodes = {
        root_config: {
            'lat':           float(loc[0]),
            'lon':           float(loc[1]),
            'iface_idx':     0,
            'parent':        None,
            'children':      set(),
            'cluster_total': 0,
        }
    }

    # ── Discover parent attribute from first available λ₁ pkl ────────────────
    parent_attr = None
    iface1_dir  = ic_dir / '1'
    if iface1_dir.exists():
        for pkl_file in list(iface1_dir.glob('*.pkl'))[:20]:
            cfg  = _load_pkl(pkl_file)
            if cfg is None:
                continue
            attr = _find_parent_attr(cfg)
            if attr:
                parent_attr = attr
                print(f'  Parent attribute discovered: cfg.{parent_attr!r}')
                break

    if parent_attr is None:
        print('  WARNING: could not find parent config attribute in λ₁ pkls.')
        print('  Available attributes on first λ₁ pkl:')
        for pkl_file in list(iface1_dir.glob('*.pkl'))[:1]:
            cfg = _load_pkl(pkl_file)
            if cfg:
                try:
                    for k, v in vars(cfg).items():
                        print(f'    {k}: {type(v).__name__} = {str(v)[:80]}')
                except Exception:
                    pass
        return None, None

    # ── BFS through interface directories ────────────────────────────────────
    current_parents = {root_config}

    for iface_num in range(1, N_IFACES):
        iface_dir = ic_dir / str(iface_num)
        if not iface_dir.exists():
            print(f'  λ{iface_num} directory not found: {iface_dir}')
            break

        pkl_files = list(iface_dir.glob('*.pkl'))
        print(f'  λ{iface_num}: scanning {len(pkl_files)} pkls …', end=' ', flush=True)

        next_parents = set()
        for pkl_file in pkl_files:
            cfg = _load_pkl(pkl_file)
            if cfg is None:
                continue
            parent_name = getattr(cfg, parent_attr, None)
            if not isinstance(parent_name, str) or parent_name not in current_parents:
                continue

            cname = pkl_file.stem   # filename without .pkl  == config name
            node  = _node_from_pkl(cfg, parent_name, iface_num)
            if node is None:
                continue

            if cname not in nodes:
                nodes[cname] = node
                nodes[parent_name]['children'].add(cname)
            next_parents.add(cname)

        print(f'{len(next_parents)} children found')
        current_parents = next_parents
        if not current_parents:
            print(f'  No children found at λ{iface_num} — stopping.')
            break

    return nodes, root_config


def subtree(nodes, root):
    """All config names reachable from root (BFS)."""
    reachable, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in reachable or n not in nodes:
            continue
        reachable.add(n)
        stack.extend(nodes[n]['children'])
    return reachable


# ── Map setup ──────────────────────────────────────────────────────────────────

def make_atlantic_axes(fig):
    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(
            central_longitude=-60.0,
            central_latitude=35.0,
            standard_parallels=(30, 50),
        )
        ax = fig.add_subplot(111, projection=proj)
        ax.set_extent([-100, -10, 5, 65], crs=ccrs.PlateCarree())
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
        ax.set_xlim(-100, -10)
        ax.set_ylim(5, 65)
        ax.grid(True, alpha=0.3)
    return ax


# ── Drawing ────────────────────────────────────────────────────────────────────

def _draw_tree(ax, nodes, visible):
    """
    Draw the full tree: edges first (lower zorder), then nodes on top.
    Edge colour = child interface level.  Seed drawn as large star.
    """
    pc = ccrs.PlateCarree() if HAS_CARTOPY else None

    # Pass 1 — edges
    for cname in visible:
        node  = nodes.get(cname)
        if node is None:
            continue
        pname = node['parent']
        if pname is None or pname not in nodes:
            continue
        parent = nodes[pname]
        iface  = min(node['iface_idx'], N_IFACES - 1)
        color  = IFACE_COLORS[iface]
        lw     = max(0.5, 1.6 - 0.18 * iface)
        alpha  = max(0.25, 0.70 - 0.07 * iface)
        xs = [parent['lon'], node['lon']]
        ys = [parent['lat'], node['lat']]
        kw = dict(color=color, alpha=alpha, linewidth=lw,
                  solid_capstyle='round', zorder=5)
        if HAS_CARTOPY:
            ax.plot(xs, ys, transform=pc, **kw)
        else:
            ax.plot(xs, ys, **kw)

    # Pass 2 — nodes
    for cname in visible:
        node    = nodes.get(cname)
        if node is None:
            continue
        iface   = min(node['iface_idx'], N_IFACES - 1)
        color   = IFACE_COLORS[iface]
        is_seed = (node['parent'] is None)
        kw = dict(
            s          = 350 if is_seed else 40,
            color      = color,
            marker     = '*' if is_seed else 'o',
            edgecolors = 'k',
            linewidths = 1.0 if is_seed else 0.4,
            alpha      = 0.95,
            zorder     = 8 if is_seed else 6,
        )
        if HAS_CARTOPY:
            ax.scatter(node['lon'], node['lat'], transform=pc, **kw)
        else:
            ax.scatter(node['lon'], node['lat'], **kw)


# ── Main figure ────────────────────────────────────────────────────────────────

def plot_tree(ic_dir: Path, plot_dir: Path, root_config: str = None):
    # Get ic_time from JSON
    jp = ic_dir / 'reactive_trajectories' / 'reactive_trajectories.json'
    ic_time = ic_dir.name
    try:
        with open(jp) as f:
            ic_time = json.load(f).get('ic_time', ic_dir.name)
    except Exception:
        pass
    date_str = str(ic_time).split(' ')[0][:10]

    # Pick root if not given
    if root_config is None:
        root_config, weight = best_root_in_ic(ic_dir)
        if root_config is None:
            print(f'  No reactive trajectories found in {ic_dir}')
            return None
        print(f'  Best root: {root_config}  cluster_weight={weight}')

    print(f'Building shooting tree from {ic_dir.name} ...')
    nodes, root = build_shooting_tree(ic_dir, root_config)

    if nodes is None:
        print('  Tree build failed.')
        return None

    visible = subtree(nodes, root)
    n_per   = [sum(1 for c in visible
                   if c in nodes and nodes[c]['iface_idx'] == i)
               for i in range(N_IFACES)]
    print(f'  Subtree: {len(visible)} nodes   per interface: {n_per}')

    fig = plt.figure(figsize=(12, 10))
    ax  = make_atlantic_axes(fig)
    _draw_tree(ax, nodes, visible)

    # Legend
    handles = [
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=IFACE_COLORS[0], markersize=16,
               markeredgecolor='k', markeredgewidth=0.8,
               label=f'λ₀  {INTERFACE_PRESSURES[0]} hPa  — seed  (n={n_per[0]})'),
    ]
    for i in range(1, N_IFACES):
        handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=IFACE_COLORS[i], markersize=9,
                   markeredgecolor='k', markeredgewidth=0.5,
                   label=f'λ{i}  {INTERFACE_PRESSURES[i]} hPa  (n={n_per[i]})')
        )
    ax.legend(handles=handles, fontsize=9, loc='lower left',
              framealpha=0.92, title='Interface level', title_fontsize=10)

    n_branches_l1 = n_per[1] if len(n_per) > 1 else 0
    ax.set_title(
        f'FFS shooting tree — IC {date_str}\n'
        f'λ₀ seed → {n_branches_l1} λ₁ branches → {len(visible) - 1} total nodes',
        fontsize=12, fontweight='bold',
    )

    out = plot_dir / f'ffs_tree_{date_str}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FFS shooting tree — genealogical tree from a single λ₀ seed'
    )
    parser.add_argument('--ic_dir',     default=None)
    parser.add_argument('--ffs_csv',    default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--plot_dir',   default='./plots')
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.ic_dir:
        plot_tree(Path(args.ic_dir), plot_dir)
        return

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

    print(f'Scanning {len(ic_dirs)} ICs for highest cluster-weight root ...')
    ic_dir, root, weight = find_best_example(ic_dirs)

    if ic_dir is None:
        print('No usable IC found.')
        return

    print(f'Best: {ic_dir.name}  root={root}  cluster_weight={weight}')
    plot_tree(ic_dir, plot_dir, root_config=root)


if __name__ == '__main__':
    main()
