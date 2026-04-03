#!/usr/bin/env python
"""
Identify reactive trajectories in FFS hurricane genesis simulations.

This script processes FFS output to identify genealogically independent reactive
trajectories by detecting branching in the pathway ensemble. For each independent
reactive trajectory, atmospheric state evolution is visualized using existing
PNG snapshots of MSLP fields.

Usage:
    python reactive_pathways.py ffs.yml
    python reactive_pathways.py ffs.yml --workers 16
    python reactive_pathways.py ffs.yml --no_plot          # JSON only, no figures
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from analyze_ffs_logs import (
    load_all_logs,
    build_genealogy,
    trace_pathway,
    find_stateB_configs,
    format_time_for_path
)


def construct_descendant_map(genealogy, stateB_configs):
    """
    Build mapping from each configuration to its state B descendants.
    
    Parameters
    ----------
    genealogy : dict
        Parent-child relationships from FFS shooting
    stateB_configs : list
        Configurations reaching state B threshold
        
    Returns
    -------
    dict
        Mapping {config_name: set of state B descendant configs}
    """
    descendant_map = defaultdict(set)
    
    # Construct reverse lookup for backward tracing
    reverse_lookup = {}
    for parent, children in genealogy.items():
        for child_info in children:
            child_name = child_info['child']
            if child_name is not None:
                reverse_lookup[child_name] = parent
    
    # Trace each B-state config backward to flux generation
    for b_config_info in stateB_configs:
        b_config = b_config_info['config']
        current = b_config
        
        while current in reverse_lookup:
            parent = reverse_lookup[current]
            if parent is None:
                break
            
            descendant_map[parent].add(b_config)
            current = parent
    
    return dict(descendant_map)


def identify_branch_points(genealogy, stateB_configs, min_degree=2):
    """
    Identify configurations generating multiple state B descendants.
    
    Parameters
    ----------
    genealogy : dict
        Parent-child relationships
    stateB_configs : list
        Terminal state B configurations
    min_degree : int
        Minimum descendants to constitute branch point
        
    Returns
    -------
    dict
        {config_name: number of descendants}
    """
    descendant_map = construct_descendant_map(genealogy, stateB_configs)
    
    branch_points = {}
    for config, descendants in descendant_map.items():
        if len(descendants) >= min_degree:
            branch_points[config] = len(descendants)
    
    return branch_points


def find_earliest_branch(pathway, branch_points):
    """
    Locate earliest branching event in pathway.
    
    Parameters
    ----------
    pathway : list
        Ordered configurations from λ₀ to state B
    branch_points : dict
        Configurations with branching degree ≥ 2
        
    Returns
    -------
    tuple
        (branch_config, index) or (None, -1) if no branching
    """
    for idx, step in enumerate(pathway):
        if step['config'] in branch_points:
            return (step['config'], idx)
    return (None, -1)


def cluster_trajectories(stateB_configs, genealogy, branch_points):
    """
    Partition trajectories into genealogically independent clusters.
    
    Trajectories sharing earliest branch point constitute correlated
    family representing single reactive event.
    
    Parameters
    ----------
    stateB_configs : list
        All state B terminal configurations
    genealogy : dict
        Complete genealogy structure
    branch_points : dict
        Identified branching configurations
        
    Returns
    -------
    dict
        {
            'independent': list of unbranched trajectories,
            'clustered': {branch_config: list of correlated trajectories}
        }
    """
    independent = []
    clustered = defaultdict(list)
    
    for config_info in stateB_configs:
        terminal_config = config_info['config']
        pathway = trace_pathway(genealogy, terminal_config)
        
        earliest_branch, branch_idx = find_earliest_branch(pathway, branch_points)
        
        trajectory_info = {
            'terminal_config': terminal_config,
            'pathway': pathway,
            'mslp': config_info['mslp']
        }
        
        if earliest_branch is None:
            independent.append(trajectory_info)
        else:
            clustered[earliest_branch].append(trajectory_info)
    
    return {
        'independent': independent,
        'clustered': dict(clustered)
    }


def select_representative_trajectories(clustered_data, criterion='shortest'):
    """
    Select single representative from each correlated cluster.
    
    Parameters
    ----------
    clustered_data : dict
        Output from cluster_trajectories()
    criterion : str
        Selection method: 'shortest', 'deepest_mslp', 'first'
        
    Returns
    -------
    list
        Complete set of reactive trajectories
    """
    reactive_trajs = []
    
    # All independent trajectories are reactive
    for traj in clustered_data['independent']:
        traj['cluster_size'] = 1
        traj['is_independent'] = True
        reactive_trajs.append(traj)
    
    # Select one representative per cluster
    for branch_config, trajs in clustered_data['clustered'].items():
        if criterion == 'shortest':
            representative = min(trajs, key=lambda t: len(t['pathway']))
        elif criterion == 'deepest_mslp':
            representative = min(trajs, key=lambda t: t['mslp'])
        else:  # first
            representative = trajs[0]
        
        representative['cluster_size'] = len(trajs)
        representative['is_independent'] = False
        representative['branch_point'] = branch_config
        reactive_trajs.append(representative)
    
    return reactive_trajs


def visualize_reactive_trajectory(trajectory, ic_dir, output_path, traj_id):
    """
    Create multi-panel figure showing atmospheric evolution along reactive pathway.
    
    Each panel displays PNG snapshot of MSLP field at successive interface crossings,
    illustrating physical hurricane genesis process from initial disturbance to
    mature tropical cyclone.
    
    Parameters
    ----------
    trajectory : dict
        Reactive trajectory containing pathway information
    ic_dir : Path
        Initial condition directory containing configuration PNGs
    output_path : Path
        Output file path for figure
    traj_id : int
        Trajectory identifier for labeling
    """
    pathway = trajectory['pathway']
    n_steps = len(pathway)

    # Configure figure dimensions
    import math

    n_rows = 2
    n_cols = max(1, math.ceil(n_steps / n_rows))

    fig_width = min(6 * n_cols, 36)
    fig_height = 6 * n_rows

    # squeeze=False ensures we always get a 2-D array regardless of grid shape,
    # so axes.flatten() is always safe and axes[i] is always a single Axes.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                             squeeze=False)
    axes = axes.flatten()

    # Hide any unused panels at the end of the grid
    for j in range(n_steps, len(axes)):
        axes[j].axis('off')
    
    for i, step in enumerate(pathway):
        config_name = step['config']
        
        if config_name is None:
            axes[i].text(0.5, 0.5, "Missing\nConfiguration", 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, color='red', fontweight='bold')
            axes[i].axis('off')
            continue
        
        # Locate PNG in hierarchical directory structure
        if config_name.startswith('lambda0_'):
            png_path = ic_dir / 'flux' / f"{config_name}.png"
        elif config_name.startswith('stateB_'):
            png_path = ic_dir / 'stateB' / f"{config_name}.png"
        else:
            # Extract interface index from naming convention
            interface_num = config_name.split('_')[0].replace('lambda', '')
            png_path = ic_dir / interface_num / f"{config_name}.png"
        
        if png_path.exists():
            img = mpimg.imread(png_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Construct informative title
            if i == 0:
                title = f"λ₀ (Flux Generation)\n{config_name}"
            elif step.get('interface') == 'B':
                mslp = trajectory['mslp']
                title = f"State B\n{config_name}\nMSLP = {mslp:.1f} hPa"
            else:
                lambda_label = step.get('lambda_label', '?')
                mslp = step.get('mslp_value', 0.0)
                title = f"λ_{lambda_label}\n{config_name}\nMSLP = {mslp:.1f} hPa"
            
            axes[i].set_title(title, fontsize=9, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f"PNG not found:\n{config_name}", 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=10, color='orange')
            axes[i].axis('off')
    
    # Figure title with clustering information
    cluster_info = ""
    if not trajectory['is_independent']:
        cluster_info = f" (Representative of {trajectory['cluster_size']} correlated trajectories)"
    
    fig.suptitle(f"Reactive Trajectory {traj_id}{cluster_info}", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(reactive_trajectories, clustered_data, branch_points,
                           total_stateB, output_path):
    """
    Generate text summary of reactive trajectory analysis.
    
    Parameters
    ----------
    reactive_trajectories : list
        All identified reactive trajectories
    clustered_data : dict
        Clustering results
    branch_points : dict
        Branch point configurations
    total_stateB : int
        Total trajectories reaching state B
    output_path : Path
        Output file for summary text
    """
    n_independent = len(clustered_data['independent'])
    n_clusters = len(clustered_data['clustered'])
    n_reactive = len(reactive_trajectories)
    correlation_factor = total_stateB / n_reactive if n_reactive > 0 else 1.0
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REACTIVE TRAJECTORY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total trajectories reaching state B: {total_stateB}\n")
        f.write(f"Reactive (independent) trajectories: {n_reactive}\n")
        f.write(f"  - Unbranched trajectories: {n_independent}\n")
        f.write(f"  - Clustered families: {n_clusters}\n")
        f.write(f"Correlation factor: {correlation_factor:.3f}\n\n")
        
        f.write(f"Total branch points identified: {len(branch_points)}\n\n")
        
        if branch_points:
            f.write("Top 10 branch points by descendant count:\n")
            sorted_branches = sorted(branch_points.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
            for i, (config, degree) in enumerate(sorted_branches, 1):
                f.write(f"  {i}. {config}: {degree} descendants\n")
            f.write("\n")
        
        f.write("REACTIVE TRAJECTORY DETAILS\n")
        f.write("-"*80 + "\n\n")
        
        for i, traj in enumerate(reactive_trajectories, 1):
            f.write(f"Trajectory {i}:\n")
            f.write(f"  Terminal config: {traj['terminal_config']}\n")
            f.write(f"  Final MSLP: {traj['mslp']:.1f} hPa\n")
            f.write(f"  Pathway length: {len(traj['pathway'])} interfaces\n")
            f.write(f"  Cluster size: {traj['cluster_size']}")
            if not traj['is_independent']:
                f.write(f" (branched from {traj['branch_point']})")
            f.write("\n\n")


def _visualize_worker(args: tuple):
    """Parallel worker — renders one reactive trajectory figure."""
    traj, ic_dir, output_path, traj_id = args
    visualize_reactive_trajectory(traj, ic_dir, output_path, traj_id)


def main():
    parser = argparse.ArgumentParser(
        description='Identify and visualize reactive trajectories in FFS simulations'
    )
    parser.add_argument('config_file', type=str, help='FFS config file (ffs.yml)')
    parser.add_argument('--min-branch-degree', type=int, default=2,
                       help='Minimum branching degree (default: 2)')
    parser.add_argument('--selection', type=str, default='shortest',
                       choices=['shortest', 'deepest_mslp', 'first'],
                       help='Cluster representative selection criterion')
    parser.add_argument('--workers', type=int, default=min(8, cpu_count()),
                       help='Parallel workers for figure rendering (default: min(8, ncpus))')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip figure creation — only compute trajectories and export JSON')

    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config_file, 'r') as f:
        ffs_config = yaml.safe_load(f)
    
    output_dir = Path(ffs_config['output_dir'])
    state_B = ffs_config['state_B']
    forecast_times = ffs_config['forecast_start_times']

    interfaces = ffs_config['interfaces'].copy()
    if interfaces[-1] != state_B:
        interfaces.append(state_B)

    # Process each initial condition
    for ic_time in forecast_times:
        time_label = format_time_for_path(ic_time)
        
        print("="*80)
        print(f"Processing IC: {time_label}")
        print("="*80)
        
        ic_dir = output_dir / time_label
        logs_dir = ic_dir / 'logs'
        
        if not ic_dir.exists():
            print(f"WARNING: IC directory not found: {ic_dir}")
            continue
        
        # Create reactive trajectory directory INSIDE IC directory
        reactive_dir = ic_dir / 'reactive_trajectories'
        reactive_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading logs from {logs_dir}")
        entries = load_all_logs(logs_dir)
        print(f"Loaded {len(entries)} log entries")
        
        genealogy = build_genealogy(entries)
        print(f"Built genealogy: {len(genealogy)} parent configurations")
        
        stateB_configs = find_stateB_configs(entries, state_B)
        print(f"Found {len(stateB_configs)} trajectories reaching state B")
        
        if not stateB_configs:
            print("No successful trajectories found for this IC")
            continue
        
        # Identify branch points
        print(f"\nIdentifying branch points (min degree = {args.min_branch_degree})")
        branch_points = identify_branch_points(
            genealogy, stateB_configs, min_degree=args.min_branch_degree
        )
        print(f"Identified {len(branch_points)} branch points")
        
        if branch_points:
            sorted_branches = sorted(branch_points.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            print("\nTop 5 branch points:")
            for config, degree in sorted_branches:
                print(f"  {config}: {degree} descendants")
        
        # Cluster trajectories
        print("\nClustering correlated trajectories...")
        clustered_data = cluster_trajectories(stateB_configs, genealogy, branch_points)
        
        n_independent = len(clustered_data['independent'])
        n_clusters = len(clustered_data['clustered'])
        n_total = len(stateB_configs)
        n_reactive = n_independent + n_clusters
        
        print(f"  Independent trajectories: {n_independent}")
        print(f"  Clustered families: {n_clusters}")
        print(f"  Total reactive trajectories: {n_reactive}")
        print(f"  Correlation factor: {n_total / n_reactive:.3f}")
        
        # Select representatives
        print(f"\nSelecting representative trajectories (criterion: {args.selection})")
        reactive_trajectories = select_representative_trajectories(
            clustered_data, criterion=args.selection
        )
        
        # Generate visualizations for each reactive trajectory
        if args.no_plot:
            print("\n(visualizations skipped — --no_plot flag set)")
        else:
            n_vis = len(reactive_trajectories)
            print(f"\nGenerating {n_vis} trajectory visualizations "
                  f"({args.workers} workers)...")
            worker_args = [
                (traj, ic_dir,
                 reactive_dir / f"reactive_trajectory_{i:03d}.png", i)
                for i, traj in enumerate(reactive_trajectories, 1)
            ]
            with Pool(processes=min(args.workers, n_vis)) as pool:
                list(tqdm(
                    pool.imap_unordered(_visualize_worker, worker_args),
                    total=n_vis, desc='figures', unit='fig',
                    dynamic_ncols=True,
                ))
            print(f"All visualizations saved to: {reactive_dir}")
        
        # Generate summary report
        summary_path = reactive_dir / 'summary.txt'
        generate_summary_report(
            reactive_trajectories, clustered_data, branch_points,
            len(stateB_configs), summary_path
        )
        print(f"Summary report saved: {summary_path}")
        
        # Export detailed JSON
        json_path = reactive_dir / 'reactive_trajectories.json'
        export_data = {
            'ic_time': ic_time,
            'time_label': time_label,
            'total_stateB_trajectories': len(stateB_configs),
            'n_reactive_trajectories': n_reactive,
            'correlation_factor': n_total / n_reactive if n_reactive > 0 else 1.0,
            'branch_points': {k: v for k, v in branch_points.items()},
            'reactive_trajectories': [
                {
                    'trajectory_id': i,
                    'terminal_config': traj['terminal_config'],
                    'final_mslp': traj['mslp'],
                    'pathway_length': len(traj['pathway']),
                    'cluster_size': traj['cluster_size'],
                    'is_independent': traj['is_independent'],
                    'pathway': [step['config'] for step in traj['pathway']]
                }
                for i, traj in enumerate(reactive_trajectories, 1)
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"JSON data exported: {json_path}\n")


if __name__ == "__main__":
    main()