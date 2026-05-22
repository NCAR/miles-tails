#!/usr/bin/env python
"""
Analyze FFS logs to trace pathways from flux generation to any target interface.

This script operates on the hierarchical FFS output structure where each initial
condition has its own directory containing flux/, interface subdirectories, and logs/.

Usage:
    # Trace pathways to state B
    python analyze_ffs_logs.py ffs.yml --trace_all
    
    # Trace pathways to specific interface (e.g., λ₁)
    python analyze_ffs_logs.py ffs.yml --target_interface 1
    
    # Trace pathways to λ₂
    python analyze_ffs_logs.py ffs.yml --target_interface 2
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tails.ffs_logger import FFSLogger


class _StubUnpickler(pickle.Unpickler):
    """Fast unpickler that stubs out torch tensors so we skip loading large arrays."""
    def find_class(self, module, name):
        if 'torch' in module or name in ('Tensor', 'storage', 'LongStorage',
                                          '_rebuild_tensor_v2', 'FloatStorage'):
            return lambda *args, **kwargs: None
        return super().find_class(module, name)


def _load_stateB_location(pkl_path: Path):
    """Return (lat, lon) of a stateB event without loading the full tensor."""
    try:
        with open(pkl_path, 'rb') as f:
            obj = _StubUnpickler(f).load()
        return getattr(obj, 'feature_location', None)
    except Exception:
        return None


def _stateB_in_basin(config_name: str, stateB_dir: Path, basin: dict) -> bool:
    """Return True if the stateB pkl's feature_location is within the basin."""
    pkl = stateB_dir / f'{config_name}.pkl'
    if not pkl.exists():
        return True  # can't verify — don't exclude
    loc = _load_stateB_location(pkl)
    if loc is None:
        return True
    lat, lon = loc
    return (basin['lat_min'] <= lat <= basin['lat_max'] and
            basin['lon_min'] <= lon <= basin['lon_max'])


def format_time_for_path(time_str: str) -> str:
    """Convert '2022-08-21 00:00:00' to '2022-08-21T00Z' format."""
    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    return dt.strftime('%Y-%m-%dT%HZ')


def load_flux_configs(flux_dir: Path) -> list:
    """Load all saved flux configs and extract crossing timesteps."""
    configs = []
    config_files = sorted(flux_dir.glob('lambda0_config_*.pkl'))
    
    for config_file in config_files:
        with open(config_file, 'rb') as f:
            config = pickle.load(f)
            configs.append({
                'name': config.config_name,
                'forecast_step': config.forecast_step,
                'mslp': config.mslp_value
            })
    
    return configs


def compute_flux_from_configs(flux_configs: list) -> dict:
    """Compute flux from saved configs."""
    if not flux_configs:
        return {'crossings': 0, 'forecast_steps': 0, 'flux': 0.0, 'total_time_days': 0.0}
    
    last_forecast_step = max(c['forecast_step'] for c in flux_configs)
    n_crossings = len(flux_configs)
    
    total_time_days = last_forecast_step * 6.0 / 24.0
    flux = n_crossings / total_time_days if total_time_days > 0 else 0.0
    
    return {
        'crossings': n_crossings,
        'forecast_steps': last_forecast_step,
        'total_time_days': total_time_days,
        'flux': flux
    }


def compute_flux_from_logs(entries: list, stateB_dir: Path = None,
                           basin: dict = None) -> dict:
    """Compute flux from flux generation log entries.

    If stateB_dir and basin are provided, only stateB events whose pkl
    feature_location falls within the basin are counted toward the direct
    formation rate (out-of-basin deepening events are physically spurious).
    """
    total_crossings = 0
    total_timesteps = 0
    n_trajectories = 0
    direct_B_formations = 0
    direct_B_out_of_basin = 0

    for entry in entries:
        if entry.get('phase') != 'flux_generation':
            continue

        n_trajectories += 1
        configs = entry.get('configs_saved', [])
        # Only count λ₀ crossings for flux rate (not stateB configs)
        total_crossings += sum(1 for c in configs if c.startswith('lambda0_'))

        # Count B-events; optionally filter to in-basin only
        for c in configs:
            if not c.startswith('stateB_'):
                continue
            if stateB_dir is not None and basin is not None:
                if _stateB_in_basin(c, stateB_dir, basin):
                    direct_B_formations += 1
                else:
                    direct_B_out_of_basin += 1
            else:
                direct_B_formations += 1

        total_timesteps += entry.get('num_timesteps', 0)

    total_time_days = total_timesteps * 6.0 / 24.0
    flux = total_crossings / total_time_days if total_time_days > 0 else 0.0
    direct_rate = direct_B_formations / total_time_days if total_time_days > 0 else 0.0

    return {
        'trajectories': n_trajectories,
        'crossings': total_crossings,
        'direct_B_formations': direct_B_formations,
        'direct_B_out_of_basin': direct_B_out_of_basin,
        'total_timesteps': total_timesteps,
        'total_time_days': total_time_days,
        'flux': flux,
        'direct_rate': direct_rate
    }


def load_all_logs(log_dir: Path) -> list:
    """Load and merge all log files from all workers and ranks within IC directory."""
    all_entries = []
    
    # Look for flux logs
    flux_log_pattern = 'flux/ffs_log_world*_rank*_worker*.jsonl'
    flux_log_files = sorted(log_dir.glob(flux_log_pattern))
    
    # Look for shooting logs (in numbered subdirectories)
    shooting_log_files = []
    for interface_dir in log_dir.iterdir():
        if interface_dir.is_dir() and interface_dir.name.isdigit():
            pattern = f"{interface_dir.name}/ffs_log_world*_rank*_worker*.jsonl"
            shooting_log_files.extend(sorted(log_dir.glob(pattern)))
    
    all_log_files = flux_log_files + shooting_log_files
    
    for log_file in all_log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    all_entries.sort(key=lambda x: x.get('timestamp', ''))
    
    return all_entries


def build_genealogy(entries: list) -> dict:
    """Build parent -> children mapping from shooting attempts."""
    genealogy = defaultdict(list)
    skipped_none = 0
    
    for entry in entries:
        if entry.get('phase') != 'shooting':
            continue
        
        parent = entry.get('parent_config')
        child = entry.get('child_config')
        
        if child is None:
            skipped_none += 1
            continue
        
        if parent and child:
            genealogy[parent].append({
                'child': child,
                'interface_idx': entry.get('interface_idx'),
                'lambda_label': entry.get('lambda_label'),
                'status': entry.get('status'),
                'mslp_value': entry.get('final_mslp'),
                'timestamp': entry.get('timestamp'),
                'worker_id': entry.get('worker_id'),
                'rank': entry.get('rank')
            })
    
    return dict(genealogy)


def trace_pathway(genealogy: dict, target_config: str) -> list:
    """Trace pathway from flux generation to target config."""
    reverse_lookup = {}
    for parent, children in genealogy.items():
        for child_info in children:
            reverse_lookup[child_info['child']] = {
                'parent': parent,
                'interface_idx': child_info['interface_idx'],
                'lambda_label': child_info['lambda_label'],
                'status': child_info['status'],
                'mslp_value': child_info['mslp_value']
            }
    
    pathway = [{'config': target_config, 'interface': 'B'}]
    current = target_config
    
    while current in reverse_lookup:
        parent_info = reverse_lookup[current]
        parent = parent_info['parent']
        
        if parent is None:
            import logging
            logging.warning(f"Encountered None parent for config {current} - pathway may be incomplete")
            break
        
        pathway.append({
            'config': parent,
            'interface_idx': parent_info['interface_idx'],
            'lambda_label': parent_info['lambda_label'],
            'mslp_value': parent_info['mslp_value']
        })
        
        current = parent
    
    return list(reversed(pathway))


def compute_statistics(entries: list) -> dict:
    """Compute FFS statistics from logs."""
    stats = {
        'interface_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0, 'failures': 0})
    }
    
    for entry in entries:
        phase = entry.get('phase')
        
        if phase == 'shooting':
            interface_idx = entry.get('interface_idx')
            status = entry.get('status')
            
            if interface_idx is not None and status:
                interface_stats = stats['interface_stats'][interface_idx]
                interface_stats['attempts'] += 1
                
                if status in ['success', 'reached_B', 'instant_success']:
                    interface_stats['successes'] += 1
                elif status == 'failure':
                    interface_stats['failures'] += 1
    
    return stats


def find_stateB_configs(entries: list, state_B: float) -> list:
    """Find all configs that reached state B (from both shooting and flux phases)."""
    stateB_configs = []
    broken_entries = 0

    for entry in entries:
        if entry.get('phase') == 'shooting':
            child = entry.get('child_config')
            status = entry.get('status')
            final_mslp = entry.get('final_mslp')

            if status in ['success', 'reached_B', 'instant_success'] and final_mslp is not None:
                if final_mslp <= state_B:
                    if child is None:
                        broken_entries += 1
                        continue

                    stateB_configs.append({
                        'config': child,
                        'parent': entry.get('parent_config'),
                        'mslp': final_mslp,
                        'timestamp': entry.get('timestamp')
                    })

        # Also check flux entries for B-state crossings (storms that reached B during flux mode)
        elif entry.get('phase') == 'flux_generation':
            configs_saved = entry.get('configs_saved', [])
            for config_name in configs_saved:
                if config_name.startswith('stateB_'):
                    stateB_configs.append({
                        'config': config_name,
                        'parent': None,  # Direct flux B-event, no shooting parent
                        'mslp': entry.get('final_mslp'),
                        'timestamp': entry.get('timestamp')
                    })

    if broken_entries > 0:
        import logging
        logging.warning(f"Filtered out {broken_entries} broken state B entries with status='success' but child_config=None")

    return stateB_configs


def find_interface_configs(entries: list, target_interface_idx: int) -> list:
    """Find all configs that successfully reached a specific interface."""
    interface_configs = []
    broken_entries = 0
    
    for entry in entries:
        if entry.get('phase') == 'shooting':
            child = entry.get('child_config')
            status = entry.get('status')
            interface_idx = entry.get('interface_idx')
            final_mslp = entry.get('final_mslp')
            
            if (status in ['success', 'reached_B', 'instant_success'] and 
                interface_idx == target_interface_idx and 
                final_mslp is not None):
                
                if child is None:
                    broken_entries += 1
                    continue
                
                interface_configs.append({
                    'config': child,
                    'parent': entry.get('parent_config'),
                    'interface_idx': interface_idx,
                    'lambda_label': entry.get('lambda_label'),
                    'mslp': final_mslp,
                    'timestamp': entry.get('timestamp')
                })
    
    if broken_entries > 0:
        import logging
        logging.warning(f"Filtered out {broken_entries} broken entries with status='success' but child_config=None")
    
    return interface_configs


def plot_pathway_images(pathway: list, ic_dir: Path, output_path: Path, target_label: str = 'B'):
    """Create a figure showing PNG images for each step in the pathway (2-row layout)."""
    import math
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n_steps = len(pathway)

    # ---- force 2-row layout ----
    n_rows = 2
    n_cols = math.ceil(n_steps / n_rows)

    fig_width = min(6 * n_cols, 36)
    fig_height = 6 * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height)
    )

    axes = axes.flatten()

    for i, step in enumerate(pathway):
        ax = axes[i]
        config_name = step['config']

        if config_name is None:
            ax.text(
                0.5, 0.5,
                "Incomplete pathway\n(parent config missing)",
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12, color='red'
            )
            ax.axis('off')
            continue

        # Hierarchical directory structure
        if config_name.startswith('lambda0_'):
            png_path = ic_dir / 'flux' / f"{config_name}.png"
        elif config_name.startswith('stateB_'):
            png_path = ic_dir / 'stateB' / f"{config_name}.png"
        else:
            interface_num = config_name.split('_')[0].replace('lambda', '')
            png_path = ic_dir / interface_num / f"{config_name}.png"

        if png_path.exists():
            img = mpimg.imread(png_path)
            ax.imshow(img)
            ax.set_aspect('equal')
            ax.axis('off')

            if i == 0:
                title = f"λ₀\n{config_name}"
            elif step.get('is_target', False):
                title = f"{target_label}\n{config_name}"
                if step.get('mslp_value') is not None:
                    title += f"\n{step['mslp_value']:.1f} hPa"
            elif step.get('interface') == 'B':
                title = f"STATE B\n{config_name}"
            else:
                lambda_label = step['lambda_label']
                mslp = step['mslp_value']
                title = f"λ_{lambda_label}\n{config_name}\n{mslp:.1f} hPa"

            ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

        else:
            ax.text(
                0.5, 0.5,
                f"Image not found:\n{config_name}",
                ha='center', va='center',
                transform=ax.transAxes
            )
            ax.axis('off')

    # ---- hide unused panels ----
    for j in range(n_steps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path

def save_statistics_to_csv(ic_time, time_label, flux_stats, stats, transition_probs, 
                           stateB_configs, total_ffs_sim_days, shooting_time_days,
                           interfaces, output_path):
    """Save all FFS statistics to CSV file."""
    
    # Main summary row
    summary = {
        'ic_time': ic_time,
        'time_label': time_label,
        
        # Flux statistics
        'flux_trajectories': flux_stats['trajectories'],
        'flux_lambda0_crossings': flux_stats['crossings'],
        'flux_direct_B_formations': flux_stats['direct_B_formations'],
        'flux_direct_B_out_of_basin': flux_stats.get('direct_B_out_of_basin', 0),
        'flux_total_timesteps': flux_stats['total_timesteps'],
        'flux_total_time_days': flux_stats['total_time_days'],
        'flux_rate_per_day': flux_stats['flux'],
        'direct_formation_rate_per_day': flux_stats['direct_rate'],
        
        # Shooting statistics
        'shooting_time_days': shooting_time_days,
        'total_ffs_sim_days': total_ffs_sim_days,
        'n_stateB_arrivals': len(stateB_configs),
        
        # FFS rate calculation
        'ffs_rate_per_day': flux_stats['flux'] * np.prod(transition_probs) if transition_probs else 0.0,
    }
    
    # Add per-interface shooting stats
    for interface_idx in sorted(stats['interface_stats'].keys()):
        interface_stats = stats['interface_stats'][interface_idx]
        lambda_label = interface_idx

        summary[f'lambda{lambda_label}_attempts'] = interface_stats['attempts']
        summary[f'lambda{lambda_label}_successes'] = interface_stats['successes']
        summary[f'lambda{lambda_label}_failures'] = interface_stats['failures']
        summary[f'lambda{lambda_label}_extratropical'] = interface_stats['attempts'] - interface_stats['successes'] - interface_stats['failures']
        
        if interface_stats['attempts'] > 0:
            P = interface_stats['successes'] / interface_stats['attempts']
            summary[f'lambda{lambda_label}_P_forward'] = P
        else:
            summary[f'lambda{lambda_label}_P_forward'] = np.nan
    
    # Efficiency metrics
    if len(stateB_configs) > 0:
        summary['cost_per_event_days'] = total_ffs_sim_days / len(stateB_configs)
    else:
        summary['cost_per_event_days'] = np.nan
    
    if flux_stats['direct_rate'] > 0:
        bf_days_per_event = 1.0 / flux_stats['direct_rate']
        bf_total_for_n_events = bf_days_per_event * len(stateB_configs)
        
        forecast_length_days = 15.0
        ensemble_members_per_event = bf_days_per_event / forecast_length_days
        total_ensemble_members = ensemble_members_per_event * len(stateB_configs)
        
        speedup = bf_total_for_n_events / total_ffs_sim_days if total_ffs_sim_days > 0 else np.nan
        
        summary['bf_days_per_event'] = bf_days_per_event
        summary['bf_total_cost_days'] = bf_total_for_n_events
        summary['bf_ensemble_members_per_event'] = ensemble_members_per_event
        summary['bf_total_ensemble_members'] = total_ensemble_members
        summary['ffs_speedup'] = speedup
    else:
        # Lower bounds
        min_bf_total = flux_stats['total_time_days'] * len(stateB_configs)
        forecast_length_days = 15.0
        min_ensemble_per_event = flux_stats['total_time_days'] / forecast_length_days
        min_speedup = min_bf_total / total_ffs_sim_days if total_ffs_sim_days > 0 else np.nan
        
        summary['bf_days_per_event'] = f">{flux_stats['total_time_days']:.0f}"
        summary['bf_total_cost_days'] = f">{min_bf_total:.0f}"
        summary['bf_ensemble_members_per_event'] = f">{min_ensemble_per_event:.0f}"
        summary['bf_total_ensemble_members'] = f">{min_ensemble_per_event * len(stateB_configs):.0f}"
        summary['ffs_speedup'] = f">{min_speedup:.1f}"
    
    # Convert to DataFrame and save
    df = pd.DataFrame([summary])
    df.to_csv(output_path, index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze FFS logs')
    parser.add_argument('config_file', type=str, help='FFS config file (ffs.yml)')
    parser.add_argument('--trace_all', action='store_true',
                       help='Trace all pathways to state B')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip pathway PNG figure generation (only write statistics CSV)')
    parser.add_argument('--target_interface', type=int, default=None,
                       help='Trace pathways to specific interface (e.g., 1 for λ₁, 2 for λ₂)')
    
    args = parser.parse_args()
    
    import yaml
    with open(args.config_file, 'r') as f:
        ffs_config = yaml.safe_load(f)
    
    output_dir = Path(ffs_config['output_dir'])
    state_A = ffs_config['state_A']
    state_B = ffs_config['state_B']
    forecast_times = ffs_config['forecast_start_times']
    basin = ffs_config.get('basin', {'lat_min': 10.0, 'lat_max': 45.0,
                                      'lon_min': -98.0, 'lon_max': -20.0})

    interfaces = ffs_config['interfaces'].copy()
    if interfaces[-1] != state_B:
        interfaces.append(state_B)

    lambda0 = interfaces[0]

    # Collect all IC dataframes
    all_ic_dfs = []

    # Process each initial condition
    for ic_time in forecast_times:
        time_label = format_time_for_path(ic_time)
        
        # Hierarchical structure: output_dir/IC_TIME/
        ic_dir = output_dir / time_label
        flux_dir = ic_dir / 'flux'
        logs_dir = ic_dir / 'logs'
        
        if not ic_dir.exists():
            print(f"WARNING: IC directory not found: {ic_dir}")
            continue
        
        # Initialize logger for this IC analysis
        analysis_log_dir = ic_dir / 'analysis_logs'
        analysis_log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = FFSLogger(
            output_dir=analysis_log_dir,
            worker_id='analyzer',
            rank=0,
            world_size=1,
            ic_dirname=None  # Already in IC-specific directory
        )
        
        logger.log_phase_start('analysis', ic_time=ic_time, time_label=time_label)
        logger.info(f"Analyzing FFS results for IC: {time_label}")
        logger.info(f"IC directory: {ic_dir}")
        
        # Load logs from hierarchical structure
        logger.info(f"Loading log files from {logs_dir}")
        entries = load_all_logs(logs_dir)
        logger.info(f"Loaded {len(entries)} total log entries")
        
        stateB_dir = ic_dir / 'stateB'
        flux_stats = compute_flux_from_logs(entries, stateB_dir=stateB_dir, basin=basin)

        logger.info("="*80)
        logger.info("FLUX GENERATION STATISTICS")
        logger.info("="*80)
        logger.info(f"Trajectories run: {flux_stats['trajectories']}")
        logger.info(f"λ₀ crossings: {flux_stats['crossings']}")
        logger.info(f"Direct B formations (in-basin): {flux_stats['direct_B_formations']}")
        logger.info(f"Direct B formations (out-of-basin, excluded): {flux_stats['direct_B_out_of_basin']}")
        logger.info(f"Total timesteps: {flux_stats['total_timesteps']}")
        logger.info(f"Total time: {flux_stats['total_time_days']:.1f} days")
        logger.info(f"FFS Flux Rate: Φ₀ = {flux_stats['flux']:.6f} crossings/day")
        logger.info(f"Direct Formation Rate: Φ_direct = {flux_stats['direct_rate']:.6e} formations/day")
        
        if flux_stats['direct_rate'] > 0 and flux_stats['flux'] > 0:
            ratio = flux_stats['flux'] / flux_stats['direct_rate']
            logger.info(f"FFS flux / Direct rate = {ratio:.2f}x")
        
        genealogy = build_genealogy(entries)
        logger.info(f"Built genealogy: {len(genealogy)} parent configs, "
                   f"{sum(len(children) for children in genealogy.values())} total children")
        
        stateB_configs = find_stateB_configs(entries, state_B)
        logger.info(f"Found {len(stateB_configs)} configs that reached state B:")
        # for config_info in stateB_configs:
        #     logger.info(f"  {config_info['config']} (MSLP={config_info['mslp']:.1f} hPa)")
        
        if args.trace_all and stateB_configs:
            logger.info("="*80)
            logger.info("TRACING PATHWAYS TO STATE B")
            logger.info("="*80)

            pathway_figs_dir = ic_dir / 'pathway_figures' / 'stateB'
            pathway_figs_dir.mkdir(parents=True, exist_ok=True)
            
            for i, config_info in enumerate(stateB_configs):
                logger.info(f"Pathway {i+1}/{len(stateB_configs)}:")
                pathway = trace_pathway(genealogy, config_info['config'])
                
                for j, step in enumerate(pathway):
                    step_config = step['config']
                    if j == 0:
                        logger.info(f"  {j}. {step_config} (λ₀ - flux generation)")
                    elif step.get('interface') == 'B':
                        logger.info(f"  {j}. {step_config} (STATE B)")
                    else:
                        lambda_label = step['lambda_label']
                        mslp = step['mslp_value']
                        logger.info(f"  {j}. {step_config} (λ_{lambda_label}, MSLP={mslp:.1f} hPa)")

                if not args.no_plot:
                    fig_path = pathway_figs_dir / f"pathway_{i+1:03d}.png"
                    plot_pathway_images(pathway, ic_dir, fig_path, target_label='STATE B')
                    logger.info(f"  Pathway figure saved: {fig_path}")
        
        if args.target_interface is not None:
            target_idx = args.target_interface
            target_lambda_label = target_idx - 1
            
            if target_idx < 1 or target_idx >= len(interfaces):
                logger.error(f"Invalid target_interface={target_idx}. Valid range: 1 to {len(interfaces)-1}")
                continue
            
            logger.info("="*80)
            logger.info(f"TRACING PATHWAYS TO λ_{target_lambda_label} (interface_idx={target_idx})")
            logger.info("="*80)
            
            target_configs = find_interface_configs(entries, target_idx)
            logger.info(f"Found {len(target_configs)} configs that reached λ_{target_lambda_label}:")
            for config_info in target_configs:
                logger.info(f"  {config_info['config']} (MSLP={config_info['mslp']:.1f} hPa)")
            
            if target_configs:
                pathway_figs_dir = ic_dir / 'pathway_figures' / f'lambda{target_lambda_label}'
                pathway_figs_dir.mkdir(parents=True, exist_ok=True)
                
                for i, config_info in enumerate(target_configs):
                    logger.info(f"Pathway {i+1}/{len(target_configs)}:")
                    pathway = trace_pathway(genealogy, config_info['config'])
                    
                    if len(pathway) > 0:
                        pathway[-1]['is_target'] = True
                        pathway[-1]['mslp_value'] = config_info['mslp']
                    
                    for j, step in enumerate(pathway):
                        step_config = step['config']
                        if j == 0:
                            logger.info(f"  {j}. {step_config} (λ₀ - flux generation)")
                        elif step.get('is_target', False):
                            logger.info(f"  {j}. {step_config} (λ_{target_lambda_label} - TARGET)")
                        else:
                            lambda_label = step.get('lambda_label', '?')
                            mslp = step.get('mslp_value', 0.0)
                            logger.info(f"  {j}. {step_config} (λ_{lambda_label}, MSLP={mslp:.1f} hPa)")
                    
                    if not args.no_plot:
                        fig_path = pathway_figs_dir / f"pathway_{i+1:03d}.png"
                        plot_pathway_images(pathway, ic_dir, fig_path,
                                          target_label=f'λ_{target_lambda_label}')
                        logger.info(f"  Pathway figure saved: {fig_path}")
        
        stats = compute_statistics(entries)

        logger.info("="*80)
        logger.info("SHOOTING STATISTICS")
        logger.info("="*80)

        expected_interfaces = list(range(1, len(interfaces)))
        actual_interfaces = sorted(stats['interface_stats'].keys())

        valid_interfaces = [idx for idx in actual_interfaces if idx in expected_interfaces]
        invalid_interfaces = [idx for idx in actual_interfaces if idx not in expected_interfaces]

        if invalid_interfaces:
            logger.warning(f"Ignoring unexpected shooting data for interfaces: {invalid_interfaces}")
            logger.warning("(These may be from incomplete runs or errors)")

        missing = set(expected_interfaces) - set(actual_interfaces)
        if missing:
            logger.warning(f"Missing shooting data for interfaces: {sorted(missing)}")
            logger.warning("FFS run may be incomplete!")

        transition_probs = []
        
        logger.info("Shooting phases:")
        for interface_idx in valid_interfaces:
            interface_stats = stats['interface_stats'][interface_idx]
            lambda_label = interface_idx - 1
            next_lambda_label = interface_idx
            
            lambda_mslp = interfaces[lambda_label]
            next_mslp = interfaces[next_lambda_label]
            
            attempts = interface_stats['attempts']
            successes = interface_stats['successes']
            failures = interface_stats['failures']
            
            if attempts > 0:
                P_forward = successes / attempts
                transition_probs.append(P_forward)
                
                if interface_idx == len(interfaces) - 1:
                    logger.info(f"  λ_{lambda_label} ({lambda_mslp} hPa) → STATE B ({next_mslp} hPa):")
                else:
                    logger.info(f"  λ_{lambda_label} ({lambda_mslp} hPa) → λ_{next_lambda_label} ({next_mslp} hPa):")
                    
                logger.info(f"    Attempts: {attempts}, Successes: {successes}, Failures: {failures}")
                logger.info(f"    P_forward: {P_forward:.4f}")

        ffs_rate = flux_stats['flux']
        for p in transition_probs:
            ffs_rate *= p

        # SPEEDUP CALCULATION
        # After computing transition probabilities and stats

        # Count actual B-state arrivals
        n_stateB_arrivals = len(stateB_configs)

        # Compute total FFS computational cost
        total_ffs_sim_days = flux_stats['total_time_days']  # Flux generation time

        # Add shooting time - need to extract from logs
        shooting_time_days = 0.0
        for entry in entries:
            if entry.get('phase') == 'shooting':
                shooting_time_days += entry.get('num_timesteps', 0) * 6.0 / 24.0

        total_ffs_sim_days += shooting_time_days

        logger.info("="*80)
        logger.info("FFS EFFICIENCY vs BRUTE FORCE")
        logger.info("="*80)
        logger.info(f"\nFFS Results:")
        logger.info(f"  State B arrivals: {n_stateB_arrivals}")
        logger.info(f"  Total simulation time: {total_ffs_sim_days:.1f} days")
        logger.info(f"    Flux generation: {flux_stats['total_time_days']:.1f} days")
        logger.info(f"    Shooting phases: {shooting_time_days:.1f} days")
        if n_stateB_arrivals > 0:
            logger.info(f"  Cost per event: {total_ffs_sim_days / n_stateB_arrivals:.1f} simulation-days")

        if flux_stats['direct_rate'] > 0:
            bf_days_per_event = 1.0 / flux_stats['direct_rate']
            bf_total_for_n_events = bf_days_per_event * n_stateB_arrivals
            
            logger.info(f"\nBrute Force (estimated):")
            logger.info(f"  Direct formation rate: {flux_stats['direct_rate']:.2e} per day")
            logger.info(f"  Days per event: {bf_days_per_event:.0f}")
            logger.info(f"  Cost for {n_stateB_arrivals} events: {bf_total_for_n_events:.0f} simulation-days")
            
            # ENSEMBLE MEMBER ESTIMATE
            forecast_length_days = 15.0
            ensemble_members_per_event = bf_days_per_event / forecast_length_days
            total_ensemble_members = ensemble_members_per_event * n_stateB_arrivals
            logger.info(f"\n  Ensemble members needed per event (10-day forecasts): {ensemble_members_per_event:.0f}")
            logger.info(f"  Total ensemble members for {n_stateB_arrivals} events: {total_ensemble_members:.0f}")
            
            speedup = bf_total_for_n_events / total_ffs_sim_days
            logger.info(f"\n✓ FFS SPEEDUP: {speedup:.1f}x")
            
        else:
            logger.info(f"\nBrute Force (lower bound):")
            logger.info(f"  Observed 0 events in {flux_stats['total_time_days']:.1f} days")
            logger.info(f"  Minimum days per event: >{flux_stats['total_time_days']:.0f}")
            min_bf_total = flux_stats['total_time_days'] * n_stateB_arrivals
            logger.info(f"  Minimum cost for {n_stateB_arrivals} events: >{min_bf_total:.0f} simulation-days")
            
            # ENSEMBLE MEMBER LOWER BOUND
            forecast_length_days = 15.0
            min_ensemble_per_event = flux_stats['total_time_days'] / forecast_length_days
            min_total_ensemble = min_ensemble_per_event * n_stateB_arrivals
            logger.info(f"\n  Minimum ensemble members per event (10-day forecasts): >{min_ensemble_per_event:.0f}")
            logger.info(f"  Minimum total ensemble members for {n_stateB_arrivals} events: >{min_total_ensemble:.0f}")
            
            min_speedup = min_bf_total / total_ffs_sim_days if total_ffs_sim_days > 0 else np.nan
            logger.info(f"\n✓ MINIMUM FFS SPEEDUP: >{min_speedup:.1f}x")

        logger.log_final_results(
            flux_estimate=flux_stats['flux'],
            transition_probs=transition_probs,
            total_prob=ffs_rate,
            direct_B_count=flux_stats['direct_B_formations'],
            direct_formation_rate=flux_stats['direct_rate']
        )

        # After all statistics are computed, save to CSV
        csv_path = ic_dir / 'ffs_statistics.csv'
        df = save_statistics_to_csv(
            ic_time=ic_time,
            time_label=time_label,
            flux_stats=flux_stats,
            stats=stats,
            transition_probs=transition_probs,
            stateB_configs=stateB_configs,
            total_ffs_sim_days=total_ffs_sim_days,
            shooting_time_days=shooting_time_days,
            interfaces=interfaces,
            output_path=csv_path
        )
        logger.info(f"Statistics saved to: {csv_path}")

        # Collect for concatenation
        all_ic_dfs.append(df)
        
        logger.log_phase_end('analysis')
        logger.close()

    # After the loop, concatenate and save
    if all_ic_dfs:
        combined_df = pd.concat(all_ic_dfs, ignore_index=True)
        combined_csv_path = output_dir / 'ffs_statistics_all_ics.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n{'='*80}")
        print(f"Combined statistics saved to: {combined_csv_path}")
        print(f"Total ICs processed: {len(all_ic_dfs)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


# #!/usr/bin/env python
# """
# Analyze FFS logs to trace pathways from flux generation to any target interface.

# This script operates on the hierarchical FFS output structure where each initial
# condition has its own directory containing flux/, interface subdirectories, and logs/.

# Usage:
#     # Trace pathways to state B
#     python analyze_ffs_logs.py ffs.yml --trace_all
    
#     # Trace pathways to specific interface (e.g., λ₁)
#     python analyze_ffs_logs.py ffs.yml --target_interface 1
    
#     # Trace pathways to λ₂
#     python analyze_ffs_logs.py ffs.yml --target_interface 2
# """

# import json
# import pickle
# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from collections import defaultdict
# from datetime import datetime
# from tails.ffs_logger import FFSLogger


# def format_time_for_path(time_str: str) -> str:
#     """Convert '2022-08-21 00:00:00' to '2022-08-21T00Z' format."""
#     dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
#     return dt.strftime('%Y-%m-%dT%HZ')


# def load_flux_configs(flux_dir: Path) -> list:
#     """Load all saved flux configs and extract crossing timesteps."""
#     configs = []
#     config_files = sorted(flux_dir.glob('lambda0_config_*.pkl'))
    
#     for config_file in config_files:
#         with open(config_file, 'rb') as f:
#             config = pickle.load(f)
#             configs.append({
#                 'name': config.config_name,
#                 'forecast_step': config.forecast_step,
#                 'mslp': config.mslp_value
#             })
    
#     return configs


# def compute_flux_from_configs(flux_configs: list) -> dict:
#     """Compute flux from saved configs."""
#     if not flux_configs:
#         return {'crossings': 0, 'forecast_steps': 0, 'flux': 0.0, 'total_time_days': 0.0}
    
#     last_forecast_step = max(c['forecast_step'] for c in flux_configs)
#     n_crossings = len(flux_configs)
    
#     total_time_days = last_forecast_step * 6.0 / 24.0
#     flux = n_crossings / total_time_days if total_time_days > 0 else 0.0
    
#     return {
#         'crossings': n_crossings,
#         'forecast_steps': last_forecast_step,
#         'total_time_days': total_time_days,
#         'flux': flux
#     }


# def compute_flux_from_logs(entries: list) -> dict:
#     """Compute flux from flux generation log entries."""
#     total_crossings = 0
#     total_timesteps = 0
#     n_trajectories = 0
#     direct_B_formations = 0
    
#     for entry in entries:
#         if entry.get('phase') != 'flux_generation':
#             continue
        
#         n_trajectories += 1
#         total_crossings += len(entry.get('configs_saved', []))
        
#         if entry.get('is_direct_B', False):
#             direct_B_formations += 1
        
#         total_timesteps += entry.get('num_timesteps', 0)
    
#     total_time_days = total_timesteps * 6.0 / 24.0
#     flux = total_crossings / total_time_days if total_time_days > 0 else 0.0
#     direct_rate = direct_B_formations / total_time_days if total_time_days > 0 else 0.0
    
#     return {
#         'trajectories': n_trajectories,
#         'crossings': total_crossings,
#         'direct_B_formations': direct_B_formations,
#         'total_timesteps': total_timesteps,
#         'total_time_days': total_time_days,
#         'flux': flux,
#         'direct_rate': direct_rate
#     }


# def load_all_logs(log_dir: Path) -> list:
#     """Load and merge all log files from all workers and ranks within IC directory."""
#     all_entries = []
    
#     # Look for flux logs
#     flux_log_pattern = 'flux/ffs_log_world*_rank*_worker*.jsonl'
#     flux_log_files = sorted(log_dir.glob(flux_log_pattern))
    
#     # Look for shooting logs (in numbered subdirectories)
#     shooting_log_files = []
#     for interface_dir in log_dir.iterdir():
#         if interface_dir.is_dir() and interface_dir.name.isdigit():
#             pattern = f"{interface_dir.name}/ffs_log_world*_rank*_worker*.jsonl"
#             shooting_log_files.extend(sorted(log_dir.glob(pattern)))
    
#     all_log_files = flux_log_files + shooting_log_files
    
#     for log_file in all_log_files:
#         with open(log_file, 'r') as f:
#             for line in f:
#                 if line.startswith('#'):
#                     continue
#                 try:
#                     entry = json.loads(line)
#                     all_entries.append(entry)
#                 except json.JSONDecodeError:
#                     continue
    
#     all_entries.sort(key=lambda x: x.get('timestamp', ''))
    
#     return all_entries


# def build_genealogy(entries: list) -> dict:
#     """Build parent -> children mapping from shooting attempts."""
#     genealogy = defaultdict(list)
#     skipped_none = 0
    
#     for entry in entries:
#         if entry.get('phase') != 'shooting':
#             continue
        
#         parent = entry.get('parent_config')
#         child = entry.get('child_config')
        
#         if child is None:
#             skipped_none += 1
#             continue
        
#         if parent and child:
#             genealogy[parent].append({
#                 'child': child,
#                 'interface_idx': entry.get('interface_idx'),
#                 'lambda_label': entry.get('lambda_label'),
#                 'status': entry.get('status'),
#                 'mslp_value': entry.get('final_mslp'),
#                 'timestamp': entry.get('timestamp'),
#                 'worker_id': entry.get('worker_id'),
#                 'rank': entry.get('rank')
#             })
    
#     return dict(genealogy)


# def trace_pathway(genealogy: dict, target_config: str) -> list:
#     """Trace pathway from flux generation to target config."""
#     reverse_lookup = {}
#     for parent, children in genealogy.items():
#         for child_info in children:
#             reverse_lookup[child_info['child']] = {
#                 'parent': parent,
#                 'interface_idx': child_info['interface_idx'],
#                 'lambda_label': child_info['lambda_label'],
#                 'status': child_info['status'],
#                 'mslp_value': child_info['mslp_value']
#             }
    
#     pathway = [{'config': target_config, 'interface': 'B'}]
#     current = target_config
    
#     while current in reverse_lookup:
#         parent_info = reverse_lookup[current]
#         parent = parent_info['parent']
        
#         if parent is None:
#             import logging
#             logging.warning(f"Encountered None parent for config {current} - pathway may be incomplete")
#             break
        
#         pathway.append({
#             'config': parent,
#             'interface_idx': parent_info['interface_idx'],
#             'lambda_label': parent_info['lambda_label'],
#             'mslp_value': parent_info['mslp_value']
#         })
        
#         current = parent
    
#     return list(reversed(pathway))


# def compute_statistics(entries: list) -> dict:
#     """Compute FFS statistics from logs."""
#     stats = {
#         'interface_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0, 'failures': 0})
#     }
    
#     for entry in entries:
#         phase = entry.get('phase')
        
#         if phase == 'shooting':
#             interface_idx = entry.get('interface_idx')
#             status = entry.get('status')
            
#             if interface_idx is not None and status:
#                 interface_stats = stats['interface_stats'][interface_idx]
#                 interface_stats['attempts'] += 1
                
#                 if status in ['success', 'reached_B', 'instant_success']:
#                     interface_stats['successes'] += 1
#                 elif status == 'failure':
#                     interface_stats['failures'] += 1
    
#     return stats


# def find_stateB_configs(entries: list, state_B: float) -> list:
#     """Find all configs that reached state B."""
#     stateB_configs = []
#     broken_entries = 0
    
#     for entry in entries:
#         if entry.get('phase') == 'shooting':
#             child = entry.get('child_config')
#             status = entry.get('status')
#             final_mslp = entry.get('final_mslp')
            
#             if status in ['success', 'reached_B', 'instant_success'] and final_mslp is not None:
#                 if final_mslp <= state_B:
#                     if child is None:
#                         broken_entries += 1
#                         continue
                    
#                     stateB_configs.append({
#                         'config': child,
#                         'parent': entry.get('parent_config'),
#                         'mslp': final_mslp,
#                         'timestamp': entry.get('timestamp')
#                     })
    
#     if broken_entries > 0:
#         import logging
#         logging.warning(f"Filtered out {broken_entries} broken state B entries with status='success' but child_config=None")
    
#     return stateB_configs


# def find_interface_configs(entries: list, target_interface_idx: int) -> list:
#     """Find all configs that successfully reached a specific interface."""
#     interface_configs = []
#     broken_entries = 0
    
#     for entry in entries:
#         if entry.get('phase') == 'shooting':
#             child = entry.get('child_config')
#             status = entry.get('status')
#             interface_idx = entry.get('interface_idx')
#             final_mslp = entry.get('final_mslp')
            
#             if (status in ['success', 'reached_B', 'instant_success'] and 
#                 interface_idx == target_interface_idx and 
#                 final_mslp is not None):
                
#                 if child is None:
#                     broken_entries += 1
#                     continue
                
#                 interface_configs.append({
#                     'config': child,
#                     'parent': entry.get('parent_config'),
#                     'interface_idx': interface_idx,
#                     'lambda_label': entry.get('lambda_label'),
#                     'mslp': final_mslp,
#                     'timestamp': entry.get('timestamp')
#                 })
    
#     if broken_entries > 0:
#         import logging
#         logging.warning(f"Filtered out {broken_entries} broken entries with status='success' but child_config=None")
    
#     return interface_configs


# def plot_pathway_images(pathway: list, ic_dir: Path, output_path: Path, target_label: str = 'B'):
#     """Create a figure showing PNG images for each step in the pathway (2-row layout)."""
#     import math
#     import matplotlib.pyplot as plt
#     import matplotlib.image as mpimg

#     n_steps = len(pathway)

#     # ---- force 2-row layout ----
#     n_rows = 2
#     n_cols = math.ceil(n_steps / n_rows)

#     fig_width = min(6 * n_cols, 36)
#     fig_height = 6 * n_rows

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=(fig_width, fig_height)
#     )

#     axes = axes.flatten()

#     for i, step in enumerate(pathway):
#         ax = axes[i]
#         config_name = step['config']

#         if config_name is None:
#             ax.text(
#                 0.5, 0.5,
#                 "Incomplete pathway\n(parent config missing)",
#                 ha='center', va='center',
#                 transform=ax.transAxes,
#                 fontsize=12, color='red'
#             )
#             ax.axis('off')
#             continue

#         # Hierarchical directory structure
#         if config_name.startswith('lambda0_'):
#             png_path = ic_dir / 'flux' / f"{config_name}.png"
#         elif config_name.startswith('stateB_'):
#             png_path = ic_dir / 'stateB' / f"{config_name}.png"
#         else:
#             interface_num = config_name.split('_')[0].replace('lambda', '')
#             png_path = ic_dir / interface_num / f"{config_name}.png"

#         if png_path.exists():
#             img = mpimg.imread(png_path)
#             ax.imshow(img)
#             ax.set_aspect('equal')
#             ax.axis('off')

#             if i == 0:
#                 title = f"λ₀\n{config_name}"
#             elif step.get('is_target', False):
#                 title = f"{target_label}\n{config_name}"
#                 if step.get('mslp_value') is not None:
#                     title += f"\n{step['mslp_value']:.1f} hPa"
#             elif step.get('interface') == 'B':
#                 title = f"STATE B\n{config_name}"
#             else:
#                 lambda_label = step['lambda_label']
#                 mslp = step['mslp_value']
#                 title = f"λ_{lambda_label}\n{config_name}\n{mslp:.1f} hPa"

#             ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

#         else:
#             ax.text(
#                 0.5, 0.5,
#                 f"Image not found:\n{config_name}",
#                 ha='center', va='center',
#                 transform=ax.transAxes
#             )
#             ax.axis('off')

#     # ---- hide unused panels ----
#     for j in range(n_steps, len(axes)):
#         axes[j].axis('off')

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

#     return output_path

# def save_statistics_to_csv(ic_time, time_label, flux_stats, stats, transition_probs, 
#                            stateB_configs, total_ffs_sim_days, shooting_time_days,
#                            interfaces, output_path):
#     """Save all FFS statistics to CSV file."""
    
#     # Main summary row
#     summary = {
#         'ic_time': ic_time,
#         'time_label': time_label,
        
#         # Flux statistics
#         'flux_trajectories': flux_stats['trajectories'],
#         'flux_lambda0_crossings': flux_stats['crossings'],
#         'flux_direct_B_formations': flux_stats['direct_B_formations'],
#         'flux_total_timesteps': flux_stats['total_timesteps'],
#         'flux_total_time_days': flux_stats['total_time_days'],
#         'flux_rate_per_day': flux_stats['flux'],
#         'direct_formation_rate_per_day': flux_stats['direct_rate'],
        
#         # Shooting statistics
#         'shooting_time_days': shooting_time_days,
#         'total_ffs_sim_days': total_ffs_sim_days,
#         'n_stateB_arrivals': len(stateB_configs),
        
#         # FFS rate calculation
#         'ffs_rate_per_day': flux_stats['flux'] * np.prod(transition_probs) if transition_probs else 0.0,
#     }
    
#     # Add per-interface shooting stats
#     for interface_idx in sorted(stats['interface_stats'].keys()):
#         interface_stats = stats['interface_stats'][interface_idx]
#         lambda_label = interface_idx - 1

#         summary[f'lambda{lambda_label}_attempts'] = interface_stats['attempts']
#         summary[f'lambda{lambda_label}_successes'] = interface_stats['successes']
#         summary[f'lambda{lambda_label}_failures'] = interface_stats['failures']
#         summary[f'lambda{lambda_label}_extratropical'] = interface_stats['attempts'] - interface_stats['successes'] - interface_stats['failures']
        
#         if interface_stats['attempts'] > 0:
#             P = interface_stats['successes'] / interface_stats['attempts']
#             summary[f'lambda{lambda_label}_P_forward'] = P
#         else:
#             summary[f'lambda{lambda_label}_P_forward'] = np.nan
    
#     # Efficiency metrics
#     if len(stateB_configs) > 0:
#         summary['cost_per_event_days'] = total_ffs_sim_days / len(stateB_configs)
#     else:
#         summary['cost_per_event_days'] = np.nan
    
#     if flux_stats['direct_rate'] > 0:
#         bf_days_per_event = 1.0 / flux_stats['direct_rate']
#         bf_total_for_n_events = bf_days_per_event * len(stateB_configs)
        
#         forecast_length_days = 15.0
#         ensemble_members_per_event = bf_days_per_event / forecast_length_days
#         total_ensemble_members = ensemble_members_per_event * len(stateB_configs)
        
#         speedup = bf_total_for_n_events / total_ffs_sim_days if total_ffs_sim_days > 0 else np.nan
        
#         summary['bf_days_per_event'] = bf_days_per_event
#         summary['bf_total_cost_days'] = bf_total_for_n_events
#         summary['bf_ensemble_members_per_event'] = ensemble_members_per_event
#         summary['bf_total_ensemble_members'] = total_ensemble_members
#         summary['ffs_speedup'] = speedup
#     else:
#         # Lower bounds
#         min_bf_total = flux_stats['total_time_days'] * len(stateB_configs)
#         forecast_length_days = 15.0
#         min_ensemble_per_event = flux_stats['total_time_days'] / forecast_length_days
#         min_speedup = min_bf_total / total_ffs_sim_days if total_ffs_sim_days > 0 else np.nan
        
#         summary['bf_days_per_event'] = f">{flux_stats['total_time_days']:.0f}"
#         summary['bf_total_cost_days'] = f">{min_bf_total:.0f}"
#         summary['bf_ensemble_members_per_event'] = f">{min_ensemble_per_event:.0f}"
#         summary['bf_total_ensemble_members'] = f">{min_ensemble_per_event * len(stateB_configs):.0f}"
#         summary['ffs_speedup'] = f">{min_speedup:.1f}"
    
#     # Convert to DataFrame and save
#     df = pd.DataFrame([summary])
#     df.to_csv(output_path, index=False)
    
#     return df

# def main():
#     parser = argparse.ArgumentParser(description='Analyze FFS logs')
#     parser.add_argument('config_file', type=str, help='FFS config file (ffs.yml)')
#     parser.add_argument('--trace_all', action='store_true', 
#                        help='Trace all pathways to state B')
#     parser.add_argument('--target_interface', type=int, default=None,
#                        help='Trace pathways to specific interface (e.g., 1 for λ₁, 2 for λ₂)')
    
#     args = parser.parse_args()
    
#     import yaml
#     with open(args.config_file, 'r') as f:
#         ffs_config = yaml.safe_load(f)
    
#     output_dir = Path(ffs_config['output_dir'])
#     state_A = ffs_config['state_A']
#     state_B = ffs_config['state_B']
#     forecast_times = ffs_config['forecast_times']
    
#     interfaces = ffs_config['interfaces'].copy()
#     if interfaces[-1] != state_B:
#         interfaces.append(state_B)
    
#     lambda0 = interfaces[0]

#     # Collect all IC dataframes
#     all_ic_dfs = []
    
#     # Process each initial condition
#     for time_range in forecast_times:
#         ic_time = time_range[0]
#         time_label = format_time_for_path(ic_time)
        
#         # Hierarchical structure: output_dir/IC_TIME/
#         ic_dir = output_dir / time_label
#         flux_dir = ic_dir / 'flux'
#         logs_dir = ic_dir / 'logs'
        
#         if not ic_dir.exists():
#             print(f"WARNING: IC directory not found: {ic_dir}")
#             continue
        
#         # Initialize logger for this IC analysis
#         analysis_log_dir = ic_dir / 'analysis_logs'
#         analysis_log_dir.mkdir(parents=True, exist_ok=True)
        
#         logger = FFSLogger(
#             output_dir=analysis_log_dir,
#             worker_id='analyzer',
#             rank=0,
#             world_size=1,
#             ic_dirname=None  # Already in IC-specific directory
#         )
        
#         logger.log_phase_start('analysis', ic_time=ic_time, time_label=time_label)
#         logger.info(f"Analyzing FFS results for IC: {time_label}")
#         logger.info(f"IC directory: {ic_dir}")
        
#         # Load logs from hierarchical structure
#         logger.info(f"Loading log files from {logs_dir}")
#         entries = load_all_logs(logs_dir)
#         logger.info(f"Loaded {len(entries)} total log entries")
        
#         flux_stats = compute_flux_from_logs(entries)
        
#         logger.info("="*80)
#         logger.info("FLUX GENERATION STATISTICS")
#         logger.info("="*80)
#         logger.info(f"Trajectories run: {flux_stats['trajectories']}")
#         logger.info(f"λ₀ crossings: {flux_stats['crossings']}")
#         logger.info(f"Direct B formations: {flux_stats['direct_B_formations']}")
#         logger.info(f"Total timesteps: {flux_stats['total_timesteps']}")
#         logger.info(f"Total time: {flux_stats['total_time_days']:.1f} days")
#         logger.info(f"FFS Flux Rate: Φ₀ = {flux_stats['flux']:.6f} crossings/day")
#         logger.info(f"Direct Formation Rate: Φ_direct = {flux_stats['direct_rate']:.6e} formations/day")
        
#         if flux_stats['direct_rate'] > 0 and flux_stats['flux'] > 0:
#             ratio = flux_stats['flux'] / flux_stats['direct_rate']
#             logger.info(f"FFS flux / Direct rate = {ratio:.2f}x")
        
#         genealogy = build_genealogy(entries)
#         logger.info(f"Built genealogy: {len(genealogy)} parent configs, "
#                    f"{sum(len(children) for children in genealogy.values())} total children")
        
#         stateB_configs = find_stateB_configs(entries, state_B)
#         logger.info(f"Found {len(stateB_configs)} configs that reached state B:")
#         # for config_info in stateB_configs:
#         #     logger.info(f"  {config_info['config']} (MSLP={config_info['mslp']:.1f} hPa)")
        
#         if args.trace_all and stateB_configs:
#             logger.info("="*80)
#             logger.info("TRACING PATHWAYS TO STATE B")
#             logger.info("="*80)

#             pathway_figs_dir = ic_dir / 'pathway_figures' / 'stateB'
#             pathway_figs_dir.mkdir(parents=True, exist_ok=True)
            
#             for i, config_info in enumerate(stateB_configs):
#                 logger.info(f"Pathway {i+1}/{len(stateB_configs)}:")
#                 pathway = trace_pathway(genealogy, config_info['config'])
                
#                 for j, step in enumerate(pathway):
#                     step_config = step['config']
#                     if j == 0:
#                         logger.info(f"  {j}. {step_config} (λ₀ - flux generation)")
#                     elif step.get('interface') == 'B':
#                         logger.info(f"  {j}. {step_config} (STATE B)")
#                     else:
#                         lambda_label = step['lambda_label']
#                         mslp = step['mslp_value']
#                         logger.info(f"  {j}. {step_config} (λ_{lambda_label}, MSLP={mslp:.1f} hPa)")

#                 fig_path = pathway_figs_dir / f"pathway_{i+1:03d}.png"
#                 plot_pathway_images(pathway, ic_dir, fig_path, target_label='STATE B')
#                 logger.info(f"  Pathway figure saved: {fig_path}")
        
#         if args.target_interface is not None:
#             target_idx = args.target_interface
#             target_lambda_label = target_idx - 1
            
#             if target_idx < 1 or target_idx >= len(interfaces):
#                 logger.error(f"Invalid target_interface={target_idx}. Valid range: 1 to {len(interfaces)-1}")
#                 continue
            
#             logger.info("="*80)
#             logger.info(f"TRACING PATHWAYS TO λ_{target_lambda_label} (interface_idx={target_idx})")
#             logger.info("="*80)
            
#             target_configs = find_interface_configs(entries, target_idx)
#             logger.info(f"Found {len(target_configs)} configs that reached λ_{target_lambda_label}:")
#             for config_info in target_configs:
#                 logger.info(f"  {config_info['config']} (MSLP={config_info['mslp']:.1f} hPa)")
            
#             if target_configs:
#                 pathway_figs_dir = ic_dir / 'pathway_figures' / f'lambda{target_lambda_label}'
#                 pathway_figs_dir.mkdir(parents=True, exist_ok=True)
                
#                 for i, config_info in enumerate(target_configs):
#                     logger.info(f"Pathway {i+1}/{len(target_configs)}:")
#                     pathway = trace_pathway(genealogy, config_info['config'])
                    
#                     if len(pathway) > 0:
#                         pathway[-1]['is_target'] = True
#                         pathway[-1]['mslp_value'] = config_info['mslp']
                    
#                     for j, step in enumerate(pathway):
#                         step_config = step['config']
#                         if j == 0:
#                             logger.info(f"  {j}. {step_config} (λ₀ - flux generation)")
#                         elif step.get('is_target', False):
#                             logger.info(f"  {j}. {step_config} (λ_{target_lambda_label} - TARGET)")
#                         else:
#                             lambda_label = step.get('lambda_label', '?')
#                             mslp = step.get('mslp_value', 0.0)
#                             logger.info(f"  {j}. {step_config} (λ_{lambda_label}, MSLP={mslp:.1f} hPa)")
                    
#                     fig_path = pathway_figs_dir / f"pathway_{i+1:03d}.png"
#                     plot_pathway_images(pathway, ic_dir, fig_path, 
#                                       target_label=f'λ_{target_lambda_label}')
#                     logger.info(f"  Pathway figure saved: {fig_path}")
        
#         stats = compute_statistics(entries)

#         logger.info("="*80)
#         logger.info("SHOOTING STATISTICS")
#         logger.info("="*80)

#         expected_interfaces = list(range(1, len(interfaces)))
#         actual_interfaces = sorted(stats['interface_stats'].keys())

#         valid_interfaces = [idx for idx in actual_interfaces if idx in expected_interfaces]
#         invalid_interfaces = [idx for idx in actual_interfaces if idx not in expected_interfaces]

#         if invalid_interfaces:
#             logger.warning(f"Ignoring unexpected shooting data for interfaces: {invalid_interfaces}")
#             logger.warning("(These may be from incomplete runs or errors)")

#         missing = set(expected_interfaces) - set(actual_interfaces)
#         if missing:
#             logger.warning(f"Missing shooting data for interfaces: {sorted(missing)}")
#             logger.warning("FFS run may be incomplete!")

#         transition_probs = []
        
#         logger.info("Shooting phases:")
#         for interface_idx in valid_interfaces:
#             interface_stats = stats['interface_stats'][interface_idx]
#             lambda_label = interface_idx - 1
#             next_lambda_label = interface_idx
            
#             lambda_mslp = interfaces[lambda_label]
#             next_mslp = interfaces[next_lambda_label]
            
#             attempts = interface_stats['attempts']
#             successes = interface_stats['successes']
#             failures = interface_stats['failures']
            
#             if attempts > 0:
#                 P_forward = successes / attempts
#                 transition_probs.append(P_forward)
                
#                 if interface_idx == len(interfaces) - 1:
#                     logger.info(f"  λ_{lambda_label} ({lambda_mslp} hPa) → STATE B ({next_mslp} hPa):")
#                 else:
#                     logger.info(f"  λ_{lambda_label} ({lambda_mslp} hPa) → λ_{next_lambda_label} ({next_mslp} hPa):")
                    
#                 logger.info(f"    Attempts: {attempts}, Successes: {successes}, Failures: {failures}")
#                 logger.info(f"    P_forward: {P_forward:.4f}")

#         ffs_rate = flux_stats['flux']
#         for p in transition_probs:
#             ffs_rate *= p

#         # SPEEDUP CALCULATION
#         # After computing transition probabilities and stats

#         # Count actual B-state arrivals
#         n_stateB_arrivals = len(stateB_configs)

#         # Compute total FFS computational cost
#         total_ffs_sim_days = flux_stats['total_time_days']  # Flux generation time

#         # Add shooting time - need to extract from logs
#         shooting_time_days = 0.0
#         for entry in entries:
#             if entry.get('phase') == 'shooting':
#                 shooting_time_days += entry.get('num_timesteps', 0) * 6.0 / 24.0

#         total_ffs_sim_days += shooting_time_days

#         logger.info("="*80)
#         logger.info("FFS EFFICIENCY vs BRUTE FORCE")
#         logger.info("="*80)
#         logger.info(f"\nFFS Results:")
#         logger.info(f"  State B arrivals: {n_stateB_arrivals}")
#         logger.info(f"  Total simulation time: {total_ffs_sim_days:.1f} days")
#         logger.info(f"    Flux generation: {flux_stats['total_time_days']:.1f} days")
#         logger.info(f"    Shooting phases: {shooting_time_days:.1f} days")
#         if n_stateB_arrivals > 0:
#             logger.info(f"  Cost per event: {total_ffs_sim_days / n_stateB_arrivals:.1f} simulation-days")

#         if flux_stats['direct_rate'] > 0:
#             bf_days_per_event = 1.0 / flux_stats['direct_rate']
#             bf_total_for_n_events = bf_days_per_event * n_stateB_arrivals
            
#             logger.info(f"\nBrute Force (estimated):")
#             logger.info(f"  Direct formation rate: {flux_stats['direct_rate']:.2e} per day")
#             logger.info(f"  Days per event: {bf_days_per_event:.0f}")
#             logger.info(f"  Cost for {n_stateB_arrivals} events: {bf_total_for_n_events:.0f} simulation-days")
            
#             # ENSEMBLE MEMBER ESTIMATE
#             forecast_length_days = 15.0
#             ensemble_members_per_event = bf_days_per_event / forecast_length_days
#             total_ensemble_members = ensemble_members_per_event * n_stateB_arrivals
#             logger.info(f"\n  Ensemble members needed per event (10-day forecasts): {ensemble_members_per_event:.0f}")
#             logger.info(f"  Total ensemble members for {n_stateB_arrivals} events: {total_ensemble_members:.0f}")
            
#             speedup = bf_total_for_n_events / total_ffs_sim_days
#             logger.info(f"\n✓ FFS SPEEDUP: {speedup:.1f}x")
            
#         else:
#             logger.info(f"\nBrute Force (lower bound):")
#             logger.info(f"  Observed 0 events in {flux_stats['total_time_days']:.1f} days")
#             logger.info(f"  Minimum days per event: >{flux_stats['total_time_days']:.0f}")
#             min_bf_total = flux_stats['total_time_days'] * n_stateB_arrivals
#             logger.info(f"  Minimum cost for {n_stateB_arrivals} events: >{min_bf_total:.0f} simulation-days")
            
#             # ENSEMBLE MEMBER LOWER BOUND
#             forecast_length_days = 15.0
#             min_ensemble_per_event = flux_stats['total_time_days'] / forecast_length_days
#             min_total_ensemble = min_ensemble_per_event * n_stateB_arrivals
#             logger.info(f"\n  Minimum ensemble members per event (10-day forecasts): >{min_ensemble_per_event:.0f}")
#             logger.info(f"  Minimum total ensemble members for {n_stateB_arrivals} events: >{min_total_ensemble:.0f}")
            
#             min_speedup = min_bf_total / total_ffs_sim_days
#             logger.info(f"\n✓ MINIMUM FFS SPEEDUP: >{min_speedup:.1f}x")

#         logger.log_final_results(
#             flux_estimate=flux_stats['flux'],
#             transition_probs=transition_probs,
#             total_prob=ffs_rate,
#             direct_B_count=flux_stats['direct_B_formations'],
#             direct_formation_rate=flux_stats['direct_rate']
#         )

#         # After all statistics are computed, save to CSV
#         csv_path = ic_dir / 'ffs_statistics.csv'
#         df = save_statistics_to_csv(
#             ic_time=ic_time,
#             time_label=time_label,
#             flux_stats=flux_stats,
#             stats=stats,
#             transition_probs=transition_probs,
#             stateB_configs=stateB_configs,
#             total_ffs_sim_days=total_ffs_sim_days,
#             shooting_time_days=shooting_time_days,
#             interfaces=interfaces,
#             output_path=csv_path
#         )
#         logger.info(f"Statistics saved to: {csv_path}")

#         # Collect for concatenation
#         all_ic_dfs.append(df)
        
#         logger.log_phase_end('analysis')
#         logger.close()

#     # After the loop, concatenate and save
#     if all_ic_dfs:
#         combined_df = pd.concat(all_ic_dfs, ignore_index=True)
#         combined_csv_path = output_dir / 'ffs_statistics_all_ics.csv'
#         combined_df.to_csv(combined_csv_path, index=False)
#         print(f"\n{'='*80}")
#         print(f"Combined statistics saved to: {combined_csv_path}")
#         print(f"Total ICs processed: {len(all_ic_dfs)}")
#         print(f"{'='*80}\n")


# if __name__ == "__main__":
#     main()