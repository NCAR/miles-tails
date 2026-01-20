#!/usr/bin/env python
"""
Parallel Forward Flux Sampling for Hurricane Genesis with CPS

This script runs FFS in phases that can be executed independently:
1. Flux generation (--phase flux)
2. Shooting from each interface (--phase shoot --interface N)

Multiple jobs can work on the same IC simultaneously. Each job checks
if enough work has been completed before doing more.

Walltime management: Use --walltime (hours) to stop processing before 
hitting a time limit. The script will stop launching new simulations
when approaching the limit (default 30 min buffer).

Usage:
    # Flux generation (submit many jobs)
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux
    
    # Flux with 6-hour walltime limit
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux --walltime 6.0
    
    # Shooting from interface 1 (λ₀) with custom buffer
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 1 --walltime 12.0 --walltime_buffer 45
    
    # Shooting from interface 2 (λ₁)
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 2
"""

import yaml
import os
import sys
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import pickle
from datetime import datetime
import time
from credit.distributed import get_rank_info
from credit.rare_events.hurricane_genesis_ffs import HurricaneGenesisFFS_Tracked as HurricaneGenesisFFS
from credit.rare_events.hurricane_genesis_ffs_cps import HurricaneGenesisFFS_CPS


def format_ic_dirname(ic_time_str: str) -> str:
    """Convert '2022-08-28 00:00:00' to '2022-08-28T00Z'"""
    dt = datetime.strptime(ic_time_str, '%Y-%m-%d %H:%M:%S')
    return dt.strftime('%Y-%m-%dT%HZ')


def check_walltime_remaining(start_time: float, walltime_hours: float, buffer_minutes: float = 30.0) -> tuple:
    """
    Check if we have enough time remaining before hitting walltime.
    
    Args:
        start_time: Time when script started (from time.time())
        walltime_hours: Total walltime limit in hours
        buffer_minutes: Safety buffer in minutes before walltime
        
    Returns:
        (has_time_remaining: bool, elapsed_hours: float, remaining_hours: float)
    """
    elapsed_seconds = time.time() - start_time
    elapsed_hours = elapsed_seconds / 3600.0
    
    walltime_seconds = walltime_hours * 3600.0
    buffer_seconds = buffer_minutes * 60.0
    effective_limit_seconds = walltime_seconds - buffer_seconds
    
    has_time = elapsed_seconds < effective_limit_seconds
    remaining_hours = (walltime_seconds - elapsed_seconds) / 3600.0
    
    return has_time, elapsed_hours, remaining_hours


def count_configs_in_directory(directory: Path, pattern: str) -> int:
    """Count number of config files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def flux_generation_worker(worker_id: int,
                           ffs_config: dict,
                           model_config: dict,
                           ic_start: str,
                           ic_end: str,
                           n_trials_total: int,
                           rank: int,
                           world_size: int) -> dict:
    """Worker for flux generation phase."""
    import torch
    from credit.models import load_model
    from credit.transforms import Normalize_ERA5_and_Forcing, load_transforms
    from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
    from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
    from credit.parser import credit_main_parser
    from credit.datasets import setup_data_loading
    import numpy as np

    # Compute unique seed
    base_seed = model_config.get('seed', 42)
    
    # Hash IC time to add temporal variation
    ic_hash = hash(ic_start) % 10000
    
    # Combine rank, worker_id, and IC
    unique_seed = base_seed + rank * 100000 + worker_id * 1000 + ic_hash
    
    # Set all random seeds
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    np.random.seed(unique_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = 'cuda:0'
    torch.cuda.empty_cache()
    
    # Load model using save_loc from model config
    conf = model_config.copy()
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)

    model = load_model(conf)
    save_loc = os.path.expandvars(conf["save_loc"])
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location="cpu")
    _ = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    
    model = model.eval()
    model = model.to(device)
    
    # Create dataset parameters
    dataset_params = {
        'varname_upper_air': data_config["varname_upper_air"],
        'varname_surface': data_config["varname_surface"],
        'varname_dyn_forcing': data_config["varname_dyn_forcing"],
        'varname_forcing': data_config["varname_forcing"],
        'varname_static': data_config["varname_static"],
        'varname_diagnostic': data_config["varname_diagnostic"],
        'filenames': data_config["all_ERA_files"],
        'filename_surface': data_config["surface_files"],
        'filename_dyn_forcing': data_config["dyn_forcing_files"],
        'filename_forcing': data_config["forcing_files"],
        'filename_static': data_config["static_files"],
        'filename_diagnostic': data_config["diagnostic_files"],
        'lead_time_periods': 6,
        'history_len': data_config["history_len"],
        'skip_periods': data_config["skip_periods"],
        'transform': load_transforms(conf),
        'sst_forcing': data_config["sst_forcing"],
        'batch_size': 1,
        'rank': rank,
        'world_size': world_size,
    }

    # Create dataset with initial forecast times
    forecast_times = [[ic_start, ic_end]]
    dataset = Predict_Dataset_Batcher(
        **dataset_params,
        fcst_datetime=forecast_times
    )

    loader = BatchForecastLenDataLoader(dataset)

    # Setup output directory with IC-specific path
    ic_dirname = format_ic_dirname(ic_start)
    output_dir = Path(ffs_config['output_dir']) / ic_dirname / 'flux'
    
    # Initialize base FFS
    ffs_base = HurricaneGenesisFFS(
        model=model,
        state_transformer=Normalize_ERA5_and_Forcing(conf),
        config=conf,
        initial_dataset=dataset,
        dataset_params=dataset_params,
        output_dir=str(output_dir.parent.parent),
        state_A=ffs_config['state_A'],
        state_B=ffs_config['state_B'],
        interfaces=ffs_config['interfaces'],
        worker_id=worker_id,
        rank=rank,
        world_size=world_size,
        ic_dirname=ic_dirname
    )
    
    # Wrap with CPS
    ffs = HurricaneGenesisFFS_CPS(ffs_base)
    
    # Override flux_dir to be IC-specific
    ffs.flux_dir = output_dir
    ffs.flux_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"[Worker {worker_id}] Starting for {ic_dirname}")
    logging.info(f"[Worker {worker_id}] ✓ CPS-enhanced FFS initialized")
    
    # Run flux generation - will check global count
    ffs.generate_flux_at_lambda0(loader, n_trials=n_trials_total)
    
    torch.cuda.empty_cache()
    
    # Collect generated config paths
    config_paths = list(output_dir.glob('lambda0_config_*.pkl'))
    
    return {
        'worker_id': worker_id,
        'phase': 'flux',
        'configs_generated': len(config_paths)
    }

def shooting_worker(worker_id: int,
                   ffs_config: dict,
                   model_config: dict,
                   ic_start: str,
                   ic_end: str,
                   interface_idx: int,
                   n_trials_total: int,
                   shared_config_pool: list,
                   rank: int,
                   world_size: int) -> dict:
    """Worker for shooting phase at specific interface."""
    import torch
    from credit.models import load_model
    from credit.transforms import Normalize_ERA5_and_Forcing, load_transforms
    from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
    from credit.parser import credit_main_parser
    from credit.datasets import setup_data_loading
    import numpy as np
    
    # Compute unique seed
    base_seed = model_config.get('seed', 42)
    
    # Hash IC time to add temporal variation
    ic_hash = hash(ic_start) % 10000
    
    # Combine rank, worker_id, and IC
    unique_seed = base_seed + rank * 100000 + worker_id * 1000 + ic_hash
    
    # Set all random seeds
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    np.random.seed(unique_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = 'cuda:0'
    torch.cuda.empty_cache()
    
    # Load model using save_loc from model config
    conf = model_config.copy()
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)
    
    model = load_model(conf)
    save_loc = os.path.expandvars(conf["save_loc"])
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location="cpu")
    _ = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    
    model = model.eval()
    model = model.to(device)

    dataset_params = {
        'varname_upper_air': data_config["varname_upper_air"],
        'varname_surface': data_config["varname_surface"],
        'varname_dyn_forcing': data_config["varname_dyn_forcing"],
        'varname_forcing': data_config["varname_forcing"],
        'varname_static': data_config["varname_static"],
        'varname_diagnostic': data_config["varname_diagnostic"],
        'filenames': data_config["all_ERA_files"],
        'filename_surface': data_config["surface_files"],
        'filename_dyn_forcing': data_config["dyn_forcing_files"],
        'filename_forcing': data_config["forcing_files"],
        'filename_static': data_config["static_files"],
        'filename_diagnostic': data_config["diagnostic_files"],
        'lead_time_periods': 6,
        'history_len': data_config["history_len"],
        'skip_periods': data_config["skip_periods"],
        'transform': load_transforms(conf),
        'sst_forcing': data_config["sst_forcing"],
        'batch_size': 1,
        'rank': rank,
        'world_size': world_size,
    }
    
    # Create dataset
    forecast_times = [[ic_start, ic_end]]
    dataset = Predict_Dataset_Batcher(
        **dataset_params,
        fcst_datetime=forecast_times
    )
    
    # Setup output directory with IC-specific path
    ic_dirname = format_ic_dirname(ic_start)
    shooting_output_dir = Path(ffs_config['output_dir']) / ic_dirname / str(interface_idx)
    flux_output_dir = Path(ffs_config['output_dir']) / ic_dirname / 'flux'
    
    # Initialize base FFS with base output dir
    ffs_base = HurricaneGenesisFFS(
        model=model,
        state_transformer=Normalize_ERA5_and_Forcing(conf),
        config=conf,
        initial_dataset=dataset,
        dataset_params=dataset_params,
        output_dir=str(Path(ffs_config['output_dir'])),
        state_A=ffs_config['state_A'],
        state_B=ffs_config['state_B'],
        interfaces=ffs_config['interfaces'],
        worker_id=worker_id,
        rank=rank,
        world_size=world_size,
        ic_dirname=ic_dirname
    )
    
    # Wrap with CPS
    ffs = HurricaneGenesisFFS_CPS(ffs_base)
    
    # CRITICAL: Override BOTH directories to be IC-specific
    ffs.shoot_dir = shooting_output_dir
    ffs.shoot_dir.mkdir(parents=True, exist_ok=True)
    
    ffs.flux_dir = flux_output_dir
    # flux_dir should already exist from flux generation, but check
    if not ffs.flux_dir.exists():
        logging.warning(f"WARNING: flux_dir {ffs.flux_dir} does not exist!")
    
    # Load shared configs
    loaded_configs = []
    for config_path in shared_config_pool:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            loaded_configs.append(config)
    
    ffs.interface_configs[interface_idx] = loaded_configs
    
    lambda_label = interface_idx - 1
    logging.info(f"[SHOOT Worker {worker_id}] Starting λ_{lambda_label}, pool size: {len(loaded_configs)}")
    logging.info(f"[SHOOT Worker {worker_id}] ✓ CPS-enhanced FFS initialized")
    
    # Run shooting - will check global count
    ffs.shoot_from_interface(interface_idx, n_trials=n_trials_total)
    
    torch.cuda.empty_cache()
    
    return {
        'worker_id': worker_id,
        'phase': 'shooting',
        'interface_idx': interface_idx
    }


def run_flux_phase(model_config: dict,
                  ffs_config: dict,
                  ic_start: str,
                  ic_end: str,
                  num_workers: int,
                  rank: int,
                  world_size: int,
                  start_time: float = None,
                  walltime_hours: float = None,
                  buffer_minutes: float = 30.0):
    """Run flux generation phase."""
    # Check walltime before starting
    if start_time is not None and walltime_hours is not None:
        has_time, elapsed, remaining = check_walltime_remaining(start_time, walltime_hours, buffer_minutes)
        if not has_time:
            logging.warning("⏰ WALLTIME LIMIT APPROACHING - Skipping flux phase")
            logging.warning(f"   Elapsed: {elapsed:.2f}h, Remaining: {remaining:.2f}h")
            return
        else:
            logging.info(f"⏱ Time check: Elapsed {elapsed:.2f}h, Remaining {remaining:.2f}h")
    
    ic_dirname = format_ic_dirname(ic_start)
    flux_dir = Path(ffs_config['output_dir']) / ic_dirname / 'flux'

    existing_configs = count_configs_in_directory(flux_dir, 'lambda0_config_*.pkl')
    n_flux_total = ffs_config['n_flux_total']
    
    if rank == 0:
        logging.info(f"\n{'='*80}")
        logging.info(f"FLUX GENERATION (CPS-Enhanced) - {ic_dirname}")
        logging.info(f"{'='*80}")
        logging.info(f"Target configs: {n_flux_total}")
        logging.info(f"Existing configs: {existing_configs}")
    
        if existing_configs >= n_flux_total:
            logging.info("✓ Flux generation already complete, skipping")
            logging.info(f"{'='*80}\n")
            return
        
        logging.info(f"Workers: {num_workers}")
        logging.info(f"{'='*80}\n")

    if num_workers == 0:
        logging.info("Running serially on main process (num_workers=0)")
        result = flux_generation_worker(
            worker_id=0,
            ffs_config=ffs_config,
            model_config=model_config,
            ic_start=ic_start,
            ic_end=ic_end,
            n_trials_total=n_flux_total,
            rank=rank,
            world_size=world_size
        )
        flux_results = [result]

    else:
        mp.set_start_method('spawn', force=True)
        
        flux_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    flux_generation_worker,
                    worker_id=i,
                    ffs_config=ffs_config,
                    model_config=model_config,
                    ic_start=ic_start,
                    ic_end=ic_end,
                    n_trials_total=n_flux_total,
                    rank=rank,
                    world_size=world_size
                )
                for i in range(num_workers)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                flux_results.append(result)
                logging.info(f"✓ Worker {result['worker_id']} finished")
    
    final_count = count_configs_in_directory(flux_dir, 'lambda0_config_*.pkl')
    logging.info("\n✓ FLUX GENERATION COMPLETE")
    logging.info(f"Total configs: {final_count}/{n_flux_total}")
    logging.info(f"{'='*80}\n")


def run_shoot_phase(model_config: dict,
                   ffs_config: dict,
                   ic_start: str,
                   ic_end: str,
                   interface_idx: int,
                   num_workers: int,
                   rank: int,
                   world_size: int,
                   start_time: float = None,
                   walltime_hours: float = None,
                   buffer_minutes: float = 30.0):
    """Run shooting phase for a specific interface."""
    # Check walltime before starting
    if start_time is not None and walltime_hours is not None:
        has_time, elapsed, remaining = check_walltime_remaining(start_time, walltime_hours, buffer_minutes)
        if not has_time:
            logging.warning("⏰ WALLTIME LIMIT APPROACHING - Skipping shoot phase")
            logging.warning(f"   Elapsed: {elapsed:.2f}h, Remaining: {remaining:.2f}h")
            return
        else:
            logging.info(f"⏱ Time check: Elapsed {elapsed:.2f}h, Remaining {remaining:.2f}h")
    
    ic_dirname = format_ic_dirname(ic_start)
    shoot_dir = Path(ffs_config['output_dir']) / ic_dirname / str(interface_idx)

    lambda_label = interface_idx - 1
    
    existing_configs = count_configs_in_directory(shoot_dir, f'lambda{interface_idx}_config_*.pkl')
    n_shoot_total = ffs_config['n_shoot_per_interface']
    
    if rank == 0:
        logging.info(f"\n{'='*80}")
        logging.info(f"SHOOTING (CPS-Enhanced) λ_{lambda_label} → λ_{interface_idx} - {ic_dirname}")
        logging.info(f"{'='*80}")
        logging.info(f"Target configs: {n_shoot_total}")
        logging.info(f"Existing configs: {existing_configs}")
        
        if existing_configs >= n_shoot_total:
            logging.info("✓ Shooting already complete, skipping")
            logging.info(f"{'='*80}\n")
            return
    
    # Get source configs
    if interface_idx == 1:
        source_dir = Path(ffs_config['output_dir']) / ic_dirname / 'flux'
        source_pattern = 'lambda0_config_*.pkl'
    else:
        source_dir = Path(ffs_config['output_dir']) / ic_dirname / str(interface_idx - 1)
        source_pattern = f'lambda{interface_idx-1}_config_*.pkl'
    
    source_configs = list(source_dir.glob(source_pattern))
    
    if len(source_configs) == 0:
        logging.warning("⚠ No source configs found")
        logging.info(f"{'='*80}\n")
        return
    
    logging.info(f"Source pool size: {len(source_configs)}")
    logging.info(f"Workers: {num_workers}")
    logging.info(f"{'='*80}\n")
    
    if num_workers == 0:
        logging.info("Running serially on main process (num_workers=0)")
        result = shooting_worker(
            worker_id=0,
            ffs_config=ffs_config,
            model_config=model_config,
            ic_start=ic_start,
            ic_end=ic_end,
            interface_idx=interface_idx,
            n_trials_total=n_shoot_total,
            shared_config_pool=[str(p) for p in source_configs],
            rank=rank,
            world_size=world_size
        )
        shoot_results = [result]
    else:
        mp.set_start_method('spawn', force=True)
        
        shoot_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    shooting_worker,
                    worker_id=j,
                    ffs_config=ffs_config,
                    model_config=model_config,
                    ic_start=ic_start,
                    ic_end=ic_end,
                    interface_idx=interface_idx,
                    n_trials_total=n_shoot_total,
                    shared_config_pool=[str(p) for p in source_configs],
                    rank=rank,
                    world_size=world_size
                )
                for j in range(num_workers)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                shoot_results.append(result)
                logging.info(f"✓ Worker {result['worker_id']} finished")
    
    final_count = count_configs_in_directory(shoot_dir, f'lambda{interface_idx}_config_*.pkl')
    
    logging.info(f"\n✓ SHOOTING λ_{lambda_label} COMPLETE")
    logging.info(f"Total configs: {final_count}/{n_shoot_total}")
    logging.info(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Run parallel FFS for hurricane genesis with CPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # SINGLE IC MODE (all 4 GPUs work on ONE IC specified by ic_index):
        # In ffs.yml: single_ic_mode: true, ic_index: 0
        python run_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux --walltime 12.0
        
        # Override ic_index from command line:
        python run_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux --ic_index 2 --walltime 12.0
        
        # DISTRIBUTED IC MODE (each GPU works on different ICs):
        # In ffs.yml: single_ic_mode: false
        python run_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux --walltime 12.0
        
        # SLURM job array example (single IC mode):
        # sbatch --array=0-9 submit_ffs.sh  # Each job processes different IC
        # In submit script: --ic_index ${SLURM_ARRAY_TASK_ID}
        """
    )
    
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model.yml')
    parser.add_argument('--ffs_config', type=str, required=True,
                       help='Path to ffs.yml')
    parser.add_argument('--phase', type=str, required=True, choices=['flux', 'shoot'],
                       help='Which phase to run: flux or shoot')
    parser.add_argument('--interface', type=int, default=None,
                       help='Interface index for shooting (required if phase=shoot)')
    parser.add_argument('--ic_index', type=int, default=None,
                       help='IC index to process in single_ic_mode (overrides config file)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--rank', type=int, default=None,
                       help='Rank of this process (overrides MPI detection)')
    parser.add_argument('--world_size', type=int, default=None,
                       help='Total number of distributed processes (overrides MPI detection)')
    parser.add_argument('--walltime', type=float, default=None,
                       help='Walltime limit in hours (optional)')
    parser.add_argument('--walltime_buffer', type=float, default=30.0,
                       help='Safety buffer in minutes before walltime (default: 30)')
    
    args = parser.parse_args()
    
    # Record start time for walltime tracking
    start_time = time.time()
    
    # Validate arguments
    if args.phase == 'shoot' and args.interface is None:
        parser.error("--interface is required when --phase=shoot")
    
    # Load configs
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.ffs_config, 'r') as f:
        ffs_config = yaml.safe_load(f)

    # Get MPI-detected values
    _, mpi_world_rank, mpi_world_size = get_rank_info(model_config.get("trainer", {}).get("mode", "fsdp"))
    
    # Use parser overrides if provided, otherwise use MPI values
    rank = args.rank if args.rank is not None else mpi_world_rank
    world_size = args.world_size if args.world_size is not None else mpi_world_size
    
    # Check for single IC mode
    single_ic_mode = ffs_config.get('single_ic_mode', False)
    
    # Get all forecast times
    all_forecast_times = ffs_config['forecast_times']
    
    # Select forecast times based on mode
    if single_ic_mode:
        # SINGLE IC MODE: Get ic_index from args or config
        ic_index = args.ic_index if args.ic_index is not None else ffs_config.get('ic_index', 0)
        
        # Validate ic_index
        if ic_index < 0 or ic_index >= len(all_forecast_times):
            raise ValueError(f"ic_index {ic_index} out of range [0, {len(all_forecast_times)-1}]")
        
        # All ranks process the SAME single IC
        forecast_subset = [all_forecast_times[ic_index]]
        
        if rank == 0:
            logging.info(f"\n{'='*80}")
            logging.info("SINGLE IC MODE ENABLED (CPS-Enhanced)")
            logging.info(f"{'='*80}")
            logging.info(f"All {world_size} GPUs will work on IC index {ic_index}")
            logging.info(f"IC: {forecast_subset[0][0]} → {forecast_subset[0][1]}")
            logging.info(f"Total ICs available: {len(all_forecast_times)}")
            logging.info(f"Each IC will use all {world_size} GPUs in parallel")
            logging.info(f"Physics-based extratropical filtering enabled")
            logging.info(f"{'='*80}")
    else:
        # DISTRIBUTED IC MODE: Each rank gets different ICs
        forecast_subset = [all_forecast_times[i] for i in range(len(all_forecast_times)) 
                          if i % world_size == rank]
        
        if len(forecast_subset) == 0:
            logging.info(f"[Rank {rank}] No forecast times assigned, exiting")
            return
        
        if rank == 0:
            logging.info(f"\n{'='*80}")
            logging.info("DISTRIBUTED IC MODE (CPS-Enhanced)")
            logging.info(f"{'='*80}")
            logging.info(f"Total ICs: {len(all_forecast_times)}")
            logging.info("Each GPU works on different ICs independently")
            logging.info(f"ICs per GPU: ~{len(all_forecast_times) // world_size}")
            logging.info(f"Physics-based extratropical filtering enabled")
            logging.info(f"{'='*80}")
    
    # Print job info
    if rank == 0:
        logging.info(f"\n{'='*80}")
        logging.info("PARALLEL FFS CONFIGURATION")
        logging.info(f"{'='*80}")
        logging.info(f"Phase: {args.phase}")
        if args.phase == 'shoot':
            logging.info(f"Interface: {args.interface}")
        if args.walltime is not None:
            logging.info(f"Walltime: {args.walltime:.2f}h (buffer: {args.walltime_buffer:.0f} min)")
        logging.info(f"World size: {world_size} GPUs")
        logging.info(f"Workers per GPU: {args.num_workers}")
        logging.info(f"{'='*80}\n")
    
    logging.info(f"[Rank {rank}] Forecast times assigned: {len(forecast_subset)}")
    for ic_idx, (ic_start, ic_end) in enumerate(forecast_subset):
        logging.info(f"  [{rank}] IC {ic_idx}: {ic_start} → {ic_end}")
    
    # Process each IC assigned to this rank
    for ic_idx, (ic_start, ic_end) in enumerate(forecast_subset):
        # Check walltime before processing next IC (only relevant in distributed mode)
        if args.walltime is not None and not single_ic_mode:
            has_time, elapsed, remaining = check_walltime_remaining(
                start_time, args.walltime, args.walltime_buffer
            )
            if not has_time:
                logging.warning(f"\n{'='*80}")
                logging.warning(f"[Rank {rank}] ⏰ WALLTIME LIMIT APPROACHING")
                logging.warning(f"{'='*80}")
                logging.warning(f"Elapsed: {elapsed:.2f}h / {args.walltime:.2f}h")
                logging.warning(f"Remaining: {remaining:.2f}h (< {args.walltime_buffer:.0f} min buffer)")
                logging.warning(f"Stopping before IC {ic_idx+1}/{len(forecast_subset)}: {ic_start}")
                logging.warning(f"Processed {ic_idx}/{len(forecast_subset)} ICs")
                logging.warning(f"{'='*80}\n")
                break
        
        if rank == 0:
            logging.info(f"\n{'='*80}")
            if single_ic_mode:
                logging.info(f"ALL RANKS: Processing IC index {ffs_config.get('ic_index', args.ic_index)}")
            else:
                logging.info(f"[Rank {rank}] Processing IC {ic_idx+1}/{len(forecast_subset)}")
            logging.info(f"IC: {ic_start} → {ic_end}")
            logging.info(f"{'='*80}\n")
        
        if args.phase == 'flux':
            run_flux_phase(
                model_config=model_config,
                ffs_config=ffs_config,
                ic_start=ic_start,
                ic_end=ic_end,
                num_workers=args.num_workers,
                rank=rank,
                world_size=world_size,
                start_time=start_time,
                walltime_hours=args.walltime,
                buffer_minutes=args.walltime_buffer
            )
        
        elif args.phase == 'shoot':
            run_shoot_phase(
                model_config=model_config,
                ffs_config=ffs_config,
                ic_start=ic_start,
                ic_end=ic_end,
                interface_idx=args.interface,
                num_workers=args.num_workers,
                rank=rank,
                world_size=world_size,
                start_time=start_time,
                walltime_hours=args.walltime,
                buffer_minutes=args.walltime_buffer
            )
    
    # Final time report
    if args.walltime is not None:
        final_elapsed = (time.time() - start_time) / 3600.0
        logging.info(f"\n{'='*80}")
        logging.info(f"[Rank {rank}] COMPLETED - Total elapsed: {final_elapsed:.2f}h / {args.walltime:.2f}h")
        logging.info(f"{'='*80}\n")
    else:
        logging.info(f"\n[Rank {rank}] All ICs processed, exiting")


if __name__ == "__main__":
    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    # see if we are in debug mode to set logging level
    gettrace = getattr(sys, "gettrace", None)
    debug = gettrace()
    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    logging.debug("logging set to DEBUG level")
    main()