#!/usr/bin/env python
"""
Parallel Forward Flux Sampling for Hurricane Genesis with CPS

INTERFACE INDEXING:
- interface_idx=0 means λ₀ (first interface)
  - Reads from: flux/lambda0_config_*.pkl
  - Shoots to: 1/lambda1_config_*.pkl
  
- interface_idx=1 means λ₁ (second interface)
  - Reads from: 1/lambda1_config_*.pkl  
  - Shoots to: 2/lambda2_config_*.pkl

Usage:
    # Flux generation
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux
    
    # Shooting from λ₀ (interface_idx=0)
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 0
    
    # Shooting from λ₁ (interface_idx=1)
    python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 1
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
from tails.hurricane_genesis_ffs import HurricaneGenesisFFS


def format_ic_dirname(ic_time_str: str) -> str:
    """Convert '2022-08-28 00:00:00' to '2022-08-28T00Z'"""
    dt = datetime.strptime(ic_time_str, '%Y-%m-%d %H:%M:%S')
    return dt.strftime('%Y-%m-%dT%HZ')


def check_walltime_remaining(start_time: float, walltime_hours: float, buffer_minutes: float = 30.0) -> tuple:
    """Check if we have enough time remaining before hitting walltime."""
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
    ic_hash = hash(ic_start) % 10000
    unique_seed = base_seed + rank * 100000 + worker_id * 1000 + ic_hash
    
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    np.random.seed(unique_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = 'cuda:0'
    torch.cuda.empty_cache()
    
    # Load model
    conf = model_config.copy()
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)

    model = load_model(conf)
    save_loc = os.path.expandvars(conf["save_loc"])
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location="cpu")
    _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
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

    # Create dataset
    from datetime import datetime, timedelta
    flux_length_days = ffs_config.get('flux_length_days', 15)
    dataset_days = flux_length_days * 3  # run dataset 3x flux window so storms can resolve
    ic_start_dt = datetime.strptime(ic_start, '%Y-%m-%d %H:%M:%S')
    ic_end_extended = (ic_start_dt + timedelta(days=dataset_days)).strftime('%Y-%m-%d %H:%M:%S')
    forecast_times = [[ic_start, ic_end_extended]]

    logging.info(f"[Worker {worker_id}] Flux generation: {ic_start} → {ic_end_extended} ({dataset_days} days)")
    logging.info(f"[Worker {worker_id}] New storm tracking stops after {flux_length_days} days")
    logging.info(f"[Worker {worker_id}] Existing storms tracked until dissipation or B-state (up to day {dataset_days} from IC)")
    logging.info(f"[Worker {worker_id}] This ensures proper flux statistics: genesis events counted in first {flux_length_days} days, storms allowed to resolve")

    dataset = Predict_Dataset_Batcher(**dataset_params, fcst_datetime=forecast_times)
    loader = BatchForecastLenDataLoader(dataset)

    # Setup output directory
    ic_dirname = format_ic_dirname(ic_start)
    
    # Initialize FFS with CPS enabled
    use_cps = ffs_config.get('use_cps', False)
    ffs = HurricaneGenesisFFS(
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
        ic_dirname=ic_dirname,
        use_cps=use_cps,
        flux_length_days=flux_length_days,
        shoot_length_days=ffs_config.get('shoot_length_days', 10),
    )

    logging.info(f"[Worker {worker_id}] Starting for {ic_dirname}")
    if use_cps:
        logging.info(f"[Worker {worker_id}] ✓ CPS-enhanced FFS initialized")
    
    ffs.generate_flux(loader, n_trials=n_trials_total)
    
    torch.cuda.empty_cache()
    
    config_paths = list(ffs.flux_dir.glob('lambda0_config_*.pkl'))
    
    return {
        'worker_id': worker_id,
        'phase': 'flux',
        'configs_generated': len(config_paths)
    }


def shooting_worker(worker_id: int,
                   ffs_config: dict,
                   model_config: dict,
                   ic_start: str,
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
    ic_hash = hash(ic_start) % 10000
    unique_seed = base_seed + rank * 100000 + worker_id * 1000 + ic_hash
    
    torch.manual_seed(unique_seed)
    torch.cuda.manual_seed_all(unique_seed)
    np.random.seed(unique_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = 'cuda:0'
    torch.cuda.empty_cache()
    
    # Load model
    conf = model_config.copy()
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)
    
    model = load_model(conf)
    save_loc = os.path.expandvars(conf["save_loc"])
    ckpt = os.path.join(save_loc, "checkpoint.pt")
    checkpoint = torch.load(ckpt, map_location="cpu")
    _ = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
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
    from datetime import datetime, timedelta
    shoot_length_days = ffs_config.get('shoot_length_days', 10)
    ic_start_dt = datetime.strptime(ic_start, '%Y-%m-%d %H:%M:%S')
    ic_end = (ic_start_dt + timedelta(days=shoot_length_days)).strftime('%Y-%m-%d %H:%M:%S')
    forecast_times = [[ic_start, ic_end]]
    dataset = Predict_Dataset_Batcher(**dataset_params, fcst_datetime=forecast_times)

    # Setup output directory
    ic_dirname = format_ic_dirname(ic_start)

    # Initialize FFS with CPS enabled
    use_cps = ffs_config.get('use_cps', False)
    ffs = HurricaneGenesisFFS(
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
        ic_dirname=ic_dirname,
        use_cps=use_cps,
        flux_length_days=ffs_config.get('flux_length_days', 15),
        shoot_length_days=shoot_length_days,
    )
    
    # Load shared configs
    loaded_configs = []
    for config_path in shared_config_pool:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
            loaded_configs.append(config)
    
    ffs.interface_configs[interface_idx] = loaded_configs
    
    lambda_label = interface_idx
    logging.info(f"[SHOOT Worker {worker_id}] Starting λ_{lambda_label}, pool size: {len(loaded_configs)}")
    if use_cps:
        logging.info(f"[SHOOT Worker {worker_id}] ✓ CPS-enhanced FFS initialized")
    
    # Run shooting
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
                  num_workers: int,
                  rank: int,
                  world_size: int,
                  start_time: float = None,
                  walltime_hours: float = None,
                  buffer_minutes: float = 30.0):
    """Run flux generation phase."""
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
        use_cps = ffs_config.get('use_cps', False)
        cps_str = " (CPS-Enhanced)" if use_cps else ""
        logging.info(f"FLUX GENERATION{cps_str} - {ic_dirname}")
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
                   interface_idx: int,
                   num_workers: int,
                   rank: int,
                   world_size: int,
                   start_time: float = None,
                   walltime_hours: float = None,
                   buffer_minutes: float = 30.0):
    """Run shooting phase for a specific interface."""
    if start_time is not None and walltime_hours is not None:
        has_time, elapsed, remaining = check_walltime_remaining(start_time, walltime_hours, buffer_minutes)
        if not has_time:
            logging.warning("⏰ WALLTIME LIMIT APPROACHING - Skipping shoot phase")
            logging.warning(f"   Elapsed: {elapsed:.2f}h, Remaining: {remaining:.2f}h")
            return
        else:
            logging.info(f"⏱ Time check: Elapsed {elapsed:.2f}h, Remaining {remaining:.2f}h")
    
    ic_dirname = format_ic_dirname(ic_start)
    next_interface = interface_idx + 1
    shoot_dir = Path(ffs_config['output_dir']) / ic_dirname / str(next_interface)

    lambda_label = interface_idx
    
    existing_configs = count_configs_in_directory(shoot_dir, f'lambda{next_interface}_config_*.pkl')
    n_shoot_total = ffs_config['n_shoot_per_interface']
    
    if rank == 0:
        logging.info(f"\n{'='*80}")
        use_cps = ffs_config.get('use_cps', False)
        cps_str = " (CPS-Enhanced)" if use_cps else ""
        logging.info(f"SHOOTING{cps_str} λ_{lambda_label} → λ_{next_interface} - {ic_dirname}")
        logging.info(f"{'='*80}")
        logging.info(f"Target configs: {n_shoot_total}")
        logging.info(f"Existing configs: {existing_configs}")
        
        if existing_configs >= n_shoot_total:
            logging.info("✓ Shooting already complete, skipping")
            logging.info(f"{'='*80}\n")
            return
    
    # Get source configs
    if interface_idx == 0:
        source_dir = Path(ffs_config['output_dir']) / ic_dirname / 'flux'
        source_pattern = 'lambda0_config_*.pkl'
    else:
        source_dir = Path(ffs_config['output_dir']) / ic_dirname / str(interface_idx)
        source_pattern = f'lambda{interface_idx}_config_*.pkl'
    
    source_configs = list(source_dir.glob(source_pattern))
    
    if len(source_configs) == 0:
        logging.warning(f"⚠ No source configs found in {source_dir}")
        logging.warning(f"   Looking for pattern: {source_pattern}")
        logging.info(f"{'='*80}\n")
        return
    
    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Source pattern: {source_pattern}")
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
    
    final_count = count_configs_in_directory(shoot_dir, f'lambda{next_interface}_config_*.pkl')
    
    logging.info(f"\n✓ SHOOTING λ_{lambda_label} COMPLETE")
    logging.info(f"Total configs: {final_count}/{n_shoot_total}")
    logging.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run parallel FFS for hurricane genesis with optional CPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Flux generation
        python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase flux
        
        # Shooting from λ₀ (interface_idx=0)
        python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 0
        
        # Shooting from λ₁ (interface_idx=1)
        python run_parallel_ffs.py --model_config model.yml --ffs_config ffs.yml --phase shoot --interface 1
        """
    )
    
    parser.add_argument('--model_config', type=str, required=True, help='Path to model.yml')
    parser.add_argument('--ffs_config', type=str, required=True, help='Path to ffs.yml')
    parser.add_argument('--phase', type=str, required=True, choices=['flux', 'shoot'],
                       help='Which phase to run: flux or shoot')
    parser.add_argument('--interface', type=int, default=None,
                       help='Interface index for shooting (0=λ₀, 1=λ₁, etc)')
    parser.add_argument('--ic_index', type=int, default=None,
                       help='IC index to process in single_ic_mode')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--rank', type=int, default=None,
                       help='Rank of this process')
    parser.add_argument('--world_size', type=int, default=None,
                       help='Total number of distributed processes')
    parser.add_argument('--walltime', type=float, default=None,
                       help='Walltime limit in hours')
    parser.add_argument('--walltime_buffer', type=float, default=30.0,
                       help='Safety buffer in minutes before walltime')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.phase == 'shoot' and args.interface is None:
        parser.error("--interface is required when --phase=shoot")
    
    # Load configs
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.ffs_config, 'r') as f:
        ffs_config = yaml.safe_load(f)

    # Get rank info
    _, mpi_world_rank, mpi_world_size = get_rank_info(model_config.get("trainer", {}).get("mode", "fsdp"))
    rank = args.rank if args.rank is not None else mpi_world_rank
    world_size = args.world_size if args.world_size is not None else mpi_world_size
    
    # Check for single IC mode
    single_ic_mode = ffs_config.get('single_ic_mode', False)
    all_forecast_start_times = ffs_config['forecast_start_times']

    if single_ic_mode:
        ic_index = args.ic_index if args.ic_index is not None else ffs_config.get('ic_index', 0)

        if ic_index < 0 or ic_index >= len(all_forecast_start_times):
            raise ValueError(f"ic_index {ic_index} out of range [0, {len(all_forecast_start_times)-1}]")

        forecast_subset = [all_forecast_start_times[ic_index]]

        if rank == 0:
            logging.info(f"\n{'='*80}")
            use_cps = ffs_config.get('use_cps', False)
            cps_str = " (CPS-Enhanced)" if use_cps else ""
            logging.info(f"SINGLE IC MODE{cps_str}")
            logging.info(f"{'='*80}")
            logging.info(f"All {world_size} GPUs working on IC index {ic_index}")
            logging.info(f"IC start: {forecast_subset[0]}")
            logging.info(f"{'='*80}")
    else:
        forecast_subset = [all_forecast_start_times[i] for i in range(len(all_forecast_start_times))
                          if i % world_size == rank]

        if len(forecast_subset) == 0:
            logging.info(f"[Rank {rank}] No forecast times assigned, exiting")
            return

        if rank == 0:
            logging.info(f"\n{'='*80}")
            use_cps = ffs_config.get('use_cps', False)
            cps_str = " (CPS-Enhanced)" if use_cps else ""
            logging.info(f"DISTRIBUTED IC MODE{cps_str}")
            logging.info(f"{'='*80}")
            logging.info(f"Total ICs: {len(all_forecast_start_times)}")
            logging.info(f"{'='*80}")

    if rank == 0:
        logging.info(f"\n{'='*80}")
        logging.info("PARALLEL FFS CONFIGURATION")
        logging.info(f"{'='*80}")
        logging.info(f"Phase: {args.phase}")
        if args.phase == 'shoot':
            lambda_label = args.interface
            logging.info(f"Interface: {args.interface} (λ_{lambda_label})")
        logging.info(f"Flux length: {ffs_config.get('flux_length_days', 15)} days")
        logging.info(f"Shoot length: {ffs_config.get('shoot_length_days', 10)} days")
        if args.walltime is not None:
            logging.info(f"Walltime: {args.walltime:.2f}h (buffer: {args.walltime_buffer:.0f} min)")
        logging.info(f"World size: {world_size} GPUs")
        logging.info(f"Workers per GPU: {args.num_workers}")
        logging.info(f"{'='*80}\n")

    # Process each IC
    for ic_idx, ic_start in enumerate(forecast_subset):
        if args.walltime is not None and not single_ic_mode:
            has_time, elapsed, remaining = check_walltime_remaining(
                start_time, args.walltime, args.walltime_buffer
            )
            if not has_time:
                logging.warning(f"\n[Rank {rank}] ⏰ WALLTIME LIMIT - Stopping")
                logging.warning(f"Processed {ic_idx}/{len(forecast_subset)} ICs")
                break

        if args.phase == 'flux':
            run_flux_phase(
                model_config=model_config,
                ffs_config=ffs_config,
                ic_start=ic_start,
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
                interface_idx=args.interface,
                num_workers=args.num_workers,
                rank=rank,
                world_size=world_size,
                start_time=start_time,
                walltime_hours=args.walltime,
                buffer_minutes=args.walltime_buffer
            )
    
    if args.walltime is not None:
        final_elapsed = (time.time() - start_time) / 3600.0
        logging.info(f"\n[Rank {rank}] COMPLETED - Elapsed: {final_elapsed:.2f}h / {args.walltime:.2f}h")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    ch = logging.StreamHandler()
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