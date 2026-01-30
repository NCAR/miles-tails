from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import time


class FFSLogger:
    """Logger for tracking FFS trajectories and config genealogy."""

    def __init__(self, output_dir: Path, rank: int = 0, world_size: int = 1, 
                 worker_id: int = 0, ic_dirname: str = None):
        """
        Initialize logger with IC-specific directory structure.
        
        Args:
            output_dir: Base output directory
            rank: MPI rank
            world_size: Total MPI processes
            worker_id: Worker ID within rank
            ic_dirname: IC date string (e.g., '2022-08-28T00Z'), if None uses 'default'
        """
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.worker_id = worker_id
        
        # Generate unique 2-character suffix
        self.char_id = self._generate_unique_char_id()
        
        # Create IC-specific logs directory
        # ic_dir = ic_dirname if ic_dirname is not None else 'default'
        self.log_dir = output_dir # / 'logs' / ic_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files go in IC-specific directory
        self.log_file = self.log_dir / f'ffs_log_world{world_size}_rank{rank}_worker{worker_id}_{self.char_id}.jsonl'
        self.summary_file = self.log_dir / f'ffs_summary_world{world_size}_rank{rank}_worker{worker_id}_{self.char_id}.json'
        
        # Initialize log file
        with open(self.log_file, 'a+') as f:
            f.write(f"# FFS Run Started: {datetime.now().isoformat()}\n")
            if ic_dirname:
                f.write(f"# IC: {ic_dirname}\n")

    def _generate_unique_char_id(self) -> str:
        """Generate unique 2-character alphanumeric ID."""
        import random
        import string
        
        chars = string.ascii_lowercase + string.digits  # a-z, 0-9
        max_attempts = 1000
        
        for _ in range(max_attempts):
            # Generate random 2-character ID
            char_id = ''.join(random.choices(chars, k=2))
            
            # Check if this combination already exists in the log directory
            # Note: log_dir might not exist yet on first attempt
            if hasattr(self, 'log_dir') and self.log_dir.exists():
                test_file = self.log_dir / f'ffs_log_world{self.world_size}_rank{self.rank}_worker{self.worker_id}_{char_id}.jsonl'
            else:
                # If log_dir doesn't exist yet, check in output_dir temporarily
                test_file = self.output_dir / f'ffs_log_world{self.world_size}_rank{self.rank}_worker{self.worker_id}_{char_id}.jsonl'
            
            if not test_file.exists():
                return char_id
        
        # Fallback: use millisecond timestamp
        return f"{int(time.time() * 1000) % 100:02d}"

    def log_flux_trajectory(self, trajectory_num: int, result: Dict, configs_saved: List[str]):
        """Log flux generation trajectory."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'flux_generation',
            'trajectory_num': trajectory_num,
            'status': result['status'],
            'final_mslp': result['final_mslp'],
            'num_timesteps': len(result['mslp_trajectory']),
            'configs_saved': configs_saved,
            'is_direct_B': result['status'] == 'reached_B' and len(configs_saved) == 0,
            'mslp_trajectory': result['mslp_trajectory']
        }
        self._write_entry(entry)
    
    def log_shooting_attempt(self, interface_idx: int, attempt_num: int, 
                            parent_config: str, result: Dict, 
                            child_config: Optional[str] = None):
        """Log shooting attempt with parent-child linkage."""
        lambda_label = interface_idx - 1
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'shooting',
            'interface_idx': interface_idx,
            'lambda_label': lambda_label,
            'attempt_num': attempt_num,
            'parent_config': parent_config,
            'child_config': child_config,
            'status': result['status'],
            'final_mslp': result['final_mslp'],
            'num_timesteps': len(result['mslp_trajectory']),
            'pathway': self._extract_pathway(result),
            'mslp_trajectory': result['mslp_trajectory']
        }
        
        # Add CPS failure information if present
        if 'failure_reason' in result and result['failure_reason']:
            entry['failure_reason'] = result['failure_reason']
        
        if 'failure_cps' in result and result['failure_cps']:
            entry['failure_cps'] = result['failure_cps']
        
        self._write_entry(entry)
    
    def log_instant_success(self, interface_idx: int, parent_config: str, 
                           child_config: str, mslp_value: float):
        """Log instant success (config already satisfies next interface)."""
        lambda_label = interface_idx - 1
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'shooting',
            'interface_idx': interface_idx,
            'lambda_label': lambda_label,
            'parent_config': parent_config,
            'child_config': child_config,
            'status': 'instant_success',
            'mslp_value': mslp_value,
            'note': 'Config already satisfies next interface'
        }
        self._write_entry(entry)
    
    def log_interface_summary(self, interface_idx: int, successes: int, 
                             failures: int, total_attempts: int, 
                             P_forward: float, configs_generated: int):
        """Log summary statistics for an interface."""
        lambda_label = interface_idx - 1
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'interface_summary',
            'interface_idx': interface_idx,
            'lambda_label': lambda_label,
            'successes': successes,
            'failures': failures,
            'total_attempts': total_attempts,
            'P_forward': P_forward,
            'configs_generated': configs_generated
        }
        self._write_entry(entry)
    
    def log_final_results(self, flux_estimate: float, transition_probs: List[float], 
                         total_prob: float, direct_B_count: int = 0, 
                         direct_formation_rate: float = 0.0):
        """Log final FFS results with optional direct formation statistics.
        
        Parameters
        ----------
        flux_estimate : float
            Estimated flux Φ₀ (crossings per day)
        transition_probs : List[float]
            Forward transition probabilities for each interface
        total_prob : float
            Total FFS probability (flux × product of transition probabilities)
        direct_B_count : int, optional
            Number of direct formations to state B during flux generation
        direct_formation_rate : float, optional
            Direct formation rate (formations per day)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'final_results',
            'flux_estimate': flux_estimate,
            'transition_probs': transition_probs,
            'total_probability': total_prob,
            'enhancement_factor': 1/total_prob if total_prob > 0 else None,
            'direct_B_count': direct_B_count,
            'direct_formation_rate': direct_formation_rate,
            'ffs_to_direct_ratio': total_prob / direct_formation_rate if direct_formation_rate > 0 else None
        }
        self._write_entry(entry)
        
        # Console output for analysis runs
        if hasattr(self, '_console_output_enabled'):
            self.info("="*80)
            self.info("FFS FINAL RESULTS")
            self.info("="*80)
            self.info(f"Flux estimate (Φ₀): {flux_estimate:.6f} crossings/day")
            
            for i, p in enumerate(transition_probs):
                self.info(f"P(λ_{i} → λ_{i+1}): {p:.4f}")
            
            self.info(f"FFS Rate: {total_prob:.6e} events/day")
            
            if direct_B_count > 0:
                self.info(f"Direct B formations: {direct_B_count}")
                self.info(f"Direct formation rate: {direct_formation_rate:.6e} events/day")
                
                if direct_formation_rate > 0 and total_prob > 0:
                    ratio = total_prob / direct_formation_rate
                    self.info(f"FFS rate / Direct rate = {ratio:.2f}x")
        
        # Also save to summary file
        with open(self.summary_file, 'w') as f:
            json.dump(entry, f, indent=2)
    
    def log_phase_start(self, phase: str, **kwargs):
        """Log the start of an analysis phase."""
        self._console_output_enabled = True
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': f'{phase}_start',
            **kwargs
        }
        self._write_entry(entry)
        
        # Console output
        print(f"{'='*80}")
        print(f"PHASE: {phase.upper()}")
        if kwargs:
            for key, value in kwargs.items():
                print(f"  {key}: {value}")
        print(f"{'='*80}")
    
    def log_phase_end(self, phase: str):
        """Log the end of an analysis phase."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': f'{phase}_end'
        }
        self._write_entry(entry)
    
    def info(self, message: str):
        """Log informational message with console output."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'level': 'info',
            'message': message
        }
        self._write_entry(entry)
        print(message)
    
    def warning(self, message: str):
        """Log warning message with console output."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'level': 'warning',
            'message': message
        }
        self._write_entry(entry)
        print(f"⚠ WARNING: {message}")
    
    def close(self):
        """Close the logger and finalize log file."""
        with open(self.log_file, 'a') as f:
            f.write(f"# FFS Run Ended: {datetime.now().isoformat()}\n")
    
    def _extract_pathway(self, result: Dict) -> List[Dict]:
        """Extract pathway of interface crossings from result."""
        pathway = []
        for crossing in result.get('crossings', []):
            pathway.append({
                'interface_idx': crossing.interface_idx,
                'lambda_label': crossing.interface_idx - 1,
                'mslp_value': crossing.mslp_value,
                'config_name': crossing.config_name
            })
        return pathway
    
    def _write_entry(self, entry: Dict):
        """Write entry to log file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def build_genealogy_tree(self) -> Dict:
        """Build complete genealogy tree from log file."""
        genealogy = {}
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if entry.get('phase') == 'shooting':
                    parent = entry.get('parent_config')
                    child = entry.get('child_config')
                    
                    if parent and child:
                        if parent not in genealogy:
                            genealogy[parent] = []
                        genealogy[parent].append({
                            'child': child,
                            'interface': entry.get('interface_idx'),
                            'status': entry.get('status'),
                            'timestamp': entry.get('timestamp')
                        })
        
        return genealogy
    
    def trace_pathway_to_B(self, stateB_config: str) -> List[str]:
        """Trace pathway from initial state to state B."""
        genealogy = self.build_genealogy_tree()
        
        # Build reverse lookup (child -> parent)
        reverse_lookup = {}
        for parent, children in genealogy.items():
            for child_info in children:
                reverse_lookup[child_info['child']] = parent
        
        # Trace backwards from B to start
        pathway = [stateB_config]
        current = stateB_config
        
        while current in reverse_lookup:
            parent = reverse_lookup[current]
            pathway.append(parent)
            current = parent
        
        return list(reversed(pathway))

    def log_early_stop(self, reason: str, step: int, mslp: float, 
                    interface_idx: Optional[int] = None, 
                    location: Optional[Tuple[float, float]] = None,
                    cps_params: Optional[Dict] = None,
                    mode: str = 'shoot',
                    parent_config: Optional[str] = None):
        """
        Log early stopping of trajectory.
        
        Parameters
        ----------
        reason : str
            Reason for stopping: 'extratropical', 'reached_B', 'returned_A', 
            'returned_backward', 'completed', 'failure'
        step : int
            Forecast step where stopping occurred
        mslp : float
            MSLP value at stopping point (hPa)
        interface_idx : int, optional
            Current interface index if relevant
        location : tuple, optional
            (lat, lon) location if available
        cps_params : dict, optional
            CPS parameters if extratropical: {B, VTL, VTU, phase}
        mode : str
            'flux' or 'shoot'
        parent_config : str, optional
            Parent configuration name for shoot mode
        """
        lambda_label = interface_idx - 1 if interface_idx is not None and interface_idx > 0 else None
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'early_stop',
            'mode': mode,
            'reason': reason,
            'step': step,
            'mslp': mslp,
            'interface_idx': interface_idx,
            'lambda_label': lambda_label,
            'parent_config': parent_config
        }
        
        if location:
            entry['location'] = {'lat': location[0], 'lon': location[1]}
        
        if cps_params:
            entry['cps'] = cps_params
        
        self._write_entry(entry)
        
        # Console output for debugging
        if hasattr(self, '_console_output_enabled'):
            reason_str = reason.upper().replace('_', ' ')
            msg = f"  ⏹ STOP: {reason_str} at step {step}, MSLP={mslp:.1f} hPa"
            
            if interface_idx is not None:
                msg += f", λ={interface_idx}"
            
            if location:
                lat, lon = location
                msg += f", ({lat:.1f}°N, {abs(lon):.1f}°W)"
            
            if cps_params:
                msg += f", {cps_params.get('phase', 'Unknown')} (B={cps_params.get('B', 0):.1f}m)"
            
            print(msg)


    def log_trajectory_stats(self, mode: str, duration_steps: int, 
                            max_mslp: float, min_mslp: float,
                            interface_crossings: int = 0):
        """
        Log summary statistics for a completed trajectory.
        
        Parameters
        ----------
        mode : str
            'flux' or 'shoot'
        duration_steps : int
            Number of forecast steps
        max_mslp : float
            Maximum MSLP reached (hPa)
        min_mslp : float
            Minimum MSLP reached (hPa)
        interface_crossings : int
            Number of interface crossings
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size,
            'worker_id': self.worker_id,
            'phase': 'trajectory_stats',
            'mode': mode,
            'duration_steps': duration_steps,
            'max_mslp': max_mslp,
            'min_mslp': min_mslp,
            'mslp_range': max_mslp - min_mslp,
            'interface_crossings': interface_crossings
        }
        self._write_entry(entry)