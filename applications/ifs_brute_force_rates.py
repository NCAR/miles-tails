"""
Compute brute force hurricane genesis rates from IFS ensemble forecasts with CPS filtering.

Uses TOBAC tracking to detect and track low pressure systems, counting
how many trajectories successfully develop into hurricanes. Implements
adaptive spatial tracking with jump detection and CPS-based extratropical filtering.

Rate = N_success / N_total_trajectories

Usage:
    python compute_ifs_brute_force_rates.py --ffs_config ffs.yml
"""

import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Optional

import yaml
import argparse
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing as mp

try:
    import tobac
    import iris
    TOBAC_AVAILABLE = True
except ImportError:
    TOBAC_AVAILABLE = False
    print("WARNING: tobac/iris not available, using fallback MSLP extraction")

from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings('ignore')

# Import CPS tracker if available
try:
    from tails.cyclone_phase_tracker import CyclonePhaseTracker
    CPS_AVAILABLE = True
except ImportError:
    CPS_AVAILABLE = False
    print("WARNING: CPS tracker not available, extratropical filtering disabled")

# Import CREDIT interpolation if available for CPS
try:
    from credit.interp import full_state_pressure_interpolation
    CREDIT_INTERP_AVAILABLE = True
except ImportError:
    CREDIT_INTERP_AVAILABLE = False
    if CPS_AVAILABLE:
        print("WARNING: CREDIT interpolation not available, CPS will be disabled")
        CPS_AVAILABLE = False


class IFSBruteForceRateEstimator:
    """
    Estimate brute force hurricane genesis rates from IFS ensemble.
    
    Uses TOBAC tracking to identify and follow individual systems with
    adaptive spatial search, trajectory continuity validation, and optional
    CPS-based extratropical filtering.
    """
    
    def __init__(self, 
                 ifs_path: str,
                 state_A: float = 1008.0,
                 state_B: float = 982.0,
                 interfaces: list = None,
                 basin: Dict = None,
                 cps_config: Dict = None):
        """
        Initialize rate estimator.
        
        Args:
            ifs_path: Path to IFS zarr dataset
            state_A: MSLP threshold for state A (no organized system)
            state_B: MSLP threshold for state B (hurricane)
            interfaces: List of interface thresholds to track crossings
            basin: Dict with lat_min, lat_max, lon_min, lon_max
            cps_config: Configuration dict for CPS (radius_km, etc.)
        """
        self.ifs_path = Path(ifs_path)
        self.state_A = state_A
        self.state_B = state_B
        self.interfaces = interfaces if interfaces is not None else []
        self.interfaces.append(state_B)
        
        # Default to Atlantic basin
        if basin is None:
            self.basin = {
                'lat_min': 10.0,
                'lat_max': 40.0,
                'lon_min': -100.0,
                'lon_max': -20.0
            }
        else:
            self.basin = basin
        
        # CPS setup - ALWAYS ENABLED
        self.enable_cps = CPS_AVAILABLE
        if self.enable_cps:
            cps_config = cps_config or {}
            radius_km = cps_config.get('radius_km', 500)
            
            # Load latlons for CPS tracker
            try:
                latlons = xr.open_dataset(cps_config.get('latitude_weights_path', 
                    '/glade/campaign/cisl/aiml/wchapman/MLWPS/DTR/lat_lon_plus_landmask.nc')).load()
                self.cps_tracker = CyclonePhaseTracker(latlons, radius_km=radius_km)
                
                # Load static fields for pressure interpolation
                static_path = cps_config.get('static_path', 
                    '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc')
                with xr.open_dataset(static_path) as df:
                    self.surface_geopotential = df["Z_GDS4_SFC"].values
                    self.land_sea_mask = df["LSM"].values
                
                # Load CREDIT config for pressure interpolation
                credit_config_path = cps_config.get('credit_config_path')
                if credit_config_path:
                    with open(credit_config_path, 'r') as f:
                        self.credit_config = yaml.safe_load(f)
                else:
                    self.credit_config = None
                
                self.latlons = latlons
                print("✓ CPS tracker initialized (Hart 2003 formulation)")
                print("✓ Using VTL/VTU only (B disabled due to motion uncertainty)")
            except Exception as e:
                print(f"⚠ Failed to initialize CPS tracker: {e}")
                self.enable_cps = False
        else:
            self.cps_tracker = None
            print("⚠ CPS unavailable - continuing without extratropical filtering")
        
        # Track history for CPS motion calculation
        self.storm_track_history = {}
        
        self.ds = None
        
    def load_data(self, forecast_times: list = None):
        """
        Load and filter IFS dataset based on forecast times.
        
        Args:
            forecast_times: List of [start, end] datetime strings.
                           Each entry uses the start time, forecast runs 15 days.
                           e.g., [['2022-08-21 00:00:00', '2022-09-10 00:00:00'],
                                  ['2022-08-21 12:00:00', '2022-09-10 12:00:00']]
        """
        print(f"Loading IFS data from {self.ifs_path}...")
        self.ds = xr.open_zarr(self.ifs_path, consolidated=True)
        
        if forecast_times is not None and len(forecast_times) > 0:
            # Extract all start times from forecast_times list
            init_times = [pd.to_datetime(ft[0]) for ft in forecast_times]
            
            print(f"Using {len(init_times)} initialization times from config")
            print(f"Each forecast runs 15 days from initialization")
            
            # Select these specific times
            self.ds_filtered = self.ds.sel(time=init_times)
            
            # Limit to 15-day forecasts (360 hours)
            max_lead = pd.Timedelta(days=15)
            valid_leads = self.ds_filtered.prediction_timedelta <= max_lead
            self.ds_filtered = self.ds_filtered.isel(prediction_timedelta=valid_leads)
        else:
            # Default: all 2022 data at 00Z
            ds_year = self.ds.sel(time=self.ds.time.dt.year == 2022)
            self.ds_filtered = ds_year.sel(time=ds_year.time.dt.hour == 0)
        
        print(f"Loaded {len(self.ds_filtered.time)} forecast initializations")
        print(f"Ensemble size: {len(self.ds_filtered.number)}")
        print(f"Lead times: {len(self.ds_filtered.prediction_timedelta)}")
        print(f"Variables: {list(self.ds_filtered.data_vars)}")
        
        return self.ds_filtered
    
    def get_basin_mask(self, lats: np.ndarray, lons: np.ndarray,
                      parent_location: Optional[Tuple[float, float]] = None,
                      tracking_mode: bool = False) -> np.ndarray:
        """
        Create spatial mask for basin with adaptive tracking support.
        
        Args:
            lats: Latitude array
            lons: Longitude array
            parent_location: (lat, lon) of feature from previous timestep
            tracking_mode: If True, use adaptive search box around parent
            
        Returns:
            Boolean mask array
        """
        # Convert longitude to -180 to 180
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        if tracking_mode and parent_location is not None:
            parent_lat, parent_lon = parent_location
            
            # Adaptive search box: ±10° around parent, clipped to reasonable limits
            # Allows recurvature to 50°N, westward into Gulf, eastward into Atlantic
            lat_min_search = max(self.basin['lat_min'], parent_lat - 10)
            lat_max_search = min(50.0, parent_lat + 10)
            lon_min_search = max(-110.0, parent_lon - 10)
            lon_max_search = min(10.0, parent_lon + 10)
            
            lat_mask = (lats >= lat_min_search) & (lats <= lat_max_search)
            lon_mask = (lons_180 >= lon_min_search) & (lons_180 <= lon_max_search)
        else:
            # Initial detection: use standard basin
            lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
            lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
        
        basin_mask = lat_mask[:, None] & lon_mask[None, :]
        
        return basin_mask
    
    def _compute_pressure_interpolation(self, ds_timestep: xr.Dataset) -> Optional[xr.Dataset]:
        """
        Compute pressure interpolation for CPS calculation.
        
        Args:
            ds_timestep: xarray Dataset with meteorological fields
            
        Returns:
            Interpolated dataset or None if failed
        """
        if not self.enable_cps or not CREDIT_INTERP_AVAILABLE:
            return None
        
        if self.credit_config is None:
            print("⚠ No CREDIT config available for pressure interpolation")
            return None
        
        try:
            pressure_interp = full_state_pressure_interpolation(
                ds_timestep,
                self.surface_geopotential,
                **self.credit_config.get("predict", {}).get("interp_pressure", {})
            )
            return pressure_interp
        except Exception as e:
            print(f"⚠ Pressure interpolation failed: {e}")
            return None
    
    def _check_cps(self, 
                   ds_timestep: xr.Dataset,
                   location: Tuple[float, float],
                   mslp: float,
                   storm_id: int,
                   stage: str = 'genesis') -> Tuple[bool, Optional[Dict]]:
        """
        Check if storm is extratropical using CPS.
        
        Args:
            ds_timestep: xarray Dataset with meteorological fields
            location: (lat, lon) of storm center
            mslp: MSLP value
            storm_id: Storm identifier for track history
            stage: 'genesis' or 'mature'
            
        Returns:
            (is_extratropical, cps_params)
        """
        if not self.enable_cps:
            return False, None
        
        lat, lon = location
        
        # Compute pressure interpolation
        pressure_interp = self._compute_pressure_interpolation(ds_timestep)
        if pressure_interp is None:
            return False, None
        
        # Get previous position for motion calculation
        prev_lon, prev_lat = None, None
        if storm_id in self.storm_track_history:
            hist = self.storm_track_history[storm_id]
            if len(hist['lons']) >= 1:
                prev_lon = hist['lons'][-1]
                prev_lat = hist['lats'][-1]
        
        # Update track history
        if storm_id not in self.storm_track_history:
            self.storm_track_history[storm_id] = {'lons': [], 'lats': []}
        self.storm_track_history[storm_id]['lons'].append(lon)
        self.storm_track_history[storm_id]['lats'].append(lat)
        
        # Compute CPS
        try:
            cps = self.cps_tracker.compute_CPS(
                pressure_interp,
                lon, lat, mslp,
                prev_lon=prev_lon,
                prev_lat=prev_lat,
                stage=stage
            )
            
            is_extratropical = not cps['is_tropical']
            return is_extratropical, cps
        except Exception as e:
            print(f"⚠ CPS computation failed: {e}")
            return False, None
    
    def extract_mslp_tobac(self, 
                           mslp_hpa: np.ndarray,
                           lats: np.ndarray,
                           lons: np.ndarray,
                           parent_location: Optional[Tuple[float, float]] = None,
                           tracking_mode: bool = False) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Extract minimum MSLP using TOBAC feature detection with adaptive tracking.
        
        Args:
            mslp_hpa: MSLP field in hPa
            lats: Latitude coordinates
            lons: Longitude coordinates
            parent_location: (lat, lon) from previous timestep for tracking
            tracking_mode: If True, use adaptive search box around parent
        
        Returns:
            (min_mslp, (lat, lon)) or (min_mslp, None) if detection fails
        """
        if not TOBAC_AVAILABLE:
            return self._extract_mslp_fallback(mslp_hpa, lats, lons, 
                                              parent_location, tracking_mode)
        
        # Convert to -180:180
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        # Apply adaptive basin mask
        basin_mask = self.get_basin_mask(lats, lons, parent_location, tracking_mode)
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
        # Sort longitudes for tobac (requires monotonic coordinates)
        mslp_tobac = mslp_basin.copy()
        lons_tobac = lons_180.copy()
        
        if not np.all(np.diff(lons_tobac) > 0):
            sort_idx = np.argsort(lons_tobac)
            lons_tobac = lons_tobac[sort_idx]
            mslp_tobac = mslp_tobac[:, sort_idx]
        
        try:
            # Create iris cube
            lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
            lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
            time_coord = iris.coords.DimCoord([0], standard_name='time', units='hours since 2024-01-01 00:00:00')
            
            cube = iris.cube.Cube(
                mslp_tobac[np.newaxis, :, :],
                standard_name='air_pressure_at_mean_sea_level',
                units='hPa',
                dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
            )
            
            # Set up thresholds for tobac
            basin_min = np.nanmin(mslp_tobac)
            basin_max = np.nanmax(mslp_tobac)
            thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
            thresholds = sorted(thresholds, reverse=False)
            
            # Run feature detection
            features = tobac.feature_detection_multithreshold(
                field_in=cube,
                dxy=111000,  # grid spacing in meters
                threshold=thresholds,
                target='minimum',
                position_threshold='weighted_diff',
                sigma_threshold=1.5,
                n_min_threshold=3
            )
            
            if features is not None and len(features) > 0:
                # Filter by proximity to parent if tracking
                if tracking_mode and parent_location is not None:
                    parent_lat, parent_lon = parent_location
                    
                    features['lat_diff'] = np.abs(features['latitude'] - parent_lat)
                    features['lon_diff'] = np.abs(features['longitude'] - parent_lon)
                    
                    # Select features within 10° of parent
                    nearby = features[(features['lat_diff'] < 10) & (features['lon_diff'] < 10)]
                    
                    if len(nearby) > 0:
                        features_to_use = nearby
                    else:
                        features_to_use = features
                else:
                    features_to_use = features
                
                # Get strongest feature (lowest MSLP)
                strongest_idx = features_to_use['threshold_value'].idxmin()
                strongest = features_to_use.loc[strongest_idx]
                
                feature_lat = float(strongest['latitude'])
                feature_lon = float(strongest['longitude'])
                
                # Get MSLP value at feature location
                lat_idx = np.argmin(np.abs(lats - feature_lat))
                lon_idx = np.argmin(np.abs(lons_180 - feature_lon))
                
                min_mslp = float(mslp_basin[lat_idx, lon_idx])
                
                if not np.isnan(min_mslp):
                    return min_mslp, (feature_lat, feature_lon)
            
            # Fallback if no features found
            return self._extract_mslp_fallback(mslp_hpa, lats, lons,
                                              parent_location, tracking_mode)
            
        except Exception as e:
            print(f"⚠ TOBAC error: {e}, using fallback")
            return self._extract_mslp_fallback(mslp_hpa, lats, lons,
                                              parent_location, tracking_mode)
    
    def _extract_mslp_fallback(self, 
                                mslp_hpa: np.ndarray,
                                lats: np.ndarray,
                                lons: np.ndarray,
                                parent_location: Optional[Tuple[float, float]] = None,
                                tracking_mode: bool = False) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Fallback MSLP extraction using simple minimum with adaptive tracking.
        
        Args:
            mslp_hpa: MSLP field in hPa
            lats: Latitude coordinates
            lons: Longitude coordinates
            parent_location: (lat, lon) from previous timestep
            tracking_mode: If True, use adaptive search box
            
        Returns:
            (min_mslp, (lat, lon))
        """
        lons_180 = np.where(lons > 180, lons - 360, lons)
        basin_mask = self.get_basin_mask(lats, lons, parent_location, tracking_mode)
        
        # Smooth for location
        mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        mslp_masked = np.where(basin_mask, mslp_smooth, np.nan)
        
        if np.all(np.isnan(mslp_masked)):
            return np.nan, None
        
        min_idx = np.nanargmin(mslp_masked)
        min_row, min_col = np.unravel_index(min_idx, mslp_masked.shape)
        
        # Get value from unsmoothed data
        min_mslp = float(mslp_hpa[min_row, min_col])
        min_lat = lats[min_row]
        min_lon = lons_180[min_col]
        
        return min_mslp, (min_lat, min_lon)
    
    def process_single_trajectory(self,
                                  init_time,
                                  member: int) -> Dict:
        """
        Process a single trajectory (one initialization + one ensemble member)
        with adaptive tracking, jump detection, and optional CPS filtering.
        
        Returns:
            Dict with trajectory info including interface crossing status and times
        """
        # Select this trajectory
        traj = self.ds_filtered.sel(time=init_time, number=member)
        
        # Get coordinates
        lats = traj.latitude.values
        lons = traj.longitude.values
        
        # Track MSLP over time
        mslp_trajectory = []
        locations = []
        previous_location = None
        
        # Track interface crossings (boolean and time)
        interface_crossings = {i: False for i in range(len(self.interfaces))}
        interface_crossing_times = {i: None for i in range(len(self.interfaces))}
        
        # CPS rejection tracking
        cps_rejected = False
        cps_rejection_info = None
        
        # Initialize storm ID for CPS tracking
        storm_id = member  # Use member number as unique storm ID
        
        # Check initial condition (no tracking)
        mslp_init_raw = traj.mean_sea_level_pressure.isel(prediction_timedelta=0).values / 100.0
        # Ensure correct dimension order (latitude, longitude)
        if mslp_init_raw.shape != (len(lats), len(lons)):
            mslp_init = mslp_init_raw.T
        else:
            mslp_init = mslp_init_raw
        
        mslp_init_min, loc_init = self.extract_mslp_tobac(
            mslp_init, lats, lons,
            parent_location=None,
            tracking_mode=False
        )
        
        previous_location = loc_init
        starts_in_A = mslp_init_min > self.state_A
        
        # Process each lead time with adaptive tracking
        for lead_idx, lead_time in enumerate(traj.prediction_timedelta.values):
            mslp_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=lead_idx).values
            mslp_hpa_raw = mslp_pa / 100.0
            # Ensure correct dimension order (latitude, longitude)
            if mslp_hpa_raw.shape != (len(lats), len(lons)):
                mslp_hpa = mslp_hpa_raw.T
            else:
                mslp_hpa = mslp_hpa_raw
            
            # Use adaptive tracking after first timestep
            min_mslp, location = self.extract_mslp_tobac(
                mslp_hpa, lats, lons,
                parent_location=previous_location,
                tracking_mode=(lead_idx > 0)
            )
            
            # Check for location jump (>10° = lost track)
            if previous_location and location and lead_idx > 0:
                prev_lat, prev_lon = previous_location
                curr_lat, curr_lon = location
                lat_diff = abs(curr_lat - prev_lat)
                lon_diff = abs(curr_lon - prev_lon)
                
                if lat_diff > 10 or lon_diff > 10:
                    # Lost track - trajectory failed
                    max_lead_hours = lead_time / np.timedelta64(1, 'h')
                    result = {
                        'init_time': init_time,
                        'member': member,
                        'success': False,
                        'starts_in_A': starts_in_A,
                        'mslp_trajectory': mslp_trajectory,
                        'locations': locations,
                        'min_mslp': np.min(mslp_trajectory) if mslp_trajectory else np.nan,
                        'success_lead_time': None,
                        'success_lead_hours': max_lead_hours,
                        'lost_track': True,
                        'lost_track_at_hour': max_lead_hours,
                        'cps_rejected': cps_rejected,
                        'cps_rejection_info': cps_rejection_info
                    }
                    # Add interface crossings up to this point
                    for i, crossed in interface_crossings.items():
                        result[f'crossed_interface_{i}'] = crossed
                        result[f'crossing_time_interface_{i}'] = (
                            interface_crossing_times[i] if crossed else max_lead_hours
                        )
                    return result
            
            mslp_trajectory.append(min_mslp)
            locations.append(location)
            previous_location = location
            
            # CPS CHECK: Extratropical filtering at interface crossings or state B
            if self.enable_cps and location is not None:
                # Check if crossing any interface
                crossing_interface = False
                for i, interface_val in enumerate(self.interfaces):
                    if min_mslp < interface_val and not interface_crossings[i]:
                        crossing_interface = True
                        break
                
                # Perform CPS check at crossings
                if crossing_interface or min_mslp < self.state_B:
                    # Create xarray dataset for this timestep
                    try:
                        # Extract all variables at this timestep
                        ds_step = traj.isel(prediction_timedelta=lead_idx)
                        
                        # Determine stage based on MSLP
                        stage = 'genesis' if min_mslp > 995 else 'mature'
                        
                        is_ET, cps_params = self._check_cps(
                            ds_step,
                            location,
                            min_mslp,
                            storm_id,
                            stage=stage
                        )
                        
                        if is_ET:
                            # Trajectory rejected due to extratropical structure
                            max_lead_hours = lead_time / np.timedelta64(1, 'h')
                            cps_rejected = True
                            cps_rejection_info = {
                                'phase': cps_params['phase'],
                                'VTL': cps_params['VTL'],
                                'VTU': cps_params['VTU'],
                                'B': cps_params['B'],
                                'rejection_hour': max_lead_hours,
                                'rejection_mslp': min_mslp,
                                'rejection_location': location
                            }
                            
                            result = {
                                'init_time': init_time,
                                'member': member,
                                'success': False,
                                'starts_in_A': starts_in_A,
                                'mslp_trajectory': mslp_trajectory,
                                'locations': locations,
                                'min_mslp': min_mslp,
                                'success_lead_time': None,
                                'success_lead_hours': max_lead_hours,
                                'lost_track': False,
                                'lost_track_at_hour': None,
                                'cps_rejected': True,
                                'cps_rejection_info': cps_rejection_info
                            }
                            # Add interface crossings up to this point
                            for i, crossed in interface_crossings.items():
                                result[f'crossed_interface_{i}'] = crossed
                                result[f'crossing_time_interface_{i}'] = (
                                    interface_crossing_times[i] if crossed else max_lead_hours
                                )
                            return result
                    
                    except Exception as e:
                        print(f"⚠ CPS check failed at step {lead_idx}: {e}")
            
            # Check which interfaces were crossed (record first crossing time)
            for i, interface_val in enumerate(self.interfaces):
                if min_mslp < interface_val:
                    if not interface_crossings[i]:  # First time crossing this interface
                        interface_crossings[i] = True
                        interface_crossing_times[i] = lead_time / np.timedelta64(1, 'h')  # hours
            
            # Check for success (reached state B)
            if min_mslp < self.state_B:
                result = {
                    'init_time': init_time,
                    'member': member,
                    'success': True,
                    'starts_in_A': starts_in_A,
                    'mslp_trajectory': mslp_trajectory,
                    'locations': locations,
                    'min_mslp': min_mslp,
                    'success_lead_time': lead_time,
                    'success_lead_hours': lead_time / np.timedelta64(1, 'h'),
                    'lost_track': False,
                    'lost_track_at_hour': None,
                    'cps_rejected': cps_rejected,
                    'cps_rejection_info': cps_rejection_info
                }
                # Add interface crossings (boolean and times)
                for i, crossed in interface_crossings.items():
                    result[f'crossed_interface_{i}'] = crossed
                    result[f'crossing_time_interface_{i}'] = interface_crossing_times[i]
                return result
        
        # Did not reach state B - compute total simulation time
        max_lead_hours = traj.prediction_timedelta.values[-1] / np.timedelta64(1, 'h')
        
        result = {
            'init_time': init_time,
            'member': member,
            'success': False,
            'starts_in_A': starts_in_A,
            'mslp_trajectory': mslp_trajectory,
            'locations': locations,
            'min_mslp': np.min(mslp_trajectory),
            'success_lead_time': None,
            'success_lead_hours': max_lead_hours,
            'lost_track': False,
            'lost_track_at_hour': None,
            'cps_rejected': cps_rejected,
            'cps_rejection_info': cps_rejection_info
        }
        # Add interface crossings (boolean and times)
        for i, crossed in interface_crossings.items():
            result[f'crossed_interface_{i}'] = crossed
            result[f'crossing_time_interface_{i}'] = (
                interface_crossing_times[i] if crossed else max_lead_hours
            )
        return result
    
    def _process_single_init_parallel(self, init_idx: int, init_time, n_inits: int) -> Dict:
        """
        Process all members for one initialization (for parallel execution).
        
        Args:
            init_idx: Index of initialization
            init_time: Initialization time value
            n_inits: Total number of initializations
            
        Returns:
            Dictionary with statistics for this initialization
        """
        # Reset track history for each initialization
        self.storm_track_history = {}
        
        # Results for this initialization
        traj_results = []
        
        for member in self.ds_filtered.number.values:
            result = self.process_single_trajectory(init_time, member)
            traj_results.append(result)
        
        # Compute statistics for this initialization
        traj_df = pd.DataFrame(traj_results)
        
        n_members = len(self.ds_filtered.number)
        n_success = traj_df['success'].sum()
        n_starts_in_A = traj_df['starts_in_A'].sum()
        n_success_from_A = traj_df[traj_df['starts_in_A']]['success'].sum()
        n_lost_track = traj_df['lost_track'].sum()
        n_cps_rejected = traj_df['cps_rejected'].sum()
        
        # Compute actual rates using total simulation time
        total_sim_days = traj_df['success_lead_hours'].sum() / 24.0
        rate_per_day = n_success / total_sim_days if total_sim_days > 0 else 0.0
        
        # For trajectories starting in A
        if n_starts_in_A > 0:
            total_sim_days_from_A = traj_df[traj_df['starts_in_A']]['success_lead_hours'].sum() / 24.0
            rate_from_A_per_day = n_success_from_A / total_sim_days_from_A if total_sim_days_from_A > 0 else 0.0
        else:
            rate_from_A_per_day = np.nan
        
        # Keep probability metrics for reference
        prob_overall = n_success / n_members
        prob_from_A = n_success_from_A / n_starts_in_A if n_starts_in_A > 0 else np.nan
        
        mean_lead = traj_df[traj_df['success']]['success_lead_hours'].mean() if n_success > 0 else np.nan
        
        init_result = {
            'init_time': init_time,
            'n_members': n_members,
            'n_success': n_success,
            'n_starts_in_A': n_starts_in_A,
            'n_success_from_A': n_success_from_A,
            'n_lost_track': n_lost_track,
            'n_cps_rejected': n_cps_rejected,
            'total_sim_days': total_sim_days,
            'rate_per_day': rate_per_day,
            'rate_from_A_per_day': rate_from_A_per_day,
            'prob_overall': prob_overall,
            'prob_from_A': prob_from_A,
            'mean_passage_time_hours': mean_lead
        }
        
        # Add per-interface crossing statistics
        for i in range(len(self.interfaces)):
            n_crossed = traj_df[f'crossed_interface_{i}'].sum()
            
            # Compute rate for this interface
            total_sim_days_interface = traj_df[f'crossing_time_interface_{i}'].sum() / 24.0
            rate_lambda_i = n_crossed / total_sim_days_interface if total_sim_days_interface > 0 else 0.0
            
            init_result[f'n_crossed_lambda{i}'] = n_crossed
            init_result[f'rate_lambda{i}_per_day'] = rate_lambda_i
            init_result[f'prob_lambda{i}'] = n_crossed / n_members
        
        # Progress message
        interface_str = ', '.join([f"λ{i}:{init_result[f'n_crossed_lambda{i}']}" 
                                  for i in range(len(self.interfaces))])
        track_str = f", lost:{n_lost_track}" if n_lost_track > 0 else ""
        cps_str = f", CPS:{n_cps_rejected}" if self.enable_cps and n_cps_rejected > 0 else ""
        print(f"Init {init_idx+1}/{n_inits} ({init_time}): B:{n_success} ({rate_per_day:.6f}/day), {interface_str}{track_str}{cps_str}")
        
        return init_result
    
    def compute_brute_force_rates(self,
                                  n_jobs: int = -1,
                                  verbose: bool = True) -> pd.DataFrame:
        """
        Compute brute force rates per initialization time.
        Tracks crossings at each interface with adaptive spatial tracking and CPS filtering.
        
        Returns:
            DataFrame with per-initialization rates and interface crossing stats
        """
        if self.ds_filtered is None:
            raise ValueError("Must call load_data() first")
        
        n_inits = len(self.ds_filtered.time)
        n_members = len(self.ds_filtered.number)
        
        print(f"\n{'='*70}")
        print(f"BRUTE FORCE RATE ESTIMATION (PER INITIALIZATION)")
        print(f"{'='*70}")
        print(f"Initializations: {n_inits}")
        print(f"Ensemble members per init: {n_members}")
        print(f"Forecast length: 15 days from initialization")
        print(f"Lead times: {len(self.ds_filtered.prediction_timedelta)} timesteps")
        print(f"State A threshold: {self.state_A} hPa")
        print(f"State B threshold: {self.state_B} hPa")
        print(f"Interfaces: {self.interfaces}")
        print(f"Tracking: Adaptive ±10° box with jump detection")
        print(f"CPS filtering: {'ENABLED' if self.enable_cps else 'UNAVAILABLE'} (Hart 2003, VTL/VTU only)")
        
        # Determine number of parallel jobs
        if n_jobs == -1:
            n_jobs = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        elif n_jobs == 1:
            n_jobs = None  # Serial execution
        
        if n_jobs is not None:
            print(f"Using {n_jobs} parallel workers")
        else:
            print(f"Running serially (n_jobs=1)")
        
        print(f"{'='*70}\n")
        
        # Process all initializations (parallel or serial)
        if n_jobs is None or n_jobs == 1:
            # Serial execution
            init_results = []
            for init_idx, init_time in enumerate(self.ds_filtered.time.values):
                result = self._process_single_init_parallel(init_idx, init_time, n_inits)
                init_results.append(result)
        else:
            # Parallel execution
            init_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
                delayed(self._process_single_init_parallel)(
                    init_idx, init_time, n_inits
                )
                for init_idx, init_time in enumerate(self.ds_filtered.time.values)
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(init_results)
        
        # Compute aggregate statistics across initializations
        mean_rate_per_day = df['rate_per_day'].mean()
        std_rate_per_day = df['rate_per_day'].std()
        mean_rate_from_A_per_day = df['rate_from_A_per_day'].mean()
        std_rate_from_A_per_day = df['rate_from_A_per_day'].std()
        
        mean_prob_overall = df['prob_overall'].mean()
        std_prob_overall = df['prob_overall'].std()
        
        total_success = df['n_success'].sum()
        total_members = df['n_members'].sum()
        total_starts_in_A = df['n_starts_in_A'].sum()
        total_success_from_A = df['n_success_from_A'].sum()
        total_sim_days = df['total_sim_days'].sum()
        total_lost_track = df['n_lost_track'].sum()
        total_cps_rejected = df['n_cps_rejected'].sum()
        
        # Pooled rates
        pooled_rate_per_day = total_success / total_sim_days
        
        print(f"\n{'='*70}")
        print("RESULTS (ACROSS ALL INITIALIZATIONS)")
        print(f"{'='*70}")
        print(f"Total initializations: {n_inits}")
        print(f"Total trajectories: {total_members}")
        print(f"Total simulation time: {total_sim_days:.1f} days")
        print(f"Total starting in A: {total_starts_in_A}")
        print(f"Total successes to B: {total_success}")
        print(f"Total successes from A→B: {total_success_from_A}")
        print(f"Total lost track: {total_lost_track}")
        if self.enable_cps:
            print(f"Total CPS rejected: {total_cps_rejected}")
        
        print(f"\nBrute Force RATES (events per day):")
        print(f"  Per-init mean: {mean_rate_per_day:.6e} ± {std_rate_per_day:.6e} events/day")
        print(f"  Pooled rate: {total_success}/{total_sim_days:.1f} days = {pooled_rate_per_day:.6e} events/day")
        
        print(f"\nProbabilities (for reference):")
        print(f"  P(IC → B) = {total_success}/{total_members} = {total_success/total_members:.6f}")
        print(f"  Per-init mean: {mean_prob_overall:.6f} ± {std_prob_overall:.6f}")
        if total_starts_in_A > 0:
            print(f"  P(A → B) = {total_success_from_A}/{total_starts_in_A} = {total_success_from_A/total_starts_in_A:.6f}")
        
        # Print per-interface statistics
        print(f"\nInterface Crossing Statistics:")
        for i, interface_val in enumerate(self.interfaces):
            total_crossed = df[f'n_crossed_lambda{i}'].sum()
            mean_rate = df[f'rate_lambda{i}_per_day'].mean()
            std_rate = df[f'rate_lambda{i}_per_day'].std()
            mean_prob = df[f'prob_lambda{i}'].mean()
            std_prob = df[f'prob_lambda{i}'].std()
            
            print(f"  λ_{i} = {interface_val:4.0f} hPa:")
            print(f"    Count: {total_crossed}/{total_members}")
            print(f"    Rate: {mean_rate:.6e} ± {std_rate:.6e} crossings/day")
            print(f"    Prob: {mean_prob:.4f} ± {std_prob:.4f}")
        
        print(f"{'='*70}\n")
        
        # Store metadata
        df.attrs['n_inits'] = n_inits
        df.attrs['n_members_per_init'] = n_members
        df.attrs['total_sim_days'] = total_sim_days
        df.attrs['mean_rate_per_day'] = mean_rate_per_day
        df.attrs['std_rate_per_day'] = std_rate_per_day
        df.attrs['pooled_rate_per_day'] = pooled_rate_per_day
        df.attrs['mean_rate_from_A_per_day'] = mean_rate_from_A_per_day
        df.attrs['std_rate_from_A_per_day'] = std_rate_from_A_per_day
        df.attrs['mean_prob_overall'] = mean_prob_overall
        df.attrs['std_prob_overall'] = std_prob_overall
        df.attrs['total_success'] = int(total_success)
        df.attrs['total_members'] = int(total_members)
        df.attrs['total_starts_in_A'] = int(total_starts_in_A)
        df.attrs['total_success_from_A'] = int(total_success_from_A)
        df.attrs['total_lost_track'] = int(total_lost_track)
        df.attrs['total_cps_rejected'] = int(total_cps_rejected)
        df.attrs['state_A'] = self.state_A
        df.attrs['state_B'] = self.state_B
        df.attrs['interfaces'] = self.interfaces
        df.attrs['cps_enabled'] = self.enable_cps
        
        return df


def main():
    """Run brute force rate estimation on IFS ensemble."""
    
    parser = argparse.ArgumentParser(description='Compute brute force rates from IFS ensemble')
    parser.add_argument('--ffs_config', type=str, required=True,
                       help='Path to FFS config YAML file')
    parser.add_argument('--ifs_path', type=str, 
                       default='/glade/derecho/scratch/schreck/IFS.zarr',
                       help='Path to IFS zarr dataset')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel workers (-1 for all CPUs, 1 for serial)')
    
    args = parser.parse_args()
    
    # Load FFS config
    print(f"Loading config from {args.ffs_config}")
    with open(args.ffs_config, 'r') as f:
        ffs_config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(ffs_config['output_dir']) / 'IFS'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  State A: {ffs_config['state_A']} hPa")
    print(f"  State B: {ffs_config['state_B']} hPa")
    print(f"  Interfaces: {ffs_config['interfaces']}")
    print(f"  Forecast times: {ffs_config['forecast_times']}")
    print(f"  Output dir: {output_dir}")
    
    # Initialize estimator
    estimator = IFSBruteForceRateEstimator(
        ifs_path=args.ifs_path,
        state_A=ffs_config['state_A'],
        state_B=ffs_config['state_B'],
        interfaces=ffs_config['interfaces'],
        cps_config=ffs_config.get('cps', {})
    )
    
    # Load data using forecast_times from config
    estimator.load_data(forecast_times=ffs_config['forecast_times'])
    
    # Compute rates
    results_df = estimator.compute_brute_force_rates(
        n_jobs=args.n_jobs,
        verbose=True
    )
    
    # Generate output filename with time range
    start_str = ffs_config['forecast_times'][0][0].replace(' ', 'T').replace(':', '')
    end_str = ffs_config['forecast_times'][0][1].replace(' ', 'T').replace(':', '')
    time_label = f"{start_str[:10]}_to_{end_str[:10]}"
    
    # Save results
    output_file = output_dir / f"ifs_brute_force_rates_{time_label}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Save summary
    summary_file = output_dir / f"ifs_brute_force_summary_{time_label}.txt"
    with open(summary_file, 'w') as f:
        f.write("IFS Ensemble Brute Force Rate Estimation\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {args.ifs_path}\n")
        f.write(f"Initialization period: Starting {ffs_config['forecast_times'][0][0]}\n")
        f.write(f"Forecast length: 15 days from each initialization\n")
        f.write(f"State A threshold: {results_df.attrs['state_A']} hPa\n")
        f.write(f"State B threshold: {results_df.attrs['state_B']} hPa\n")
        f.write(f"Interfaces: {ffs_config['interfaces']}\n")
        f.write(f"Tracking method: Adaptive ±10° box with jump detection\n")
        f.write(f"CPS filtering: ENABLED (Hart 2003, VTL/VTU only)\n\n")
        f.write(f"Total initializations: {results_df.attrs['n_inits']}\n")
        f.write(f"Total trajectories: {results_df.attrs['total_members']}\n")
        f.write(f"Total simulation time: {results_df.attrs['total_sim_days']:.1f} days\n")
        f.write(f"Total starting in A: {results_df.attrs['total_starts_in_A']}\n")
        f.write(f"Total successes to B: {results_df.attrs['total_success']}\n")
        f.write(f"Total successes from A→B: {results_df.attrs['total_success_from_A']}\n")
        f.write(f"Total lost track: {results_df.attrs['total_lost_track']}\n")
        f.write(f"Total CPS rejected: {results_df.attrs['total_cps_rejected']}\n")
        f.write("\n")
        
        f.write("Brute Force RATES (events per day):\n")
        f.write(f"  Pooled rate: {results_df.attrs['pooled_rate_per_day']:.6e} events/day\n")
        f.write(f"  Per-init mean: {results_df.attrs['mean_rate_per_day']:.6e} ± {results_df.attrs['std_rate_per_day']:.6e} events/day\n\n")
        
        f.write("Probabilities (for reference):\n")
        f.write(f"  P(IC → B): {results_df.attrs['mean_prob_overall']:.6f} ± {results_df.attrs['std_prob_overall']:.6f}\n\n")
        
        f.write("Interface Crossing Statistics:\n")
        for i, interface_val in enumerate(ffs_config['interfaces']):
            total_crossed = results_df[f'n_crossed_lambda{i}'].sum()
            mean_rate = results_df[f'rate_lambda{i}_per_day'].mean()
            std_rate = results_df[f'rate_lambda{i}_per_day'].std()
            mean_prob = results_df[f'prob_lambda{i}'].mean()
            std_prob = results_df[f'prob_lambda{i}'].std()
            f.write(f"  λ_{i} = {interface_val:4.0f} hPa:\n")
            f.write(f"    Count: {total_crossed}/{results_df.attrs['total_members']}\n")
            f.write(f"    Rate: {mean_rate:.6e} ± {std_rate:.6e} crossings/day\n")
            f.write(f"    Prob: {mean_prob:.4f} ± {std_prob:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()