"""
Compute IFS ensemble genesis rates matching FFS methodology EXACTLY.

Ports the complete multi-storm tracking system from FFS:
- Track ALL minima in basin (not just one storm per member)
- 12° merge radius
- 2-timestep lost track threshold
- 3x3 local minimum verification
- CPS tropical check at λ₀ crossing

Rate = N_lambda0_crossings / (N_members * 15_days)
"""

import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Optional
import yaml
import argparse
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
import multiprocessing as mp

try:
    from tails.cyclone_phase_tracker import CyclonePhaseTracker
    CPS_AVAILABLE = True
except ImportError:
    CPS_AVAILABLE = False
    print("WARNING: CPS not available")

class IFSRateEstimator:
    """
    IFS ensemble rate estimator matching FFS methodology.
    
    Multi-storm tracking with local minimum verification and CPS filtering.
    """
    
    def __init__(self,
                 ifs_path: str,
                 state_A: float = 1008.0,
                 state_B: float = 982.0,
                 interfaces: list = None,
                 basin: Dict = None,
                 cps_config: Dict = None,
                 output_dir: str = None,
                 save_plots: bool = True):

        self.ifs_path = Path(ifs_path)
        self.save_plots = save_plots and (output_dir is not None)
        if self.save_plots:
            self.plot_dir = Path(output_dir) / 'plots' / 'stateB'
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.plot_dir = None
        self.state_A = state_A
        self.state_B = state_B
        self.interfaces = (interfaces if interfaces is not None else [1000]).copy()
        if state_B not in self.interfaces:
            self.interfaces.append(state_B)
        self.lambda0 = self.interfaces[0]
        
        # Basin (matches FFS)
        if basin is None:
            self.basin = {
                'lat_min': 10.0,
                'lat_max': 45.0,
                'lon_min': -98.0,
                'lon_max': -20.0
            }
        else:
            self.basin = basin
        
        # CPS setup
        self.enable_cps = CPS_AVAILABLE
        self.surface_geopotential = None
        self.credit_config = None
        self.cps_tracker = None
        
        if self.enable_cps:
            cps_config = cps_config or {}
            try:
                # Load static fields
                static_path = cps_config.get('static_path', 
                    '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc')
                with xr.open_dataset(static_path) as df:
                    self.surface_geopotential = df["Z_GDS4_SFC"].values
                
                # Load CREDIT config for pressure interpolation
                credit_config_path = cps_config.get('credit_config_path')
                if credit_config_path:
                    with open(credit_config_path, 'r') as f:
                        self.credit_config = yaml.safe_load(f)
                
                print("✓ CPS setup ready (will init after data load)")
            except Exception as e:
                print(f"⚠ CPS setup failed: {e}")
                self.enable_cps = False
        
        self.ds = None
    
    def load_data(self, forecast_times: list = None):
        """Load IFS data."""
        print(f"Loading IFS data from {self.ifs_path}...")
        self.ds = xr.open_zarr(self.ifs_path, consolidated=True)
        
        if forecast_times is not None and len(forecast_times) > 0:
            init_times = [pd.to_datetime(ft[0]) for ft in forecast_times]
            self.ds_filtered = self.ds.sel(time=init_times)
            max_lead = pd.Timedelta(days=15)
            valid_leads = self.ds_filtered.prediction_timedelta <= max_lead
            self.ds_filtered = self.ds_filtered.isel(prediction_timedelta=valid_leads)
        else:
            ds_year = self.ds.sel(time=self.ds.time.dt.year == 2022)
            self.ds_filtered = ds_year.sel(time=ds_year.time.dt.hour == 0)
        
        print(f"Loaded {len(self.ds_filtered.time)} initializations")
        print(f"Ensemble size: {len(self.ds_filtered.number)}")
        
        # Initialize CPS tracker with IFS grid
        if self.enable_cps and self.cps_tracker is None:
            try:
                # Create latlons dataset from IFS grid
                latlons = xr.Dataset({
                    'latitude': self.ds_filtered.latitude,
                    'longitude': self.ds_filtered.longitude
                })
                self.cps_tracker = CyclonePhaseTracker(latlons, radius_km=500)
                print("✓ CPS tracker initialized with IFS grid")
            except Exception as e:
                print(f"⚠ CPS tracker init failed: {e}")
                self.enable_cps = False
        
        return self.ds_filtered
    
    def get_basin_mask(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Basin mask.
        
        Args:
            lats: Latitude array
            lons: Longitude array (any format)
        
        Returns:
            Boolean mask array
        """
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
        lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
        
        return lat_mask[:, None] & lon_mask[None, :]

    def _save_storm_figure(self, mslp_hpa, lats, lons_180,
                           storm_lat, storm_lon, storm_mslp,
                           init_time, member, lead_idx, lead_hours,
                           storm_id, event_type='stateB'):
        """
        Save MSLP figure for a storm event (matching FFS _save_mslp_figure style).
        """
        if not self.save_plots:
            return

        try:
            from matplotlib.colors import BoundaryNorm

            basin_mask = self.get_basin_mask(lats, np.where(lons_180 < 0, lons_180 + 360, lons_180))
            basin_rows, basin_cols = np.where(basin_mask)
            pad = 10
            row_min = max(0, basin_rows.min() - pad)
            row_max = min(mslp_hpa.shape[0], basin_rows.max() + pad)
            col_min = max(0, basin_cols.min() - pad)
            col_max = min(mslp_hpa.shape[1], basin_cols.max() + pad)

            mslp_crop = mslp_hpa[row_min:row_max, col_min:col_max]
            lat_crop = lats[row_min:row_max]
            lon_crop = lons_180[col_min:col_max]

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

            ax.set_extent([lon_crop.min(), lon_crop.max(),
                          lat_crop.min(), lat_crop.max()],
                          crs=ccrs.PlateCarree())

            levels = np.arange(960, 1030, 4)
            norm = BoundaryNorm(levels, ncolors=plt.cm.RdBu_r.N, clip=True)
            pcm = ax.pcolormesh(lon_crop, lat_crop, mslp_crop,
                               norm=norm, cmap='RdBu_r',
                               transform=ccrs.PlateCarree(),
                               shading='auto', zorder=1)

            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)

            ax.plot(storm_lon, storm_lat, 'k*', markersize=20,
                   markeredgewidth=2, markeredgecolor='yellow',
                   transform=ccrs.PlateCarree(), zorder=5)

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)
            gl.top_labels = False
            gl.right_labels = False

            lat_dir = 'N' if storm_lat >= 0 else 'S'
            lon_dir = 'W' if storm_lon < 0 else 'E'
            init_str = pd.Timestamp(init_time).strftime('%Y-%m-%d %H:%M UTC')
            valid_str = (pd.Timestamp(init_time) + pd.Timedelta(hours=lead_hours)).strftime('%Y-%m-%d %H:%M UTC')

            title = f'{valid_str} (init: {init_str}, member {member})\n'
            title += f'MSLP: {storm_mslp:.1f} hPa @ ({abs(storm_lat):.1f}\u00b0{lat_dir}, {abs(storm_lon):.1f}\u00b0{lon_dir})'
            title += f'\nStorm {storm_id} | Lead +{lead_hours:.0f}h | {event_type}'
            ax.set_title(title, fontsize=12, fontweight='bold')

            plt.colorbar(pcm, ax=ax, label='MSLP (hPa)', shrink=0.8)
            plt.tight_layout()

            init_label = pd.Timestamp(init_time).strftime('%Y%m%d_%H')
            fig_path = self.plot_dir / f'{init_label}_m{member:02d}_storm{storm_id}_{event_type}.png'
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  \u26a0 Plot save failed: {e}")
            plt.close('all')

    def _find_local_minima(self, mslp_hpa: np.ndarray,
                           lats: np.ndarray, lons: np.ndarray,
                           basin_mask: np.ndarray,
                           exclude_centers: List = None,
                           exclude_radius_deg: float = 12.0) -> List[Tuple[float, float, float]]:
        """
        Find ALL local MSLP minima (matching FFS _find_local_mslp_minima).

        Args:
            mslp_hpa: MSLP field in hPa
            lats: Latitude array
            lons: Longitude array (converted to -180:180 internally)
            basin_mask: Boolean mask
            exclude_centers: List of (lat, lon) tuples to exclude
            exclude_radius_deg: Radius around exclude_centers to mask out

        Returns:
            List of (mslp, lat, lon) tuples
        """
        # Convert lons to -180:180
        lons_180 = np.where(lons > 180, lons - 360, lons)

        # Mask to basin
        field = np.where(basin_mask, mslp_hpa, np.nan)

        # Exclude regions around tracked storms (matches FFS)
        if exclude_centers:
            lat_grid, lon_grid = np.meshgrid(lats, lons_180, indexing='ij')
            for center_lat, center_lon in exclude_centers:
                dist = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
                field = np.where(dist >= exclude_radius_deg, field, np.nan)

        # Smooth for location finding
        mslp_smooth = gaussian_filter(field, sigma=1.5)

        minima = []

        # 3x3 neighborhood check (matches FFS)
        for i in range(1, mslp_smooth.shape[0] - 1):
            for j in range(1, mslp_smooth.shape[1] - 1):
                val_smooth = mslp_smooth[i, j]
                if not np.isfinite(val_smooth):
                    continue

                nbrs = mslp_smooth[i-1:i+2, j-1:j+2]
                if np.all(val_smooth <= nbrs):
                    # Local minimum found - use RAW value
                    val_raw = mslp_hpa[i, j]
                    minima.append((val_raw, lats[i], lons_180[j]))

        return minima
    
    def _merge_nearby_minima(self, minima: List[Tuple[float, float, float]], 
                            radius_deg: float = 12.0) -> List[Tuple[float, float, float]]:
        """
        Merge minima within radius_deg (matching FFS).
        
        Keep strongest (lowest MSLP) in each cluster.
        """
        if len(minima) == 0:
            return []
        
        merged = []
        for min_mslp, min_lat, min_lon in minima:
            merged_flag = False
            for i, (m_mslp, m_lat, m_lon) in enumerate(merged):
                dist = np.sqrt((min_lat - m_lat)**2 + (min_lon - m_lon)**2)
                if dist < radius_deg:
                    # Merge - keep stronger
                    if min_mslp < m_mslp:
                        merged[i] = (min_mslp, min_lat, min_lon)
                    merged_flag = True
                    break
            
            if not merged_flag:
                merged.append((min_mslp, min_lat, min_lon))
        
        return merged
    
    def _check_cps_with_motion(self, ds_timestep: xr.Dataset, location: Tuple[float, float],
                                mslp: float, storm_id: int,
                                prev_lon: Optional[float], prev_lat: Optional[float],
                                stage: str = 'genesis') -> Tuple[bool, Dict]:
        """
        CPS tropical check for IFS data with nearest-neighbor pressure level extension.
        """
        if not self.enable_cps:
            return False, {}
        
        lat, lon = location
        
        try:
            # IFS has only 500, 700, 850 hPa - extend to 300, 600, 900
            ifs_levels = [500, 700, 850]
            target_levels = [300, 500, 600, 700, 850, 900]
            
            ds_list = []
            for target in target_levels:
                if target in ifs_levels:
                    ds_list.append(ds_timestep.sel(level=target))
                else:
                    nearest = min(ifs_levels, key=lambda x: abs(x - target))
                    ds_level = ds_timestep.sel(level=nearest).copy()
                    ds_level = ds_level.assign_coords(level=target)
                    ds_list.append(ds_level)
            
            ds_extended = xr.concat(ds_list, dim='level')
            
            # CREATE NEW LATLONS GRID FROM IFS COORDINATES
            latlons_ifs = xr.Dataset({
                'latitude': ds_extended.latitude,
                'longitude': ds_extended.longitude
            })
            
            # CREATE NEW CPS TRACKER WITH IFS GRID
            cps_tracker_ifs = CyclonePhaseTracker(latlons_ifs, radius_km=500)
            
            # RENAME to match CPS expectations
            ds_cps = ds_extended.rename({
                'geopotential': 'Z_PRES',
                'temperature': 'T_PRES',
                'level': 'pressure'
            })

            # WeatherBench geopotential is in m²/s²; CPS expects geopotential height in meters
            ds_cps['Z_PRES'] = ds_cps['Z_PRES'] / 9.80665

            for var in ['Z_PRES', 'T_PRES']:
                if var in ds_cps:
                    # Make sure dimension order is (pressure, latitude, longitude)
                    ds_cps[var] = ds_cps[var].transpose('pressure', 'latitude', 'longitude')

            # Compute CPS with IFS-specific tracker
            cps = cps_tracker_ifs.compute_CPS(
                ds_cps,
                lon, lat, mslp,
                prev_lon=prev_lon, prev_lat=prev_lat,
                stage=stage
            )
            
            is_ET = not cps['is_tropical']

            return is_ET, cps
            
        except Exception as e:
            print(f"⚠ CPS computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, {}
    
    def process_single_member(self, init_time, member: int) -> Dict:
        """
        Process one ensemble member with FFS-style multi-storm tracking.
        
        EXACTLY matches FFS flux mode:
        - Tracks ALL storms simultaneously
        - 12° merge radius
        - 2-step lost track threshold
        - 3x3 local minimum verification before saving
        - CPS check at λ₀ crossing
        - Storm track history for CPS motion
        """
        traj = self.ds_filtered.sel(time=init_time, number=member)
        
        lats = traj.latitude.values
        lons = traj.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        basin_mask = self.get_basin_mask(lats, lons)
        
        # Multi-storm tracking state (EXACTLY like FFS)
        tracked_storms = {}  # storm_id -> {location, mslp, saved, lost_count}
        next_storm_id = 0
        storm_track_history = {}  # storm_id -> {lons: [], lats: []} for CPS
        
        # Track ALL interface crossings (not just λ₀)
        interface_crossings = {i: False for i in range(len(self.interfaces))}
        interface_crossing_times = {i: None for i in range(len(self.interfaces))}
        
        lambda0_crossings = []
        cps_rejections = []
        reached_B = False
        B_lead_time = None
        
        n_timesteps = len(traj.prediction_timedelta)
        
        # Compute basin mask once (doesn't change per timestep)
        basin_mask = self.get_basin_mask(lats, lons)
        
        # Check if starts in state A
        mslp_init_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=0).values
        mslp_init_hpa = mslp_init_pa / 100.0
        if mslp_init_hpa.shape != (len(lats), len(lons)):
            mslp_init_hpa = mslp_init_hpa.T
        try:
            mslp_init_basin = np.where(basin_mask, mslp_init_hpa, np.nan)
        except:
            print(init_time, member)
            raise
        starts_in_A = np.nanmin(mslp_init_basin) > self.state_A

        # Track which storms existed at t=0 (pre-existing)
        initial_storm_ids = set()  # Storm IDs that exist at t=0
        
        for lead_idx in range(n_timesteps):
            mslp_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=lead_idx).values
            # Remove any extra dimensions
            while mslp_pa.ndim > 2:
                mslp_pa = mslp_pa[0]
            mslp_hpa_raw = mslp_pa / 100.0
            
            # Ensure correct dimensions
            if mslp_hpa_raw.shape == (len(lons), len(lats)):
                mslp_hpa = mslp_hpa_raw.T
            else:
                mslp_hpa = mslp_hpa_raw
            
            # STEP 1: Update tracked storms via 12° radius on unmasked field (matches FFS)
            tracked_minima = []
            if len(tracked_storms) > 0:
                mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
                lat_grid, lon_grid = np.meshgrid(lats, lons_180, indexing='ij')
                for storm_id, storm_info in tracked_storms.items():
                    storm_lat, storm_lon = storm_info['location']
                    dist = np.sqrt((lat_grid - storm_lat)**2 + (lon_grid - storm_lon)**2)
                    local = np.where(dist < 12.0, mslp_smooth, np.nan)

                    if np.all(np.isnan(local)):
                        continue

                    idx = np.nanargmin(local)
                    i, j = np.unravel_index(idx, local.shape)
                    min_lat = float(lats[i])
                    min_lon = float(lons_180[j])
                    min_mslp = float(mslp_hpa[i, j])

                    tracked_minima.append((min_mslp, min_lat, min_lon))

            # STEP 2: Find NEW minima in basin, excluding 12° around tracked storms
            exclude_centers = [s['location'] for s in tracked_storms.values()]
            new_minima = self._find_local_minima(mslp_hpa, lats, lons, basin_mask,
                                                  exclude_centers=exclude_centers,
                                                  exclude_radius_deg=12.0)

            minima = tracked_minima + new_minima

            # STEP 3: Merge nearby minima within 12° (matches FFS)
            minima = self._merge_nearby_minima(minima, radius_deg=12.0)

            # STEP 4: Match storms to merged minima (matches FFS)
            matched_storms = set()
            matched_minima = set()

            for storm_id, storm_info in list(tracked_storms.items()):
                storm_lat, storm_lon = storm_info['location']
                best_match = None
                best_dist = float('inf')

                for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
                    if min_idx in matched_minima:
                        continue
                    dist = np.sqrt((min_lat - storm_lat)**2 + (min_lon - storm_lon)**2)
                    if dist < 12.0 and dist < best_dist:
                        best_match = min_idx
                        best_dist = dist

                if best_match is not None:
                    min_mslp, min_lat, min_lon = minima[best_match]
                    tracked_storms[storm_id]['location'] = (min_lat, min_lon)
                    tracked_storms[storm_id]['mslp'] = min_mslp
                    tracked_storms[storm_id]['lost_count'] = 0

                    if storm_id not in storm_track_history:
                        storm_track_history[storm_id] = {'lons': [], 'lats': []}
                    storm_track_history[storm_id]['lons'].append(min_lon)
                    storm_track_history[storm_id]['lats'].append(min_lat)

                    matched_storms.add(storm_id)
                    matched_minima.add(best_match)
                else:
                    tracked_storms[storm_id]['lost_count'] += 1

            # STEP 5: Add new storms from unmatched minima (matches FFS)
            for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
                if min_idx not in matched_minima:
                    storm_id = next_storm_id
                    next_storm_id += 1
                    tracked_storms[storm_id] = {
                        'location': (min_lat, min_lon),
                        'mslp': min_mslp,
                        'saved': False,
                        'lost_count': 0
                    }
                    storm_track_history[storm_id] = {
                        'lons': [min_lon],
                        'lats': [min_lat]
                    }

                    if lead_idx == 0 and min_mslp < self.lambda0:
                        initial_storm_ids.add(storm_id)

            # STEP 6: Remove storms lost for 2+ timesteps (matches FFS)
            for storm_id in list(tracked_storms.keys()):
                if tracked_storms[storm_id]['lost_count'] >= 2:
                    del tracked_storms[storm_id]
                    if storm_id in storm_track_history:
                        del storm_track_history[storm_id]
            
            # STEP 5b: Remove storms that have become extratropical (match FFS flux mode)
            if self.enable_cps:
                for storm_id in list(tracked_storms.keys()):
                    storm = tracked_storms[storm_id]
                    storm_lat, storm_lon = storm['location']

                    # Only check storms at high latitudes or far east (match FFS)
                    if storm_lat > 50.0 or storm_lon > -10.0:
                        try:
                            ds_step = traj.isel(prediction_timedelta=lead_idx)

                            prev_lon, prev_lat = None, None
                            if storm_id in storm_track_history:
                                hist = storm_track_history[storm_id]
                                if len(hist['lons']) >= 2:
                                    prev_lon = hist['lons'][-2]
                                    prev_lat = hist['lats'][-2]

                            is_ET, cps = self._check_cps_with_motion(
                                ds_step,
                                (storm_lat, storm_lon),
                                storm['mslp'],
                                storm_id,
                                prev_lon=prev_lon,
                                prev_lat=prev_lat
                            )

                            if is_ET:
                                del tracked_storms[storm_id]
                                if storm_id in storm_track_history:
                                    del storm_track_history[storm_id]
                        except Exception:
                            pass

            # STEP 7: Check for λ₀ crossings (matches FFS: dissipation → λ₀ → B-state)
            for storm_id in list(tracked_storms.keys()):
                storm = tracked_storms[storm_id]

                # Dissipation check FIRST (matches FFS order)
                if storm['mslp'] > self.state_A:
                    del tracked_storms[storm_id]
                    if storm_id in storm_track_history:
                        del storm_track_history[storm_id]
                    continue

                # Only check unsaved storms that crossed λ₀
                if not storm['saved'] and storm['mslp'] < self.lambda0:
                    # Skip storms that existed at t=0 (not genesis)
                    if storm_id in initial_storm_ids:
                        storm['saved'] = True  # Mark as saved to prevent re-checking
                        continue

                    storm_lat, storm_lon = storm['location']
                    
                    # VERIFICATION 1: 3x3 local minimum check (EXACTLY like FFS)
                    lat_idx = np.argmin(np.abs(lats - storm_lat))
                    lon_idx = np.argmin(np.abs(lons_180 - storm_lon))
                    
                    i_min = max(0, lat_idx - 1)
                    i_max = min(mslp_hpa.shape[0], lat_idx + 2)
                    j_min = max(0, lon_idx - 1)
                    j_max = min(mslp_hpa.shape[1], lon_idx + 2)
                    
                    # Known terrain artifact: Hispaniola (~18°N, 70°W)
                    # Smoothed field check — artifact vanishes when smoothed
                    # if abs(storm_lat - 18.0) < 2.0 and abs(storm_lon - (-70.0)) < 3.0:
                    #     mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
                    #     nbhd = mslp_smooth[i_min:i_max, j_min:j_max]
                    #     center_val = mslp_smooth[lat_idx, lon_idx]
                    #     if not np.all(center_val <= nbhd):
                    #         storm['saved'] = True
                    #         continue

                    # VERIFICATION 2: Reject storms that drifted outside the basin
                    if not (self.basin['lat_min'] <= storm_lat <= self.basin['lat_max'] and
                            self.basin['lon_min'] <= storm_lon <= self.basin['lon_max']):
                        storm['saved'] = True
                        continue

                    # VERIFICATION 3: CPS tropical check (EXACTLY like FFS)
                    if self.enable_cps:
                        try:
                            ds_step = traj.isel(prediction_timedelta=lead_idx)
                            
                            # Get previous position for CPS motion
                            prev_lon, prev_lat = None, None
                            if storm_id in storm_track_history:
                                hist = storm_track_history[storm_id]
                                if len(hist['lons']) >= 2:  # Need at least 2 points
                                    prev_lon = hist['lons'][-2]
                                    prev_lat = hist['lats'][-2]
                            
                            is_ET, cps = self._check_cps_with_motion(
                                ds_step, 
                                (storm_lat, storm_lon),
                                storm['mslp'],
                                storm_id,
                                prev_lon=prev_lon,
                                prev_lat=prev_lat
                            )
                            
                            if is_ET:
                                # Extratropical - reject
                                storm['saved'] = True
                                cps_rejections.append({
                                    'storm_id': storm_id,
                                    'lead_idx': lead_idx,
                                    'mslp': storm['mslp'],
                                    'lat': storm_lat,
                                    'lon': storm_lon,
                                    'phase': cps.get('phase', 'unknown')
                                })
                                continue
                        except Exception as e:
                            # CPS check failed - reject crossing (match FFS behavior)
                            storm['saved'] = True
                            continue
                    
                    # VALID λ₀ CROSSING - save it and mark this storm as having crossed λ₀
                    lambda0_crossings.append({
                        'storm_id': storm_id,
                        'mslp': storm['mslp'],
                        'lat': storm_lat,
                        'lon': storm_lon,
                        'lead_idx': lead_idx,
                        'lead_hours': float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                    })
                    storm['saved'] = True
                    storm['crossed_lambda0'] = True  # Mark that this storm crossed λ₀
                
                # Check remaining interface crossings (λ₁, λ₂, ...) - λ₀ already handled above
                if storm_id not in initial_storm_ids and storm.get('crossed_lambda0', False):
                    for i in range(1, len(self.interfaces)):  # Skip index 0 (λ₀)
                        interface_val = self.interfaces[i]
                        if storm['mslp'] < interface_val:
                            if not interface_crossings[i]:
                                interface_crossings[i] = True
                                interface_crossing_times[i] = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                
                # Check if reached state B AND had crossed λ₀ (genesis -> B event)
                if storm['mslp'] < self.state_B and not reached_B:
                    if storm.get('crossed_lambda0', False):  # Only count if it crossed λ₀ first
                        if storm_id not in initial_storm_ids:
                            # CPS check at B-state (match FFS behavior: stage='mature')
                            b_is_tropical = True
                            if self.enable_cps:
                                try:
                                    ds_step = traj.isel(prediction_timedelta=lead_idx)
                                    prev_lon, prev_lat = None, None
                                    if storm_id in storm_track_history:
                                        hist = storm_track_history[storm_id]
                                        if len(hist['lons']) >= 2:
                                            prev_lon = hist['lons'][-2]
                                            prev_lat = hist['lats'][-2]
                                    storm_lat, storm_lon = storm['location']
                                    is_ET, cps = self._check_cps_with_motion(
                                        ds_step,
                                        (storm_lat, storm_lon),
                                        storm['mslp'],
                                        storm_id,
                                        prev_lon=prev_lon,
                                        prev_lat=prev_lat,
                                        stage='mature'
                                    )
                                    if is_ET:
                                        b_is_tropical = False
                                except Exception as e:
                                    b_is_tropical = False

                            if b_is_tropical:
                                reached_B = True
                                B_lead_time = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                                storm_lat, storm_lon = storm['location']
                                self._save_storm_figure(
                                    mslp_hpa, lats, lons_180,
                                    storm_lat, storm_lon, storm['mslp'],
                                    init_time, member, lead_idx, B_lead_time,
                                    storm_id, event_type='stateB'
                                )

        return {
            'init_time': init_time,
            'member': member,
            'starts_in_A': starts_in_A,
            'reached_B': reached_B,
            'B_lead_time': B_lead_time,
            'n_lambda0_crossings': len(lambda0_crossings),
            'n_cps_rejected': len(cps_rejections),
            'crossings': lambda0_crossings,
            'cps_rejections': cps_rejections,
            'interface_crossings': interface_crossings,
            'interface_crossing_times': interface_crossing_times
        }
    
    def _process_single_init(self, init_idx: int, init_time, n_inits: int) -> Dict:
        """
        Process all members for one initialization time.
        This is the unit of parallelization.
        """
        n_members = len(self.ds_filtered.number)
        
        member_results = []
        for member in self.ds_filtered.number.values:
            result = self.process_single_member(init_time, member)
            member_results.append(result)
        
        # Aggregate for this IC
        total_lambda0 = sum(r['n_lambda0_crossings'] for r in member_results)
        total_cps_rejected = sum(r['n_cps_rejected'] for r in member_results)
        n_starts_in_A = sum(r['starts_in_A'] for r in member_results)
        n_reached_B = sum(r['reached_B'] for r in member_results)
        n_B_from_A = sum(1 for r in member_results if r['reached_B'] and r['starts_in_A'])
        
        # Interface crossing counts
        interface_counts = {}
        for i in range(len(self.interfaces)):
            interface_counts[i] = sum(r['interface_crossings'][i] for r in member_results)
        
        # FIXED denominator for FLUX: always 50 members × 15 days = 750 days
        total_sim_days = n_members * 15.0
        rate_lambda0_flux = total_lambda0 / total_sim_days
        
        # BF rate calculation
        total_sim_days = n_members * 15.0
        n_reached_B = sum(r['reached_B'] for r in member_results)
        rate_B_bf = n_reached_B / total_sim_days  # Simple: events/time

        # Still useful to track mean time to B for diagnostics
        if n_reached_B > 0:
            times_to_B = [r['B_lead_time'] for r in member_results if r['reached_B']]
            mean_time_to_B_hours = np.mean(times_to_B)
            mean_time_to_B_days = mean_time_to_B_hours / 24.0
        else:
            mean_time_to_B_hours = np.nan
            mean_time_to_B_days = np.nan

        result = {
            'init_time': init_time,
            'n_members': n_members,
            'n_starts_in_A': n_starts_in_A,
            'n_reached_B': n_reached_B,
            'n_B_from_A': n_B_from_A,
            'mean_time_to_B_hours': mean_time_to_B_hours,
            'mean_time_to_B_days': mean_time_to_B_days,
            'n_lambda0_crossings': total_lambda0,
            'n_cps_rejected': total_cps_rejected,
            'total_sim_days': total_sim_days,
            'rate_lambda0_flux_per_day': rate_lambda0_flux,
            'rate_B_bf_per_day': rate_B_bf,
            'prob_B': n_reached_B / n_members,
            'prob_B_from_A': n_B_from_A / n_starts_in_A if n_starts_in_A > 0 else np.nan
        }

        # Add per-interface counts and rates
        for i in range(len(self.interfaces)):
            result[f'n_crossed_lambda{i}'] = interface_counts[i]
            result[f'rate_lambda{i}_per_day'] = interface_counts[i] / total_sim_days
            result[f'prob_lambda{i}'] = interface_counts[i] / n_members
        # Progress message
        cps_str = f", CPS rejected={total_cps_rejected}" if self.enable_cps else ""
        # Show λ₀ from lambda0_crossings, then λ₁, λ₂, ... from interface counts
        interface_parts = [f"λ0:{result['n_lambda0_crossings']}"]
        interface_parts.extend([f"λ{i}:{result[f'n_crossed_lambda{i}']}" for i in range(1, len(self.interfaces))])
        interface_str = ', '.join(interface_parts)
        print(f"Init {init_idx+1}/{n_inits} ({init_time}): B={result['n_reached_B']}, {interface_str}{cps_str}, flux_λ0={result['rate_lambda0_flux_per_day']:.6f}/day")
        return result
    
    def compute_rates(self, n_jobs: int = 1) -> pd.DataFrame:
        """Compute rates per initialization."""
        n_inits = len(self.ds_filtered.time)
        n_members = len(self.ds_filtered.number)
        
        print(f"\n{'='*70}")
        print("IFS RATE ESTIMATION (matching FFS methodology)")
        print(f"{'='*70}")
        print(f"Initializations: {n_inits}")
        print(f"Members per init: {n_members}")
        print(f"λ₀ threshold: {self.lambda0} hPa")
        print(f"Multi-storm tracking: YES")
        print(f"Local minimum verification: YES (3x3)")
        print(f"Storm merge radius: 12°")
        print(f"Lost track threshold: 2 timesteps")
        print(f"CPS filtering: {'YES' if self.enable_cps else 'NO'}")
        
        # Determine number of parallel jobs
        if n_jobs == -1:
            n_jobs_actual = max(1, mp.cpu_count() - 1)
        else:
            n_jobs_actual = n_jobs
        
        if n_jobs_actual > 1:
            print(f"Parallel workers: {n_jobs_actual} (parallelizing across init times)")
        else:
            print(f"Running serially")
        
        print(f"{'='*70}\n")
        
        # Process all inits (parallel or serial)
        if n_jobs_actual > 1:
            # Parallel across init times
            results = Parallel(n_jobs=n_jobs_actual, backend='loky', verbose=10)(
                delayed(self._process_single_init)(init_idx, init_time, n_inits)
                for init_idx, init_time in enumerate(self.ds_filtered.time.values)
            )
        else:
            # Serial
            results = []
            for init_idx, init_time in enumerate(self.ds_filtered.time.values):
                result = self._process_single_init(init_idx, init_time, n_inits)
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # Summary
        total_B = df['n_reached_B'].sum()
        total_crossings = df['n_lambda0_crossings'].sum()
        total_cps_rejected = df['n_cps_rejected'].sum()
        total_sim_days = df['total_sim_days'].sum()
        total_starts_in_A = df['n_starts_in_A'].sum()
        total_B_from_A = df['n_B_from_A'].sum()

        pooled_rate_lambda0_flux = total_crossings / total_sim_days
        pooled_rate_B_bf = total_B / total_sim_days

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total trajectories: {n_inits * n_members}")
        print(f"Total starting in A: {total_starts_in_A}")
        print(f"Total reached B (genesis->B events): {total_B}")
        print(f"Total B from A: {total_B_from_A}")
        print(f"Total λ₀ crossings: {total_crossings}")
        if self.enable_cps:
            print(f"Total CPS rejected: {total_cps_rejected}")
        print(f"Total simulation time: {total_sim_days:.0f} days")
        print(f"\nRates:")
        print(f"  λ₀ FLUX rate: {pooled_rate_lambda0_flux:.6e} crossings/day")
        print(f"  State B rate: {pooled_rate_B_bf:.6e} events/day")
        if not df['mean_time_to_B_days'].isna().all():
            print(f"  Mean time to B (diagnostic): {df['mean_time_to_B_days'].mean():.1f} days")
        print(f"\nInterface Statistics:")
        # λ₀ from lambda0_crossings
        print(f"  λ_0 = {self.interfaces[0]:4.0f} hPa: {total_crossings} crossings, {pooled_rate_lambda0_flux:.6e}/day")
        # Remaining interfaces
        for i in range(1, len(self.interfaces)):
            interface_val = self.interfaces[i]
            total_crossed = df[f'n_crossed_lambda{i}'].sum()
            pooled_rate = df[f'n_crossed_lambda{i}'].sum() / total_sim_days
            print(f"  λ_{i} = {interface_val:4.0f} hPa: {total_crossed} crossings, {pooled_rate:.6e}/day")
        print(f"{'='*70}\n")

        df.attrs['pooled_rate_lambda0_flux'] = pooled_rate_lambda0_flux
        df.attrs['pooled_rate_B_bf'] = pooled_rate_B_bf
        df.attrs['total_crossings'] = int(total_crossings)
        df.attrs['total_B'] = int(total_B)
        df.attrs['total_cps_rejected'] = int(total_cps_rejected)
        df.attrs['total_sim_days'] = float(total_sim_days)

        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffs_config', type=str, required=True)
    parser.add_argument('--ifs_path', type=str, 
                       default='/glade/derecho/scratch/schreck/IFS.zarr')
    parser.add_argument('--n_jobs', type=int, default=1)
    
    args = parser.parse_args()
    
    with open(args.ffs_config, 'r') as f:
        ffs_config = yaml.safe_load(f)
    
    output_dir = Path(ffs_config['output_dir']) / 'IFS'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    estimator = IFSRateEstimator(
        ifs_path=args.ifs_path,
        state_A=ffs_config['state_A'],
        state_B=ffs_config['state_B'],
        interfaces=ffs_config['interfaces'],
        cps_config={
            'static_path': '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc',
            'credit_config_path': ffs_config.get('model_config')  # Path to CREDIT config
        },
        output_dir=str(output_dir)
    )
    
    estimator.load_data(forecast_times=ffs_config['forecast_times'])
    
    df = estimator.compute_rates(n_jobs=args.n_jobs)
    
    # Save
    start_str = ffs_config['forecast_times'][0][0].replace(' ', 'T').replace(':', '')[:10]
    end_str = ffs_config['forecast_times'][0][1].replace(' ', 'T').replace(':', '')[:10]
    time_label = f"{start_str}_to_{end_str}"
    
    output_file = output_dir / f"ifs_rates_FFS.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()


# """
# Compute IFS ensemble genesis rates matching FFS methodology EXACTLY.

# Ports the complete multi-storm tracking system from FFS:
# - Track ALL minima in basin (not just one storm per member)
# - 12° merge radius
# - 2-timestep lost track threshold
# - 3x3 local minimum verification
# - CPS tropical check at λ₀ crossing

# Rate = N_lambda0_crossings / (N_members * 15_days)
# """

# import numpy as np
# import xarray as xr
# from pathlib import Path
# import pandas as pd
# from typing import Dict, Tuple, List, Optional
# import yaml
# import argparse
# from scipy.ndimage import gaussian_filter
# import warnings
# warnings.filterwarnings('ignore')
# from joblib import Parallel, delayed
# import multiprocessing as mp

# try:
#     from tails.cyclone_phase_tracker import CyclonePhaseTracker
#     CPS_AVAILABLE = True
# except ImportError:
#     CPS_AVAILABLE = False
#     print("WARNING: CPS not available")

# class IFSRateEstimator:
#     """
#     IFS ensemble rate estimator matching FFS methodology.
    
#     Multi-storm tracking with local minimum verification and CPS filtering.
#     """
    
#     def __init__(self, 
#                  ifs_path: str,
#                  state_A: float = 1008.0,
#                  state_B: float = 982.0,
#                  interfaces: list = None,
#                  basin: Dict = None,
#                  cps_config: Dict = None):
        
#         self.ifs_path = Path(ifs_path)
#         self.state_A = state_A
#         self.state_B = state_B
#         self.interfaces = (interfaces if interfaces is not None else [1000]).copy()
#         if state_B not in self.interfaces:
#             self.interfaces.append(state_B)
#         self.lambda0 = self.interfaces[0]
        
#         # Basin
#         if basin is None:
#             self.basin = {
#                 'lat_min': 10.0,
#                 'lat_max': 45.0,
#                 'lon_min': -100.0,
#                 'lon_max': -20.0
#             }
#         else:
#             self.basin = basin
        
#         # CPS setup
#         self.enable_cps = CPS_AVAILABLE
#         self.surface_geopotential = None
#         self.credit_config = None
#         self.cps_tracker = None
        
#         if self.enable_cps:
#             cps_config = cps_config or {}
#             try:
#                 # Load static fields
#                 static_path = cps_config.get('static_path', 
#                     '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc')
#                 with xr.open_dataset(static_path) as df:
#                     self.surface_geopotential = df["Z_GDS4_SFC"].values
                
#                 # Load CREDIT config for pressure interpolation
#                 credit_config_path = cps_config.get('credit_config_path')
#                 if credit_config_path:
#                     with open(credit_config_path, 'r') as f:
#                         self.credit_config = yaml.safe_load(f)
                
#                 print("✓ CPS setup ready (will init after data load)")
#             except Exception as e:
#                 print(f"⚠ CPS setup failed: {e}")
#                 self.enable_cps = False
        
#         self.ds = None
    
#     def load_data(self, forecast_times: list = None):
#         """Load IFS data."""
#         print(f"Loading IFS data from {self.ifs_path}...")
#         self.ds = xr.open_zarr(self.ifs_path, consolidated=True)
        
#         if forecast_times is not None and len(forecast_times) > 0:
#             init_times = [pd.to_datetime(ft[0]) for ft in forecast_times]
#             self.ds_filtered = self.ds.sel(time=init_times)
#             max_lead = pd.Timedelta(days=15)
#             valid_leads = self.ds_filtered.prediction_timedelta <= max_lead
#             self.ds_filtered = self.ds_filtered.isel(prediction_timedelta=valid_leads)
#         else:
#             ds_year = self.ds.sel(time=self.ds.time.dt.year == 2022)
#             self.ds_filtered = ds_year.sel(time=ds_year.time.dt.hour == 0)
        
#         print(f"Loaded {len(self.ds_filtered.time)} initializations")
#         print(f"Ensemble size: {len(self.ds_filtered.number)}")
        
#         # Initialize CPS tracker with IFS grid
#         if self.enable_cps and self.cps_tracker is None:
#             try:
#                 # Create latlons dataset from IFS grid
#                 latlons = xr.Dataset({
#                     'latitude': self.ds_filtered.latitude,
#                     'longitude': self.ds_filtered.longitude
#                 })
#                 self.cps_tracker = CyclonePhaseTracker(latlons, radius_km=500)
#                 print("✓ CPS tracker initialized with IFS grid")
#             except Exception as e:
#                 print(f"⚠ CPS tracker init failed: {e}")
#                 self.enable_cps = False
        
#         return self.ds_filtered
    
#     def get_basin_mask(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
#         """
#         Basin mask.
        
#         Args:
#             lats: Latitude array
#             lons: Longitude array (any format)
        
#         Returns:
#             Boolean mask array
#         """
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
#         lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
        
#         return lat_mask[:, None] & lon_mask[None, :]
    
#     def _find_local_minima(self, mslp_hpa: np.ndarray, 
#                            lats: np.ndarray, lons: np.ndarray,
#                            basin_mask: np.ndarray) -> List[Tuple[float, float, float]]:
#         """
#         Find ALL local MSLP minima (matching FFS flux mode).
        
#         Args:
#             mslp_hpa: MSLP field in hPa
#             lats: Latitude array
#             lons: Longitude array (converted to -180:180 internally)
#             basin_mask: Boolean mask
        
#         Returns:
#             List of (mslp, lat, lon) tuples
#         """
#         # Convert lons to -180:180
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         # Mask to basin
#         field = np.where(basin_mask, mslp_hpa, np.nan)
        
#         # Smooth for location finding
#         mslp_smooth = gaussian_filter(field, sigma=1.5)
        
#         minima = []
        
#         # 3x3 neighborhood check (EXACTLY like FFS)
#         for i in range(1, mslp_smooth.shape[0] - 1):
#             for j in range(1, mslp_smooth.shape[1] - 1):
#                 val_smooth = mslp_smooth[i, j]
#                 if not np.isfinite(val_smooth):
#                     continue
                
#                 nbrs = mslp_smooth[i-1:i+2, j-1:j+2]
#                 if np.all(val_smooth <= nbrs):
#                     # Local minimum found - use RAW value
#                     val_raw = mslp_hpa[i, j]
#                     minima.append((val_raw, lats[i], lons_180[j]))
        
#         return minima
    
#     def _merge_nearby_minima(self, minima: List[Tuple[float, float, float]], 
#                             radius_deg: float = 12.0) -> List[Tuple[float, float, float]]:
#         """
#         Merge minima within radius_deg (matching FFS).
        
#         Keep strongest (lowest MSLP) in each cluster.
#         """
#         if len(minima) == 0:
#             return []
        
#         merged = []
#         for min_mslp, min_lat, min_lon in minima:
#             merged_flag = False
#             for i, (m_mslp, m_lat, m_lon) in enumerate(merged):
#                 dist = np.sqrt((min_lat - m_lat)**2 + (min_lon - m_lon)**2)
#                 if dist < radius_deg:
#                     # Merge - keep stronger
#                     if min_mslp < m_mslp:
#                         merged[i] = (min_mslp, min_lat, min_lon)
#                     merged_flag = True
#                     break
            
#             if not merged_flag:
#                 merged.append((min_mslp, min_lat, min_lon))
        
#         return merged
    
#     def _check_cps_with_motion(self, ds_timestep: xr.Dataset, location: Tuple[float, float],
#                                 mslp: float, storm_id: int,
#                                 prev_lon: Optional[float], prev_lat: Optional[float]) -> Tuple[bool, Dict]:
#         """
#         CPS tropical check for IFS data with nearest-neighbor pressure level extension.
#         """
#         if not self.enable_cps:
#             return False, {}
        
#         lat, lon = location
        
#         try:
#             # IFS has only 500, 700, 850 hPa - extend to 300, 600, 900
#             ifs_levels = [500, 700, 850]
#             target_levels = [300, 500, 600, 700, 850, 900]
            
#             ds_list = []
#             for target in target_levels:
#                 if target in ifs_levels:
#                     ds_list.append(ds_timestep.sel(level=target))
#                 else:
#                     nearest = min(ifs_levels, key=lambda x: abs(x - target))
#                     ds_level = ds_timestep.sel(level=nearest).copy()
#                     ds_level = ds_level.assign_coords(level=target)
#                     ds_list.append(ds_level)
            
#             ds_extended = xr.concat(ds_list, dim='level')
            
#             # CREATE NEW LATLONS GRID FROM IFS COORDINATES
#             latlons_ifs = xr.Dataset({
#                 'latitude': ds_extended.latitude,
#                 'longitude': ds_extended.longitude
#             })
            
#             # CREATE NEW CPS TRACKER WITH IFS GRID
#             cps_tracker_ifs = CyclonePhaseTracker(latlons_ifs, radius_km=500)
            
#             # RENAME to match CPS expectations
#             ds_cps = ds_extended.rename({
#                 'geopotential': 'Z_PRES',
#                 'temperature': 'T_PRES',
#                 'level': 'pressure'
#             })

#             for var in ['Z_PRES', 'T_PRES']:
#                 if var in ds_cps:
#                     # Make sure dimension order is (pressure, latitude, longitude)
#                     ds_cps[var] = ds_cps[var].transpose('pressure', 'latitude', 'longitude')

#             # Compute CPS with IFS-specific tracker
#             cps = cps_tracker_ifs.compute_CPS(
#                 ds_cps,
#                 lon, lat, mslp,
#                 prev_lon=prev_lon, prev_lat=prev_lat,
#                 stage='genesis'
#             )
            
#             is_ET = not cps['is_tropical']

#             return is_ET, cps
            
#         except Exception as e:
#             print(f"⚠ CPS computation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return False, {}
    
#     def process_single_member(self, init_time, member: int) -> Dict:
#         """
#         Process one ensemble member with FFS-style multi-storm tracking.
        
#         EXACTLY matches FFS flux mode:
#         - Tracks ALL storms simultaneously
#         - 12° merge radius
#         - 2-step lost track threshold
#         - 3x3 local minimum verification before saving
#         - CPS check at λ₀ crossing
#         - Storm track history for CPS motion
#         """
#         traj = self.ds_filtered.sel(time=init_time, number=member)
        
#         lats = traj.latitude.values
#         lons = traj.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         basin_mask = self.get_basin_mask(lats, lons)
        
#         # Multi-storm tracking state (EXACTLY like FFS)
#         tracked_storms = {}  # storm_id -> {location, mslp, saved, lost_count}
#         next_storm_id = 0
#         storm_track_history = {}  # storm_id -> {lons: [], lats: []} for CPS
        
#         # Track ALL interface crossings (not just λ₀)
#         interface_crossings = {i: False for i in range(len(self.interfaces))}
#         interface_crossing_times = {i: None for i in range(len(self.interfaces))}
        
#         lambda0_crossings = []
#         cps_rejections = []
#         reached_B = False
#         B_lead_time = None
        
#         n_timesteps = len(traj.prediction_timedelta)
        
#         # Compute basin mask once (doesn't change per timestep)
#         basin_mask = self.get_basin_mask(lats, lons)
        
#         # Check if starts in state A
#         mslp_init_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=0).values
#         mslp_init_hpa = mslp_init_pa / 100.0
#         if mslp_init_hpa.shape != (len(lats), len(lons)):
#             mslp_init_hpa = mslp_init_hpa.T
#         try:
#             mslp_init_basin = np.where(basin_mask, mslp_init_hpa, np.nan)
#         except:
#             print(init_time, member)
#             raise
#         starts_in_A = np.nanmin(mslp_init_basin) > self.state_A

#         # Track which storms existed at t=0 (pre-existing)
#         initial_storm_ids = set()  # Storm IDs that exist at t=0
        
#         for lead_idx in range(n_timesteps):
#             mslp_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=lead_idx).values
#             # Remove any extra dimensions
#             while mslp_pa.ndim > 2:
#                 mslp_pa = mslp_pa[0]
#             mslp_hpa_raw = mslp_pa / 100.0
            
#             # Ensure correct dimensions
#             if mslp_hpa_raw.shape == (len(lons), len(lats)):
#                 mslp_hpa = mslp_hpa_raw.T
#             else:
#                 mslp_hpa = mslp_hpa_raw
            
#             # STEP 1: Find ALL local minima (EXACTLY like FFS _find_local_mslp_minima)
#             minima = self._find_local_minima(mslp_hpa, lats, lons, basin_mask)
            
#             # STEP 2: Merge nearby minima within 12° (EXACTLY like FFS)
#             minima = self._merge_nearby_minima(minima, radius_deg=12.0)
            
#             # STEP 3: Match minima to tracked storms (EXACTLY like FFS)
#             matched_storms = set()
#             matched_minima = set()
            
#             for storm_id, storm_info in list(tracked_storms.items()):
#                 storm_lat, storm_lon = storm_info['location']
#                 best_match = None
#                 best_dist = float('inf')
                
#                 for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
#                     if min_idx in matched_minima:
#                         continue
#                     dist = np.sqrt((min_lat - storm_lat)**2 + (min_lon - storm_lon)**2)
#                     if dist < 12.0 and dist < best_dist:
#                         best_match = min_idx
#                         best_dist = dist
                
#                 if best_match is not None:
#                     # Storm tracked successfully
#                     min_mslp, min_lat, min_lon = minima[best_match]
#                     tracked_storms[storm_id]['location'] = (min_lat, min_lon)
#                     tracked_storms[storm_id]['mslp'] = min_mslp
#                     tracked_storms[storm_id]['lost_count'] = 0
                    
#                     # Update track history for CPS
#                     if storm_id not in storm_track_history:
#                         storm_track_history[storm_id] = {'lons': [], 'lats': []}
#                     storm_track_history[storm_id]['lons'].append(min_lon)
#                     storm_track_history[storm_id]['lats'].append(min_lat)
                    
#                     matched_storms.add(storm_id)
#                     matched_minima.add(best_match)
#                 else:
#                     # Storm lost track
#                     tracked_storms[storm_id]['lost_count'] += 1
            
#             # STEP 4: Add new storms from unmatched minima (EXACTLY like FFS)
#             for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
#                 if min_idx not in matched_minima:
#                     storm_id = next_storm_id
#                     next_storm_id += 1
#                     tracked_storms[storm_id] = {
#                         'location': (min_lat, min_lon),
#                         'mslp': min_mslp,
#                         'saved': False,
#                         'lost_count': 0
#                     }
#                     # Initialize track history
#                     storm_track_history[storm_id] = {
#                         'lons': [min_lon],
#                         'lats': [min_lat]
#                     }

#                     # Mark storms that exist at t=0
#                     if lead_idx == 0:
#                         initial_storm_ids.add(storm_id)
            
#             # STEP 5: Remove storms lost for 2+ timesteps (EXACTLY like FFS)
#             for storm_id in list(tracked_storms.keys()):
#                 if tracked_storms[storm_id]['lost_count'] >= 2:
#                     del tracked_storms[storm_id]
#                     if storm_id in storm_track_history:
#                         del storm_track_history[storm_id]
            
#             # STEP 6: Check for λ₀ crossings (EXACTLY like FFS flux mode)
#             dissipated_storms = []
#             for storm_id in list(tracked_storms.keys()):
#                 storm = tracked_storms[storm_id]
                
#                 # Only check unsaved storms that crossed λ₀
#                 if not storm['saved'] and storm['mslp'] < self.lambda0:
#                     # Skip storms that existed at t=0 (not genesis)
#                     if storm_id in initial_storm_ids:
#                         storm['saved'] = True  # Mark as saved to prevent re-checking
#                         continue

#                     storm_lat, storm_lon = storm['location']
                    
#                     # VERIFICATION 1: 3x3 local minimum check (EXACTLY like FFS)
#                     lat_idx = np.argmin(np.abs(lats - storm_lat))
#                     lon_idx = np.argmin(np.abs(lons_180 - storm_lon))
                    
#                     i_min = max(0, lat_idx - 1)
#                     i_max = min(mslp_hpa.shape[0], lat_idx + 2)
#                     j_min = max(0, lon_idx - 1)
#                     j_max = min(mslp_hpa.shape[1], lon_idx + 2)
                    
#                     nbhd = mslp_hpa[i_min:i_max, j_min:j_max]
#                     center_val = mslp_hpa[lat_idx, lon_idx]
                    
#                     if not np.all(center_val <= nbhd):
#                         # Not a local minimum - reject
#                         storm['saved'] = True
#                         continue
                    
#                     # VERIFICATION 2: CPS tropical check (EXACTLY like FFS)
#                     if self.enable_cps:
#                         try:
#                             ds_step = traj.isel(prediction_timedelta=lead_idx)
                            
#                             # Get previous position for CPS motion
#                             prev_lon, prev_lat = None, None
#                             if storm_id in storm_track_history:
#                                 hist = storm_track_history[storm_id]
#                                 if len(hist['lons']) >= 2:  # Need at least 2 points
#                                     prev_lon = hist['lons'][-2]
#                                     prev_lat = hist['lats'][-2]
                            
#                             is_ET, cps = self._check_cps_with_motion(
#                                 ds_step, 
#                                 (storm_lat, storm_lon),
#                                 storm['mslp'],
#                                 storm_id,
#                                 prev_lon=prev_lon,
#                                 prev_lat=prev_lat
#                             )
                            
#                             if is_ET:
#                                 # Extratropical - reject
#                                 storm['saved'] = True
#                                 cps_rejections.append({
#                                     'storm_id': storm_id,
#                                     'lead_idx': lead_idx,
#                                     'mslp': storm['mslp'],
#                                     'lat': storm_lat,
#                                     'lon': storm_lon,
#                                     'phase': cps.get('phase', 'unknown')
#                                 })
#                                 continue
#                         except Exception as e:
#                             # CPS check failed - allow crossing (conservative)
#                             pass
                    
#                     # VALID λ₀ CROSSING - save it and mark this storm as having crossed λ₀
#                     lambda0_crossings.append({
#                         'storm_id': storm_id,
#                         'mslp': storm['mslp'],
#                         'lat': storm_lat,
#                         'lon': storm_lon,
#                         'lead_idx': lead_idx,
#                         'lead_hours': float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
#                     })
#                     storm['saved'] = True
#                     storm['crossed_lambda0'] = True  # Mark that this storm crossed λ₀
                
#                 # Check remaining interface crossings (λ₁, λ₂, ...) - λ₀ already handled above
#                 if storm_id not in initial_storm_ids:
#                     for i in range(1, len(self.interfaces)):  # Skip index 0 (λ₀)
#                         interface_val = self.interfaces[i]
#                         if storm['mslp'] < interface_val:
#                             if not interface_crossings[i]:
#                                 interface_crossings[i] = True
#                                 interface_crossing_times[i] = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                
#                 # Check if reached state B AND had crossed λ₀ (genesis -> B event)
#                 if storm['mslp'] < self.state_B and not reached_B:
#                     if storm.get('crossed_lambda0', False):  # Only count if it crossed λ₀ first
#                         if storm_id not in initial_storm_ids:
#                             reached_B = True
#                             B_lead_time = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                
#                 # Mark dissipated storms for removal (MSLP > state_A)
#                 if storm['mslp'] > self.state_A:
#                     dissipated_storms.append(storm_id)
            
#             # Remove dissipated storms
#             for storm_id in dissipated_storms:
#                 del tracked_storms[storm_id]
#                 if storm_id in storm_track_history:
#                     del storm_track_history[storm_id]
        
#         return {
#             'init_time': init_time,
#             'member': member,
#             'starts_in_A': starts_in_A,
#             'reached_B': reached_B,
#             'B_lead_time': B_lead_time,
#             'n_lambda0_crossings': len(lambda0_crossings),
#             'n_cps_rejected': len(cps_rejections),
#             'crossings': lambda0_crossings,
#             'cps_rejections': cps_rejections,
#             'interface_crossings': interface_crossings,
#             'interface_crossing_times': interface_crossing_times
#         }
    
#     def _process_single_init(self, init_idx: int, init_time, n_inits: int) -> Dict:
#         """
#         Process all members for one initialization time.
#         This is the unit of parallelization.
#         """
#         n_members = len(self.ds_filtered.number)
        
#         member_results = []
#         for member in self.ds_filtered.number.values:
#             result = self.process_single_member(init_time, member)
#             member_results.append(result)
        
#         # Aggregate for this IC
#         total_lambda0 = sum(r['n_lambda0_crossings'] for r in member_results)
#         total_cps_rejected = sum(r['n_cps_rejected'] for r in member_results)
#         n_starts_in_A = sum(r['starts_in_A'] for r in member_results)
#         n_reached_B = sum(r['reached_B'] for r in member_results)
#         n_B_from_A = sum(1 for r in member_results if r['reached_B'] and r['starts_in_A'])
        
#         # Interface crossing counts
#         interface_counts = {}
#         for i in range(len(self.interfaces)):
#             interface_counts[i] = sum(r['interface_crossings'][i] for r in member_results)
        
#         # FIXED denominator for FLUX: always 50 members × 15 days = 750 days
#         total_sim_days = n_members * 15.0
#         rate_lambda0_flux = total_lambda0 / total_sim_days
        
#         # BF rate calculation
#         total_sim_days = n_members * 15.0
#         n_reached_B = sum(r['reached_B'] for r in member_results)
#         rate_B_bf = n_reached_B / total_sim_days  # Simple: events/time

#         # Still useful to track mean time to B for diagnostics
#         if n_reached_B > 0:
#             times_to_B = [r['B_lead_time'] for r in member_results if r['reached_B']]
#             mean_time_to_B_hours = np.mean(times_to_B)
#             mean_time_to_B_days = mean_time_to_B_hours / 24.0
#         else:
#             mean_time_to_B_hours = np.nan
#             mean_time_to_B_days = np.nan

#         result = {
#             'init_time': init_time,
#             'n_members': n_members,
#             'n_starts_in_A': n_starts_in_A,
#             'n_reached_B': n_reached_B,
#             'n_B_from_A': n_B_from_A,
#             'mean_time_to_B_hours': mean_time_to_B_hours,
#             'mean_time_to_B_days': mean_time_to_B_days,
#             'n_lambda0_crossings': total_lambda0,
#             'n_cps_rejected': total_cps_rejected,
#             'total_sim_days': total_sim_days,
#             'rate_lambda0_flux_per_day': rate_lambda0_flux,
#             'rate_B_bf_per_day': rate_B_bf,
#             'prob_B': n_reached_B / n_members,
#             'prob_B_from_A': n_B_from_A / n_starts_in_A if n_starts_in_A > 0 else np.nan
#         }

#         # Add per-interface counts and rates
#         for i in range(len(self.interfaces)):
#             result[f'n_crossed_lambda{i}'] = interface_counts[i]
#             result[f'rate_lambda{i}_per_day'] = interface_counts[i] / total_sim_days
#             result[f'prob_lambda{i}'] = interface_counts[i] / n_members
#         # Progress message
#         cps_str = f", CPS rejected={total_cps_rejected}" if self.enable_cps else ""
#         # Show λ₀ from lambda0_crossings, then λ₁, λ₂, ... from interface counts
#         interface_parts = [f"λ0:{result['n_lambda0_crossings']}"]
#         interface_parts.extend([f"λ{i}:{result[f'n_crossed_lambda{i}']}" for i in range(1, len(self.interfaces))])
#         interface_str = ', '.join(interface_parts)
#         print(f"Init {init_idx+1}/{n_inits} ({init_time}): B={result['n_reached_B']}, {interface_str}{cps_str}, flux_λ0={result['rate_lambda0_flux_per_day']:.6f}/day")
#         return result
    
#     def compute_rates(self, n_jobs: int = 1) -> pd.DataFrame:
#         """Compute rates per initialization."""
#         n_inits = len(self.ds_filtered.time)
#         n_members = len(self.ds_filtered.number)
        
#         print(f"\n{'='*70}")
#         print("IFS RATE ESTIMATION (matching FFS methodology)")
#         print(f"{'='*70}")
#         print(f"Initializations: {n_inits}")
#         print(f"Members per init: {n_members}")
#         print(f"λ₀ threshold: {self.lambda0} hPa")
#         print(f"Multi-storm tracking: YES")
#         print(f"Local minimum verification: YES (3x3)")
#         print(f"Storm merge radius: 12°")
#         print(f"Lost track threshold: 2 timesteps")
#         print(f"CPS filtering: {'YES' if self.enable_cps else 'NO'}")
        
#         # Determine number of parallel jobs
#         if n_jobs == -1:
#             n_jobs_actual = max(1, mp.cpu_count() - 1)
#         else:
#             n_jobs_actual = n_jobs
        
#         if n_jobs_actual > 1:
#             print(f"Parallel workers: {n_jobs_actual} (parallelizing across init times)")
#         else:
#             print(f"Running serially")
        
#         print(f"{'='*70}\n")
        
#         # Process all inits (parallel or serial)
#         if n_jobs_actual > 1:
#             # Parallel across init times
#             results = Parallel(n_jobs=n_jobs_actual, backend='loky', verbose=10)(
#                 delayed(self._process_single_init)(init_idx, init_time, n_inits)
#                 for init_idx, init_time in enumerate(self.ds_filtered.time.values)
#             )
#         else:
#             # Serial
#             results = []
#             for init_idx, init_time in enumerate(self.ds_filtered.time.values):
#                 result = self._process_single_init(init_idx, init_time, n_inits)
#                 results.append(result)
        
#         df = pd.DataFrame(results)
        
#         # Summary
#         total_B = df['n_reached_B'].sum()
#         total_crossings = df['n_lambda0_crossings'].sum()
#         total_cps_rejected = df['n_cps_rejected'].sum()
#         total_sim_days = df['total_sim_days'].sum()
#         total_starts_in_A = df['n_starts_in_A'].sum()
#         total_B_from_A = df['n_B_from_A'].sum()

#         pooled_rate_lambda0_flux = total_crossings / total_sim_days
#         pooled_rate_B_bf = total_B / total_sim_days

#         print(f"\n{'='*70}")
#         print("SUMMARY")
#         print(f"{'='*70}")
#         print(f"Total trajectories: {n_inits * n_members}")
#         print(f"Total starting in A: {total_starts_in_A}")
#         print(f"Total reached B (genesis->B events): {total_B}")
#         print(f"Total B from A: {total_B_from_A}")
#         print(f"Total λ₀ crossings: {total_crossings}")
#         if self.enable_cps:
#             print(f"Total CPS rejected: {total_cps_rejected}")
#         print(f"Total simulation time: {total_sim_days:.0f} days")
#         print(f"\nRates:")
#         print(f"  λ₀ FLUX rate: {pooled_rate_lambda0_flux:.6e} crossings/day")
#         print(f"  State B rate: {pooled_rate_B_bf:.6e} events/day")
#         if not df['mean_time_to_B_days'].isna().all():
#             print(f"  Mean time to B (diagnostic): {df['mean_time_to_B_days'].mean():.1f} days")
#         print(f"\nInterface Statistics:")
#         # λ₀ from lambda0_crossings
#         print(f"  λ_0 = {self.interfaces[0]:4.0f} hPa: {total_crossings} crossings, {pooled_rate_lambda0_flux:.6e}/day")
#         # Remaining interfaces
#         for i in range(1, len(self.interfaces)):
#             interface_val = self.interfaces[i]
#             total_crossed = df[f'n_crossed_lambda{i}'].sum()
#             pooled_rate = df[f'n_crossed_lambda{i}'].sum() / total_sim_days
#             print(f"  λ_{i} = {interface_val:4.0f} hPa: {total_crossed} crossings, {pooled_rate:.6e}/day")
#         print(f"{'='*70}\n")

#         df.attrs['pooled_rate_lambda0_flux'] = pooled_rate_lambda0_flux
#         df.attrs['pooled_rate_B_bf'] = pooled_rate_B_bf
#         df.attrs['total_crossings'] = int(total_crossings)
#         df.attrs['total_B'] = int(total_B)
#         df.attrs['total_cps_rejected'] = int(total_cps_rejected)
#         df.attrs['total_sim_days'] = float(total_sim_days)

#         return df


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ffs_config', type=str, required=True)
#     parser.add_argument('--ifs_path', type=str, 
#                        default='/glade/derecho/scratch/schreck/IFS.zarr')
#     parser.add_argument('--n_jobs', type=int, default=1)
    
#     args = parser.parse_args()
    
#     with open(args.ffs_config, 'r') as f:
#         ffs_config = yaml.safe_load(f)
    
#     output_dir = Path(ffs_config['output_dir']) / 'IFS'
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     estimator = IFSRateEstimator(
#         ifs_path=args.ifs_path,
#         state_A=ffs_config['state_A'],
#         state_B=ffs_config['state_B'],
#         interfaces=ffs_config['interfaces'],
#         cps_config={
#             'static_path': '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc',
#             'credit_config_path': ffs_config.get('model_config')  # Path to CREDIT config
#         }
#     )
    
#     estimator.load_data(forecast_times=ffs_config['forecast_times'])
    
#     df = estimator.compute_rates(n_jobs=args.n_jobs)
    
#     # Save
#     start_str = ffs_config['forecast_times'][0][0].replace(' ', 'T').replace(':', '')[:10]
#     end_str = ffs_config['forecast_times'][0][1].replace(' ', 'T').replace(':', '')[:10]
#     time_label = f"{start_str}_to_{end_str}"
    
#     output_file = output_dir / f"ifs_rates_FFS_methodology_{time_label}.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Results saved to: {output_file}")


# if __name__ == "__main__":
#     main()


# """
# Compute IFS ensemble genesis rates matching FFS methodology EXACTLY.

# Ports the complete multi-storm tracking system from FFS:
# - Track ALL minima in basin (not just one storm per member)
# - 12° merge radius
# - 2-timestep lost track threshold
# - 3x3 local minimum verification
# - CPS tropical check at λ₀ crossing

# Rate = N_lambda0_crossings / (N_members * 15_days)
# """

# import numpy as np
# import xarray as xr
# from pathlib import Path
# import pandas as pd
# from typing import Dict, Tuple, List, Optional
# import yaml
# import argparse
# from scipy.ndimage import gaussian_filter
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import warnings
# warnings.filterwarnings('ignore')
# from joblib import Parallel, delayed
# import multiprocessing as mp

# try:
#     from tails.cyclone_phase_tracker import CyclonePhaseTracker
#     CPS_AVAILABLE = True
# except ImportError:
#     CPS_AVAILABLE = False
#     print("WARNING: CPS not available")

# class IFSRateEstimator:
#     """
#     IFS ensemble rate estimator matching FFS methodology.
    
#     Multi-storm tracking with local minimum verification and CPS filtering.
#     """
    
#     def __init__(self,
#                  ifs_path: str,
#                  state_A: float = 1008.0,
#                  state_B: float = 982.0,
#                  interfaces: list = None,
#                  basin: Dict = None,
#                  cps_config: Dict = None,
#                  output_dir: str = None,
#                  save_plots: bool = True):

#         self.ifs_path = Path(ifs_path)
#         self.save_plots = save_plots and (output_dir is not None)
#         if self.save_plots:
#             self.plot_dir = Path(output_dir) / 'plots' / 'stateB'
#             self.plot_dir.mkdir(parents=True, exist_ok=True)
#         else:
#             self.plot_dir = None
#         self.state_A = state_A
#         self.state_B = state_B
#         self.interfaces = (interfaces if interfaces is not None else [1000]).copy()
#         if state_B not in self.interfaces:
#             self.interfaces.append(state_B)
#         self.lambda0 = self.interfaces[0]
        
#         # Basin
#         if basin is None:
#             self.basin = {
#                 'lat_min': 10.0,
#                 'lat_max': 45.0,
#                 'lon_min': -98.0,
#                 'lon_max': -20.0
#             }
#         else:
#             self.basin = basin
        
#         # CPS setup
#         self.enable_cps = CPS_AVAILABLE
#         self.surface_geopotential = None
#         self.credit_config = None
#         self.cps_tracker = None
        
#         if self.enable_cps:
#             cps_config = cps_config or {}
#             try:
#                 # Load static fields
#                 static_path = cps_config.get('static_path', 
#                     '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc')
#                 with xr.open_dataset(static_path) as df:
#                     self.surface_geopotential = df["Z_GDS4_SFC"].values
                
#                 # Load CREDIT config for pressure interpolation
#                 credit_config_path = cps_config.get('credit_config_path')
#                 if credit_config_path:
#                     with open(credit_config_path, 'r') as f:
#                         self.credit_config = yaml.safe_load(f)
                
#                 print("✓ CPS setup ready (will init after data load)")
#             except Exception as e:
#                 print(f"⚠ CPS setup failed: {e}")
#                 self.enable_cps = False
        
#         self.ds = None
    
#     def load_data(self, forecast_times: list = None):
#         """Load IFS data."""
#         print(f"Loading IFS data from {self.ifs_path}...")
#         self.ds = xr.open_zarr(self.ifs_path, consolidated=True)
        
#         if forecast_times is not None and len(forecast_times) > 0:
#             init_times = [pd.to_datetime(ft[0]) for ft in forecast_times]
#             self.ds_filtered = self.ds.sel(time=init_times)
#             max_lead = pd.Timedelta(days=15)
#             valid_leads = self.ds_filtered.prediction_timedelta <= max_lead
#             self.ds_filtered = self.ds_filtered.isel(prediction_timedelta=valid_leads)
#         else:
#             ds_year = self.ds.sel(time=self.ds.time.dt.year == 2022)
#             self.ds_filtered = ds_year.sel(time=ds_year.time.dt.hour == 0)
        
#         print(f"Loaded {len(self.ds_filtered.time)} initializations")
#         print(f"Ensemble size: {len(self.ds_filtered.number)}")
        
#         # Initialize CPS tracker with IFS grid
#         if self.enable_cps and self.cps_tracker is None:
#             try:
#                 # Create latlons dataset from IFS grid
#                 latlons = xr.Dataset({
#                     'latitude': self.ds_filtered.latitude,
#                     'longitude': self.ds_filtered.longitude
#                 })
#                 self.cps_tracker = CyclonePhaseTracker(latlons, radius_km=500)
#                 print("✓ CPS tracker initialized with IFS grid")
#             except Exception as e:
#                 print(f"⚠ CPS tracker init failed: {e}")
#                 self.enable_cps = False
        
#         return self.ds_filtered
    
#     def get_basin_mask(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
#         """
#         Basin mask.
        
#         Args:
#             lats: Latitude array
#             lons: Longitude array (any format)
        
#         Returns:
#             Boolean mask array
#         """
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
#         lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
        
#         return lat_mask[:, None] & lon_mask[None, :]

#     def _save_storm_figure(self, mslp_hpa, lats, lons_180,
#                            storm_lat, storm_lon, storm_mslp,
#                            init_time, member, lead_idx, lead_hours,
#                            storm_id, event_type='stateB'):
#         """
#         Save MSLP figure for a storm event (matching FFS _save_mslp_figure style).
#         """
#         if not self.save_plots:
#             return

#         try:
#             from matplotlib.colors import BoundaryNorm

#             basin_mask = self.get_basin_mask(lats, np.where(lons_180 < 0, lons_180 + 360, lons_180))
#             basin_rows, basin_cols = np.where(basin_mask)
#             pad = 10
#             row_min = max(0, basin_rows.min() - pad)
#             row_max = min(mslp_hpa.shape[0], basin_rows.max() + pad)
#             col_min = max(0, basin_cols.min() - pad)
#             col_max = min(mslp_hpa.shape[1], basin_cols.max() + pad)

#             mslp_crop = mslp_hpa[row_min:row_max, col_min:col_max]
#             lat_crop = lats[row_min:row_max]
#             lon_crop = lons_180[col_min:col_max]

#             fig = plt.figure(figsize=(12, 8))
#             ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

#             ax.set_extent([lon_crop.min(), lon_crop.max(),
#                           lat_crop.min(), lat_crop.max()],
#                           crs=ccrs.PlateCarree())

#             levels = np.arange(960, 1030, 2)
#             norm = BoundaryNorm(levels, ncolors=plt.cm.RdBu_r.N, clip=True)
#             pcm = ax.pcolormesh(lon_crop, lat_crop, mslp_crop,
#                                norm=norm, cmap='RdBu_r',
#                                transform=ccrs.PlateCarree(),
#                                shading='auto', zorder=1)

#             ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
#             ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
#             ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)

#             # ax.plot(storm_lon, storm_lat, 'k*', markersize=20,
#             #        markeredgewidth=2, markeredgecolor='yellow',
#             #        transform=ccrs.PlateCarree(), zorder=5)

#             gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)
#             gl.top_labels = False
#             gl.right_labels = False

#             lat_dir = 'N' if storm_lat >= 0 else 'S'
#             lon_dir = 'W' if storm_lon < 0 else 'E'
#             init_str = pd.Timestamp(init_time).strftime('%Y-%m-%d %H:%M UTC')
#             valid_str = (pd.Timestamp(init_time) + pd.Timedelta(hours=lead_hours)).strftime('%Y-%m-%d %H:%M UTC')

#             title = f'{valid_str} (init: {init_str}, member {member})\n'
#             title += f'MSLP: {storm_mslp:.1f} hPa @ ({abs(storm_lat):.1f}\u00b0{lat_dir}, {abs(storm_lon):.1f}\u00b0{lon_dir})'
#             title += f'\nStorm {storm_id} | Lead +{lead_hours:.0f}h | {event_type}'
#             ax.set_title(title, fontsize=12, fontweight='bold')

#             plt.colorbar(pcm, ax=ax, label='MSLP (hPa)', shrink=0.8)
#             plt.tight_layout()

#             init_label = pd.Timestamp(init_time).strftime('%Y%m%d_%H')
#             fig_path = self.plot_dir / f'{init_label}_m{member:02d}_storm{storm_id}_{event_type}.png'
#             plt.savefig(fig_path, dpi=200, bbox_inches='tight')
#             plt.close(fig)
#         except Exception as e:
#             print(f"  \u26a0 Plot save failed: {e}")
#             plt.close('all')

#     def _find_local_minima(self, mslp_hpa: np.ndarray,
#                            lats: np.ndarray, lons: np.ndarray,
#                            basin_mask: np.ndarray) -> List[Tuple[float, float, float]]:
#         """
#         Find ALL local MSLP minima (matching FFS flux mode).
        
#         Args:
#             mslp_hpa: MSLP field in hPa
#             lats: Latitude array
#             lons: Longitude array (converted to -180:180 internally)
#             basin_mask: Boolean mask
        
#         Returns:
#             List of (mslp, lat, lon) tuples
#         """
#         # Convert lons to -180:180
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         # Mask to basin
#         field = np.where(basin_mask, mslp_hpa, np.nan)
        
#         # Smooth for location finding
#         mslp_smooth = gaussian_filter(field, sigma=1.5)
        
#         minima = []
        
#         # 3x3 neighborhood check (EXACTLY like FFS)
#         for i in range(1, mslp_smooth.shape[0] - 1):
#             for j in range(1, mslp_smooth.shape[1] - 1):
#                 val_smooth = mslp_smooth[i, j]
#                 if not np.isfinite(val_smooth):
#                     continue
                
#                 nbrs = mslp_smooth[i-1:i+2, j-1:j+2]
#                 if np.all(val_smooth <= nbrs):
#                     # Local minimum found - use RAW value
#                     val_raw = mslp_hpa[i, j]
#                     minima.append((val_raw, lats[i], lons_180[j]))
        
#         return minima
    
#     def _merge_nearby_minima(self, minima: List[Tuple[float, float, float]], 
#                             radius_deg: float = 12.0) -> List[Tuple[float, float, float]]:
#         """
#         Merge minima within radius_deg (matching FFS).
        
#         Keep strongest (lowest MSLP) in each cluster.
#         """
#         if len(minima) == 0:
#             return []
        
#         merged = []
#         for min_mslp, min_lat, min_lon in minima:
#             merged_flag = False
#             for i, (m_mslp, m_lat, m_lon) in enumerate(merged):
#                 dist = np.sqrt((min_lat - m_lat)**2 + (min_lon - m_lon)**2)
#                 if dist < radius_deg:
#                     # Merge - keep stronger
#                     if min_mslp < m_mslp:
#                         merged[i] = (min_mslp, min_lat, min_lon)
#                     merged_flag = True
#                     break
            
#             if not merged_flag:
#                 merged.append((min_mslp, min_lat, min_lon))
        
#         return merged
    
#     def _check_cps_with_motion(self, ds_timestep: xr.Dataset, location: Tuple[float, float],
#                                 mslp: float, storm_id: int,
#                                 prev_lon: Optional[float], prev_lat: Optional[float]) -> Tuple[bool, Dict]:
#         """
#         CPS tropical check for IFS data with nearest-neighbor pressure level extension.
#         """
#         if not self.enable_cps:
#             return False, {}
        
#         lat, lon = location
        
#         try:
#             # IFS has only 500, 700, 850 hPa - extend to 300, 600, 900
#             ifs_levels = [500, 700, 850]
#             target_levels = [300, 500, 600, 700, 850, 900]
            
#             ds_list = []
#             for target in target_levels:
#                 if target in ifs_levels:
#                     ds_list.append(ds_timestep.sel(level=target))
#                 else:
#                     nearest = min(ifs_levels, key=lambda x: abs(x - target))
#                     ds_level = ds_timestep.sel(level=nearest).copy()
#                     ds_level = ds_level.assign_coords(level=target)
#                     ds_list.append(ds_level)
            
#             ds_extended = xr.concat(ds_list, dim='level')
            
#             # CREATE NEW LATLONS GRID FROM IFS COORDINATES
#             latlons_ifs = xr.Dataset({
#                 'latitude': ds_extended.latitude,
#                 'longitude': ds_extended.longitude
#             })
            
#             # CREATE NEW CPS TRACKER WITH IFS GRID
#             cps_tracker_ifs = CyclonePhaseTracker(latlons_ifs, radius_km=500)
            
#             # RENAME to match CPS expectations
#             ds_cps = ds_extended.rename({
#                 'geopotential': 'Z_PRES',
#                 'temperature': 'T_PRES',
#                 'level': 'pressure'
#             })

#             for var in ['Z_PRES', 'T_PRES']:
#                 if var in ds_cps:
#                     # Make sure dimension order is (pressure, latitude, longitude)
#                     ds_cps[var] = ds_cps[var].transpose('pressure', 'latitude', 'longitude')

#             # Compute CPS with IFS-specific tracker
#             cps = cps_tracker_ifs.compute_CPS(
#                 ds_cps,
#                 lon, lat, mslp,
#                 prev_lon=prev_lon, prev_lat=prev_lat,
#                 stage='genesis'
#             )
            
#             is_ET = not cps['is_tropical']

#             return is_ET, cps
            
#         except Exception as e:
#             print(f"⚠ CPS computation failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return False, {}
    
#     def process_single_member(self, init_time, member: int) -> Dict:
#         """
#         Process one ensemble member with FFS-style multi-storm tracking.
        
#         EXACTLY matches FFS flux mode:
#         - Tracks ALL storms simultaneously
#         - 12° merge radius
#         - 2-step lost track threshold
#         - 3x3 local minimum verification before saving
#         - CPS check at λ₀ crossing
#         - Storm track history for CPS motion
#         """
#         traj = self.ds_filtered.sel(time=init_time, number=member)
        
#         lats = traj.latitude.values
#         lons = traj.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         basin_mask = self.get_basin_mask(lats, lons)
        
#         # Multi-storm tracking state (EXACTLY like FFS)
#         tracked_storms = {}  # storm_id -> {location, mslp, saved, lost_count}
#         next_storm_id = 0
#         storm_track_history = {}  # storm_id -> {lons: [], lats: []} for CPS
        
#         # Track ALL interface crossings (not just λ₀)
#         interface_crossings = {i: False for i in range(len(self.interfaces))}
#         interface_crossing_times = {i: None for i in range(len(self.interfaces))}
        
#         lambda0_crossings = []
#         cps_rejections = []
#         reached_B = False
#         B_lead_time = None
        
#         n_timesteps = len(traj.prediction_timedelta)
        
#         # Compute basin mask once (doesn't change per timestep)
#         basin_mask = self.get_basin_mask(lats, lons)
        
#         # Check if starts in state A
#         mslp_init_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=0).values
#         mslp_init_hpa = mslp_init_pa / 100.0
#         if mslp_init_hpa.shape != (len(lats), len(lons)):
#             mslp_init_hpa = mslp_init_hpa.T
#         try:
#             mslp_init_basin = np.where(basin_mask, mslp_init_hpa, np.nan)
#         except:
#             print(init_time, member)
#             raise
#         starts_in_A = np.nanmin(mslp_init_basin) > self.state_A

#         # Track which storms existed at t=0 (pre-existing)
#         initial_storm_ids = set()  # Storm IDs that exist at t=0
        
#         for lead_idx in range(n_timesteps):
#             mslp_pa = traj.mean_sea_level_pressure.isel(prediction_timedelta=lead_idx).values
#             # Remove any extra dimensions
#             while mslp_pa.ndim > 2:
#                 mslp_pa = mslp_pa[0]
#             mslp_hpa_raw = mslp_pa / 100.0
            
#             # Ensure correct dimensions
#             if mslp_hpa_raw.shape == (len(lons), len(lats)):
#                 mslp_hpa = mslp_hpa_raw.T
#             else:
#                 mslp_hpa = mslp_hpa_raw
            
#             # STEP 1: Find ALL local minima (EXACTLY like FFS _find_local_mslp_minima)
#             minima = self._find_local_minima(mslp_hpa, lats, lons, basin_mask)
            
#             # STEP 2: Merge nearby minima within 12° (EXACTLY like FFS)
#             minima = self._merge_nearby_minima(minima, radius_deg=12.0)
            
#             # STEP 3: Match minima to tracked storms (EXACTLY like FFS)
#             matched_storms = set()
#             matched_minima = set()
            
#             for storm_id, storm_info in list(tracked_storms.items()):
#                 storm_lat, storm_lon = storm_info['location']
#                 best_match = None
#                 best_dist = float('inf')
                
#                 for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
#                     if min_idx in matched_minima:
#                         continue
#                     dist = np.sqrt((min_lat - storm_lat)**2 + (min_lon - storm_lon)**2)
#                     if dist < 12.0 and dist < best_dist:
#                         best_match = min_idx
#                         best_dist = dist
                
#                 if best_match is not None:
#                     # Storm tracked successfully
#                     min_mslp, min_lat, min_lon = minima[best_match]
#                     tracked_storms[storm_id]['location'] = (min_lat, min_lon)
#                     tracked_storms[storm_id]['mslp'] = min_mslp
#                     tracked_storms[storm_id]['lost_count'] = 0
                    
#                     # Update track history for CPS
#                     if storm_id not in storm_track_history:
#                         storm_track_history[storm_id] = {'lons': [], 'lats': []}
#                     storm_track_history[storm_id]['lons'].append(min_lon)
#                     storm_track_history[storm_id]['lats'].append(min_lat)
                    
#                     matched_storms.add(storm_id)
#                     matched_minima.add(best_match)
#                 else:
#                     # Storm lost track in basin - but if it crossed λ₀, try tracking
#                     # outside basin (match FFS: existing storms tracked via 12° radius
#                     # without basin mask, only new detection uses basin mask)
#                     if storm_info.get('crossed_lambda0', False):
#                         storm_lat, storm_lon = storm_info['location']
#                         mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
#                         lat_grid, lon_grid = np.meshgrid(lats, lons_180, indexing='ij')
#                         dist = np.sqrt((lat_grid - storm_lat)**2 + (lon_grid - storm_lon)**2)
#                         local = np.where(dist < 12.0, mslp_smooth, np.nan)

#                         if not np.all(np.isnan(local)):
#                             idx = np.nanargmin(local)
#                             i, j = np.unravel_index(idx, local.shape)
#                             min_mslp = float(mslp_hpa[i, j])
#                             min_lat = float(lats[i])
#                             min_lon = float(lons_180[j])

#                             tracked_storms[storm_id]['location'] = (min_lat, min_lon)
#                             tracked_storms[storm_id]['mslp'] = min_mslp
#                             tracked_storms[storm_id]['lost_count'] = 0

#                             if storm_id not in storm_track_history:
#                                 storm_track_history[storm_id] = {'lons': [], 'lats': []}
#                             storm_track_history[storm_id]['lons'].append(min_lon)
#                             storm_track_history[storm_id]['lats'].append(min_lat)

#                             matched_storms.add(storm_id)
#                         else:
#                             tracked_storms[storm_id]['lost_count'] += 1
#                     else:
#                         tracked_storms[storm_id]['lost_count'] += 1

#             # STEP 4: Add new storms from unmatched minima (EXACTLY like FFS)
#             for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
#                 if min_idx not in matched_minima:
#                     storm_id = next_storm_id
#                     next_storm_id += 1
#                     tracked_storms[storm_id] = {
#                         'location': (min_lat, min_lon),
#                         'mslp': min_mslp,
#                         'saved': False,
#                         'lost_count': 0
#                     }
#                     # Initialize track history
#                     storm_track_history[storm_id] = {
#                         'lons': [min_lon],
#                         'lats': [min_lat]
#                     }

#                     # Mark storms that exist at t=0
#                     if lead_idx == 0:
#                         initial_storm_ids.add(storm_id)
            
#             # STEP 5: Remove storms lost for 2+ timesteps (EXACTLY like FFS)
#             for storm_id in list(tracked_storms.keys()):
#                 if tracked_storms[storm_id]['lost_count'] >= 2:
#                     del tracked_storms[storm_id]
#                     if storm_id in storm_track_history:
#                         del storm_track_history[storm_id]
            
#             # STEP 5b: Remove storms that have become extratropical (match FFS flux mode)
#             if self.enable_cps:
#                 for storm_id in list(tracked_storms.keys()):
#                     storm = tracked_storms[storm_id]
#                     storm_lat, storm_lon = storm['location']

#                     # Only check storms at high latitudes or far east (match FFS)
#                     if storm_lat > 50.0 or storm_lon > -10.0:
#                         try:
#                             ds_step = traj.isel(prediction_timedelta=lead_idx)

#                             prev_lon, prev_lat = None, None
#                             if storm_id in storm_track_history:
#                                 hist = storm_track_history[storm_id]
#                                 if len(hist['lons']) >= 2:
#                                     prev_lon = hist['lons'][-2]
#                                     prev_lat = hist['lats'][-2]

#                             is_ET, cps = self._check_cps_with_motion(
#                                 ds_step,
#                                 (storm_lat, storm_lon),
#                                 storm['mslp'],
#                                 storm_id,
#                                 prev_lon=prev_lon,
#                                 prev_lat=prev_lat
#                             )

#                             if is_ET:
#                                 del tracked_storms[storm_id]
#                                 if storm_id in storm_track_history:
#                                     del storm_track_history[storm_id]
#                         except Exception:
#                             pass

#             # STEP 6: Check for λ₀ crossings (EXACTLY like FFS flux mode)
#             dissipated_storms = []
#             for storm_id in list(tracked_storms.keys()):
#                 storm = tracked_storms[storm_id]

#                 # Only check unsaved storms that crossed λ₀
#                 if not storm['saved'] and storm['mslp'] < self.lambda0:
#                     # Skip storms that existed at t=0 (not genesis)
#                     if storm_id in initial_storm_ids:
#                         storm['saved'] = True  # Mark as saved to prevent re-checking
#                         continue

#                     storm_lat, storm_lon = storm['location']
                    
#                     # VERIFICATION 1: 3x3 local minimum check (EXACTLY like FFS)
#                     lat_idx = np.argmin(np.abs(lats - storm_lat))
#                     lon_idx = np.argmin(np.abs(lons_180 - storm_lon))
                    
#                     i_min = max(0, lat_idx - 1)
#                     i_max = min(mslp_hpa.shape[0], lat_idx + 2)
#                     j_min = max(0, lon_idx - 1)
#                     j_max = min(mslp_hpa.shape[1], lon_idx + 2)
                    
#                     nbhd = mslp_hpa[i_min:i_max, j_min:j_max]
#                     center_val = mslp_hpa[lat_idx, lon_idx]
                    
#                     if not np.all(center_val <= nbhd):
#                         # Not a local minimum - reject
#                         storm['saved'] = True
#                         continue
                    
#                     # VERIFICATION 2: CPS tropical check (EXACTLY like FFS)
#                     if self.enable_cps:
#                         try:
#                             ds_step = traj.isel(prediction_timedelta=lead_idx)
                            
#                             # Get previous position for CPS motion
#                             prev_lon, prev_lat = None, None
#                             if storm_id in storm_track_history:
#                                 hist = storm_track_history[storm_id]
#                                 if len(hist['lons']) >= 2:  # Need at least 2 points
#                                     prev_lon = hist['lons'][-2]
#                                     prev_lat = hist['lats'][-2]
                            
#                             is_ET, cps = self._check_cps_with_motion(
#                                 ds_step, 
#                                 (storm_lat, storm_lon),
#                                 storm['mslp'],
#                                 storm_id,
#                                 prev_lon=prev_lon,
#                                 prev_lat=prev_lat
#                             )
                            
#                             if is_ET:
#                                 # Extratropical - reject
#                                 storm['saved'] = True
#                                 cps_rejections.append({
#                                     'storm_id': storm_id,
#                                     'lead_idx': lead_idx,
#                                     'mslp': storm['mslp'],
#                                     'lat': storm_lat,
#                                     'lon': storm_lon,
#                                     'phase': cps.get('phase', 'unknown')
#                                 })
#                                 continue
#                         except Exception as e:
#                             # CPS check failed - reject crossing (match FFS behavior)
#                             storm['saved'] = True
#                             continue
                    
#                     # VALID λ₀ CROSSING - save it and mark this storm as having crossed λ₀
#                     lambda0_crossings.append({
#                         'storm_id': storm_id,
#                         'mslp': storm['mslp'],
#                         'lat': storm_lat,
#                         'lon': storm_lon,
#                         'lead_idx': lead_idx,
#                         'lead_hours': float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
#                     })
#                     storm['saved'] = True
#                     storm['crossed_lambda0'] = True  # Mark that this storm crossed λ₀
                
#                 # Check remaining interface crossings (λ₁, λ₂, ...) - λ₀ already handled above
#                 if storm_id not in initial_storm_ids and storm.get('crossed_lambda0', False):
#                     for i in range(1, len(self.interfaces)):  # Skip index 0 (λ₀)
#                         interface_val = self.interfaces[i]
#                         if storm['mslp'] < interface_val:
#                             if not interface_crossings[i]:
#                                 interface_crossings[i] = True
#                                 interface_crossing_times[i] = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
                
#                 # Check if reached state B AND had crossed λ₀ (genesis -> B event)
#                 if storm['mslp'] < self.state_B and not reached_B:
#                     if storm.get('crossed_lambda0', False):  # Only count if it crossed λ₀ first
#                         if storm_id not in initial_storm_ids:
#                             reached_B = True
#                             B_lead_time = float(traj.prediction_timedelta.values[lead_idx] / np.timedelta64(1, 'h'))
#                             storm_lat, storm_lon = storm['location']
#                             self._save_storm_figure(
#                                 mslp_hpa, lats, lons_180,
#                                 storm_lat, storm_lon, storm['mslp'],
#                                 init_time, member, lead_idx, B_lead_time,
#                                 storm_id, event_type='stateB'
#                             )
                
#                 # Mark dissipated storms for removal (MSLP > state_A)
#                 if storm['mslp'] > self.state_A:
#                     dissipated_storms.append(storm_id)
            
#             # Remove dissipated storms
#             for storm_id in dissipated_storms:
#                 del tracked_storms[storm_id]
#                 if storm_id in storm_track_history:
#                     del storm_track_history[storm_id]
        
#         return {
#             'init_time': init_time,
#             'member': member,
#             'starts_in_A': starts_in_A,
#             'reached_B': reached_B,
#             'B_lead_time': B_lead_time,
#             'n_lambda0_crossings': len(lambda0_crossings),
#             'n_cps_rejected': len(cps_rejections),
#             'crossings': lambda0_crossings,
#             'cps_rejections': cps_rejections,
#             'interface_crossings': interface_crossings,
#             'interface_crossing_times': interface_crossing_times
#         }
    
#     def _process_single_init(self, init_idx: int, init_time, n_inits: int) -> Dict:
#         """
#         Process all members for one initialization time.
#         This is the unit of parallelization.
#         """
#         n_members = len(self.ds_filtered.number)
        
#         member_results = []
#         for member in self.ds_filtered.number.values:
#             result = self.process_single_member(init_time, member)
#             member_results.append(result)
        
#         # Aggregate for this IC
#         total_lambda0 = sum(r['n_lambda0_crossings'] for r in member_results)
#         total_cps_rejected = sum(r['n_cps_rejected'] for r in member_results)
#         n_starts_in_A = sum(r['starts_in_A'] for r in member_results)
#         n_reached_B = sum(r['reached_B'] for r in member_results)
#         n_B_from_A = sum(1 for r in member_results if r['reached_B'] and r['starts_in_A'])
        
#         # Interface crossing counts
#         interface_counts = {}
#         for i in range(len(self.interfaces)):
#             interface_counts[i] = sum(r['interface_crossings'][i] for r in member_results)
        
#         # FIXED denominator for FLUX: always 50 members × 15 days = 750 days
#         total_sim_days = n_members * 15.0
#         rate_lambda0_flux = total_lambda0 / total_sim_days
        
#         # BF rate calculation
#         total_sim_days = n_members * 15.0
#         n_reached_B = sum(r['reached_B'] for r in member_results)
#         rate_B_bf = n_reached_B / total_sim_days  # Simple: events/time

#         # Still useful to track mean time to B for diagnostics
#         if n_reached_B > 0:
#             times_to_B = [r['B_lead_time'] for r in member_results if r['reached_B']]
#             mean_time_to_B_hours = np.mean(times_to_B)
#             mean_time_to_B_days = mean_time_to_B_hours / 24.0
#         else:
#             mean_time_to_B_hours = np.nan
#             mean_time_to_B_days = np.nan

#         result = {
#             'init_time': init_time,
#             'n_members': n_members,
#             'n_starts_in_A': n_starts_in_A,
#             'n_reached_B': n_reached_B,
#             'n_B_from_A': n_B_from_A,
#             'mean_time_to_B_hours': mean_time_to_B_hours,
#             'mean_time_to_B_days': mean_time_to_B_days,
#             'n_lambda0_crossings': total_lambda0,
#             'n_cps_rejected': total_cps_rejected,
#             'total_sim_days': total_sim_days,
#             'rate_lambda0_flux_per_day': rate_lambda0_flux,
#             'rate_B_bf_per_day': rate_B_bf,
#             'prob_B': n_reached_B / n_members,
#             'prob_B_from_A': n_B_from_A / n_starts_in_A if n_starts_in_A > 0 else np.nan
#         }

#         # Add per-interface counts and rates
#         for i in range(len(self.interfaces)):
#             result[f'n_crossed_lambda{i}'] = interface_counts[i]
#             result[f'rate_lambda{i}_per_day'] = interface_counts[i] / total_sim_days
#             result[f'prob_lambda{i}'] = interface_counts[i] / n_members
#         # Progress message
#         cps_str = f", CPS rejected={total_cps_rejected}" if self.enable_cps else ""
#         # Show λ₀ from lambda0_crossings, then λ₁, λ₂, ... from interface counts
#         interface_parts = [f"λ0:{result['n_lambda0_crossings']}"]
#         interface_parts.extend([f"λ{i}:{result[f'n_crossed_lambda{i}']}" for i in range(1, len(self.interfaces))])
#         interface_str = ', '.join(interface_parts)
#         print(f"Init {init_idx+1}/{n_inits} ({init_time}): B={result['n_reached_B']}, {interface_str}{cps_str}, flux_λ0={result['rate_lambda0_flux_per_day']:.6f}/day")
#         return result
    
#     def compute_rates(self, n_jobs: int = 1) -> pd.DataFrame:
#         """Compute rates per initialization."""
#         n_inits = len(self.ds_filtered.time)
#         n_members = len(self.ds_filtered.number)
        
#         print(f"\n{'='*70}")
#         print("IFS RATE ESTIMATION (matching FFS methodology)")
#         print(f"{'='*70}")
#         print(f"Initializations: {n_inits}")
#         print(f"Members per init: {n_members}")
#         print(f"λ₀ threshold: {self.lambda0} hPa")
#         print(f"Multi-storm tracking: YES")
#         print(f"Local minimum verification: YES (3x3)")
#         print(f"Storm merge radius: 12°")
#         print(f"Lost track threshold: 2 timesteps")
#         print(f"CPS filtering: {'YES' if self.enable_cps else 'NO'}")
        
#         # Determine number of parallel jobs
#         if n_jobs == -1:
#             n_jobs_actual = max(1, mp.cpu_count() - 1)
#         else:
#             n_jobs_actual = n_jobs
        
#         if n_jobs_actual > 1:
#             print(f"Parallel workers: {n_jobs_actual} (parallelizing across init times)")
#         else:
#             print(f"Running serially")
        
#         print(f"{'='*70}\n")
        
#         # Process all inits (parallel or serial)
#         if n_jobs_actual > 1:
#             # Parallel across init times
#             results = Parallel(n_jobs=n_jobs_actual, backend='loky', verbose=10)(
#                 delayed(self._process_single_init)(init_idx, init_time, n_inits)
#                 for init_idx, init_time in enumerate(self.ds_filtered.time.values)
#             )
#         else:
#             # Serial
#             results = []
#             for init_idx, init_time in enumerate(self.ds_filtered.time.values):
#                 result = self._process_single_init(init_idx, init_time, n_inits)
#                 results.append(result)
        
#         df = pd.DataFrame(results)
        
#         # Summary
#         total_B = df['n_reached_B'].sum()
#         total_crossings = df['n_lambda0_crossings'].sum()
#         total_cps_rejected = df['n_cps_rejected'].sum()
#         total_sim_days = df['total_sim_days'].sum()
#         total_starts_in_A = df['n_starts_in_A'].sum()
#         total_B_from_A = df['n_B_from_A'].sum()

#         pooled_rate_lambda0_flux = total_crossings / total_sim_days
#         pooled_rate_B_bf = total_B / total_sim_days

#         print(f"\n{'='*70}")
#         print("SUMMARY")
#         print(f"{'='*70}")
#         print(f"Total trajectories: {n_inits * n_members}")
#         print(f"Total starting in A: {total_starts_in_A}")
#         print(f"Total reached B (genesis->B events): {total_B}")
#         print(f"Total B from A: {total_B_from_A}")
#         print(f"Total λ₀ crossings: {total_crossings}")
#         if self.enable_cps:
#             print(f"Total CPS rejected: {total_cps_rejected}")
#         print(f"Total simulation time: {total_sim_days:.0f} days")
#         print(f"\nRates:")
#         print(f"  λ₀ FLUX rate: {pooled_rate_lambda0_flux:.6e} crossings/day")
#         print(f"  State B rate: {pooled_rate_B_bf:.6e} events/day")
#         if not df['mean_time_to_B_days'].isna().all():
#             print(f"  Mean time to B (diagnostic): {df['mean_time_to_B_days'].mean():.1f} days")
#         print(f"\nInterface Statistics:")
#         # λ₀ from lambda0_crossings
#         print(f"  λ_0 = {self.interfaces[0]:4.0f} hPa: {total_crossings} crossings, {pooled_rate_lambda0_flux:.6e}/day")
#         # Remaining interfaces
#         for i in range(1, len(self.interfaces)):
#             interface_val = self.interfaces[i]
#             total_crossed = df[f'n_crossed_lambda{i}'].sum()
#             pooled_rate = df[f'n_crossed_lambda{i}'].sum() / total_sim_days
#             print(f"  λ_{i} = {interface_val:4.0f} hPa: {total_crossed} crossings, {pooled_rate:.6e}/day")
#         print(f"{'='*70}\n")

#         df.attrs['pooled_rate_lambda0_flux'] = pooled_rate_lambda0_flux
#         df.attrs['pooled_rate_B_bf'] = pooled_rate_B_bf
#         df.attrs['total_crossings'] = int(total_crossings)
#         df.attrs['total_B'] = int(total_B)
#         df.attrs['total_cps_rejected'] = int(total_cps_rejected)
#         df.attrs['total_sim_days'] = float(total_sim_days)

#         return df


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ffs_config', type=str, required=True)
#     parser.add_argument('--ifs_path', type=str, 
#                        default='/glade/derecho/scratch/schreck/IFS.zarr')
#     parser.add_argument('--n_jobs', type=int, default=1)
    
#     args = parser.parse_args()
    
#     with open(args.ffs_config, 'r') as f:
#         ffs_config = yaml.safe_load(f)
    
#     output_dir = Path(ffs_config['output_dir']) / 'IFS'
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     estimator = IFSRateEstimator(
#         ifs_path=args.ifs_path,
#         state_A=ffs_config['state_A'],
#         state_B=ffs_config['state_B'],
#         interfaces=ffs_config['interfaces'],
#         cps_config={
#             'static_path': '/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc',
#             'credit_config_path': ffs_config.get('model_config')  # Path to CREDIT config
#         },
#         output_dir=str(output_dir)
#     )
    
#     estimator.load_data(forecast_times=ffs_config['forecast_times'])
    
#     df = estimator.compute_rates(n_jobs=args.n_jobs)
    
#     # Save
#     start_str = ffs_config['forecast_times'][0][0].replace(' ', 'T').replace(':', '')[:10]
#     end_str = ffs_config['forecast_times'][0][1].replace(' ', 'T').replace(':', '')[:10]
#     time_label = f"{start_str}_to_{end_str}"
    
#     output_file = output_dir / f"ifs_rates_FFS_methodology_{time_label}.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Results saved to: {output_file}")


# if __name__ == "__main__":
#     main()
