import torch
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import copy
import random
import string
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

from credit.output import make_xarray
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.data import concat_and_reshape, reshape_only
from credit.interp import full_state_pressure_interpolation, mean_sea_level_pressure_simple as mslp_simple
from tails.ffs_logger import FFSLogger
from tails.cyclone_phase_tracker import CyclonePhaseTracker

import warnings
warnings.filterwarnings('ignore')


@dataclass
class InterfaceConfig:
    """Configuration saved at interface crossing."""
    input_state: torch.Tensor
    latents: Optional[torch.Tensor]
    forecast_step: int
    mslp_value: float
    interface_idx: int
    timestamp: str
    restart_datetime: datetime
    config_name: str
    parent_config: Optional[str] = None
    track_id: Optional[int] = None
    feature_location: Optional[Tuple[float, float]] = None
    cps_params: Optional[Dict] = None


class HurricaneGenesisFFS:
    """
    Forward Flux Sampling for hurricane genesis with integrated CPS filtering.
    
    Single unified class - no inheritance, no delegation, no monkey patching.
    All functionality in one place with clean conditional logic for CPS.
    """
    
    def __init__(self, 
                 model, 
                 state_transformer, 
                 config, 
                 initial_dataset,
                 dataset_params,
                 output_dir='./ffs_output',
                 state_A=1008, 
                 state_B=982, 
                 interfaces=[1000, 988, 980, 975, 970],
                 decorrelation_interface=None,
                 max_attempts_without_success=1000,
                 worker_id=0,
                 rank=0,
                 world_size=1,
                 ic_dirname=None,
                 use_cps=True):
        
        self.model = model
        self.state_transformer = state_transformer
        self.config = config
        self.initial_dataset = initial_dataset
        self.dataset_params = dataset_params
        self.worker_id = worker_id
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        self.max_attempts_without_success = max_attempts_without_success
        
        # Output directories
        self.output_dir = Path(output_dir)
        self.ic_base_dir = self.output_dir / ic_dirname if ic_dirname else self.output_dir
        
        self.logs_dir = self.ic_base_dir / 'logs'
        self.flux_dir = self.ic_base_dir / 'flux'
        self.stateB_dir = self.ic_base_dir / 'stateB'
        self.failed_dir = self.ic_base_dir / 'failed_trajectories'
        
        for d in [self.logs_dir, self.flux_dir, self.stateB_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load static data
        self.latlons = xr.open_dataset(config["loss"]["latitude_weights"]).load()
        with xr.open_dataset('/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc') as df:
            self.surface_geopotential = df["Z_GDS4_SFC"].values
            self.land_sea_mask = df["LSM"].values
        
        # Basin definition
        self.basin = {
            'lat_min': 10.0,
            'lat_max': 45.0,
            'lon_min': -100.0,
            'lon_max': -20.0
        }
        
        # FFS thresholds
        self.interfaces = sorted(interfaces, reverse=True)
        self.interfaces.append(state_B)
        self.state_A_threshold = state_A
        self.state_B_threshold = state_B
        self.decorrelation_interface = decorrelation_interface
        
        # Multi-storm tracking
        self.tracked_storms = {}
        self.next_storm_id = 0
        
        # Storage
        self.interface_configs: Dict[int, List[InterfaceConfig]] = {
            i: [] for i in range(len(self.interfaces))
        }
        
        # Statistics
        self.flux_estimate = None
        self.transition_probs = []
        self.direct_B_count = 0
        self.direct_B_rate = None
        
        # Visualization
        self.visualize_mslp = False
        self.fig = None
        self.ax = None
        self._colorbar_added = False
        
        # CPS integration
        self.use_cps = use_cps
        if self.use_cps:
            self.cps_tracker = CyclonePhaseTracker(self.latlons, radius_km=500)
            self.storm_track_history = {}
            print("✓ CPS tracker initialized (Hart 2003 formulation)")
    
    def enable_visualization(self):
        """Enable real-time MSLP visualization."""
        self.visualize_mslp = True
        print("✓ Visualization enabled")
    
    def disable_visualization(self):
        """Disable visualization."""
        self.visualize_mslp = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        print("✓ Visualization disabled")
    
    def _plot_mslp(self, mslp_hpa, basin_mask, datetime_str, center_lat, center_lon, 
                   center_val, show_marker=False, marker_label=None):
        """Plot MSLP field with optional crossing marker."""
        from IPython.display import display, clear_output
        
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(16, 10))
            self.ax = self.fig.add_subplot(1, 1, 1, 
                projection=ccrs.LambertConformal(
                    central_longitude=-60.0,
                    central_latitude=35.0,
                    standard_parallels=(30, 50)
                ))
        
        self.ax.clear()
        
        lats = self.latlons.latitude.values
        lons_180 = np.where(self.latlons.longitude.values > 180, 
                        self.latlons.longitude.values - 360, 
                        self.latlons.longitude.values)
        
        # Periodic boundary fix
        n_pad = 20
        left_pad = mslp_hpa[:, -n_pad:]
        right_pad = mslp_hpa[:, :n_pad]
        mslp_extended = np.concatenate([left_pad, mslp_hpa, right_pad], axis=1)
        mslp_extended_smooth = gaussian_filter(mslp_extended, sigma=3.0)
        mslp_smooth = mslp_extended_smooth[:, n_pad:-n_pad]
        
        self.ax.set_extent([-100, -10, 0, 85], crs=ccrs.PlateCarree())
        
        from matplotlib.colors import BoundaryNorm
        levels = np.arange(960, 1032, 4)
        norm = BoundaryNorm(levels, ncolors=plt.cm.RdBu_r.N, clip=True)
        
        pcm = self.ax.pcolormesh(lons_180, lats, mslp_smooth, 
                                norm=norm,
                                cmap='RdBu_r', 
                                transform=ccrs.PlateCarree(), 
                                shading='auto', zorder=1)
        
        self.ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0)
        self.ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, alpha=0.6)
        
        if show_marker and center_lat is not None and center_lon is not None:
            self.ax.plot(center_lon, center_lat, marker='*', markersize=24,
                    markeredgecolor='yellow', markeredgewidth=2.5, color='red',
                    transform=ccrs.PlateCarree(), zorder=10,
                    label=marker_label or 'Crossing')
        
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        title_parts = []
        if datetime_str:
            title_parts.append(datetime_str)
        
        if center_lat is not None and center_lon is not None:
            lat_dir = 'N' if center_lat >= 0 else 'S'
            lon_dir = 'W' if center_lon < 0 else 'E'
            title_parts.append(f"MSLP: {center_val:.1f} hPa @ ({abs(center_lat):.1f}°{lat_dir}, {abs(center_lon):.1f}°{lon_dir})")
        else:
            title_parts.append(f"MSLP: {center_val:.1f} hPa")
        
        self.ax.set_title('\n'.join(title_parts), fontsize=14, fontweight="bold")
        
        if show_marker:
            self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        if not self._colorbar_added:
            cbar = plt.colorbar(pcm, ax=self.ax, label="MSLP (hPa)", shrink=0.8)
            cbar.set_ticks(np.arange(960, 1032, 8))
            self._colorbar_added = True
        
        clear_output(wait=True)
        display(self.fig)
        plt.pause(0.01)
    
    def get_basin_mask(self) -> np.ndarray:
        """Create basin mask."""
        lats = self.latlons.latitude.values
        lons = np.where(self.latlons.longitude.values > 180, 
                       self.latlons.longitude.values - 360, 
                       self.latlons.longitude.values)
        
        lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
        lon_mask = (lons >= self.basin['lon_min']) & (lons <= self.basin['lon_max'])
        
        return lat_mask[:, None] & lon_mask[None, :]
    
    def calculate_mslp_wrapper(self, y_pred_phys, batch, simple_mslp=False):
        """Calculate MSLP from model output."""
        datetime_str = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
        
        if simple_mslp:
            surface_pressure_pa = y_pred_phys[0, 64, 0].cpu().numpy()
            temperature_k = y_pred_phys[0, 65, 0].cpu().numpy()
            mslp_pa = mslp_simple(surface_pressure_pa, temperature_k, self.surface_geopotential)
            mslp_pa = torch.from_numpy(mslp_pa).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            return torch.cat([y_pred_phys, mslp_pa], dim=1)

        darray_upper, darray_single = make_xarray(
            y_pred_phys,
            datetime_str,
            self.latlons.latitude.values,
            self.latlons.longitude.values,
            self.config,
        )
        
        ds_merged = xr.merge([
            darray_upper.to_dataset(dim="vars"),
            darray_single.to_dataset(dim="vars")
        ])
        
        pressure_interp = full_state_pressure_interpolation(
            ds_merged,
            self.surface_geopotential,
            **self.config["predict"]["interp_pressure"]
        )
        
        mslp = torch.from_numpy(
            pressure_interp['mean_sea_level_pressure'].values
        ).unsqueeze(1).unsqueeze(2)
        
        return torch.cat([y_pred_phys, mslp], dim=1)
    
    def _compute_pressure_interp(self, y_phys, batch):
        """Compute full pressure interpolation for CPS."""
        datetime_str = datetime.fromtimestamp(
            batch["datetime"][0].item()
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        darray_upper, darray_single = make_xarray(
            y_phys,
            datetime_str,
            self.latlons.latitude.values,
            self.latlons.longitude.values,
            self.config,
        )
        
        ds_merged = xr.merge([
            darray_upper.to_dataset(dim="vars"),
            darray_single.to_dataset(dim="vars")
        ])
        
        return full_state_pressure_interpolation(
            ds_merged,
            self.surface_geopotential,
            **self.config["predict"]["interp_pressure"]
        )
    
    def _get_previous_position(self, storm_id):
        """Get previous position for CPS motion calculation."""
        if not self.use_cps or storm_id not in self.storm_track_history:
            return None, None
        
        hist = self.storm_track_history[storm_id]
        if len(hist['lons']) < 1:
            return None, None
        
        return hist['lons'][-1], hist['lats'][-1]
    
    def _update_track_history(self, storm_id, lon, lat):
        """Update track history for CPS."""
        if not self.use_cps:
            return
        
        if storm_id not in self.storm_track_history:
            self.storm_track_history[storm_id] = {'lons': [], 'lats': []}
        
        self.storm_track_history[storm_id]['lons'].append(lon)
        self.storm_track_history[storm_id]['lats'].append(lat)
    
    def _check_cps(self, pressure_interp, location, mslp, storm_id, stage='genesis'):
        """Check if storm is extratropical using CPS."""
        if not self.use_cps:
            return False, {}
        
        lat, lon = location
        
        prev_lon, prev_lat = self._get_previous_position(storm_id)
        self._update_track_history(storm_id, lon, lat)
        
        cps = self.cps_tracker.compute_CPS(
            pressure_interp, 
            lon, lat, mslp,
            prev_lon=prev_lon,
            prev_lat=prev_lat,
            stage=stage
        )
        
        is_ET = bool(not cps['is_tropical'])
        cps_json = {}
        for k, v in cps.items():
            if isinstance(v, np.bool_):
                cps_json[k] = bool(v)
            elif isinstance(v, np.floating):
                cps_json[k] = float(v)
            elif isinstance(v, np.integer):
                cps_json[k] = int(v)
            else:
                cps_json[k] = v
        
        return is_ET, cps_json
    
    def extract_mslp(self, y_phys, batch=None, forecast_step=None,
                     parent_location=None, mode="flux"):
        """Extract minimum sea level pressure."""
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0

        lats = self.latlons.latitude.values
        lons = np.where(
            self.latlons.longitude.values > 180,
            self.latlons.longitude.values - 360,
            self.latlons.longitude.values,
        )

        basin_mask = self.get_basin_mask()

        # SHOOT MODE: Track single storm
        if mode == "shoot" and parent_location is not None:
            lat0, lon0 = parent_location
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            dist = np.sqrt((lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2)

            mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
            local = np.where(dist < 12.0, mslp_smooth, np.nan)
            idx = np.nanargmin(local)
            i, j = np.unravel_index(idx, local.shape)
            
            min_mslp = float(mslp_hpa[i, j])
            lat = float(lats[i])
            lon = float(lons[j])
            
            if self.visualize_mslp:
                datetime_str = None
                if batch and "datetime" in batch:
                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                self._plot_mslp(mslp_hpa, basin_mask, datetime_str, lat, lon, min_mslp, 
                            show_marker=False)

            return min_mslp, 0, (lat, lon)

        # FLUX MODE: Multi-storm detection
        minima = self._find_local_mslp_minima(
            mslp_hpa, lats, lons, basin_mask,
            exclude_center=None,
            exclude_radius_deg=12.0,
        )

        if len(minima) == 0:
            for storm_id in self.tracked_storms:
                self.tracked_storms[storm_id]['lost_count'] += 1
            return self._extract_mslp_fallback(y_phys, batch), None, None
        
        # Merge nearby minima
        merged_minima = []
        for min_mslp, min_lat, min_lon in minima:
            merged = False
            for i, (m_mslp, m_lat, m_lon) in enumerate(merged_minima):
                dist = np.sqrt((min_lat - m_lat)**2 + (min_lon - m_lon)**2)
                if dist < 12.0:
                    if min_mslp < m_mslp:
                        merged_minima[i] = (min_mslp, min_lat, min_lon)
                    merged = True
                    break
            if not merged:
                merged_minima.append((min_mslp, min_lat, min_lon))
        
        minima = merged_minima
        
        # Match to tracked storms
        matched_storms = set()
        matched_minima = set()
        
        for storm_id, storm_info in list(self.tracked_storms.items()):
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
                self.tracked_storms[storm_id]['location'] = (min_lat, min_lon)
                self.tracked_storms[storm_id]['mslp'] = min_mslp
                self.tracked_storms[storm_id]['lost_count'] = 0
                matched_storms.add(storm_id)
                matched_minima.add(best_match)
            else:
                self.tracked_storms[storm_id]['lost_count'] += 1
        
        # Add new storms (track ALL minima, not just those below λ₀)
        for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
            if min_idx not in matched_minima:
                storm_id = self.next_storm_id
                self.next_storm_id += 1
                self.tracked_storms[storm_id] = {
                    'location': (min_lat, min_lon),
                    'mslp': min_mslp,
                    'saved': False,
                    'lost_count': 0
                }
        
        # Remove lost storms
        for storm_id in list(self.tracked_storms.keys()):
            if self.tracked_storms[storm_id]['lost_count'] >= 2:
                del self.tracked_storms[storm_id]
        
        if len(minima) > 0:
            min_mslp, lat, lon = min(minima, key=lambda x: x[0])
            
            if self.visualize_mslp:
                datetime_str = None
                if batch and "datetime" in batch:
                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                self._plot_mslp(mslp_hpa, basin_mask, datetime_str, lat, lon, min_mslp,
                            show_marker=False)
            
            return float(min_mslp), 0, (float(lat), float(lon))
        else:
            return self._extract_mslp_fallback(y_phys, batch), None, None
    
    def _extract_mslp_fallback(self, y_phys, batch=None):
        """Fallback MSLP extraction."""
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        mslp_smooth = gaussian_filter(mslp_basin, sigma=1.5)
        
        min_idx = np.nanargmin(mslp_smooth)
        min_row, min_col = np.unravel_index(min_idx, mslp_smooth.shape)
        min_mslp = float(mslp_hpa[min_row, min_col])

        return min_mslp
    
    def _find_local_mslp_minima(self, mslp_hpa, lats, lons, basin_mask,
                                exclude_center=None, exclude_radius_deg=12.0,
                                smooth_sigma=1.5):
        """Find local MSLP minima - returns ALL minima."""
        field = np.where(basin_mask, mslp_hpa, np.nan)
        
        if exclude_center is not None:
            lat0, lon0 = exclude_center
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            dist = np.sqrt((lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2)
            field = np.where(dist >= exclude_radius_deg, field, np.nan)
        
        mslp_smooth = gaussian_filter(field, sigma=smooth_sigma)

        minima = []

        for i in range(1, mslp_smooth.shape[0] - 1):
            for j in range(1, mslp_smooth.shape[1] - 1):
                val_smooth = mslp_smooth[i, j]
                if not np.isfinite(val_smooth):
                    continue

                nbrs = mslp_smooth[i - 1 : i + 2, j - 1 : j + 2]
                if np.all(val_smooth <= nbrs):
                    val_raw = mslp_hpa[i, j]
                    minima.append((val_raw, lats[i], lons[j]))

        return minima
    
    def check_trajectory_status(self, mslp_value, current_interface, mode='flux'):
        """Check trajectory status."""
        next_interface = current_interface + 1
        if next_interface < len(self.interfaces):
            if mslp_value < self.interfaces[next_interface]:
                return 'crossed_forward', next_interface
        
        if mslp_value < self.state_B_threshold:
            return 'reached_B', None
        
        if mode == 'shoot':
            if mslp_value > self.state_A_threshold:
                return 'returned_A', None
            return 'continue', None
        
        if current_interface >= 0:
            if mslp_value > self.interfaces[current_interface]:
                for i in range(current_interface - 1, -1, -1):
                    if mslp_value < self.interfaces[i]:
                        return 'returned_backward', i
                return 'returned_backward', -1
        
        return 'continue', None
    
    def _create_crossing(self, state, step, mslp, interface_idx, batch, feat_idx, 
                        location, parent, y_phys_with_mslp, cps_params=None):
        """Create InterfaceConfig."""
        if interface_idx == -1:
            config_name = f"stateB_config_{len(list(self.stateB_dir.glob('stateB_config_*.pkl')))+1:04d}"
        elif interface_idx == 0:
            config_name = f"lambda0_config_{len(list(self.flux_dir.glob('lambda0_config_*.pkl')))+1:04d}"
        else:
            save_dir = self.ic_base_dir / str(interface_idx)
            config_name = f"lambda{interface_idx}_config_{len(list(save_dir.glob(f'lambda{interface_idx}_config_*.pkl')))+1:04d}"
        
        config_name += '_' + ''.join(random.choices(string.ascii_uppercase, k=2))
        
        restart_dt = datetime.fromtimestamp(batch["datetime"][0].item()) + timedelta(hours=6)
        
        crossing = InterfaceConfig(
            input_state=state,
            latents=None,
            forecast_step=step,
            mslp_value=mslp,
            interface_idx=interface_idx,
            timestamp=datetime.now().isoformat(),
            restart_datetime=restart_dt,
            config_name=config_name,
            parent_config=parent,
            track_id=None,
            feature_location=location,
            cps_params=cps_params
        )
        
        crossing._feature_idx = feat_idx
        crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
        crossing._y_phys = y_phys_with_mslp.cpu().clone()
        
        mslp_channel_idx = 71
        mslp_raw = y_phys_with_mslp[0, mslp_channel_idx, 0].cpu().numpy() / 100.0
        crossing._mslp_field = gaussian_filter(mslp_raw, sigma=1.5)
        
        return crossing
    
    
    def _save_mslp_figure(self, crossing, save_dir):
        """Save MSLP figure for crossing."""
        if not hasattr(crossing, '_y_phys'):
            return
        
        if hasattr(crossing, '_mslp_field'):
            mslp_smooth = crossing._mslp_field
        else:
            mslp_channel_idx = 71
            mslp_pa = crossing._y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
            mslp_hpa = mslp_pa / 100.0
            mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
        basin_mask = self.get_basin_mask()
        
        lats = self.latlons.latitude.values
        lons_180 = np.where(self.latlons.longitude.values > 180, 
                           self.latlons.longitude.values - 360, 
                           self.latlons.longitude.values)
        
        if crossing.feature_location:
            feat_lat, feat_lon = crossing.feature_location
        else:
            mslp_masked = np.where(basin_mask, mslp_smooth, np.nan)
            min_idx = np.nanargmin(mslp_masked)
            lat_idx, lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
            feat_lat = lats[lat_idx]
            feat_lon = lons_180[lon_idx]
        
        basin_rows, basin_cols = np.where(basin_mask)
        pad = 10
        row_min = max(0, basin_rows.min() - pad)
        row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
        col_min = max(0, basin_cols.min() - pad)
        col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
        
        mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
        lat_crop = lats[row_min:row_max]
        lon_crop = lons_180[col_min:col_max]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        ax.set_extent([lon_crop.min(), lon_crop.max(), 
                      lat_crop.min(), lat_crop.max()], 
                      crs=ccrs.PlateCarree())
        
        levels = np.arange(960, 1030, 4)
        pcm = ax.contourf(lon_crop, lat_crop, mslp_crop, levels=levels,
                         cmap='RdBu_r', extend='both', 
                         transform=ccrs.PlateCarree(), zorder=1)
        
        cs = ax.contour(lon_crop, lat_crop, mslp_crop,
                       levels=np.arange(960, 1030, 8),
                       colors='k', linewidths=0.5, 
                       transform=ccrs.PlateCarree(), zorder=2)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
        
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)
        
        ax.plot(feat_lon, feat_lat, 'k*', markersize=20, 
               markeredgewidth=2, markeredgecolor='yellow',
               transform=ccrs.PlateCarree(), zorder=5)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        
        datetime_str = None
        if hasattr(crossing, '_datetime_obj'):
            datetime_str = crossing._datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
        lat_dir = 'N' if feat_lat >= 0 else 'S'
        lon_dir = 'W' if feat_lon < 0 else 'E'
        title = f'MSLP: {crossing.mslp_value:.1f} hPa @ ({abs(feat_lat):.1f}°{lat_dir}, {abs(feat_lon):.1f}°{lon_dir})'
        if datetime_str:
            title = f'{datetime_str}\n{title}'
        title += f'\n{crossing.config_name}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.colorbar(pcm, ax=ax, label='MSLP (hPa)', shrink=0.8)
        plt.tight_layout()
        
        fig_path = save_dir / f"{crossing.config_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _save_crossing(self, crossing, mode):
        """Save crossing to disk (PKL + PNG)."""
        if crossing.interface_idx == -1:
            save_dir = self.stateB_dir
        elif mode == 'flux':
            save_dir = self.flux_dir
        else:
            save_dir = self.ic_base_dir / str(crossing.interface_idx)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PNG first
        png_saved = False
        try:
            self._save_mslp_figure(crossing, save_dir)
            png_saved = True
        except Exception as e:
            print(f"  ✗ ERROR saving PNG: {e}")
        
        # Save PKL
        pkl_saved = False
        try:
            config_path = save_dir / f"{crossing.config_name}.pkl"
            with open(config_path, 'wb') as f:
                pickle.dump(crossing, f)
            pkl_saved = True
        except Exception as e:
            print(f"  ✗ ERROR saving PKL: {e}")
        
        if png_saved and pkl_saved:
            print(f"  → SAVED: {crossing.config_name} ({crossing.mslp_value:.1f} hPa)")
        else:
            print(f"  ✗ INCOMPLETE: {crossing.config_name} (PNG:{png_saved}, PKL:{pkl_saved})")
    
    def rollout(self, loader, mode='flux', initial_state=None, start_interface=-1, parent_config=None):
        """Run trajectory rollout."""
        if mode == 'flux':
            self.tracked_storms = {}
            self.next_storm_id = 0
            if self.use_cps:
                self.storm_track_history = {}
        
        previous_location = None
        if parent_config and parent_config.feature_location:
            previous_location = parent_config.feature_location
            if self.use_cps and mode == 'shoot':
                storm_id = 0
                self.storm_track_history[storm_id] = {
                    'lons': [previous_location[1]],
                    'lats': [previous_location[0]]
                }
                print(f"  → Initialized shoot track at ({previous_location[0]:.1f}°N, {abs(previous_location[1]):.1f}°W)")
        
        trajectory_mslp = []
        crossings = []
        current_interface = start_interface
        status = 'ongoing'
        saved_states = {}
        failure_reason = None
        failure_cps = None
        
        with torch.no_grad():
            for batch in loader:
                step = batch["forecast_step"].item()
                
                if step == 1:
                    if initial_state is not None:
                        x = initial_state.to(self.device).float()
                    else:
                        if "x_surf" in batch:
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device).float()
                        else:
                            x = reshape_only(batch["x"]).to(self.device).float()
                
                if "x_forcing_static" in batch:
                    x_forcing = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4).float()
                    x = torch.cat((x, x_forcing), dim=1)
                
                y_pred = self.model(x, forecast_step=step - 1)
                y_phys = self.state_transformer.inverse_transform(y_pred.cpu())
                y_with_mslp = self.calculate_mslp_wrapper(y_phys, batch, simple_mslp=True)
                pressure_interp = None
                
                mslp, feat_idx, location = self.extract_mslp(
                    y_with_mslp, batch, step, previous_location, mode
                )
                
                trajectory_mslp.append(mslp)
                
                # SHOOT MODE: Geographic bounds check
                if mode == 'shoot' and location:
                    storm_lat, storm_lon = location
                    outside_normal_bounds = False
                    
                    if storm_lat > 50.0:
                        outside_normal_bounds = True
                    elif storm_lon > -10.0:
                        outside_normal_bounds = True
                    elif storm_lat > 30.0:
                        lats = self.latlons.latitude.values
                        lons = np.where(
                            self.latlons.longitude.values > 180,
                            self.latlons.longitude.values - 360,
                            self.latlons.longitude.values
                        )
                        lat_idx = np.argmin(np.abs(lats - storm_lat))
                        lon_idx = np.argmin(np.abs(lons - storm_lon))
                        lsm_value = self.land_sea_mask[lat_idx, lon_idx]
                        if lsm_value > 0.5:
                            outside_normal_bounds = True
                    
                    if outside_normal_bounds:
                        if self.use_cps:
                            try:
                                if pressure_interp is None:
                                    print("  → Computing pressure interp for CPS verification...")
                                    pressure_interp = self._compute_pressure_interp(y_phys, batch)
                                
                                is_ET, cps = self._check_cps(
                                    pressure_interp, location, mslp, 
                                    storm_id=0, stage='mature'
                                )
                                
                                if is_ET:
                                    print("  → FAILURE: Extratropical at boundary")
                                    print(f"     Phase: {cps['phase']}, -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                    status = 'extratropical'
                                    failure_reason = 'extratropical_at_boundary'
                                    failure_cps = cps
                                    break
                                else:
                                    print(f"  → Still TROPICAL: -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                    print("     Continuing trajectory...")
                            except Exception as e:
                                print(f"  ⚠ CPS check failed: {e}")
                                status = 'failure'
                                failure_reason = 'cps_check_failed'
                                break
                        else:
                            print("  → FAILURE: Outside tropical bounds (no CPS check)")
                            status = 'failure'
                            break
                    
                    previous_location = location
                
                y_norm = self.state_transformer.transform_array(y_phys).to(self.device)
                
                if batch.get("y_diag") is not None:
                    varnum_diag = batch["y_diag"].shape[1]
                    saved_states[step] = y_norm[:, :-varnum_diag, ...].cpu().clone()
                else:
                    saved_states[step] = y_norm.cpu().clone()
                
                traj_status, interface_idx = self.check_trajectory_status(mslp, current_interface, mode)
                
                # FLUX MODE: Check tracked storms for λ₀ crossing
                if mode == 'flux':
                    lambda_0 = self.interfaces[0]
                    
                    for storm_id in list(self.tracked_storms.keys()):
                        storm = self.tracked_storms[storm_id]
                        
                        if not storm['saved'] and storm['mslp'] < lambda_0:
                            storm_lat, storm_lon = storm['location']
                            
                            # Verify local minimum
                            lats = self.latlons.latitude.values
                            lons = np.where(
                                self.latlons.longitude.values > 180,
                                self.latlons.longitude.values - 360,
                                self.latlons.longitude.values
                            )
                            lat_idx = np.argmin(np.abs(lats - storm_lat))
                            lon_idx = np.argmin(np.abs(lons - storm_lon))
                            
                            mslp_channel_idx = 71
                            mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                            mslp_hpa = mslp_pa / 100.0
                            
                            i_min = max(0, lat_idx - 1)
                            i_max = min(mslp_hpa.shape[0], lat_idx + 2)
                            j_min = max(0, lon_idx - 1)
                            j_max = min(mslp_hpa.shape[1], lon_idx + 2)
                            
                            nbhd = mslp_hpa[i_min:i_max, j_min:j_max]
                            center_val = mslp_hpa[lat_idx, lon_idx]
                            
                            if not np.all(center_val <= nbhd):
                                print(f"  → Storm {storm_id} location has no local minimum - REJECTING")
                                storm['saved'] = True
                                continue
                            
                            # CPS check if enabled
                            if self.use_cps:
                                try:
                                    if pressure_interp is None:
                                        print("  → Computing pressure interp for λ₀ crossing...")
                                        pressure_interp = self._compute_pressure_interp(y_phys, batch)
                                    
                                    is_ET, cps = self._check_cps(
                                        pressure_interp, 
                                        storm['location'],
                                        storm['mslp'],
                                        storm_id,
                                        stage='genesis'
                                    )
                                    
                                    # Log CPS values
                                    cps_log_path = self.flux_dir / 'cps_values.txt'
                                    with open(cps_log_path, 'a') as f:
                                        timestamp = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
                                        f.write(f"\n{'='*60}\n")
                                        f.write(f"Storm {storm_id} λ₀ crossing at step {step}\n")
                                        f.write(f"Time: {timestamp}\n")
                                        f.write(f"Location: ({storm_lat:.1f}°N, {abs(storm_lon):.1f}°W)\n")
                                        f.write(f"MSLP: {storm['mslp']:.1f} hPa\n")
                                        f.write(f"B: {cps['B']:.1f} m\n")
                                        f.write(f"-VT^L: {cps['VTL']:.1f} m\n")
                                        f.write(f"-VT^U: {cps['VTU']:.1f} m\n")
                                        f.write(f"Phase: {cps['phase']}\n")
                                        f.write(f"Decision: {'REJECT' if is_ET else 'ACCEPT'}\n")
                                    
                                    if is_ET:
                                        print(f"  → Storm {storm_id} {cps['phase']} at λ₀: NOT SAVED")
                                        storm['saved'] = True
                                        continue
                                    
                                    print(f"  → Storm {storm_id} TROPICAL at λ₀: SAVED")
                                
                                except Exception as e:
                                    print(f"  ⚠ CPS check failed for storm {storm_id}: {e}")
                                    storm['saved'] = True
                                    continue
                            
                            crossing = self._create_crossing(
                                saved_states[step], step, storm['mslp'], 0,
                                batch, None, storm['location'], None, y_with_mslp,
                                cps_params=cps if self.use_cps else None
                            )
                            crossings.append(crossing)
                            storm['saved'] = True
                            print(f"  → Storm {storm_id} crossed λ₀: {storm['mslp']:.1f} hPa")
                            
                            if self.visualize_mslp:
                                datetime_str = None
                                if batch and "datetime" in batch:
                                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                                
                                mslp_channel_idx = 71
                                mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                                mslp_hpa = mslp_pa / 100.0
                                
                                self._plot_mslp(
                                    mslp_hpa, 
                                    self.get_basin_mask(),
                                    datetime_str,
                                    storm_lat, storm_lon, storm['mslp'],
                                    show_marker=True,
                                    marker_label=f'λ₀ CROSSING (Storm {storm_id})'
                                )
                                plt.pause(1.0)
                        
                        if storm['mslp'] > self.state_A_threshold:
                            print(f"  → Storm {storm_id} dissipated: {storm['mslp']:.1f} hPa")
                            del self.tracked_storms[storm_id]
                
                # Handle interface crossings
                if traj_status == 'crossed_forward':
                    if mode == 'shoot':
                        next_idx = current_interface + 1
                        
                        if self.use_cps:
                            try:
                                if pressure_interp is None:
                                    print(f"  → Computing pressure interp for λ{next_idx} crossing...")
                                    pressure_interp = self._compute_pressure_interp(y_phys, batch)
                                
                                is_ET, cps = self._check_cps(
                                    pressure_interp, location, mslp, 
                                    storm_id=0, stage='mature'
                                )
                                
                                # Log CPS values
                                cps_log_path = self.ic_base_dir / str(next_idx) / 'cps_values.txt'
                                cps_log_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(cps_log_path, 'a') as f:
                                    timestamp = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
                                    f.write(f"\n{'='*60}\n")
                                    f.write(f"Interface crossing λ{current_interface}→λ{next_idx} at step {step}\n")
                                    f.write(f"Time: {timestamp}\n")
                                    f.write(f"Location: ({location[0]:.1f}°N, {abs(location[1]):.1f}°W)\n")
                                    f.write(f"MSLP: {mslp:.1f} hPa\n")
                                    f.write(f"B: {cps['B']:.1f} m\n")
                                    f.write(f"-VT^L: {cps['VTL']:.1f} m\n")
                                    f.write(f"-VT^U: {cps['VTU']:.1f} m\n")
                                    f.write(f"Phase: {cps['phase']}\n")
                                    f.write(f"Decision: {'REJECT' if is_ET else 'ACCEPT'}\n")
                                
                                if is_ET:
                                    print("  → FAILURE: Extratropical at crossing")
                                    print(f"     Phase: {cps['phase']}, -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                    status = 'extratropical'
                                    failure_reason = 'extratropical_at_crossing'
                                    failure_cps = cps
                                    break
                                
                                print(f"  → SUCCESS: Tropical at λ{next_idx} crossing")
                            
                            except Exception as e:
                                print(f"  ⚠ CPS check failed at crossing: {e}")
                                status = 'failure'
                                failure_reason = 'cps_check_failed'
                                break
                        
                        crossing = self._create_crossing(
                            saved_states[step], step, mslp, next_idx,
                            batch, feat_idx, location, 
                            parent_config.config_name if parent_config else None,
                            y_with_mslp,
                            cps_params=cps if self.use_cps else None
                        )
                        crossings.append(crossing)
                        status = 'success'
                        print(f"  → Saved {crossing.config_name}: {mslp:.1f} hPa")
                        
                        if self.visualize_mslp and location:
                            datetime_str = None
                            if batch and "datetime" in batch:
                                dt = datetime.fromtimestamp(batch["datetime"][0].item())
                                datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                            
                            mslp_channel_idx = 71
                            mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                            mslp_hpa = mslp_pa / 100.0
                            
                            self._plot_mslp(
                                mslp_hpa, 
                                self.get_basin_mask(),
                                datetime_str,
                                location[0], location[1], mslp,
                                show_marker=True,
                                marker_label=f'λ{next_idx} CROSSING'
                            )
                            plt.pause(1.5)
                        
                        break
                    
                    if mode == 'flux':
                        current_interface = interface_idx
                
                elif traj_status == 'reached_B':
                    crossing = self._create_crossing(
                        saved_states[step], step, mslp, -1,
                        batch, feat_idx, location,
                        parent_config.config_name if parent_config else None,
                        y_with_mslp
                    )
                    crossings.append(crossing)
                    status = 'reached_B'
                    
                    if self.visualize_mslp and location:
                        datetime_str = None
                        if batch and "datetime" in batch:
                            dt = datetime.fromtimestamp(batch["datetime"][0].item())
                            datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                        
                        mslp_channel_idx = 71
                        mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                        mslp_hpa = mslp_pa / 100.0
                        
                        self._plot_mslp(
                            mslp_hpa, 
                            self.get_basin_mask(),
                            datetime_str,
                            location[0], location[1], mslp,
                            show_marker=True,
                            marker_label='STATE B REACHED!'
                        )
                        plt.pause(2.0)
                    
                    break
                
                elif traj_status == 'returned_A':
                    status = 'failure'
                    failure_reason = 'returned_A'
                    break
                
                elif traj_status == 'returned_backward':
                    current_interface = interface_idx if interface_idx is not None else -1
                
                if batch.get("y_diag") is not None:
                    varnum_diag = batch["y_diag"].shape[1]
                    x = y_norm[:, :-varnum_diag, ...].detach()
                else:
                    x = y_norm.detach()
                
                if batch.get("stop_forecast", torch.tensor(False)).item():
                    if status == 'ongoing':
                        status = 'completed' if mode == 'flux' else 'failure'
                    break
        
        for crossing in crossings:
            self._save_crossing(crossing, mode)
        
        return {
            'status': status,
            'mslp_trajectory': trajectory_mslp,
            'crossings': crossings,
            'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None,
            'failure_reason': failure_reason,
            'failure_cps': failure_cps
        }
    
    def generate_flux(self, initial_loader, n_trials):
        """Phase 0: Flux generation."""
        logger = FFSLogger(self.logs_dir / 'flux', rank=self.rank, 
                          world_size=self.world_size, worker_id=self.worker_id, 
                          ic_dirname=None)
        
        print(f"PHASE 0: FLUX GENERATION (λ₀={self.interfaces[0]} hPa)")
        if self.use_cps:
            print("✓ CPS filtering enabled\n")
        
        traj_count = 0
        worker_crossings = 0
        total_days = 0
        direct_B = 0
        
        while True:
            global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
            
            if global_count >= n_trials:
                break
            
            traj_count += 1
            loader = copy.deepcopy(initial_loader)
            result = self.rollout(loader, mode='flux')
            
            n_crossings = len(result['crossings'])
            worker_crossings += n_crossings
            
            if result['status'] == 'reached_B' and n_crossings == 0:
                direct_B += 1
            
            config_names = [c.config_name for c in result['crossings']]
            logger.log_flux_trajectory(traj_count, result, config_names)
            
            total_days += len(result['mslp_trajectory']) * 6 / 24
        
        self.flux_estimate = worker_crossings / total_days if total_days > 0 else 0
        self.direct_B_count = direct_B
        self.direct_B_rate = direct_B / total_days if total_days > 0 else 0
        
        logger.close()
    
    def shoot_trajectory(self, config, interface_idx):
        """Shoot single trajectory."""
        restart_time = config.restart_datetime
        forecast_times = [[
            (restart_time + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
            (restart_time + timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
        ]]
        
        restart_dataset = Predict_Dataset_Batcher(
            **self.dataset_params,
            fcst_datetime=forecast_times
        )
        restart_loader = BatchForecastLenDataLoader(restart_dataset)
        
        return self.rollout(
            restart_loader,
            mode='shoot',
            initial_state=config.input_state,
            start_interface=interface_idx,
            parent_config=config
        )
    
    def shoot_from_interface(self, interface_idx, n_trials):
        """
        Shooting phase: shoot from interface λᵢ to reach λᵢ₊₁.
        
        Args:
            interface_idx: Interface index to shoot FROM (0-indexed)
                          0 = shoot from λ₀ to λ₁
                          1 = shoot from λ₁ to λ₂, etc.
            n_trials: Target number of successful crossings to next interface
        """
        # Load configs to shoot FROM
        # For interface_idx=0: read from flux/ (already in interface_configs[0])
        # For interface_idx>0: load from previous interface directory
        if interface_idx == 0:
            loaded_configs = self.interface_configs.get(0, [])
            source_pattern = "flux/lambda0_config_*.pkl"
        else:
            # Load from directory {interface_idx}
            source_dir = self.ic_base_dir / str(interface_idx)
            source_pattern = f"lambda{interface_idx}_config_*.pkl"
            config_files = list(source_dir.glob(source_pattern))
            loaded_configs = []
            for cfg_file in config_files:
                with open(cfg_file, 'rb') as f:
                    loaded_configs.append(pickle.load(f))
            self.interface_configs[interface_idx] = loaded_configs
        
        lambda_label = interface_idx
        next_interface = interface_idx + 1
        
        print(f"\nSHOOTING from λ_{lambda_label} ({self.interfaces[lambda_label]} hPa) → λ_{next_interface}\n")
        print(f"Available configs: {len(loaded_configs)}")
        
        if len(loaded_configs) == 0:
            print("⚠ No configs available")
            return
        
        logger = FFSLogger(
            self.logs_dir / str(next_interface),
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            ic_dirname=None
        )
        
        successes = 0
        failures = 0
        attempts = 0
        
        # Save TO directory for NEXT interface
        save_dir = self.ic_base_dir / str(next_interface)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            global_count = len(list(save_dir.glob(f'lambda{next_interface}_config_*.pkl')))
            
            if global_count >= n_trials:
                break

            if attempts >= self.max_attempts_without_success and successes == 0:
                logger.log_early_stop(next_interface, attempts)
                break
            
            attempts += 1
            config = np.random.choice(loaded_configs)
            
            # Shoot with interface_idx as START interface
            result = self.shoot_trajectory(config, interface_idx)
            
            child_config = None
            if result['crossings']:
                child_config = result['crossings'][0].config_name
            
            logger.log_shooting_attempt(
                next_interface,
                attempts,
                config.config_name,
                result,
                child_config
            )
            
            if result['status'] == 'success':
                successes += 1
            else:
                failures += 1
        
        P = successes / attempts if attempts > 0 else 0
        self.transition_probs.append(P)
        
        logger.log_interface_summary(
            next_interface, successes, failures, attempts, P, successes
        )
        logger.close()
        
        return P
    
    def run_ffs(self, initial_loader, n_flux_trials=20, n_shoot_trials=10):
        """Run full FFS algorithm."""
        self.generate_flux(initial_loader, n_flux_trials)
        
        flux_configs = list(self.flux_dir.glob('lambda0_config_*.pkl'))
        loaded = []
        for cfg_file in flux_configs:
            with open(cfg_file, 'rb') as f:
                loaded.append(pickle.load(f))
        self.interface_configs[0] = loaded
        
        for i in range(len(self.interfaces) - 1):
            self.shoot_from_interface(i, n_shoot_trials)
        
        summary_logger = FFSLogger(self.logs_dir, rank=self.rank, 
                                   world_size=self.world_size, 
                                   worker_id=self.worker_id, ic_dirname=None)
        
        ffs_prob = self.flux_estimate * np.prod(self.transition_probs)
        
        summary_logger.log_final_results(
            self.flux_estimate,
            self.transition_probs,
            ffs_prob,
            self.direct_B_count,
            self.direct_B_rate
        )
        
        summary_logger.close()

        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Flux: {self._base.flux_estimate:.6f}/day")
        for i, p in enumerate(self._base.transition_probs):
            print(f"P(λ_{i}→λ_{i+1}): {p:.4f}")
        print(f"\nP_FFS: {ffs_prob:.2e}/day")
        print(f"P_direct: {self._base.direct_B_rate:.2e}/day")
        print(f"{'='*70}\n")
        
        return ffs_prob