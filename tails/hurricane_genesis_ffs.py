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
from credit.transforms import Normalize_ERA5_and_Forcing
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
    cps_params: Optional[Dict] = None  # ← ADD THIS LINE


class HurricaneGenesisFFS:
    """
    Clean FFS implementation with feature tracking.
    No inheritance mess, all functionality in one place.
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
                 ic_dirname=None):
        
        self.model = model
        self.state_transformer = state_transformer
        self.config = config
        self.initial_dataset = initial_dataset
        self.dataset_params = dataset_params
        self.worker_id = worker_id
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        # Early stopping
        self.max_attempts_without_success = max_attempts_without_success
        
        # Output directories
        self.output_dir = Path(output_dir)
        if ic_dirname:
            self.ic_base_dir = self.output_dir / ic_dirname
        else:
            self.ic_base_dir = self.output_dir
        
        self.logs_dir = self.ic_base_dir / 'logs'
        self.flux_dir = self.ic_base_dir / 'flux'
        self.stateB_dir = self.ic_base_dir / 'stateB'
        self.failed_dir = self.ic_base_dir / 'failed_trajectories'
        
        for d in [self.logs_dir, self.flux_dir, self.stateB_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
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
        
        # Thresholds
        self.interfaces = sorted(interfaces, reverse=True)  # Ensure descending order
        self.interfaces.append(state_B)  # Add final state B threshold
        self.state_A_threshold = state_A
        self.state_B_threshold = state_B
        self.decorrelation_interface = decorrelation_interface
        
        # Multi-storm tracking for flux mode
        self.tracked_storms = {}  # {storm_id: {'location': (lat,lon), 'mslp': float, 'saved': bool, 'lost_count': int}}
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
    
    def _extract_mslp_with_tobac(
        self,
        mslp_hpa: np.ndarray,
        lats: np.ndarray,
        lons_180: np.ndarray,
        basin_mask: np.ndarray,
        forecast_step: int = 0
    ) -> Tuple[float, Optional[int], Optional[Tuple[float, float]]]:
        """
        Extract MSLP using tobac feature detection for flux mode.
        Returns (min_mslp, feature_idx, (lat, lon))
        """
        try:
            import tobac
            import iris
        except ImportError:
            print("\n⚠ tobac not available, falling back to simple detection")
            return None, None, None
        
        # Mask basin
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
        # Sort coordinates for tobac
        lons_tobac = lons_180.copy()
        mslp_tobac = mslp_basin.copy()
        
        if not np.all(np.diff(lons_tobac) > 0):
            sort_idx = np.argsort(lons_tobac)
            lons_tobac = lons_tobac[sort_idx]
            mslp_tobac = mslp_tobac[:, sort_idx]
        
        try:
            # Create iris cube
            lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
            lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
            time_coord = iris.coords.DimCoord([forecast_step], standard_name='time', units='hours since 2000-01-01 00:00:00')
            
            cube = iris.cube.Cube(
                mslp_tobac[np.newaxis, :, :],
                standard_name='air_pressure_at_mean_sea_level',
                units='hPa',
                dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
            )
            
            # Define thresholds
            basin_min = np.nanmin(mslp_tobac)
            basin_max = np.nanmax(mslp_tobac)
            thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
            thresholds = sorted(thresholds, reverse=False)
            
            # Detect features
            features = tobac.feature_detection_multithreshold(
                field_in=cube,
                dxy=111000,
                threshold=thresholds,
                target='minimum',
                position_threshold='weighted_diff',
                sigma_threshold=1.5,
                n_min_threshold=3
            )

            print(features)
            
            if features is not None and len(features) > 0:
                # Get strongest feature
                strongest_idx = features['threshold_value'].idxmin()
                strongest = features.loc[strongest_idx]
                
                # Get position in features list
                feature_position = features.index.get_loc(strongest_idx)
                
                # Get lat/lon from feature
                feature_lat = float(strongest['latitude'])
                feature_lon = float(strongest['longitude'])
                
                # Find nearest grid point for ACTUAL MSLP value
                lat_idx = np.argmin(np.abs(lats - feature_lat))
                lon_idx = np.argmin(np.abs(lons_180 - feature_lon))
                
                # Return RAW MSLP value at detected location
                min_mslp = float(mslp_hpa[lat_idx, lon_idx])
                
                return min_mslp, feature_position, (feature_lat, feature_lon)
            else:
                return None, None, None
        
        except Exception as e:
            print(f"\n⚠ tobac error: {e}")
            return None, None, None

    def extract_mslp(
        self,
        y_phys,
        batch: Dict = None,
        forecast_step: int = None,
        parent_location: Optional[Tuple[float, float]] = None,
        mode: str = "flux",
    ):
        """
        Robust Tempest-style MSLP extraction.
        Tracks existing storms deterministically and detects new ones safely.
        """

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

        # ---- SHOOT MODE: Track existing storm deterministically ----
        if mode == "shoot" and parent_location is not None:
            lat0, lon0 = parent_location
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            dist = np.sqrt((lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2)

            # Apply basin mask BEFORE smoothing
            # mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
            # mslp_smooth = gaussian_filter(mslp_basin, sigma=1.5)
            mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
            
            # Increased radius to 12° for fast-moving storms
            local = np.where(dist < 12.0, mslp_smooth, np.nan)
            idx = np.nanargmin(local)
            i, j = np.unravel_index(idx, local.shape)
            
            # RETURN RAW VALUE at smoothed location
            min_mslp = float(mslp_hpa[i, j])
            lat = float(lats[i])
            lon = float(lons[j])
            
            # Visualize if enabled
            if self.visualize_mslp:
                datetime_str = None
                if batch and "datetime" in batch:
                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                self._plot_mslp(mslp_hpa, basin_mask, datetime_str, lat, lon, min_mslp, 
                            show_marker=False)

            return min_mslp, 0, (lat, lon)

        # ---- FLUX MODE: Use old detection logic (NO TOBAC) ----
        minima = self._find_local_mslp_minima(
            mslp_hpa,
            lats,
            lons,
            basin_mask,
            exclude_center=None,
            exclude_radius_deg=12.0,
            max_mslp_threshold=1010.0,
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
        
        # Add new storms
        lambda_0 = self.interfaces[0]
        for min_idx, (min_mslp, min_lat, min_lon) in enumerate(minima):
            if min_idx not in matched_minima and min_mslp < lambda_0:
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
        
        # Return strongest storm
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
    
    def _extract_mslp_fallback(self, y_phys: torch.Tensor, batch: Dict = None) -> float:
        """Fallback MSLP extraction using simple minimum."""
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        
        # FIXED: Apply basin mask BEFORE smoothing
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        mslp_smooth = gaussian_filter(mslp_basin, sigma=1.5)
        
        min_idx = np.nanargmin(mslp_smooth)
        min_row, min_col = np.unravel_index(min_idx, mslp_smooth.shape)
        min_mslp = float(mslp_hpa[min_row, min_col])

        return min_mslp
    
    
    def _find_local_mslp_minima(
        self,
        mslp_hpa,
        lats,
        lons,
        basin_mask,
        exclude_center=None,
        exclude_radius_deg=12.0,
        smooth_sigma=1.5,
        max_mslp_threshold=1010.0,
    ):
        """
        Tempest-style local MSLP minimum detector.
        Returns list of (mslp, lat, lon) with RAW MSLP values.
        Only returns minima below max_mslp_threshold (filters weak systems).
        """
        # FIXED: Apply basin mask BEFORE smoothing
        field = np.where(basin_mask, mslp_hpa, np.nan)
        
        # Exclude existing storm explicitly
        if exclude_center is not None:
            lat0, lon0 = exclude_center
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            dist = np.sqrt((lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2)
            field = np.where(dist >= exclude_radius_deg, field, np.nan)
        
        # Smooth the masked field
        mslp_smooth = gaussian_filter(field, sigma=smooth_sigma)

        minima = []

        # 3x3 neighborhood test on SMOOTHED field (for stable detection)
        for i in range(1, mslp_smooth.shape[0] - 1):
            for j in range(1, mslp_smooth.shape[1] - 1):
                val_smooth = mslp_smooth[i, j]
                if not np.isfinite(val_smooth):
                    continue

                nbrs = mslp_smooth[i - 1 : i + 2, j - 1 : j + 2]
                if np.all(val_smooth <= nbrs):
                    # Found minimum location - now get RAW value
                    val_raw = mslp_hpa[i, j]
                    
                    # Check RAW value against threshold
                    if val_raw >= max_mslp_threshold:
                        continue
                    
                    minima.append((val_raw, lats[i], lons[j]))

        return minima
    
    def _plot_mslp(self, mslp_hpa, basin_mask, datetime_str, center_lat, center_lon, 
                center_val, show_marker=False, marker_label=None):
        """Plot MSLP field with optional crossing marker."""
        from IPython.display import display, clear_output
        
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(16, 10))
            # Lambert Conformal
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
        
        # PERIODIC BOUNDARY FIX: Extend field periodically, smooth, then extract
        n_pad = 20  # Pad with 20 pixels on each side
        
        # Wrap the field: append right edge to left, left edge to right
        left_pad = mslp_hpa[:, -n_pad:]
        right_pad = mslp_hpa[:, :n_pad]
        mslp_extended = np.concatenate([left_pad, mslp_hpa, right_pad], axis=1)
        
        # Smooth the extended field
        mslp_extended_smooth = gaussian_filter(mslp_extended, sigma=3.0)
        
        # Extract the central portion (original size)
        mslp_smooth = mslp_extended_smooth[:, n_pad:-n_pad]
        
        # Extended extent all the way to North Pole - show Iceland and Greenland
        self.ax.set_extent([-100, -10, 0, 85], crs=ccrs.PlateCarree())
        
        # Discrete levels and norm for discrete colorbar
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
        
        # ONLY plot star during actual crossings
        if show_marker and center_lat is not None and center_lon is not None:
            self.ax.plot(center_lon, center_lat, marker='*', markersize=24,
                    markeredgecolor='yellow', markeredgewidth=2.5, color='red',
                    transform=ccrs.PlateCarree(), zorder=10,
                    label=marker_label or 'Crossing')
        
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Title
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
    
    def _save_mslp_figure(self, crossing: InterfaceConfig, save_dir: Path):
        """Save MSLP figure for crossing."""
        if not hasattr(crossing, '_y_phys'):
            return
        
        # Use the exact MSLP field that was used during extraction
        if hasattr(crossing, '_mslp_field'):
            mslp_smooth = crossing._mslp_field
        else:
            # Fallback: extract and smooth
            mslp_channel_idx = 71
            mslp_pa = crossing._y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
            mslp_hpa = mslp_pa / 100.0
            mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
        basin_mask = self.get_basin_mask()
        
        lats = self.latlons.latitude.values
        lons_180 = np.where(self.latlons.longitude.values > 180, 
                           self.latlons.longitude.values - 360, 
                           self.latlons.longitude.values)
        
        # Use stored feature location
        if crossing.feature_location:
            feat_lat, feat_lon = crossing.feature_location
        else:
            mslp_masked = np.where(basin_mask, mslp_smooth, np.nan)
            min_idx = np.nanargmin(mslp_masked)
            lat_idx, lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
            feat_lat = lats[lat_idx]
            feat_lon = lons_180[lon_idx]
        
        # Crop to basin
        basin_rows, basin_cols = np.where(basin_mask)
        pad = 10
        row_min = max(0, basin_rows.min() - pad)
        row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
        col_min = max(0, basin_cols.min() - pad)
        col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
        
        mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
        lat_crop = lats[row_min:row_max]
        lon_crop = lons_180[col_min:col_max]
        
        # Create figure
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
        
        # Title
        datetime_str = None
        if hasattr(crossing, '_datetime_obj'):
            datetime_str = crossing._datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
        # Format coordinates to match axis labels (N/S for lat, E/W for lon)
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
        
        print(f"  → PNG: {fig_path.name}")
    
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
    
    def check_trajectory_status(self, mslp_value: float, 
                               current_interface: int,
                               mode: str = 'flux') -> Tuple[str, Optional[int]]:
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
    
    def run_ffs(self, initial_loader, n_flux_trials=20, n_shoot_trials=10):
        """Run full FFS algorithm."""
        print(f"\n{'='*70}")
        print()
        print(f"Interfaces: {self.interfaces}")
        print(f"Flux trials: {n_flux_trials}, Shoot trials: {n_shoot_trials}")
        print(f"{'='*70}\n")
        
        # Phase 0: Flux generation
        self.generate_flux(initial_loader, n_flux_trials)
        
        # Shooting phases
        for i in range(len(self.interfaces) - 1):
            self.shoot_from_interface(i, n_shoot_trials)
        
        # Create summary logger
        summary_logger = FFSLogger(self.logs_dir, rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        # Final results
        ffs_prob = self.flux_estimate * np.prod(self.transition_probs)
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Flux: {self.flux_estimate:.6f}/day")
        for i, p in enumerate(self.transition_probs):
            print(f"P(λ_{i}→λ_{i+1}): {p:.4f}")
        print(f"\nP_FFS: {ffs_prob:.2e}/day")
        print(f"P_direct: {self.direct_B_rate:.2e}/day")
        print(f"{'='*70}\n")
        
        summary_logger.log_final_results(
            self.flux_estimate,
            self.transition_probs,
            ffs_prob,
            self.direct_B_count,
            self.direct_B_rate
        )
        
        summary_logger.close()
        
        return ffs_prob
    
    def generate_flux(self, initial_loader, n_trials):
        """Generate flux at lambda_0."""
        # Create flux-specific logger
        logger = FFSLogger(self.logs_dir / 'flux', rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        print(f"PHASE 0: FLUX GENERATION (λ₀={self.interfaces[0]} hPa)\n")
        
        traj_count = 0
        worker_crossings = 0
        total_days = 0
        direct_B = 0
        
        while True:
            global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
            
            if global_count >= n_trials:
                print(f"✓ Target reached ({global_count}/{n_trials})")
                break
            
            traj_count += 1
            print(f"Traj {traj_count} (global={global_count}/{n_trials}): ", end='')
            
            loader = copy.deepcopy(initial_loader)
            result = self.rollout(loader, mode='flux')
            
            n_crossings = len(result['crossings'])
            worker_crossings += n_crossings
            
            if result['status'] == 'reached_B' and n_crossings == 0:
                direct_B += 1
            
            config_names = [c.config_name for c in result['crossings']]
            logger.log_flux_trajectory(traj_count, result, config_names)
            
            if n_crossings > 0:
                print(f"({n_crossings} crossings)")
            else:
                print("(no crossings)")
            
            total_days += len(result['mslp_trajectory']) * 6 / 24
        
        self.flux_estimate = worker_crossings / total_days if total_days > 0 else 0
        self.direct_B_count = direct_B
        self.direct_B_rate = direct_B / total_days if total_days > 0 else 0
        
        # VERIFY: Check PNG/PKL match
        final_pkl_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
        final_png_count = len(list(self.flux_dir.glob('lambda0_config_*.png')))
        if final_pkl_count != final_png_count:
            print(f"  ⚠ WARNING: {final_pkl_count} PKL files but {final_png_count} PNG files in flux/")
        
        print("\n✓ Flux generation complete")
        print(f"Worker λ₀ crossings: {worker_crossings}")
        print(f"Global configs saved: {final_pkl_count}")
        print(f"Φ₀ = {self.flux_estimate:.6f}/day (based on worker crossings)")
        print(f"Direct B: {self.direct_B_rate:.2e}/day\n")
        
        logger.close()
    
    def generate_flux_at_lambda0(self, loader, n_trials):
        """Alias for generate_flux - called by run_parallel_ffs.py."""
        return self.generate_flux(loader, n_trials)
    
    def shoot_from_interface(self, interface_idx, n_trials):
        """Shoot from interface."""
        # Create interface-specific logger - use NEXT interface number to match save dir
        logger = FFSLogger(self.logs_dir / str(interface_idx + 1), rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        print(f"\nSHOOTING from λ_{interface_idx} ({self.interfaces[interface_idx]} hPa)\n")
        
        # Load configs
        if interface_idx == 0:
            config_dir = self.flux_dir
            configs = list(config_dir.glob('lambda0_config_*.pkl'))
        else:
            config_dir = self.ic_base_dir / str(interface_idx)
            configs = list(config_dir.glob(f'lambda{interface_idx}_config_*.pkl'))
        
        loaded_configs = []
        for cfg_file in configs:
            with open(cfg_file, 'rb') as f:
                loaded_configs.append(pickle.load(f))
        
        print(f"Available configs: {len(loaded_configs)}")
        
        successes = 0
        failures = 0
        attempts = 0
        
        save_dir = self.ic_base_dir / str(interface_idx + 1)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        initial_count = len(list(save_dir.glob(f'lambda{interface_idx+1}_config_*.pkl')))
        
        while True:
            global_count = len(list(save_dir.glob(f'lambda{interface_idx+1}_config_*.pkl')))
            
            if global_count >= n_trials:
                print(f"✓ Target reached ({global_count}/{n_trials})")
                break

            # EARLY STOPPING: If we've tried many times with zero successes, stop
            if attempts >= self.max_attempts_without_success and successes == 0:
                print(f"\n⚠ EARLY STOP: {attempts} attempts with 0 successes")
                print(f"Interface λ_{interface_idx}→λ_{interface_idx+1} appears unreachable")
                logger.log_early_stop(interface_idx, attempts)
                break
            
            attempts += 1
            config = np.random.choice(loaded_configs)
            
            print(f"Attempt {attempts} (global={global_count}/{n_trials}): ", end='')
            
            # Shoot
            result = self.shoot_trajectory(config, interface_idx)
            
            child_config = None
            if result['crossings']:
                child_config = result['crossings'][0].config_name
            
            logger.log_shooting_attempt(
                interface_idx,
                attempts,
                config.config_name,
                result,
                child_config
            )
            
            # VERIFY: status=='success' must match having valid crossings
            if result['status'] == 'success':
                if len(result['crossings']) == 0:
                    print("ERROR: status='success' but no crossings saved!")
                    failures += 1
                else:
                    successes += 1
                    print("SUCCESS")
            else:
                if len(result['crossings']) > 0:
                    print(f"WARNING: status='{result['status']}' but {len(result['crossings'])} crossings saved")
                failures += 1
                print(f"{result['status']}")
        
        P = successes / attempts if attempts > 0 else 0
        self.transition_probs.append(P)
        
        final_count = len(list(save_dir.glob(f'lambda{interface_idx+1}_config_*.pkl')))
        png_count = len(list(save_dir.glob(f'lambda{interface_idx+1}_config_*.png')))
        configs_saved = final_count - initial_count
        
        # VERIFY: configs_saved should match successes
        if configs_saved != successes:
            print(f"  ⚠ WARNING: Mismatch! Logged {successes} successes but saved {configs_saved} PKL files")
        
        # VERIFY: PNG count should match PKL count
        if png_count != final_count:
            print(f"  ⚠ WARNING: {final_count} PKL files but {png_count} PNG files!")
        
        logger.log_interface_summary(
            interface_idx,
            successes,
            failures,
            attempts,
            P,
            configs_saved
        )
        
        print("\n✓ Shooting complete")
        print(f"Attempts: {attempts}")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")
        print(f"Configs saved: {configs_saved} (initial: {initial_count}, final: {final_count})")
        print(f"P(λ_{interface_idx}→λ_{interface_idx+1}) = {P:.4f}\n")
        
        logger.close()
        
        return P
    
    def shoot_trajectory(self, config: InterfaceConfig, interface_idx: int) -> Dict:
        """Shoot single trajectory from config."""
        # Create loader from config
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
    
    def rollout(self, loader, mode='flux', initial_state=None, start_interface=-1, parent_config=None):
        """Run trajectory rollout."""
        # Reset multi-storm tracking for flux mode
        if mode == 'flux':
            self.tracked_storms = {}
            self.next_storm_id = 0
        
        previous_location = None
        if parent_config and parent_config.feature_location:
            previous_location = parent_config.feature_location
        
        trajectory_mslp = []
        crossings = []
        current_interface = start_interface
        status = 'ongoing'
        saved_states = {}
        
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
                # y_with_mslp = self.calculate_mslp_wrapper(y_phys, batch)
                y_with_mslp = self.calculate_mslp_wrapper(y_phys, batch, simple_mslp=(mode=='flux'))
                
                mslp, feat_idx, location = self.extract_mslp(
                    y_with_mslp, batch, step, previous_location, mode
                )
                
                trajectory_mslp.append(mslp)
                
                # EXTRATROPICAL EARLY TERMINATION - SHOOT MODE ONLY
                if mode == 'shoot' and location:
                    storm_lat, storm_lon = location
                    
                    # Too far north
                    if storm_lat > 50.0:
                        print(f"  → FAILURE: Extratropical ({storm_lat:.1f}°N > 50°N)")
                        status = 'failure'
                        break  # EXIT IMMEDIATELY
                    
                    # Heading toward Europe
                    if storm_lon > -10.0:
                        print(f"  → FAILURE: Recurving toward Europe ({abs(storm_lon):.1f}°W)")
                        status = 'failure'
                        break  # EXIT IMMEDIATELY
                    
                    # North of 30°N must be over ocean
                    if storm_lat > 30.0:
                        lats = self.latlons.latitude.values
                        lons = np.where(
                            self.latlons.longitude.values > 180,
                            self.latlons.longitude.values - 360,
                            self.latlons.longitude.values
                        )
                        lat_idx = np.argmin(np.abs(lats - storm_lat))
                        lon_idx = np.argmin(np.abs(lons - storm_lon))
                        lsm_value = self.land_sea_mask[lat_idx, lon_idx]
                        
                        if lsm_value > 0.5:  # Over land
                            print(f"  → FAILURE: Over land ({storm_lat:.1f}°N, {abs(storm_lon):.1f}°W)")
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
                
                # Check all tracked storms for λ₀ crossings and dissipation
                if mode == 'flux':
                    lambda_0 = self.interfaces[0]
                    
                    for storm_id in list(self.tracked_storms.keys()):
                        storm = self.tracked_storms[storm_id]
                        
                        # Check for λ₀ crossing
                        if not storm['saved'] and storm['mslp'] < lambda_0:
                            storm_lat, storm_lon = storm['location']
                            
                            # EXTRATROPICAL CHECKS - don't save if extratropical
                            # RULE 1: Too far north (> 50°N)
                            if storm_lat > 50.0:
                                print(f"  → Storm {storm_id} extratropical ({storm_lat:.1f}°N), not saved")
                                storm['saved'] = True  # Mark as "saved" so we don't check again
                                continue
                            
                            # RULE 2: Heading to Europe (east of 10°W)
                            if storm_lon > -10.0:
                                print(f"  → Storm {storm_id} heading to Europe ({abs(storm_lon):.1f}°W), not saved")
                                storm['saved'] = True  # Mark as "saved" so we don't check again
                                continue
                            
                            # RULE 3: North of 30°N must be over ocean
                            if storm_lat > 30.0:
                                # Get land/sea mask value at storm location
                                lats = self.latlons.latitude.values
                                lons = np.where(
                                    self.latlons.longitude.values > 180,
                                    self.latlons.longitude.values - 360,
                                    self.latlons.longitude.values
                                )
                                lat_idx = np.argmin(np.abs(lats - storm_lat))
                                lon_idx = np.argmin(np.abs(lons - storm_lon))
                                lsm_value = self.land_sea_mask[lat_idx, lon_idx]
                                
                                if lsm_value > 0.5:  # Over land
                                    print(f"  → Storm {storm_id} over land ({storm_lat:.1f}°N, {abs(storm_lon):.1f}°W), not saved")
                                    storm['saved'] = True
                                    continue
                            
                            # Valid tropical genesis - save it
                            crossing = self._create_crossing(
                                saved_states[step], step, storm['mslp'], 0,
                                batch, None, storm['location'], None, y_with_mslp
                            )
                            crossings.append(crossing)
                            storm['saved'] = True
                            print(f"  → Storm {storm_id} crossed λ₀: {storm['mslp']:.1f} hPa")
                            
                            # SHOW CROSSING MARKER for flux mode
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
                                    storm_lat,
                                    storm_lon,
                                    storm['mslp'],
                                    show_marker=True,
                                    marker_label=f'λ₀ CROSSING (Storm {storm_id})'
                                )
                                plt.pause(1.0)  # Pause longer on crossings
                        
                        # Remove dissipated storms (returned to state A)
                        if storm['mslp'] > self.state_A_threshold:
                            print(f"  → Storm {storm_id} dissipated: {storm['mslp']:.1f} hPa")
                            del self.tracked_storms[storm_id]
                
                if traj_status == 'crossed_forward':
                    # Shoot mode: save crossing for next interface
                    if mode == 'shoot':
                        # Save crossing with exact location/MSLP from extract_mslp()
                        next_idx = current_interface + 1
                        crossing = self._create_crossing(
                            saved_states[step], step, mslp, next_idx,
                            batch, feat_idx, location, 
                            parent_config.config_name if parent_config else None,
                            y_with_mslp
                        )
                        crossings.append(crossing)
                        status = 'success'
                        
                        # SHOW CROSSING MARKER for shoot mode
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
                                location[0],
                                location[1],
                                mslp,
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
                    
                    # SHOW CROSSING MARKER for state B
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
                            location[0],
                            location[1],
                            mslp,
                            show_marker=True,
                            marker_label='STATE B REACHED!'
                        )
                        plt.pause(2.0)  # Pause even longer for state B
                    
                    break
                
                elif traj_status == 'returned_A':
                    status = 'failure'
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
        
        # Save crossings
        for crossing in crossings:
            self._save_crossing(crossing, mode)
        
        return {
            'status': status,
            'mslp_trajectory': trajectory_mslp,
            'crossings': crossings,
            'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None
        }
    
    def _create_crossing(self, state, step, mslp, interface_idx, batch, feat_idx, 
                        location, parent, y_phys_with_mslp, cps_params=None):  # ← ADD cps_params parameter
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
        
        # STORE DATA FOR PLOTTING - save exact MSLP field used for extraction
        crossing._feature_idx = feat_idx
        crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
        crossing._y_phys = y_phys_with_mslp.cpu().clone()
        
        # Store the smoothed MSLP field that was actually used to find the minimum
        mslp_channel_idx = 71
        mslp_raw = y_phys_with_mslp[0, mslp_channel_idx, 0].cpu().numpy() / 100.0
        crossing._mslp_field = gaussian_filter(mslp_raw, sigma=1.5)
        
        return crossing
    
    def _save_crossing(self, crossing, mode):
        """Save crossing to disk."""
        if crossing.interface_idx == -1:
            save_dir = self.stateB_dir
        elif mode == 'flux':
            save_dir = self.flux_dir
        else:
            # Shoot mode - use numbered interface directory
            save_dir = self.ic_base_dir / str(crossing.interface_idx)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # SAVE PNG FIRST
        try:
            self._save_mslp_figure(crossing, save_dir)
            png_saved = True
        except Exception as e:
            print(f"  ✗ ERROR saving PNG: {e}")
            png_saved = False
        
        # THEN SAVE PICKLE
        try:
            config_path = save_dir / f"{crossing.config_name}.pkl"
            with open(config_path, 'wb') as f:
                pickle.dump(crossing, f)
            pkl_saved = True
        except Exception as e:
            print(f"  ✗ ERROR saving PKL: {e}")
            pkl_saved = False
        
        if png_saved and pkl_saved:
            print(f"  → SAVED: {crossing.config_name} ({crossing.mslp_value:.1f} hPa)")
        else:
            print(f"  ✗ INCOMPLETE SAVE: {crossing.config_name} (PNG:{png_saved}, PKL:{pkl_saved})")


class HurricaneGenesisFFS_CPS:
    """CPS-enhanced FFS with proper track history management."""
    
    def __init__(self, base_ffs_instance):
        self._base = base_ffs_instance
        self.cps_tracker = CyclonePhaseTracker(self._base.latlons, radius_km=500)
        self.storm_track_history = {}
        
        print("✓ CPS tracker initialized (Hart 2003 formulation)")
        print("✓ Using VTL/VTU only (B disabled due to motion uncertainty)")
    
    def __getattr__(self, name):
        return getattr(self._base, name)
    
    def _compute_pressure_interp(self, y_phys, batch):
        """Compute pressure interpolation for CPS."""
        datetime_str = datetime.fromtimestamp(
            batch["datetime"][0].item()
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        darray_upper, darray_single = make_xarray(
            y_phys,
            datetime_str,
            self._base.latlons.latitude.values,
            self._base.latlons.longitude.values,
            self._base.config,
        )
        
        ds_merged = xr.merge([
            darray_upper.to_dataset(dim="vars"),
            darray_single.to_dataset(dim="vars")
        ])
        
        return full_state_pressure_interpolation(
            ds_merged,
            self._base.surface_geopotential,
            **self._base.config["predict"]["interp_pressure"]
        )

    def _extend_forecast_for_active_storms(self, last_datetime, last_state):
        """
        Extend forecast if there are still active tracked storms.
        Returns new loader or None if no extension needed.
        """
        # Check if any storms are still active (not dissipated, not saved)
        active_storms = [
            storm_id for storm_id, storm in self._base.tracked_storms.items()
            if storm['mslp'] < self._base.state_A_threshold
        ]
        
        if len(active_storms) == 0:
            return None
        
        print(f"  → {len(active_storms)} active storm(s) at forecast end - extending...")
        
        from datetime import timedelta
        from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
        from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
        
        # Extend from last datetime + 6 hours for another 15 days
        restart_time = last_datetime
        forecast_times = [[
            (restart_time + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
            (restart_time + timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        ]]
        
        extended_dataset = Predict_Dataset_Batcher(
            **self._base.dataset_params,
            fcst_datetime=forecast_times
        )
        extended_loader = BatchForecastLenDataLoader(extended_dataset)
        
        return extended_loader
    
    def _get_previous_position(self, storm_id):
        """Get previous position for motion calculation."""
        if storm_id not in self.storm_track_history:
            return None, None
        
        hist = self.storm_track_history[storm_id]
        if len(hist['lons']) < 1:
            return None, None
        
        return hist['lons'][-1], hist['lats'][-1]
    
    def _update_track_history(self, storm_id, lon, lat):
        """Update track history for a storm."""
        if storm_id not in self.storm_track_history:
            self.storm_track_history[storm_id] = {'lons': [], 'lats': []}
        
        self.storm_track_history[storm_id]['lons'].append(lon)
        self.storm_track_history[storm_id]['lats'].append(lat)
    
    def _check_cps(self, pressure_interp, location, mslp, storm_id, stage='genesis'):
        """Check if storm is extratropical using CPS."""
        lat, lon = location
        
        # Get previous position BEFORE updating
        prev_lon, prev_lat = self._get_previous_position(storm_id)
        
        # Update track history
        self._update_track_history(storm_id, lon, lat)
        
        # Compute CPS
        cps = self.cps_tracker.compute_CPS(
            pressure_interp, 
            lon, lat, mslp,
            prev_lon=prev_lon,
            prev_lat=prev_lat,
            stage=stage
        )
        
        return not cps['is_tropical'], cps

    def rollout(self, loader, mode='flux', initial_state=None, start_interface=-1, parent_config=None):
        """Run trajectory rollout with CPS checks only at decision points."""
        if mode == 'flux':
            self._base.tracked_storms = {}
            self._base.next_storm_id = 0
            self.storm_track_history = {}
        
        previous_location = None
        if parent_config and parent_config.feature_location:
            previous_location = parent_config.feature_location
        
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
                        x = initial_state.to(self._base.device).float()
                    else:
                        if "x_surf" in batch:
                            from credit.data import concat_and_reshape
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self._base.device).float()
                        else:
                            from credit.data import reshape_only
                            x = reshape_only(batch["x"]).to(self._base.device).float()
                
                if "x_forcing_static" in batch:
                    x_forcing = batch["x_forcing_static"].to(self._base.device).permute(0, 2, 1, 3, 4).float()
                    x = torch.cat((x, x_forcing), dim=1)
                
                y_pred = self._base.model(x, forecast_step=step - 1)
                y_phys = self._base.state_transformer.inverse_transform(y_pred.cpu())
                
                # ALWAYS use simple MSLP (fast!)
                y_with_mslp = self._base.calculate_mslp_wrapper(y_phys, batch, simple_mslp=True)
                pressure_interp = None  # Lazy computation - only when needed
                
                mslp, feat_idx, location = self._base.extract_mslp(
                    y_with_mslp, batch, step, previous_location, mode
                )
                
                trajectory_mslp.append(mslp)
                
                # === SHOOT MODE: GEOGRAPHIC BOUNDS CHECK ===
                if mode == 'shoot' and location:
                    storm_lat, storm_lon = location
                    
                    # Check if outside normal bounds - triggers CPS verification
                    outside_normal_bounds = False
                    
                    if storm_lat > 50.0:
                        print(f"  → Outside normal bounds: lat={storm_lat:.1f}°N > 50°N")
                        outside_normal_bounds = True
                    
                    elif storm_lon > -10.0:
                        print(f"  → Outside normal bounds: lon={abs(storm_lon):.1f}°W (too far east)")
                        outside_normal_bounds = True
                    
                    elif storm_lat > 30.0:
                        # Check land mask
                        lats = self._base.latlons.latitude.values
                        lons = np.where(
                            self._base.latlons.longitude.values > 180,
                            self._base.latlons.longitude.values - 360,
                            self._base.latlons.longitude.values
                        )
                        lat_idx = np.argmin(np.abs(lats - storm_lat))
                        lon_idx = np.argmin(np.abs(lons - storm_lon))
                        lsm_value = self._base.land_sea_mask[lat_idx, lon_idx]
                        
                        if lsm_value > 0.5:
                            print(f"  → Over land at ({storm_lat:.1f}°N, {abs(storm_lon):.1f}°W)")
                            outside_normal_bounds = True
                    
                    # IF OUTSIDE BOUNDS: Verify tropical structure with CPS
                    if outside_normal_bounds:
                        try:
                            if pressure_interp is None:
                                print(f"  → Computing pressure interp for CPS verification...")
                                pressure_interp = self._compute_pressure_interp(y_phys, batch)
                            
                            is_ET, cps = self._check_cps(
                                pressure_interp, 
                                location, 
                                mslp, 
                                storm_id=0, 
                                stage='mature'
                            )
                            
                            if is_ET:
                                print(f"  → FAILURE: Extratropical at boundary")
                                print(f"     Phase: {cps['phase']}, -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                status = 'extratropical'
                                failure_reason = 'extratropical_at_boundary'
                                failure_cps = cps
                                break
                            else:
                                print(f"  → Still TROPICAL: -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                print(f"     Continuing trajectory...")
                        
                        except Exception as e:
                            print(f"  ⚠ CPS check failed: {e}")
                            status = 'failure'
                            failure_reason = 'cps_check_failed'
                            break
                    
                    previous_location = location
                
                y_norm = self._base.state_transformer.transform_array(y_phys).to(self._base.device)
                
                if batch.get("y_diag") is not None:
                    varnum_diag = batch["y_diag"].shape[1]
                    saved_states[step] = y_norm[:, :-varnum_diag, ...].cpu().clone()
                else:
                    saved_states[step] = y_norm.cpu().clone()
                
                traj_status, interface_idx = self._base.check_trajectory_status(mslp, current_interface, mode)
                
                # === FLUX MODE: CPS CHECK AT λ₀ CROSSING ===
                if mode == 'flux':
                    lambda_0 = self._base.interfaces[0]
                    
                    for storm_id in list(self._base.tracked_storms.keys()):
                        storm = self._base.tracked_storms[storm_id]
                        
                        if not storm['saved'] and storm['mslp'] < lambda_0:
                            storm_lat, storm_lon = storm['location']
                            
                            # Verify local minimum
                            lats = self._base.latlons.latitude.values
                            lons = np.where(
                                self._base.latlons.longitude.values > 180,
                                self._base.latlons.longitude.values - 360,
                                self._base.latlons.longitude.values
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
                            
                            # CPS CHECK AT λ₀ CROSSING
                            try:
                                if pressure_interp is None:
                                    print(f"  → Computing pressure interp for λ₀ crossing...")
                                    pressure_interp = self._compute_pressure_interp(y_phys, batch)
                                
                                is_ET, cps = self._check_cps(
                                    pressure_interp, 
                                    storm['location'],
                                    storm['mslp'],
                                    storm_id,
                                    stage='genesis'
                                )
                                
                                # LOG
                                cps_log_path = self._base.flux_dir / 'cps_values.txt'
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
                            
                            # VALID TROPICAL GENESIS - SAVE WITH CPS DATA
                            crossing = self._base._create_crossing(
                                saved_states[step], step, storm['mslp'], 0,
                                batch, None, storm['location'], None, y_with_mslp,
                                cps_params=cps
                            )
                            crossings.append(crossing)
                            storm['saved'] = True
                            
                            if self._base.visualize_mslp:
                                datetime_str = None
                                if batch and "datetime" in batch:
                                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                                
                                mslp_channel_idx = 71
                                mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                                mslp_hpa = mslp_pa / 100.0
                                
                                self._base._plot_mslp(
                                    mslp_hpa, 
                                    self._base.get_basin_mask(),
                                    datetime_str,
                                    storm_lat, storm_lon, storm['mslp'],
                                    show_marker=True,
                                    marker_label=f'λ₀ CROSSING (Storm {storm_id})'
                                )
                                import matplotlib.pyplot as plt
                                plt.pause(1.0)
                        
                        if storm['mslp'] > self._base.state_A_threshold:
                            print(f"  → Storm {storm_id} dissipated: {storm['mslp']:.1f} hPa")
                            del self._base.tracked_storms[storm_id]
                
                # === SHOOT MODE: CPS CHECK AT INTERFACE CROSSING ===
                if traj_status == 'crossed_forward':
                    if mode == 'shoot':
                        next_idx = current_interface + 1
                        
                        # FULL CPS CHECK AT CROSSING
                        try:
                            if pressure_interp is None:
                                print(f"  → Computing pressure interp for λ{next_idx} crossing...")
                                pressure_interp = self._compute_pressure_interp(y_phys, batch)
                            
                            is_ET, cps = self._check_cps(
                                pressure_interp, 
                                location, 
                                mslp, 
                                storm_id=0, 
                                stage='mature'
                            )
                            
                            # LOG
                            cps_log_path = self._base.ic_base_dir / str(next_idx) / 'cps_values.txt'
                            with open(cps_log_path, 'a+') as f:
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
                                print(f"  → FAILURE: Extratropical at crossing")
                                print(f"     Phase: {cps['phase']}, -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                                status = 'extratropical'
                                failure_reason = 'extratropical_at_crossing'
                                failure_cps = cps
                                break
                            
                            print(f"  → SUCCESS: Tropical at λ{next_idx} crossing")
                            
                            # SAVE WITH CPS DATA
                            crossing = self._base._create_crossing(
                                saved_states[step], step, mslp, next_idx,
                                batch, feat_idx, location, 
                                parent_config.config_name if parent_config else None,
                                y_with_mslp,
                                cps_params=cps
                            )
                            crossings.append(crossing)
                            status = 'success'
                            
                            if self._base.visualize_mslp and location:
                                datetime_str = None
                                if batch and "datetime" in batch:
                                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                                
                                mslp_channel_idx = 71
                                mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                                mslp_hpa = mslp_pa / 100.0
                                
                                self._base._plot_mslp(
                                    mslp_hpa, 
                                    self._base.get_basin_mask(),
                                    datetime_str,
                                    location[0], location[1], mslp,
                                    show_marker=True,
                                    marker_label=f'λ{next_idx} CROSSING'
                                )
                                import matplotlib.pyplot as plt
                                plt.pause(1.5)
                            
                            break
                        
                        except Exception as e:
                            print(f"  ⚠ CPS check failed at crossing: {e}")
                            status = 'failure'
                            failure_reason = 'cps_check_failed'
                            break
                    
                    if mode == 'flux':
                        current_interface = interface_idx
                
                elif traj_status == 'reached_B':
                    crossing = self._base._create_crossing(
                        saved_states[step], step, mslp, -1,
                        batch, feat_idx, location,
                        parent_config.config_name if parent_config else None,
                        y_with_mslp
                    )
                    crossings.append(crossing)
                    status = 'reached_B'
                    
                    if self._base.visualize_mslp and location:
                        datetime_str = None
                        if batch and "datetime" in batch:
                            dt = datetime.fromtimestamp(batch["datetime"][0].item())
                            datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                        
                        mslp_channel_idx = 71
                        mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
                        mslp_hpa = mslp_pa / 100.0
                        
                        self._base._plot_mslp(
                            mslp_hpa, 
                            self._base.get_basin_mask(),
                            datetime_str,
                            location[0], location[1], mslp,
                            show_marker=True,
                            marker_label='STATE B REACHED!'
                        )
                        import matplotlib.pyplot as plt
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
                    # FLUX MODE: Check for active storms that need extended forecast
                    if mode == 'flux':
                        active_storms = [
                            storm_id for storm_id, storm in self._base.tracked_storms.items()
                            if storm['mslp'] < self._base.state_A_threshold
                        ]
                        
                        if len(active_storms) > 0:
                            # Get last datetime and state for extension
                            last_datetime = datetime.fromtimestamp(batch["datetime"][0].item())
                            last_state = x  # Current state tensor
                            
                            extended_loader = self._extend_forecast_for_active_storms(
                                last_datetime, 
                                last_state
                            )
                            
                            if extended_loader is not None:
                                print(f"  → Extending forecast from {last_datetime.strftime('%Y-%m-%d %H:%M')}")
                                
                                # Continue rollout with extended loader
                                for ext_batch in extended_loader:
                                    ext_step = ext_batch["forecast_step"].item()
                                    
                                    if "x_forcing_static" in ext_batch:
                                        x_forcing = ext_batch["x_forcing_static"].to(self._base.device).permute(0, 2, 1, 3, 4).float()
                                        x_with_forcing = torch.cat((x, x_forcing), dim=1)
                                    else:
                                        x_with_forcing = x
                                    
                                    y_pred = self._base.model(x_with_forcing, forecast_step=ext_step - 1)
                                    y_phys = self._base.state_transformer.inverse_transform(y_pred.cpu())
                                    y_with_mslp = self._base.calculate_mslp_wrapper(y_phys, ext_batch, simple_mslp=True)
                                    
                                    mslp, _, location = self._base.extract_mslp(
                                        y_with_mslp, ext_batch, ext_step, None, mode
                                    )
                                    
                                    trajectory_mslp.append(mslp)
                                    
                                    y_norm = self._base.state_transformer.transform_array(y_phys).to(self._base.device)
                                    
                                    # Check for state B crossings
                                    if mslp < self._base.state_B_threshold:
                                        for storm_id in list(self._base.tracked_storms.keys()):
                                            storm = self._base.tracked_storms[storm_id]
                                            if storm['mslp'] < self._base.state_B_threshold:
                                                print(f"  → Storm {storm_id} reached STATE B during extension!")
                                                # This is a direct B formation (crossed λ₀ then reached B)
                                                # Don't save as crossing, but mark for direct_B_count
                                                del self._base.tracked_storms[storm_id]
                                    
                                    # Check for dissipation
                                    for storm_id in list(self._base.tracked_storms.keys()):
                                        storm = self._base.tracked_storms[storm_id]
                                        if storm['mslp'] > self._base.state_A_threshold:
                                            print(f"  → Storm {storm_id} dissipated during extension: {storm['mslp']:.1f} hPa")
                                            del self._base.tracked_storms[storm_id]
                                    
                                    # Check if all storms resolved
                                    if len(self._base.tracked_storms) == 0:
                                        print(f"  → All storms resolved - ending extension")
                                        break
                                    
                                    # Update x for next step
                                    if ext_batch.get("y_diag") is not None:
                                        varnum_diag = ext_batch["y_diag"].shape[1]
                                        x = y_norm[:, :-varnum_diag, ...].detach()
                                    else:
                                        x = y_norm.detach()
                                    
                                    if ext_batch.get("stop_forecast", torch.tensor(False)).item():
                                        print(f"  → Extended forecast ended")
                                        break
                    
                    if status == 'ongoing':
                        status = 'completed' if mode == 'flux' else 'failure'
                    break

    
        for crossing in crossings:
            self._base._save_crossing(crossing, mode)
        
        return {
            'status': status,
            'mslp_trajectory': trajectory_mslp,
            'crossings': crossings,
            'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None,
            'failure_reason': failure_reason,
            'failure_cps': failure_cps
        }
    
    # def rollout(self, loader, mode='flux', initial_state=None, start_interface=-1, parent_config=None):
    #     """Run trajectory rollout with CPS-based filtering."""
    #     if mode == 'flux':
    #         self._base.tracked_storms = {}
    #         self._base.next_storm_id = 0
    #         self.storm_track_history = {}
        
    #     previous_location = None
    #     if parent_config and parent_config.feature_location:
    #         previous_location = parent_config.feature_location
        
    #     trajectory_mslp = []
    #     crossings = []
    #     current_interface = start_interface
    #     status = 'ongoing'
    #     saved_states = {}
        
    #     with torch.no_grad():
    #         for batch in loader:
    #             step = batch["forecast_step"].item()
                
    #             if step == 1:
    #                 if initial_state is not None:
    #                     x = initial_state.to(self._base.device).float()
    #                 else:
    #                     if "x_surf" in batch:
    #                         from credit.data import concat_and_reshape
    #                         x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self._base.device).float()
    #                     else:
    #                         from credit.data import reshape_only
    #                         x = reshape_only(batch["x"]).to(self._base.device).float()
                
    #             if "x_forcing_static" in batch:
    #                 x_forcing = batch["x_forcing_static"].to(self._base.device).permute(0, 2, 1, 3, 4).float()
    #                 x = torch.cat((x, x_forcing), dim=1)
                
    #             y_pred = self._base.model(x, forecast_step=step - 1)
    #             y_phys = self._base.state_transformer.inverse_transform(y_pred.cpu())
                
    #             y_with_mslp = self._base.calculate_mslp_wrapper(y_phys, batch, simple_mslp=True)
    #             pressure_interp = None
                
    #             mslp, feat_idx, location = self._base.extract_mslp(
    #                 y_with_mslp, batch, step, previous_location, mode
    #             )
                
    #             trajectory_mslp.append(mslp)
                
    #             # CPS CHECK - SHOOT MODE
    #             if mode == 'shoot' and location is not None:
    #                 lat, lon = location
    #                 try:
    #                     if pressure_interp is None:
    #                         pressure_interp = self._compute_pressure_interp(y_phys, batch)
                        
    #                     is_ET, cps = self._check_cps(pressure_interp, location, mslp, storm_id=0, stage='mature')

    #                     cps_log_path = self._base.ic_base_dir / str(start_interface + 1) / 'cps_values.txt'
    #                     with open(cps_log_path, 'a+') as f:
    #                         timestamp = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
    #                         f.write(f"\n{'='*60}\n")
    #                         f.write(f"Shoot mode at step {step}\n")
    #                         f.write(f"Time: {timestamp}\n")
    #                         f.write(f"Location: ({lat:.1f}°N, {abs(lon):.1f}°W)\n")
    #                         f.write(f"MSLP: {mslp:.1f} hPa\n")
    #                         f.write(f"Motion: {cps['motion_dir']:.1f}°\n")
    #                         f.write(f"B: {cps['B']:.1f} m (threshold: {cps['threshold_B']:.1f} m)\n")
    #                         f.write(f"-VT^L: {cps['VTL']:.1f} m\n")
    #                         f.write(f"-VT^U: {cps['VTU']:.1f} m\n")
    #                         f.write(f"Phase: {cps['phase']}\n")
    #                         f.write(f"Decision: {'REJECT (extratropical)' if is_ET else 'ACCEPT (tropical)'}\n")
                        
    #                     if is_ET:
    #                         print(f"  → EXTRATROPICAL: {cps['phase']} (heading {cps['motion_dir']:.0f}°)")
    #                         print(f"     -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
    #                         print(f"     Location: ({lat:.1f}°N, {abs(lon):.1f}°W), MSLP={mslp:.1f} hPa")
    #                         status = 'extratropical'
    #                         break
                    
    #                 except Exception as e:
    #                     print(f"  ⚠ CPS check failed: {e}")
    #                     print(f"  → REJECTING - cannot verify tropical structure")
    #                     status = 'extratropical'
    #                     break
                    
    #                 previous_location = location
                
    #             y_norm = self._base.state_transformer.transform_array(y_phys).to(self._base.device)
                
    #             if batch.get("y_diag") is not None:
    #                 varnum_diag = batch["y_diag"].shape[1]
    #                 saved_states[step] = y_norm[:, :-varnum_diag, ...].cpu().clone()
    #             else:
    #                 saved_states[step] = y_norm.cpu().clone()
                
    #             traj_status, interface_idx = self._base.check_trajectory_status(mslp, current_interface, mode)
                
    #             # CPS CHECK - FLUX MODE
    #             if mode == 'flux':
    #                 lambda_0 = self._base.interfaces[0]
                    
    #                 for storm_id in list(self._base.tracked_storms.keys()):
    #                     storm = self._base.tracked_storms[storm_id]
                        
    #                     if not storm['saved'] and storm['mslp'] < lambda_0:
    #                         storm_lat, storm_lon = storm['location']
                            
    #                         # Verify local minimum
    #                         lats = self._base.latlons.latitude.values
    #                         lons = np.where(
    #                             self._base.latlons.longitude.values > 180,
    #                             self._base.latlons.longitude.values - 360,
    #                             self._base.latlons.longitude.values
    #                         )
    #                         lat_idx = np.argmin(np.abs(lats - storm_lat))
    #                         lon_idx = np.argmin(np.abs(lons - storm_lon))
                            
    #                         mslp_channel_idx = 71
    #                         mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
    #                         mslp_hpa = mslp_pa / 100.0
                            
    #                         i_min = max(0, lat_idx - 1)
    #                         i_max = min(mslp_hpa.shape[0], lat_idx + 2)
    #                         j_min = max(0, lon_idx - 1)
    #                         j_max = min(mslp_hpa.shape[1], lon_idx + 2)
                            
    #                         nbhd = mslp_hpa[i_min:i_max, j_min:j_max]
    #                         center_val = mslp_hpa[lat_idx, lon_idx]
                            
    #                         if not np.all(center_val <= nbhd):
    #                             print(f"  → Storm {storm_id} location has no local minimum - REJECTING")
    #                             storm['saved'] = True
    #                             continue
                            
    #                         try:
    #                             if pressure_interp is None:
    #                                 print(f"  → Computing pressure interp for CPS check...")
    #                                 pressure_interp = self._compute_pressure_interp(y_phys, batch)
                                
    #                             is_ET, cps = self._check_cps(
    #                                 pressure_interp, 
    #                                 storm['location'],
    #                                 storm['mslp'],
    #                                 storm_id,
    #                                 stage='genesis'
    #                             )
    
    #                             # LOG
    #                             cps_log_path = self._base.flux_dir / 'cps_values.txt'
    #                             with open(cps_log_path, 'a') as f:
    #                                 timestamp = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
    #                                 f.write(f"\n{'='*60}\n")
    #                                 f.write(f"Storm {storm_id} at step {step}\n")
    #                                 f.write(f"Time: {timestamp}\n")
    #                                 f.write(f"Location: ({storm_lat:.1f}°N, {abs(storm_lon):.1f}°W)\n")
    #                                 f.write(f"MSLP: {storm['mslp']:.1f} hPa\n")
    #                                 f.write(f"Motion: {cps['motion_dir']:.1f}°\n")
    #                                 f.write(f"B: {cps['B']:.1f} m\n")
    #                                 f.write(f"-VT^L: {cps['VTL']:.1f} m\n")
    #                                 f.write(f"-VT^U: {cps['VTU']:.1f} m\n")
    #                                 f.write(f"Phase: {cps['phase']}\n")
    #                                 f.write(f"Decision: {'REJECT (extratropical)' if is_ET else 'ACCEPT (tropical)'}\n")
                                
    #                             if is_ET:
    #                                 print(f"  → Storm {storm_id} {cps['phase']}: -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
    #                                 print(f"     NOT SAVED - extratropical system")
    #                                 storm['saved'] = True
    #                                 continue
                                
    #                             print(f"  → Storm {storm_id} TROPICAL: -VT^L={cps['VTL']:.1f}m, -VT^U={cps['VTU']:.1f}m")
                            
    #                         except Exception as e:
    #                             print(f"  ⚠ CPS check failed for storm {storm_id}: {e}")
    #                             print(f"  → REJECTING")
    #                             storm['saved'] = True
    #                             continue
                            
    #                         crossing = self._base._create_crossing(
    #                             saved_states[step], step, storm['mslp'], 0,
    #                             batch, None, storm['location'], None, y_with_mslp
    #                         )
    #                         crossings.append(crossing)
    #                         storm['saved'] = True
    #                         print(f"  → Storm {storm_id} crossed λ₀: {storm['mslp']:.1f} hPa")
                            
    #                         if self._base.visualize_mslp:
    #                             datetime_str = None
    #                             if batch and "datetime" in batch:
    #                                 dt = datetime.fromtimestamp(batch["datetime"][0].item())
    #                                 datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                                
    #                             mslp_channel_idx = 71
    #                             mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
    #                             mslp_hpa = mslp_pa / 100.0
                                
    #                             self._base._plot_mslp(
    #                                 mslp_hpa, 
    #                                 self._base.get_basin_mask(),
    #                                 datetime_str,
    #                                 storm_lat, storm_lon, storm['mslp'],
    #                                 show_marker=True,
    #                                 marker_label=f'λ₀ CROSSING (Storm {storm_id})'
    #                             )
    #                             import matplotlib.pyplot as plt
    #                             plt.pause(1.0)
                        
    #                     if storm['mslp'] > self._base.state_A_threshold:
    #                         print(f"  → Storm {storm_id} dissipated: {storm['mslp']:.1f} hPa")
    #                         del self._base.tracked_storms[storm_id]
                
    #             if traj_status == 'crossed_forward':
    #                 if mode == 'shoot':
    #                     next_idx = current_interface + 1
    #                     crossing = self._base._create_crossing(
    #                         saved_states[step], step, mslp, next_idx,
    #                         batch, feat_idx, location, 
    #                         parent_config.config_name if parent_config else None,
    #                         y_with_mslp
    #                     )
    #                     crossings.append(crossing)
    #                     status = 'success'
                        
    #                     if self._base.visualize_mslp and location:
    #                         datetime_str = None
    #                         if batch and "datetime" in batch:
    #                             dt = datetime.fromtimestamp(batch["datetime"][0].item())
    #                             datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                            
    #                         mslp_channel_idx = 71
    #                         mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
    #                         mslp_hpa = mslp_pa / 100.0
                            
    #                         self._base._plot_mslp(
    #                             mslp_hpa, 
    #                             self._base.get_basin_mask(),
    #                             datetime_str,
    #                             location[0], location[1], mslp,
    #                             show_marker=True,
    #                             marker_label=f'λ{next_idx} CROSSING'
    #                         )
    #                         import matplotlib.pyplot as plt
    #                         plt.pause(1.5)
                        
    #                     break
                    
    #                 if mode == 'flux':
    #                     current_interface = interface_idx
                
    #             elif traj_status == 'reached_B':
    #                 crossing = self._base._create_crossing(
    #                     saved_states[step], step, mslp, -1,
    #                     batch, feat_idx, location,
    #                     parent_config.config_name if parent_config else None,
    #                     y_with_mslp
    #                 )
    #                 crossings.append(crossing)
    #                 status = 'reached_B'
                    
    #                 if self._base.visualize_mslp and location:
    #                     datetime_str = None
    #                     if batch and "datetime" in batch:
    #                         dt = datetime.fromtimestamp(batch["datetime"][0].item())
    #                         datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                        
    #                     mslp_channel_idx = 71
    #                     mslp_pa = y_with_mslp[0, mslp_channel_idx, 0].cpu().numpy()
    #                     mslp_hpa = mslp_pa / 100.0
                        
    #                     self._base._plot_mslp(
    #                         mslp_hpa, 
    #                         self._base.get_basin_mask(),
    #                         datetime_str,
    #                         location[0], location[1], mslp,
    #                         show_marker=True,
    #                         marker_label='STATE B REACHED!'
    #                     )
    #                     import matplotlib.pyplot as plt
    #                     plt.pause(2.0)
                    
    #                 break
                
    #             elif traj_status == 'returned_A':
    #                 status = 'failure'
    #                 break
                
    #             elif traj_status == 'returned_backward':
    #                 current_interface = interface_idx if interface_idx is not None else -1
                
    #             if batch.get("y_diag") is not None:
    #                 varnum_diag = batch["y_diag"].shape[1]
    #                 x = y_norm[:, :-varnum_diag, ...].detach()
    #             else:
    #                 x = y_norm.detach()
                
    #             if batch.get("stop_forecast", torch.tensor(False)).item():
    #                 if status == 'ongoing':
    #                     status = 'completed' if mode == 'flux' else 'failure'
    #                 break
        
    #     for crossing in crossings:
    #         self._base._save_crossing(crossing, mode)
        
    #     return {
    #         'status': status,
    #         'mslp_trajectory': trajectory_mslp,
    #         'crossings': crossings,
    #         'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None
    #     }

    def run_ffs(self, initial_loader, n_flux_trials=20, n_shoot_trials=10):
        """Override run_ffs to use CPS rollout."""
        print(f"\n{'='*70}")
        print(f"Running FFS with CPS-enhanced flux generation")
        print(f"Interfaces: {self._base.interfaces}")
        print(f"Flux trials: {n_flux_trials}, Shoot trials: {n_shoot_trials}")
        print(f"{'='*70}\n")
        
        self.generate_flux(initial_loader, n_flux_trials)
        
        for i in range(len(self._base.interfaces) - 1):
            original = self._base.shoot_trajectory
            self._base.shoot_trajectory = lambda c, idx: self.shoot_trajectory(c, idx)
            self._base.shoot_from_interface(i, n_shoot_trials)
            self._base.shoot_trajectory = original
        
        from tails.ffs_logger import FFSLogger
        summary_logger = FFSLogger(
            self._base.logs_dir, 
            rank=self._base.rank, 
            world_size=self._base.world_size, 
            worker_id=self._base.worker_id, 
            ic_dirname=None
        )
        
        ffs_prob = self._base.flux_estimate * np.prod(self._base.transition_probs)
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Flux: {self._base.flux_estimate:.6f}/day")
        for i, p in enumerate(self._base.transition_probs):
            print(f"P(λ_{i}→λ_{i+1}): {p:.4f}")
        print(f"\nP_FFS: {ffs_prob:.2e}/day")
        print(f"P_direct: {self._base.direct_B_rate:.2e}/day")
        print(f"{'='*70}\n")
        
        summary_logger.log_final_results(
            self._base.flux_estimate,
            self._base.transition_probs,
            ffs_prob,
            self._base.direct_B_count,
            self._base.direct_B_rate
        )
        
        summary_logger.close()
        return ffs_prob
    
    def generate_flux(self, initial_loader, n_trials):
        """Override generate_flux to use CPS rollout."""
        import copy
        from tails.ffs_logger import FFSLogger
        
        logger = FFSLogger(
            self._base.logs_dir / 'flux', 
            rank=self._base.rank, 
            world_size=self._base.world_size, 
            worker_id=self._base.worker_id, 
            ic_dirname=None
        )
        
        print(f"PHASE 0: FLUX GENERATION (λ₀={self._base.interfaces[0]} hPa)")
        print("✓ Using CPS-based extratropical filtering\n")
        
        traj_count = 0
        worker_crossings = 0
        total_days = 0
        direct_B = 0
        
        while True:
            global_count = len(list(self._base.flux_dir.glob('lambda0_config_*.pkl')))
            
            if global_count >= n_trials:
                print(f"✓ Target reached ({global_count}/{n_trials})")
                break
            
            traj_count += 1
            print(f"Traj {traj_count} (global={global_count}/{n_trials}): ", end='')
            
            loader = copy.deepcopy(initial_loader)
            result = self.rollout(loader, mode='flux')
            
            n_crossings = len(result['crossings'])
            worker_crossings += n_crossings
            
            if result['status'] == 'reached_B' and n_crossings == 0:
                direct_B += 1
            
            config_names = [c.config_name for c in result['crossings']]
            logger.log_flux_trajectory(traj_count, result, config_names)
            
            if n_crossings > 0:
                print(f"({n_crossings} crossings)")
            else:
                print("(no crossings)")
            
            total_days += len(result['mslp_trajectory']) * 6 / 24
        
        self._base.flux_estimate = worker_crossings / total_days if total_days > 0 else 0
        self._base.direct_B_count = direct_B
        self._base.direct_B_rate = direct_B / total_days if total_days > 0 else 0
        
        final_pkl_count = len(list(self._base.flux_dir.glob('lambda0_config_*.pkl')))
        final_png_count = len(list(self._base.flux_dir.glob('lambda0_config_*.png')))
        
        if final_pkl_count != final_png_count:
            print(f"  ⚠ WARNING: {final_pkl_count} PKL but {final_png_count} PNG files")
        
        print("\n✓ Flux generation complete")
        print(f"Worker λ₀ crossings: {worker_crossings}")
        print(f"Global configs saved: {final_pkl_count}")
        print(f"Φ₀ = {self._base.flux_estimate:.6f}/day")
        print(f"Direct B: {self._base.direct_B_rate:.2e}/day\n")
        
        logger.close()

    def shoot_trajectory(self, config, interface_idx):
        """Override to use CPS rollout with track initialization."""
        from datetime import timedelta
        from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
        from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
        
        # CRITICAL: Initialize track history from parent config
        if config.feature_location:
            storm_id = 0
            self.storm_track_history[storm_id] = {
                'lons': [config.feature_location[1]],  # (lat, lon) -> lon
                'lats': [config.feature_location[0]]   # lat
            }
            print(f"  → Initialized shoot track at ({config.feature_location[0]:.1f}°N, {abs(config.feature_location[1]):.1f}°W)")
        
        restart_time = config.restart_datetime
        forecast_times = [[
            (restart_time + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
            (restart_time + timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
        ]]
        
        restart_dataset = Predict_Dataset_Batcher(
            **self._base.dataset_params,
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