import torch
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import copy

# Core Python
import os
import yaml
import random
import string

# Visualization
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from credit.output import make_xarray
from scipy.ndimage import gaussian_filter

# CREDIT framework
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.parser import credit_main_parser
from credit.datasets import setup_data_loading
from credit.models import load_model
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.rare_events.ffs_logger import FFSLogger

from credit.data import concat_and_reshape, reshape_only
from credit.interp import full_state_pressure_interpolation
from pathlib import Path
import pickle
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


class HurricaneGenesisFFS:
    """
    Forward Flux Sampling for hurricane genesis in the Atlantic.
    
    Directory structure:
    output_dir/
    └── IC_TIME/
        ├── logs/                    # Log files
        ├── flux/                    # λ₀ crossings
        ├── 1/                       # λ₁ crossings
        ├── 2/                       # λ₂ crossings
        ├── stateB/                  # State B arrivals
        └── failed_trajectories/     # Failed tracking diagnostics
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
                 interfaces=[1000, 997, 994, 991, 988, 985],
                 decorrelation_interface=None,
                 worker_id=0,
                 rank=0,
                 world_size=1,
                 ic_dirname=None):
        
        self.model = model
        self.state_transformer = state_transformer
        self.config = config
        self.initial_dataset = initial_dataset
        self.worker_id = worker_id
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
        # IC-specific base directory
        self.output_dir = Path(output_dir)
        if ic_dirname:
            self.ic_base_dir = self.output_dir / ic_dirname
        else:
            self.ic_base_dir = self.output_dir
        
        # Setup ALL directories
        self.logs_dir = self.ic_base_dir / 'logs'
        self.flux_dir = self.ic_base_dir / 'flux'
        self.stateB_dir = self.ic_base_dir / 'stateB'
        self.failed_dir = self.ic_base_dir / 'failed_trajectories'
        
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.flux_dir.mkdir(parents=True, exist_ok=True)
        self.stateB_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_params = dataset_params
        
        # Load static data
        self.latlons = xr.open_dataset(config["loss"]["latitude_weights"]).load()
        with xr.open_dataset('/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc') as df:
            self.surface_geopotential = df["Z_GDS4_SFC"].values
            self.land_sea_mask = df["LSM"].values
        
        # Basin definition
        self.basin = {
            'lat_min': 10.0,
            'lat_max': 40.0,
            'lon_min': -100.0,
            'lon_max': -20.0
        }
        
        # Thresholds
        if state_B not in interfaces:
            interfaces.append(state_B)
        self.interfaces = interfaces
        self.state_A_threshold = state_A
        self.state_B_threshold = state_B
        self.decorrelation_interface = decorrelation_interface
        
        # Storage for interface crossings
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
    
    def _get_shoot_dir(self, interface_idx: int) -> Path:
        """Get shooting directory for specific interface."""
        shoot_dir = self.ic_base_dir / str(interface_idx)
        # DON'T CREATE IT HERE
        return shoot_dir
    
    def get_basin_mask(self) -> np.ndarray:
        """Create spatial mask for Atlantic basin."""
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons = np.where(lons > 180, lons - 360, lons)
        
        lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
        lon_mask = (lons >= self.basin['lon_min']) & (lons <= self.basin['lon_max'])
        
        basin_mask = lat_mask[:, None] & lon_mask[None, :]
        return basin_mask

    def extract_mslp(self, y_phys: torch.Tensor, batch: Dict = None, 
                     forecast_step: int = None, 
                     parent_location: Optional[Tuple[float, float]] = None) -> Tuple:
        """Extract minimum MSLP in Atlantic basin."""
        try:
            import tobac
            import iris
        except ImportError:
            return self._extract_mslp_fallback(y_phys, batch), None, None
        
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
        lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
        basin_mask = lat_mask[:, None] & lon_mask[None, :]
        
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
        lons_tobac = lons_180.copy()
        mslp_tobac = mslp_basin.copy()
        
        if not np.all(np.diff(lons_tobac) > 0):
            sort_idx = np.argsort(lons_tobac)
            lons_tobac = lons_tobac[sort_idx]
            mslp_tobac = mslp_tobac[:, sort_idx]
        
        try:
            lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
            lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
            time_coord = iris.coords.DimCoord([0], standard_name='time', units='hours since 2024-01-01 00:00:00')
            
            cube = iris.cube.Cube(
                mslp_tobac[np.newaxis, :, :],
                standard_name='air_pressure_at_mean_sea_level',
                units='hPa',
                dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
            )
            
            basin_min = np.nanmin(mslp_tobac)
            basin_max = np.nanmax(mslp_tobac)
            thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
            thresholds = sorted(thresholds, reverse=False)
        
            features = tobac.feature_detection_multithreshold(
                field_in=cube,
                dxy=111000,
                threshold=thresholds,
                target='minimum',
                position_threshold='weighted_diff',
                sigma_threshold=1.5,
                n_min_threshold=3
            )
            
            if features is not None and len(features) > 0:
                strongest = features.loc[features['threshold_value'].idxmin()]
                min_row_sorted = int(strongest['hdim_1'])
                min_col_sorted = int(strongest['hdim_2'])
                min_mslp = float(mslp_tobac[min_row_sorted, min_col_sorted])
                
                if np.isnan(min_mslp):
                    return self._extract_mslp_fallback(y_phys, batch), None, None
            else:
                return self._extract_mslp_fallback(y_phys, batch), None, None
        
        except Exception as e:
            print(f"\n⚠ tobac error: {e}, using fallback")
            return self._extract_mslp_fallback(y_phys, batch), None, None
        
        return min_mslp, None, None
    
    def _extract_mslp_fallback(self, y_phys: torch.Tensor, batch: Dict = None) -> float:
        """Fallback MSLP extraction using simple minimum."""
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        mslp_smooth_masked = np.where(basin_mask, mslp_smooth, np.nan)
        
        min_idx = np.nanargmin(mslp_smooth_masked)
        min_row, min_col = np.unravel_index(min_idx, mslp_smooth_masked.shape)
        min_mslp = float(mslp_hpa[min_row, min_col])

        return min_mslp
    
    def check_trajectory_status(self, mslp_value: float, 
                               current_interface: int,
                               mode: str = 'flux') -> Tuple[str, Optional[int]]:
        """Determine trajectory status during FFS."""
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
    
    def calculate_mslp_wrapper(self, y_pred_phys, batch):
        """Wrapper for MSLP calculation."""
        datetime_str = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
            
        darray_upper_air, darray_single_level = make_xarray(
            y_pred_phys,
            datetime_str,
            self.latlons.latitude.values,
            self.latlons.longitude.values,
            self.config,
        )
        
        ds_merged = xr.merge([
            darray_upper_air.to_dataset(dim="vars"),
            darray_single_level.to_dataset(dim="vars")
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
    
    def _compute_restart_datetime(self, batch):
        """Compute datetime for restarting from this forecast step."""
        original_ic_time = datetime.fromtimestamp(batch["datetime"][0].item())
        restart_time = original_ic_time + timedelta(hours=6)
        return restart_time

    def _generate_config_name(self, interface_idx: int, mode: str = 'flux') -> str:
        """Generate unique config name based on existing files."""
        random_suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
        
        if mode == 'stateB':
            existing = len(list(self.stateB_dir.glob('stateB_config_*.pkl')))
            return f"stateB_config_{existing+1:04d}_{random_suffix}"
        
        lambda_label = interface_idx
        
        if mode == 'flux':
            existing = len(list(self.flux_dir.glob(f'lambda{lambda_label}_config_*.pkl')))
            return f"lambda{lambda_label}_config_{existing+1:04d}_{random_suffix}"
        else:
            shoot_dir = self._get_shoot_dir(interface_idx)
            existing = len(list(shoot_dir.glob(f'lambda{lambda_label}_config_*.pkl')))
            return f"lambda{lambda_label}_config_{existing+1:04d}_{random_suffix}"
    
    def _save_config(self, config: InterfaceConfig, mode: str = 'flux'):
        """Save config to disk."""
        if mode == 'stateB':
            save_dir = self.stateB_dir
        elif mode == 'flux':
            save_dir = self.flux_dir
        else:
            save_dir = self.ic_base_dir / str(config.interface_idx)
            save_dir.mkdir(parents=True, exist_ok=True)  # CREATE HERE, not in _get_shoot_dir
        
        config_path = save_dir / f"{config.config_name}.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        return config_path
    
    def _save_mslp_figure(self, y_phys_with_mslp: torch.Tensor, 
                          config_name: str, save_dir: Path,
                          datetime_obj: Optional[datetime] = None,
                          feature_idx: Optional[int] = None,
                          location: Optional[Tuple[float, float]] = None):
        """Save MSLP figure at interface crossing."""
        mslp_channel_idx = 71
        mslp_pa = y_phys_with_mslp[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)

        if location is not None:
            min_lat, min_lon = location
            lat_idx = np.argmin(np.abs(lats - min_lat))
            lon_idx = np.argmin(np.abs(lons_180 - min_lon))
        else:
            mslp_masked = np.where(basin_mask, mslp_hpa, np.nan)
            min_idx = np.nanargmin(mslp_masked)
            lat_idx, lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
            min_lat = lats[lat_idx]
            min_lon = lons_180[lon_idx]

        basin_rows, basin_cols = np.where(basin_mask)
        if len(basin_rows) > 0:
            pad = 10
            row_min = max(0, basin_rows.min() - pad)
            row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
            col_min = max(0, basin_cols.min() - pad)
            col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
        else:
            row_min, row_max = 0, mslp_smooth.shape[0]
            col_min, col_max = 0, mslp_smooth.shape[1]

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
                        cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=1)
        
        contour_levels = np.arange(960, 1030, 8)
        cs = ax.contour(lon_crop, lat_crop, mslp_crop,
                        levels=contour_levels,
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=2)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)
        
        ax.plot(min_lon, min_lat, 'k*', markersize=20, 
                markeredgewidth=2, markeredgecolor='yellow',
                transform=ccrs.PlateCarree(), zorder=5)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)
        gl.top_labels = False
        gl.right_labels = False
        
        datetime_str = None
        if datetime_obj is not None:
            datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
        min_val = mslp_hpa[lat_idx, lon_idx]
        title = f'MSLP (hPa) - Basin Min: {min_val:.1f} hPa @ ({min_lat:.1f}°N, {min_lon:.1f}°W)'
        if datetime_str is not None:
            title = f'{datetime_str}\n{title}'
        title += f'\nConfig: {config_name}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.colorbar(pcm, ax=ax, label='MSLP (hPa)', shrink=0.8)
        plt.tight_layout()
        
        fig_path = save_dir / f"{config_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig_path

    def _save_failure_figure(self, y_phys_with_mslp: torch.Tensor, 
                            failure_name: str,
                            datetime_obj: Optional[datetime] = None,
                            feature_idx: Optional[int] = None,
                            location: Optional[Tuple[float, float]] = None,
                            parent_config: Optional[InterfaceConfig] = None,
                            diagnostic_info: Optional[Dict] = None):
        """Save 3-panel MSLP figure for failed trajectory with diagnostic info."""
        mslp_channel_idx = 71
        
        mslp_pa = y_phys_with_mslp[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)

        if location is not None:
            curr_lat, curr_lon = location
            curr_lat_idx = np.argmin(np.abs(lats - curr_lat))
            curr_lon_idx = np.argmin(np.abs(lons_180 - curr_lon))
        else:
            mslp_masked = np.where(basin_mask, mslp_hpa, np.nan)
            min_idx = np.nanargmin(mslp_masked)
            curr_lat_idx, curr_lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
            curr_lat = lats[curr_lat_idx]
            curr_lon = lons_180[curr_lon_idx]

        curr_mslp = mslp_hpa[curr_lat_idx, curr_lon_idx]

        parent_mslp_hpa = None
        parent_mslp_smooth = None
        parent_lat, parent_lon = None, None
        parent_mslp_val = None
        
        if parent_config and hasattr(parent_config, '_y_phys'):
            parent_mslp_pa = parent_config._y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
            parent_mslp_hpa = parent_mslp_pa / 100.0
            parent_mslp_smooth = gaussian_filter(parent_mslp_hpa, sigma=1.5)
            
            if parent_config.feature_location:
                parent_lat, parent_lon = parent_config.feature_location
                parent_lat_idx = np.argmin(np.abs(lats - parent_lat))
                parent_lon_idx = np.argmin(np.abs(lons_180 - parent_lon))
                parent_mslp_val = parent_mslp_hpa[parent_lat_idx, parent_lon_idx]

        basin_rows, basin_cols = np.where(basin_mask)
        if len(basin_rows) > 0:
            pad = 10
            row_min = max(0, basin_rows.min() - pad)
            row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
            col_min = max(0, basin_cols.min() - pad)
            col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
        else:
            row_min, row_max = 0, mslp_smooth.shape[0]
            col_min, col_max = 0, mslp_smooth.shape[1]

        mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
        lat_crop = lats[row_min:row_max]
        lon_crop = lons_180[col_min:col_max]
        
        if parent_mslp_smooth is not None:
            parent_mslp_crop = parent_mslp_smooth[row_min:row_max, col_min:col_max]

        fig = plt.figure(figsize=(20, 7))
        
        def plot_mslp_panel(ax, mslp_data, title, marker_configs):
            ax.set_extent([lon_crop.min(), lon_crop.max(), 
                        lat_crop.min(), lat_crop.max()], 
                        crs=ccrs.PlateCarree())

            levels = np.arange(960, 1030, 4)
            pcm = ax.contourf(lon_crop, lat_crop, mslp_data, levels=levels,
                            cmap='RdBu_r', extend='both', 
                            transform=ccrs.PlateCarree(), zorder=1)
            
            contour_levels = np.arange(960, 1030, 8)
            cs = ax.contour(lon_crop, lat_crop, mslp_data,
                        levels=contour_levels, colors='k', 
                        linewidths=0.5, transform=ccrs.PlateCarree(), zorder=2)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)
            
            for marker_cfg in marker_configs:
                lon, lat, marker, size, color, edgecolor, label = marker_cfg
                ax.plot(lon, lat, marker, markersize=size, 
                    color=color, markeredgewidth=2 if marker == '*' else 3,
                    markeredgecolor=edgecolor,
                    transform=ccrs.PlateCarree(), zorder=5, label=label)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, 
                            linestyle='--', zorder=2)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            
            return pcm
        
        if parent_mslp_smooth is not None and parent_lat is not None:
            ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
            
            parent_datetime_str = ""
            if hasattr(parent_config, '_datetime_obj'):
                parent_datetime_str = parent_config._datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
            
            title1 = f"PARENT CONFIG\n{parent_datetime_str}\n"
            title1 += f"MSLP: {parent_mslp_val:.1f} hPa @ ({parent_lat:.1f}°N, {parent_lon:.1f}°W)"
            
            markers1 = [(parent_lon, parent_lat, '*', 20, 'k', 'yellow', 'Parent')]
            _ = plot_mslp_panel(ax1, parent_mslp_crop, title1, markers1)
            ax1.legend(loc='upper right')
        else:
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.text(0.5, 0.5, 'Parent config\ndata not available', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax1.axis('off')
            _ = None
        
        ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        
        curr_datetime_str = ""
        if datetime_obj is not None:
            curr_datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
        title2 = f"CURRENT CONFIG (LOST TRACK)\n{curr_datetime_str}\n"
        title2 += f"MSLP: {curr_mslp:.1f} hPa @ ({curr_lat:.1f}°N, {curr_lon:.1f}°W)"
        
        markers2 = [(curr_lon, curr_lat, 'X', 20, 'r', 'r', 'Lost track')]
        pcm2 = plot_mslp_panel(ax2, mslp_crop, title2, markers2)
        ax2.legend(loc='upper right')
        
        ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        
        if diagnostic_info:
            lat_diff, lon_diff = diagnostic_info['distance']
            title3 = f"DISPLACEMENT\n{curr_datetime_str}\n"
            title3 += f"Distance: Δlat={lat_diff:.1f}°, Δlon={lon_diff:.1f}°"
        else:
            title3 = f"OVERLAY\n{curr_datetime_str}"
        
        markers3 = [(curr_lon, curr_lat, 'X', 20, 'r', 'r', 'Lost track')]
        if parent_lat is not None:
            markers3.append((parent_lon, parent_lat, '*', 20, 'k', 'yellow', 'Parent'))
        
        pcm3 = plot_mslp_panel(ax3, mslp_crop, title3, markers3)
        
        if parent_lat is not None:
            ax3.plot([parent_lon, curr_lon], [parent_lat, curr_lat], 
                    'r--', linewidth=2, transform=ccrs.PlateCarree(), zorder=4)
        
        ax3.legend(loc='upper right')
        
        fig.subplots_adjust(right=0.92, wspace=0.3)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        _ = fig.colorbar(pcm2 if pcm2 else pcm3, cax=cbar_ax, label='MSLP (hPa)')
        
        fig.suptitle(f'❌ LOST TRACK DIAGNOSTIC: {failure_name}', 
                    fontsize=14, fontweight='bold', color='red', y=0.98)
        
        fig_path = self.failed_dir / f"{failure_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return fig_path
    
    def _load_parent_config(self, parent_config_name):
        """Load parent config from disk."""
        # Extract lambda number from config name
        if parent_config_name.startswith('lambda'):
            parts = parent_config_name.split('_')
            lambda_str = parts[0].replace('lambda', '')
            interface_idx = int(lambda_str)
            
            # Try shooting directory first
            shoot_dir = self._get_shoot_dir(interface_idx)
            parent_path = shoot_dir / f'{parent_config_name}.pkl'
            if parent_path.exists():
                with open(parent_path, 'rb') as f:
                    return pickle.load(f)
        
        # Fall back to flux directory
        parent_path = self.flux_dir / f'{parent_config_name}.pkl'
        if parent_path.exists():
            with open(parent_path, 'rb') as f:
                return pickle.load(f)
        
        return None

    def _filter_crossings(self, crossings: List[InterfaceConfig], 
                         mode: str, 
                         ic_has_preexisting: bool,
                         parent_config_name: Optional[str]) -> List[InterfaceConfig]:
        """Filter crossings based on mode and tracking criteria. Override in subclass."""
        if mode == 'flux':
            return crossings
        elif mode == 'shoot':
            return crossings
        return crossings

    def rollout_with_monitoring(self, data_loader, ensemble_size: int = 1,
                            start_interface: int = -1,
                            capture_crossings: bool = True,
                            initial_state_override: Optional[torch.Tensor] = None,
                            mode: str = 'flux',
                            parent_config_name: Optional[str] = None,
                            n_trials_target: Optional[int] = None) -> Dict:
        """
        Single trajectory rollout with interface monitoring.
        
        Args:
            n_trials_target: Maximum number of configurations to save (shooting mode only).
                            If provided, checks global count before saving each crossing.
        """
        previous_location = None
        if mode == 'shoot' and parent_config_name:
            parent_config = self._load_parent_config(parent_config_name)
            if parent_config and parent_config.feature_location:
                previous_location = parent_config.feature_location
        
        trajectory_mslp = []
        trajectory_features = []
        crossings = []
        current_interface = start_interface
        trajectory_status = 'ongoing'
        can_save_crossing = True
        saved_states = {}
        
        ic_has_preexisting_system = False
        
        # Track previous timestep for lost track diagnostics
        prev_timestep_y_phys = None
        prev_timestep_datetime = None
        prev_timestep_feature_idx = None
        prev_timestep_location = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                forecast_step = batch["forecast_step"].item()
                
                if forecast_step == 1:
                    if initial_state_override is not None:
                        x = initial_state_override.to(self.device).float()
                    else:
                        if "x_surf" in batch:
                            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device).float()
                        else:
                            x = reshape_only(batch["x"]).to(self.device).float()
                        
                        if ensemble_size > 1:
                            x = torch.repeat_interleave(x, ensemble_size, 0)
                
                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4).float()
                    if ensemble_size > 1:
                        x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
                    x = torch.cat((x, x_forcing_batch), dim=1)
                
                y_pred = self.model(x, forecast_step=forecast_step - 1)
                y_pred_phys = self.state_transformer.inverse_transform(y_pred.cpu())
                y_pred_phys_with_mslp = self.calculate_mslp_wrapper(y_pred_phys, batch)
                
                mslp, feature_idx, location = self.extract_mslp(
                    y_pred_phys_with_mslp, batch, forecast_step, previous_location, mode=mode
                )
                trajectory_mslp.append(mslp)
                trajectory_features.append((forecast_step, feature_idx, location))
                
                # CHECK FOR LOCATION JUMP (only in shoot mode with tracking)
                if mode == 'shoot' and previous_location and location and forecast_step > 1:
                    prev_lat, prev_lon = previous_location
                    curr_lat, curr_lon = location
                    lat_diff = abs(curr_lat - prev_lat)
                    lon_diff = abs(curr_lon - prev_lon)
                    
                    if lat_diff > 10 or lon_diff > 10:
                        trajectory_status = 'lost_track'
                        random_suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
                        
                        if prev_timestep_y_phys is not None:
                            mock_parent = type('obj', (object,), {
                                '_y_phys': prev_timestep_y_phys,
                                '_datetime_obj': prev_timestep_datetime,
                                '_feature_idx': prev_timestep_feature_idx,
                                'feature_location': prev_timestep_location
                            })()
                        else:
                            mock_parent = None
                        
                        diagnostic_info = {
                            'parent_location': previous_location,
                            'crossing_location': location,
                            'distance': (lat_diff, lon_diff)
                        }
                        
                        failure_name = f"LOST_TRACK_{parent_config_name}_step{forecast_step}" if parent_config_name else f"LOST_TRACK_step{forecast_step}_{random_suffix}"
                        
                        self._save_failure_figure(
                            y_pred_phys_with_mslp,
                            failure_name=failure_name,
                            datetime_obj=datetime.fromtimestamp(batch["datetime"][0].item()),
                            feature_idx=feature_idx,
                            location=location,
                            parent_config=mock_parent,
                            diagnostic_info=diagnostic_info
                        )
                        
                        print(f" LOST TRACK (Δlat={lat_diff:.1f}°, Δlon={lon_diff:.1f}° in 1 timestep)")
                        break
                
                if location:
                    previous_location = location
                
                if forecast_step == 1 and mode == 'flux':
                    if mslp < self.interfaces[0]:
                        ic_has_preexisting_system = True
                
                y_pred_norm = self.state_transformer.transform_array(y_pred_phys).to(self.device)
                
                if capture_crossings:
                    if batch.get("y_diag") is not None:
                        varnum_diag = batch["y_diag"].shape[1]
                        saved_states[forecast_step] = y_pred_norm[:, :-varnum_diag, ...].cpu().clone()
                    else:
                        saved_states[forecast_step] = y_pred_norm.cpu().clone()
                
                status, interface_idx = self.check_trajectory_status(mslp, current_interface, mode)
                
                if status == 'crossed_forward':
                    if mode == 'flux':
                        current_interface = interface_idx
                        
                        if interface_idx == 0 and can_save_crossing:
                            should_save = True
                            
                            if location:
                                feat_lat, feat_lon = location
                                lat_idx = np.argmin(np.abs(self.latlons.latitude.values - feat_lat))
                                lon_180 = np.where(self.latlons.longitude.values > 180, 
                                                self.latlons.longitude.values - 360, 
                                                self.latlons.longitude.values)
                                lon_idx = np.argmin(np.abs(lon_180 - feat_lon))
                                
                                is_over_ocean = self.land_sea_mask[lat_idx, lon_idx] < 0.5
                                
                                if feat_lat >= 30.0 and not is_over_ocean:
                                    should_save = False
                                    print(f" (skipped - feature over land at {feat_lat:.1f}°N)")
                            
                            if should_save:
                                config_name = self._generate_config_name(interface_idx, mode='flux')
                                
                                crossing = InterfaceConfig(
                                    input_state=saved_states[forecast_step],
                                    latents=None,
                                    forecast_step=forecast_step,
                                    mslp_value=mslp,
                                    interface_idx=interface_idx,
                                    timestamp=datetime.now().isoformat(),
                                    restart_datetime=self._compute_restart_datetime(batch),
                                    config_name=config_name,
                                    parent_config=parent_config_name,
                                    track_id=None,
                                    feature_location=location
                                )
                                
                                crossing._feature_idx = feature_idx
                                crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
                                crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                                
                                crossings.append(crossing)
                                can_save_crossing = False
                    
                    elif mode == 'shoot':
                        next_interface = current_interface + 1
                        config_name = self._generate_config_name(next_interface, mode='shoot')
                        
                        crossing = InterfaceConfig(
                            input_state=saved_states[forecast_step],
                            latents=None,
                            forecast_step=forecast_step,
                            mslp_value=mslp,
                            interface_idx=next_interface,
                            timestamp=datetime.now().isoformat(),
                            restart_datetime=self._compute_restart_datetime(batch),
                            config_name=config_name,
                            parent_config=parent_config_name,
                            track_id=None,
                            feature_location=location
                        )
                        
                        crossing._feature_idx = feature_idx
                        crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
                        crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                        
                        crossings.append(crossing)
                        trajectory_status = 'success'
                        break
                
                elif status == 'reached_B':
                    config_name = self._generate_config_name(interface_idx=-1, mode='stateB')
                    
                    crossing = InterfaceConfig(
                        input_state=saved_states[forecast_step],
                        latents=None,
                        forecast_step=forecast_step,
                        mslp_value=mslp,
                        interface_idx=-1,
                        timestamp=datetime.now().isoformat(),
                        restart_datetime=self._compute_restart_datetime(batch),
                        config_name=config_name,
                        parent_config=parent_config_name,
                        track_id=None,
                        feature_location=location
                    )
                    
                    crossing._feature_idx = feature_idx
                    crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
                    crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                    
                    crossings.append(crossing)
                    trajectory_status = 'reached_B'
                    print(f" → STATE B ({mslp:.1f} hPa)")
                    break
                
                elif status == 'returned_A':
                    trajectory_status = 'failure'
                    print(f" → STATE A ({mslp:.1f} hPa)")
                    break
                
                elif status == 'returned_backward':
                    if mode == 'flux':
                        if self.decorrelation_interface is not None:
                            if mslp >= self.decorrelation_interface:
                                can_save_crossing = True
                        else:
                            if mslp >= self.state_A_threshold:
                                can_save_crossing = True
                    
                    current_interface = interface_idx if interface_idx is not None else -1
                
                prev_timestep_y_phys = y_pred_phys_with_mslp.cpu().clone()
                prev_timestep_datetime = datetime.fromtimestamp(batch["datetime"][0].item())
                prev_timestep_feature_idx = feature_idx
                prev_timestep_location = location
                
                if batch.get("y_diag") is not None:
                    varnum_diag = batch["y_diag"].shape[1]
                    x = y_pred_norm[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred_norm.detach()
                
                if batch.get("stop_forecast", torch.tensor(False)).item():
                    if trajectory_status == 'ongoing':
                        trajectory_status = 'completed' if mode == 'flux' else 'failure'
                    break
        
        # Filter crossings
        valid_crossings = self._filter_crossings(
            crossings, mode, ic_has_preexisting_system, parent_config_name
        )
        
        # Handle filtered out crossings
        if mode == 'shoot' and len(crossings) > 0 and len(valid_crossings) == 0:
            print(" LOST TRACK (filtered out)")
            trajectory_status = 'lost_track'
            
            for crossing in crossings:
                failure_name = f"LOST_TRACK_{crossing.config_name}"
                
                parent_config = self._load_parent_config(parent_config_name) if parent_config_name else None
                diagnostic_info = None
                if parent_config and parent_config.feature_location and crossing.feature_location:
                    parent_lat, parent_lon = parent_config.feature_location
                    cross_lat, cross_lon = crossing.feature_location
                    lat_diff = abs(parent_lat - cross_lat)
                    lon_diff = abs(parent_lon - cross_lon)
                    diagnostic_info = {
                        'parent_location': (parent_lat, parent_lon),
                        'crossing_location': (cross_lat, cross_lon),
                        'distance': (lat_diff, lon_diff)
                    }
                
                self._save_failure_figure(
                    crossing._y_phys,
                    failure_name=failure_name,
                    datetime_obj=crossing._datetime_obj,
                    feature_idx=crossing._feature_idx,
                    location=crossing.feature_location,
                    parent_config=parent_config,
                    diagnostic_info=diagnostic_info
                )
        
        # UNIFIED SAVING LOGIC: Save valid crossings with optional global count check
        for crossing in valid_crossings:
            if crossing.interface_idx == -1:
                save_dir = self.stateB_dir
                mode_str = 'stateB'
            elif mode == 'flux':
                save_dir = self.flux_dir
                mode_str = 'flux'
            else:
                save_dir = self._get_shoot_dir(crossing.interface_idx)
                save_dir.mkdir(parents=True, exist_ok=True)
                mode_str = 'shoot'
                
                # Thread-safe global count check for shooting mode
                if n_trials_target is not None:
                    current_count = len(list(save_dir.glob(f'lambda{crossing.interface_idx}_config_*.pkl')))
                    if current_count >= n_trials_target:
                        print(f"  Target reached ({current_count}/{n_trials_target}), skipping save")
                        valid_crossings.remove(crossing)
                        trajectory_status = 'target_reached'
                        continue
            
            self._save_mslp_figure(
                crossing._y_phys,
                config_name=crossing.config_name,
                save_dir=save_dir,
                datetime_obj=crossing._datetime_obj,
                feature_idx=crossing._feature_idx,
                location=crossing.feature_location
            )
            
            self._save_config(crossing, mode=mode_str)
            print(f"  → SAVED: {crossing.config_name} (MSLP: {crossing.mslp_value:.1f} hPa)")

        return {
            'status': trajectory_status,
            'mslp_trajectory': trajectory_mslp,
            'crossings': valid_crossings,
            'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None,
            'ic_had_preexisting': ic_has_preexisting_system if mode == 'flux' else False
        }

    def generate_flux_at_lambda0(self, initial_data_loader, n_trials: int = 100):
        """Phase 0: Flux generation - collect decorrelated crossings at λ₀."""
        # Create flux-specific logger
        logger = FFSLogger(self.logs_dir / 'flux', rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        print(f"\n{'='*70}")
        print(f"PHASE 0: FLUX GENERATION at λ₀ = {self.interfaces[0]} hPa")
        if self.decorrelation_interface is not None:
            print(f"Decorrelation interface: λ₋₁ = {self.decorrelation_interface} hPa")
        else:
            print(f"Decorrelation: Return to state A (MSLP > {self.state_A_threshold} hPa)")
        print(f"Target: {n_trials} decorrelated crossings")
        print(f"{'='*70}\n")
        
        trajectory_count = 0
        total_time_days = 0.0
        worker_crossings = 0
        direct_B_formations = 0
        
        while True:
            global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
            
            if global_count >= n_trials:
                print(f"✓ Target reached: {global_count}/{n_trials}, stopping")
                break
            
            trajectory_count += 1
            print(f"Trajectory {trajectory_count} (global: {global_count}/{n_trials}): ", end='')
            
            data_loader = copy.deepcopy(initial_data_loader)
            
            result = self.rollout_with_monitoring(
                data_loader,
                start_interface=-1,
                capture_crossings=True,
                mode='flux'
            )
            
            configs_from_this_traj = len(result['crossings'])
            worker_crossings += configs_from_this_traj
            
            if result['status'] == 'reached_B' and configs_from_this_traj == 0:
                direct_B_formations += 1
            
            config_names = [c.config_name for c in result['crossings']]
            
            logger.log_flux_trajectory(trajectory_count, result, config_names)
            
            if configs_from_this_traj > 0:
                print(f" ({configs_from_this_traj} crossings)")
            elif result['status'] == 'reached_B':
                print(" (direct B formation)")
            else:
                print(" (no crossings)")
            
            num_timesteps = len(result['mslp_trajectory'])
            elapsed_days = num_timesteps * 6.0 / 24.0
            total_time_days += elapsed_days
        
        self.flux_estimate = worker_crossings / total_time_days if total_time_days > 0 else 0.0
        self.direct_B_count = direct_B_formations
        self.direct_B_rate = direct_B_formations / total_time_days if total_time_days > 0 else 0.0
        
        final_global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
        
        print(f"\n{'='*70}")
        print("✓ FLUX GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Trajectories run: {trajectory_count}")
        print(f"Worker λ₀ crossings: {worker_crossings}")
        print(f"Worker direct B formations: {direct_B_formations}")
        print(f"Total time: {total_time_days:.1f} days")
        print("\nFFS Flux Rate:")
        print(f"  Φ₀ = {worker_crossings}/{total_time_days:.1f} days = {self.flux_estimate:.6f} crossings/day")
        print("\nDirect Formation Rate (brute force):")
        print(f"  Φ_direct = {direct_B_formations}/{total_time_days:.1f} days = {self.direct_B_rate:.6e} formations/day")
        print(f"\nGlobal configs at λ₀: {final_global_count}")
        print(f"{'='*70}\n")
        
        logger.close()

    def shoot_single_trajectory(self, config: InterfaceConfig, current_interface: int, 
                            logger: FFSLogger, n_trials: int) -> Dict:
        """
        Shoot single trajectory from saved config.
        
        Args:
            n_trials: Global target count for thread-safe saving verification
        """
        if config.mslp_value < self.interfaces[current_interface]:
            print(f"   Config {config.config_name} already at {config.mslp_value:.1f} hPa < λ_{current_interface} ({self.interfaces[current_interface]} hPa) - INSTANT SUCCESS")
            
            new_config_name = self._generate_config_name(current_interface, mode='shoot')
            new_config = InterfaceConfig(
                input_state=config.input_state,
                latents=config.latents,
                forecast_step=config.forecast_step,
                mslp_value=config.mslp_value,
                interface_idx=current_interface,
                timestamp=datetime.now().isoformat(),
                restart_datetime=config.restart_datetime,
                config_name=new_config_name,
                parent_config=config.config_name,
                track_id=config.track_id,
                feature_location=config.feature_location
            )
            
            # Check global count before saving instant success
            save_dir = self._get_shoot_dir(current_interface)
            save_dir.mkdir(parents=True, exist_ok=True)
            current_count = len(list(save_dir.glob(f'lambda{current_interface}_config_*.pkl')))
            
            if current_count >= n_trials:
                print(f"   Target reached ({current_count}/{n_trials}), not saving instant success")
                return {
                    'status': 'target_reached',
                    'mslp_trajectory': [config.mslp_value],
                    'crossings': [],
                    'final_mslp': config.mslp_value
                }
            
            if hasattr(config, '_y_phys'):
                self._save_mslp_figure(
                    config._y_phys,
                    config_name=new_config_name,
                    save_dir=save_dir,
                    datetime_obj=config._datetime_obj if hasattr(config, '_datetime_obj') else None,
                    feature_idx=config._feature_idx if hasattr(config, '_feature_idx') else None,
                    location=config.feature_location
                )
            
            self._save_config(new_config, mode='shoot')
            
            return {
                'status': 'success',
                'mslp_trajectory': [config.mslp_value],
                'crossings': [new_config],
                'final_mslp': config.mslp_value
            }
        
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
        
        print(f"   Using config: {config.config_name} (MSLP: {config.mslp_value:.1f} hPa, {config.restart_datetime})", end='')

        result = self.rollout_with_monitoring(
            restart_loader,
            start_interface=current_interface - 1,
            capture_crossings=True,
            initial_state_override=config.input_state,
            mode='shoot',
            parent_config_name=config.config_name,
            n_trials_target=n_trials
        )
        
        return result
    
    def shoot_from_interface(self, interface_idx: int, n_trials: int = 100):
        """Shoot trajectories from interface λᵢ until GLOBAL target reached."""
        logger = FFSLogger(self.logs_dir / str(interface_idx), rank=self.rank, 
                        world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        if len(self.interface_configs[interface_idx]) == 0:
            raise ValueError(f"No configs available for interface {interface_idx}")
        
        lambda_label = interface_idx - 1
        next_interface_idx = interface_idx + 1
        next_lambda_label = next_interface_idx - 1
        
        if self.rank == 0 and self.worker_id == 0:
            print(f"\n{'='*70}")
            print(f"SHOOTING from λ_{lambda_label} = {self.interfaces[interface_idx]} hPa")
            print(f"Available configs: {len(self.interface_configs[interface_idx])}")
            print(f"Target configs: {n_trials}")
            print(f"{'='*70}\n")
        
        successes = 0
        failures = 0
        total_attempts = 0
        
        shoot_dir_next = self._get_shoot_dir(next_interface_idx)
        initial_count = len(list(shoot_dir_next.glob(f'lambda{next_interface_idx}_config_*.pkl')))
        
        while True:
            global_count = len(list(shoot_dir_next.glob(f'lambda{next_interface_idx}_config_*.pkl')))
            
            if global_count >= n_trials:
                print(f"\n✓ Target reached: {global_count}/{n_trials} configs exist, stopping")
                break
            
            total_attempts += 1
            config = np.random.choice(self.interface_configs[interface_idx])
            
            print(f"Attempt {total_attempts} (global: {global_count}/{n_trials}):", end=' ')
            
            result = self.shoot_single_trajectory(config, interface_idx, logger, n_trials)

            child_config = None
            if result['crossings']:
                child_config = result['crossings'][0].config_name
            
            logger.log_shooting_attempt(
                interface_idx, 
                total_attempts, 
                config.config_name, 
                result, 
                child_config
            )
            
            if result['status'] == 'success' or result['status'] == 'reached_B':
                successes += 1
            elif result['status'] == 'failure' or result['status'] == 'lost_track':
                failures += 1
            elif result['status'] == 'target_reached':
                # Don't count as success or failure
                pass

        P_forward = successes / total_attempts if total_attempts > 0 else 0.0
        self.transition_probs.append(P_forward)

        final_count = len(list(shoot_dir_next.glob(f'lambda{next_interface_idx}_config_*.pkl')))
        configs_saved = final_count - initial_count

        logger.log_interface_summary(
            interface_idx, 
            successes, 
            failures, 
            total_attempts, 
            P_forward, 
            configs_saved
        )
        
        print(f"\n{'='*70}")
        print(f"✓ SHOOTING from λ_{lambda_label} COMPLETE")
        print(f"{'='*70}")
        print(f"Total attempts: {total_attempts}")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")
        print(f"P(λ_{lambda_label} → λ_{lambda_label + 1}) = {successes}/{total_attempts} = {P_forward:.4f}")
        print(f"Configs saved at λ_{lambda_label + 1}: {configs_saved}")
        print(f"{'='*80}\n")
        
        if next_interface_idx < len(self.interfaces):
            saved_config_files = list(shoot_dir_next.glob(f'lambda{next_interface_idx}_config_*.pkl'))
            
            existing_names = {c.config_name for c in self.interface_configs[next_interface_idx]}
            
            for config_file in saved_config_files:
                with open(config_file, 'rb') as f:
                    config = pickle.load(f)
                    if config.config_name not in existing_names:
                        self.interface_configs[next_interface_idx].append(config)
                        existing_names.add(config.config_name)
        
        logger.close()
        
        return P_forward

    def run_ffs(self, initial_data_loader, n_flux_trials: int = 10, n_shoot_trials: int = 10):
        """Full FFS algorithm."""
        self.generate_flux_at_lambda0(initial_data_loader, n_trials=n_flux_trials)
        
        for i in range(0, len(self.interfaces) - 1):
            self.shoot_from_interface(i, n_trials=n_shoot_trials)
        
        # Create summary logger at base logs dir for final results
        summary_logger = FFSLogger(self.logs_dir, rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, ic_dirname=None)
        
        ffs_total_prob = self.flux_estimate * np.prod(self.transition_probs)
        
        print(f"\n{'='*70}")
        print("FINAL FFS RESULTS")
        print(f"{'='*70}")
        print("\nFFS Enhanced Estimate:")
        print(f"  Φ₀ (flux at λ₀ = {self.interfaces[0]} hPa) = {self.flux_estimate:.6f} crossings/day")
        for i, p in enumerate(self.transition_probs):
            print(f"  P(λ_{i} → λ_{i+1}) = {p:.4f}")
        print("\n  P_FFS(IC → Hurricane) = Φ₀ × Π P(λᵢ→λᵢ₊₁)")
        print(f"  P_FFS(IC → Hurricane) = {ffs_total_prob:.2e} per day")
        if ffs_total_prob > 0:
            print(f"  Enhancement factor: {1/ffs_total_prob:.1f}x over direct sampling")
        
        print("\nDirect Formation Rate (brute force baseline):")
        print(f"  Direct B formations: {self.direct_B_count}")
        print(f"  P_direct(IC → Hurricane) = {self.direct_B_rate:.2e} per day")
        
        if self.direct_B_rate > 0 and ffs_total_prob > 0:
            ratio = ffs_total_prob / self.direct_B_rate
            print(f"  FFS/Direct ratio: {ratio:.2f}")
        
        print(f"{'='*70}\n")

        summary_logger.log_final_results(
            self.flux_estimate, 
            self.transition_probs, 
            ffs_total_prob,
            self.direct_B_count,
            self.direct_B_rate
        )
        
        summary_logger.close()

        return ffs_total_prob

class HurricaneGenesisFFS_Tracked(HurricaneGenesisFFS):
    """Extended FFS class with feature tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.feature_history = []
        self.tracks = None
        self.initial_track_ids = set()
        self.feature_id_map = {}
        
        self.tracking_params = {
            'v_max': 50,
            'dt': 6 * 3600,
            'dxy': 111000,
            'memory': 2,
            'stubs': 1
        }
    
    def enable_visualization(self):
        """Enable real-time MSLP visualization during rollout."""
        self.visualize_mslp = True
        print("✓ MSLP visualization enabled")
    
    def disable_visualization(self):
        """Disable MSLP visualization and close plot."""
        self.visualize_mslp = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self._colorbar_added = False
        print("✓ MSLP visualization disabled")

    def extract_mslp(self, y_phys: torch.Tensor, batch: Dict = None,
                     forecast_step: int = None,
                     parent_location: Optional[Tuple[float, float]] = None,
                     mode: str = 'flux') -> Tuple:
        """Extract MSLP with feature detection and tracking metadata."""
        try:
            import tobac
            import iris
        except ImportError:
            mslp = self._extract_mslp_fallback(y_phys, batch)
            return mslp, None, None
        
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)
        
        # Define search area based on mode and parent location
        if mode == 'shoot' and parent_location is not None:
            parent_lat, parent_lon = parent_location
            
            # Expand box if parent is near boundaries
            lat_min_search = max(self.basin['lat_min'], parent_lat - 10)
            lat_max_search = min(50.0, parent_lat + 10)  # Allow up to 50N for recurvature
            lon_min_search = max(-110.0, parent_lon - 10)  # Allow westward into Gulf
            lon_max_search = min(10.0, parent_lon + 10)   # Allow eastward into open Atlantic
            
            lat_mask = (lats >= lat_min_search) & (lats <= lat_max_search)
            lon_mask = (lons_180 >= lon_min_search) & (lons_180 <= lon_max_search)
        else:
            # Flux mode: use standard basin
            lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
            lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])

        basin_mask = lat_mask[:, None] & lon_mask[None, :]
        mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)

        # lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
        # lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])  # pyright: ignore[reportUndefinedVariable]
        # basin_mask = lat_mask[:, None] & lon_mask[None, :]
        # mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
        lons_tobac = lons_180.copy()
        mslp_tobac = mslp_basin.copy()
        
        if not np.all(np.diff(lons_tobac) > 0):
            sort_idx = np.argsort(lons_tobac)
            lons_tobac = lons_tobac[sort_idx]
            mslp_tobac = mslp_tobac[:, sort_idx]
        
        try:
            lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
            lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
            time_coord = iris.coords.DimCoord([forecast_step], standard_name='time', units='hours since 2000-01-01 00:00:00')
            
            cube = iris.cube.Cube(
                mslp_tobac[np.newaxis, :, :],
                standard_name='air_pressure_at_mean_sea_level',
                units='hPa',
                dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
            )
            
            basin_min = np.nanmin(mslp_tobac)
            basin_max = np.nanmax(mslp_tobac)
            thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
            thresholds = sorted(thresholds, reverse=False)
        
            features = tobac.feature_detection_multithreshold(
                field_in=cube,
                dxy=self.tracking_params['dxy'],
                threshold=thresholds,
                target='minimum',
                position_threshold='weighted_diff',
                sigma_threshold=1.5,
                n_min_threshold=3
            )
            
            if features is not None and len(features) > 0:
                if parent_location is not None:
                    parent_lat, parent_lon = parent_location
                    
                    features['lat_diff'] = abs(features['latitude'] - parent_lat)
                    features['lon_diff'] = abs(features['longitude'] - parent_lon)
                    
                    nearby = features[(features['lat_diff'] < 10) & (features['lon_diff'] < 10)]
                    
                    if len(nearby) > 0:
                        features_to_use = nearby
                    else:
                        features_to_use = features
                else:
                    features_to_use = features
                
                strongest_idx = features_to_use['threshold_value'].idxmin()
                strongest = features_to_use.loc[strongest_idx]
                
                feature_position = features.index.get_loc(strongest_idx)
                
                feature_lat = float(strongest['latitude'])
                feature_lon = float(strongest['longitude'])
                
                lat_idx = np.argmin(np.abs(lats - feature_lat))
                lon_idx = np.argmin(np.abs(lons_180 - feature_lon))
                
                min_mslp = float(mslp_basin[lat_idx, lon_idx])
                
                features['forecast_step'] = forecast_step
                self.feature_history.append(features.copy())
                
                if self.visualize_mslp:
                    datetime_str = None
                    if batch is not None and "datetime" in batch:
                        try:
                            datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
                            datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
                        except Exception:
                            pass
                    
                    mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
                    self._plot_mslp(
                        mslp_field=mslp_hpa,
                        mslp_smooth=mslp_smooth,
                        basin_mask=basin_mask,
                        datetime_str=datetime_str,
                        center_lat=feature_lat,
                        center_lon=feature_lon,
                        center_val=min_mslp
                    )
                
                return min_mslp, feature_position, (feature_lat, feature_lon)
            else:
                mslp = self._extract_mslp_fallback(y_phys, batch)
                return mslp, None, None
        
        except Exception as e:
            print(f"\n⚠ tobac error: {e}, using fallback")
            mslp = self._extract_mslp_fallback(y_phys, batch)
            return mslp, None, None
    
    def _extract_mslp_fallback(self, y_phys: torch.Tensor, batch: Dict = None) -> float:
        """Fallback MSLP extraction using simple minimum."""
        mslp_channel_idx = 71
        mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
        mslp_hpa = mslp_pa / 100.0
        
        basin_mask = self.get_basin_mask()
        
        mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        mslp_smooth_masked = np.where(basin_mask, mslp_smooth, np.nan)
        
        min_idx = np.nanargmin(mslp_smooth_masked)
        min_row, min_col = np.unravel_index(min_idx, mslp_smooth_masked.shape)
        min_mslp = float(mslp_hpa[min_row, min_col])
        
        if self.visualize_mslp:
            datetime_str = None
            if batch is not None and "datetime" in batch:
                try:
                    dt = datetime.fromtimestamp(batch["datetime"][0].item())
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
                except Exception:
                    pass
        
            min_lat = self.latlons.latitude.values[min_row]
            lons_180 = np.where(self.latlons.longitude.values > 180,
                                self.latlons.longitude.values - 360,
                                self.latlons.longitude.values)
            min_lon = lons_180[min_col]
        
            self._plot_mslp(
                mslp_field=mslp_hpa,
                mslp_smooth=mslp_smooth,
                basin_mask=basin_mask,
                datetime_str=datetime_str,
                center_lat=float(min_lat),
                center_lon=float(min_lon),
                center_val=min_mslp
            )

        return min_mslp
    
    def _plot_mslp(self, mslp_field: np.ndarray, mslp_smooth: np.ndarray,
                   basin_mask: np.ndarray, datetime_str: str = None,
                   center_lat: float = None, center_lon: float = None,
                   center_val: float = None):
        """Plot MSLP field centered on the storm position."""
        from IPython.display import display, clear_output
    
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(14, 10))
            self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
        self.ax.clear()
    
        lats = self.latlons.latitude.values
        lons = self.latlons.longitude.values
        lons_180 = np.where(lons > 180, lons - 360, lons)
    
        basin_rows, basin_cols = np.where(basin_mask)
        if len(basin_rows) > 0:
            row_min = max(0, basin_rows.min() - 10)
            row_max = min(mslp_smooth.shape[0], basin_rows.max() + 10)
            col_min = max(0, basin_cols.min() - 10)
            col_max = min(mslp_smooth.shape[1], basin_cols.max() + 10)
        else:
            row_min, row_max = 0, mslp_smooth.shape[0]
            col_min, col_max = 0, mslp_smooth.shape[1]
    
        mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
        lat_crop = lats[row_min:row_max]
        lon_crop = lons_180[col_min:col_max]
    
        self.ax.set_extent(
            [lon_crop.min(), lon_crop.max(), lat_crop.min(), lat_crop.max()],
            crs=ccrs.PlateCarree()
        )
    
        levels = np.arange(960, 1030, 4)
        pcm = self.ax.contourf(
            lon_crop, lat_crop, mslp_crop,
            levels=levels, cmap='RdBu_r', extend='both',
            transform=ccrs.PlateCarree(), zorder=1
        )
    
        cs = self.ax.contour(
            lon_crop, lat_crop, mslp_crop,
            levels=np.arange(960, 1030, 8),
            colors='k', linewidths=0.5,
            transform=ccrs.PlateCarree(), zorder=4
        )
        self.ax.clabel(cs, inline=True, fontsize=8)
    
        self.ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0)
        self.ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, alpha=0.6)
        self.ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5)
    
        self.ax.plot(
            center_lon, center_lat,
            marker='*', markersize=20,
            markeredgecolor='yellow', markeredgewidth=2,
            color='k', transform=ccrs.PlateCarree(), zorder=10
        )
    
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    
        title = f"MSLP (hPa) - Min: {center_val:.1f} hPa @ ({center_lat:.1f}°, {center_lon:.1f}°)"
        if datetime_str is not None:
            title = f"{datetime_str}\n{title}"
    
        self.ax.set_title(title, fontsize=14, fontweight="bold")
    
        if not self._colorbar_added:
            plt.colorbar(pcm, ax=self.ax, label="MSLP (hPa)", shrink=0.8)
            self._colorbar_added = True
    
        clear_output(wait=True)
        display(self.fig)
        plt.pause(0.01)
    
    def perform_tracking(self):
        """Link features across timesteps using tobac tracking."""
        if len(self.feature_history) == 0:
            return
        
        try:
            import tobac
            import pandas as pd
        except ImportError:
            return
        
        all_features = pd.concat(self.feature_history, ignore_index=False)
        
        if len(all_features) == 0:
            return
        
        all_features = all_features.reset_index(drop=False)
        all_features = all_features.rename(columns={'index': 'original_idx'})
        
        self.tracks = tobac.linking_trackpy(
            all_features,
            field_in=None,
            dt=self.tracking_params['dt'],
            dxy=self.tracking_params['dxy'],
            v_max=self.tracking_params['v_max'],
            memory=self.tracking_params['memory'],
            stubs=self.tracking_params['stubs']
        )
        
        initial_features = self.tracks[self.tracks['forecast_step'] == 1]
        if len(initial_features) > 0:
            self.initial_track_ids = set(initial_features['cell'].unique())
        else:
            self.initial_track_ids = set()
        
        for _, row in self.tracks.iterrows():
            step = int(row['forecast_step'])
            original_idx = int(row['original_idx'])
            track_id = row['cell']
            
            if step not in self.feature_id_map:
                self.feature_id_map[step] = {}
            self.feature_id_map[step][original_idx] = track_id
        
    def is_new_genesis(self, forecast_step: int, feature_idx: Optional[int]) -> bool:
        """Check if feature represents NEW genesis (not present at IC)."""
        if feature_idx is None:
            return False
        
        if self.tracks is None or len(self.feature_id_map) == 0:
            return False
        
        if forecast_step not in self.feature_id_map:
            return False
        
        if feature_idx not in self.feature_id_map[forecast_step]:
            return False
        
        track_id = self.feature_id_map[forecast_step][feature_idx]
        is_new = track_id not in self.initial_track_ids
        
        return is_new
    
    def get_track_info(self, feature_idx: Optional[int], forecast_step: int) -> Dict:
        """Get detailed tracking information for a feature."""
        info = {
            'tracked': False,
            'track_id': None,
            'is_new': False,
            'track_length': 0,
            'first_detection': None
        }
        
        if feature_idx is None or self.tracks is None:
            return info
        
        if forecast_step not in self.feature_id_map:
            return info
        
        if feature_idx not in self.feature_id_map[forecast_step]:
            return info
        
        track_id = self.feature_id_map[forecast_step][feature_idx]
        track_data = self.tracks[self.tracks['cell'] == track_id]
        
        info['tracked'] = True
        info['track_id'] = track_id
        info['is_new'] = track_id not in self.initial_track_ids
        info['track_length'] = len(track_data)
        info['first_detection'] = int(track_data['forecast_step'].min())
        
        return info
    
    def reset_tracking(self):
        """Reset tracking state for a new trajectory."""
        self.feature_history = []
        self.tracks = None
        self.initial_track_ids = set()
        self.feature_id_map = {}

    def _filter_crossings(self, crossings: List[InterfaceConfig], 
                        mode: str, 
                        ic_has_preexisting: bool,
                        parent_config_name: Optional[str]) -> List[InterfaceConfig]:
        """Filter crossings: flux mode = NEW storms only, shoot mode = same storm."""
        self.perform_tracking()
        
        if mode == 'shoot':
            return crossings
        
        # FLUX MODE: Only accept NEW genesis
        valid_crossings = []
        for crossing in crossings:
            track_info = self.get_track_info(crossing._feature_idx, crossing.forecast_step)
            crossing.track_id = track_info['track_id']
            
            if track_info['tracked'] and track_info['is_new']:
                valid_crossings.append(crossing)
        
        return valid_crossings

    # def _filter_crossings(self, crossings: List[InterfaceConfig], 
    #                     mode: str, 
    #                     ic_has_preexisting: bool,
    #                     parent_config_name: Optional[str]) -> List[InterfaceConfig]:
    #     """Filter crossings based on tracking criteria."""
    #     self.perform_tracking()
        
    #     for crossing in crossings:
    #         feature_idx = crossing._feature_idx
    #         forecast_step = crossing.forecast_step
            
    #         track_info = self.get_track_info(feature_idx, forecast_step)
    #         crossing.track_id = track_info['track_id'] if track_info['tracked'] else None
        
    #     valid_crossings = []
        
    #     if mode == 'flux':
    #         if ic_has_preexisting:
    #             pass
    #         else:
    #             valid_crossings = crossings
        
    #     elif mode == 'shoot':
    #         valid_crossings = crossings
        
    #     return valid_crossings

# Add this to the END of the file after the HurricaneGenesisFFS_Tracked class:

if __name__ == "__main__":
    filepath = "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/"
    device = "cuda"

    with open(os.path.join(filepath, "model.yml"), "r") as f:
        conf = yaml.safe_load(f)

    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    data_config = setup_data_loading(conf)

    ensemble_size = 1
    conf["trainer"]["ensemble_size"] = ensemble_size
    conf["predict"]["ensemble_size"] = ensemble_size

    model = load_model(conf, load_weights=True).to("cuda")
    model = model.eval()

    forecast_times = [['2022-08-28 00:00:00', '2022-09-10 00:00:00']]

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
        'rank': 0,
        'world_size': 1,
    }

    initial_dataset = Predict_Dataset_Batcher(
        **dataset_params,
        fcst_datetime=forecast_times
    )

    initial_loader = BatchForecastLenDataLoader(initial_dataset)

    ffs = HurricaneGenesisFFS(
        model=model,
        state_transformer=Normalize_ERA5_and_Forcing(conf),
        config=conf,
        initial_dataset=initial_dataset,
        dataset_params=dataset_params,
        device='cuda'
    )

    ffs.run_ffs(initial_loader, n_flux_trials=20, n_shoot_trials=10)



# import torch
# import numpy as np
# import xarray as xr
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass
# from datetime import datetime, timedelta
# import copy

# # Core Python
# import os
# import yaml
# import random
# import string

# # Numerical and ML

# # Visualization
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from credit.output import make_xarray
# from scipy.ndimage import gaussian_filter

# # CREDIT framework
# from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher
# from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
# from credit.parser import credit_main_parser
# from credit.datasets import setup_data_loading
# from credit.models import load_model
# from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
# from credit.rare_events.ffs_logger import FFSLogger

# from credit.data import concat_and_reshape, reshape_only
# from credit.interp import full_state_pressure_interpolation
# from pathlib import Path
# import pickle
# import warnings
# warnings.filterwarnings('ignore')


# @dataclass
# class InterfaceConfig:
#     """Configuration saved at interface crossing."""
#     input_state: torch.Tensor
#     latents: Optional[torch.Tensor]
#     forecast_step: int
#     mslp_value: float
#     interface_idx: int
#     timestamp: str
#     restart_datetime: datetime
#     config_name: str
#     parent_config: Optional[str] = None
#     track_id: Optional[int] = None  # Track ID from tobac
#     feature_location: Optional[Tuple[float, float]] = None  # (lat, lon)


# class HurricaneGenesisFFS:
#     """
#     Forward Flux Sampling for hurricane genesis in the Atlantic.
    
#     State A: No hurricane (MSLP > state_A_threshold hPa)
#     State B: Hurricane formed (MSLP < state_B_threshold hPa)
    
#     Interface setup:
#         interfaces[0] = λ₀ (flux interface, e.g., 1000 hPa)
#         interfaces[1] = λ₁ (first shooting interface, e.g., 997 hPa)
#         interfaces[2] = λ₂ (second shooting interface, e.g., 994 hPa)
#         ... and so on
    
#     Optional decorrelation: If decorrelation_interface is provided (e.g., λ₋₁ = 1005 hPa),
#     trajectories must return above this threshold before saving the next λ₀ crossing.
#     If None, trajectories must return to state A instead.
#     """
    
#     def __init__(self, 
#                  model, 
#                  state_transformer, 
#                  config, 
#                  initial_dataset,
#                  dataset_params,
#                  output_dir='./ffs_output',
#                  state_A=1008, 
#                  state_B=982, 
#                  interfaces=[1000, 997, 994, 991, 988, 985],
#                  decorrelation_interface=None,
#                  worker_id=0,
#                  rank=0,
#                  world_size=1,
#                  ic_dirname=None):
#         """
#         Initialize FFS sampler.
        
#         Args:
#             model: The neural weather model
#             state_transformer: Normalization transformer
#             config: Model configuration dict
#             initial_dataset: The initial Predict_Dataset_Batcher to copy for restarts
#             output_dir: Directory for saving configs and figures
#             state_A: MSLP threshold for state A (hPa) - no organized system
#             state_B: MSLP threshold for state B (hPa) - hurricane target
#             interfaces: List of interface thresholds (hPa), decreasing order
#                        interfaces[0] = λ₀ (flux interface, e.g., 1000 hPa)
#                        interfaces[1] = λ₁ (first shooting interface, e.g., 997 hPa)
#                        interfaces[2+] = λ₂, λ₃, ... (subsequent shooting interfaces)
#             decorrelation_interface: Optional λ₋₁ threshold (hPa). If provided, trajectories
#                                     must return above this to save next λ₀ crossing.
#                                     If None, trajectories must return to state A instead.
#             device: Device to run on
#         """
#         self.model = model
#         self.state_transformer = state_transformer
#         self.config = config
#         self.initial_dataset = initial_dataset
#         self.worker_id = worker_id
#         self.rank = rank
#         self.world_size = world_size
#         self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
#         # Output directories
#         self.output_dir = Path(output_dir)
#         self.flux_dir = self.output_dir / 'flux_gen'
#         self.shoot_dir = self.output_dir / 'shooting'
#         self.stateB_dir = self.output_dir / 'stateB'
#         self.flux_dir.mkdir(parents=True, exist_ok=True)
#         self.shoot_dir.mkdir(parents=True, exist_ok=True)
#         self.stateB_dir.mkdir(parents=True, exist_ok=True)
#         self.dataset_params = dataset_params

#         self.failed_dir = self.output_dir / 'failed_trajectories'
#         self.failed_dir.mkdir(parents=True, exist_ok=True)
        
#         # Config counters for naming
#         self.config_counters = {
#             'flux': 0,
#             'shoot': {i: 0 for i in range(len(interfaces))},
#             'stateB': 0
#         }
        
#         # Load static data for MSLP calculation
#         self.latlons = xr.open_dataset(config["loss"]["latitude_weights"]).load()
#         with xr.open_dataset('/glade/campaign/cisl/aiml/credit/static_scalers/static_whole_20250416_1deg.nc') as df:
#             self.surface_geopotential = df["Z_GDS4_SFC"].values
#             self.land_sea_mask = df["LSM"].values
        
#         # Basin definition
#         self.basin = {
#             'lat_min': 10.0,
#             'lat_max': 40.0,
#             'lon_min': -100.0,
#             'lon_max': -20.0
#         }
        
#         # Thresholds
#         if state_B not in interfaces:
#             interfaces.append(state_B)
#         self.interfaces = interfaces
#         self.state_A_threshold = state_A
#         self.state_B_threshold = state_B
#         self.decorrelation_interface = decorrelation_interface
        
#         # Storage for interface crossings
#         self.interface_configs: Dict[int, List[InterfaceConfig]] = {
#             i: [] for i in range(len(self.interfaces))
#         }
        
#         # Statistics
#         self.flux_estimate = None
#         self.transition_probs = []
        
#         # Direct B formation tracking (brute force estimate)
#         self.direct_B_count = 0
#         self.direct_B_rate = None
        
#         # Visualization
#         self.visualize_mslp = False
#         self.fig = None
#         self.ax = None
#         self._colorbar_added = False

#         # Initialize logger
#         self.logger = FFSLogger(self.output_dir, rank=rank, world_size=world_size, worker_id=worker_id, ic_dirname=ic_dirname)
    
#     def get_basin_mask(self) -> np.ndarray:
#         """Create spatial mask for Atlantic basin."""
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
        
#         # Convert longitude to -180 to 180
#         lons = np.where(lons > 180, lons - 360, lons)
        
#         lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
#         lon_mask = (lons >= self.basin['lon_min']) & (lons <= self.basin['lon_max'])
        
#         basin_mask = lat_mask[:, None] & lon_mask[None, :]
        
#         return basin_mask

#     def extract_mslp(self, y_phys: torch.Tensor, batch: Dict = None, 
#                      forecast_step: int = None, 
#                      parent_location: Optional[Tuple[float, float]] = None) -> Tuple:
#         """
#         Extract minimum MSLP in Atlantic basin.
        
#         Returns:
#             Tuple of (min_mslp, feature_idx, location)
#             Base class returns (mslp, None, None)
#         """
#         try:
#             import tobac
#             import iris
#         except ImportError:
#             return self._extract_mslp_fallback(y_phys, batch), None, None
        
#         mslp_channel_idx = 71
#         mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         # Get lat/lon - keep original coordinates
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         # Create basin mask with ORIGINAL coordinate order
#         lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
#         lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
#         basin_mask = lat_mask[:, None] & lon_mask[None, :]
        
#         mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
#         # FOR TOBAC ONLY: sort if needed
#         lons_tobac = lons_180.copy()
#         mslp_tobac = mslp_basin.copy()
        
#         if not np.all(np.diff(lons_tobac) > 0):
#             sort_idx = np.argsort(lons_tobac)
#             lons_tobac = lons_tobac[sort_idx]
#             mslp_tobac = mslp_tobac[:, sort_idx]
        
#         # Create iris cube with sorted coordinates for tobac
#         try:
#             lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
#             lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
#             time_coord = iris.coords.DimCoord([0], standard_name='time', units='hours since 2024-01-01 00:00:00')
            
#             cube = iris.cube.Cube(
#                 mslp_tobac[np.newaxis, :, :],
#                 standard_name='air_pressure_at_mean_sea_level',
#                 units='hPa',
#                 dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
#             )
            
#             basin_min = np.nanmin(mslp_tobac)
#             basin_max = np.nanmax(mslp_tobac)
#             thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
#             thresholds = sorted(thresholds, reverse=False)
        
#             features = tobac.feature_detection_multithreshold(
#                 field_in=cube,
#                 dxy=111000,
#                 threshold=thresholds,
#                 target='minimum',
#                 position_threshold='weighted_diff',
#                 sigma_threshold=1.5,
#                 n_min_threshold=3
#             )
            
#             if features is not None and len(features) > 0:
#                 strongest = features.loc[features['threshold_value'].idxmin()]
#                 # Indices are for SORTED array
#                 min_row_sorted = int(strongest['hdim_1'])
#                 min_col_sorted = int(strongest['hdim_2'])
                
#                 # Get value from sorted array
#                 min_mslp = float(mslp_tobac[min_row_sorted, min_col_sorted])
                
#                 if np.isnan(min_mslp):
#                     return self._extract_mslp_fallback(y_phys, batch), None, None
#             else:
#                 return self._extract_mslp_fallback(y_phys, batch), None, None
        
#         except Exception as e:
#             print(f"\n⚠ tobac error: {e}, using fallback")
#             return self._extract_mslp_fallback(y_phys, batch), None, None
        
#         return min_mslp, None, None
    
#     def _extract_mslp_fallback(self, y_phys: torch.Tensor, batch: Dict = None) -> float:
#         """Fallback MSLP extraction using simple minimum."""
#         mslp_channel_idx = 71
#         mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         basin_mask = self.get_basin_mask()
        
#         # Smooth for location, raw for value
#         mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
#         mslp_smooth_masked = np.where(basin_mask, mslp_smooth, np.nan)
        
#         min_idx = np.nanargmin(mslp_smooth_masked)
#         min_row, min_col = np.unravel_index(min_idx, mslp_smooth_masked.shape)
#         min_mslp = float(mslp_hpa[min_row, min_col])

#         return min_mslp
    
    
#     def check_trajectory_status(self, mslp_value: float, 
#                                current_interface: int,
#                                mode: str = 'flux') -> Tuple[str, Optional[int]]:
#         """Determine trajectory status during FFS."""
        
#         # Check forward crossing - strictly less than
#         next_interface = current_interface + 1
#         if next_interface < len(self.interfaces):
#             if mslp_value < self.interfaces[next_interface]:
#                 return 'crossed_forward', next_interface
        
#         # Reached B
#         if mslp_value < self.state_B_threshold:
#             return 'reached_B', None
        
#         # SHOOT MODE: check for failure (return to state A only)
#         if mode == 'shoot':
#             if mslp_value > self.state_A_threshold:
#                 return 'returned_A', None
#             return 'continue', None
        
#         # FLUX MODE: backward crossing
#         if current_interface >= 0:
#             if mslp_value > self.interfaces[current_interface]:
#                 for i in range(current_interface - 1, -1, -1):
#                     if mslp_value < self.interfaces[i]:
#                         return 'returned_backward', i
#                 return 'returned_backward', -1
        
#         return 'continue', None
    
#     def calculate_mslp_wrapper(self, y_pred_phys, batch):
#         """Wrapper for MSLP calculation."""
#         datetime_str = datetime.fromtimestamp(batch["datetime"][0].item()).strftime('%Y-%m-%d %H:%M:%S')
            
#         darray_upper_air, darray_single_level = make_xarray(
#             y_pred_phys,
#             datetime_str,
#             self.latlons.latitude.values,
#             self.latlons.longitude.values,
#             self.config,
#         )
        
#         ds_merged = xr.merge([
#             darray_upper_air.to_dataset(dim="vars"),
#             darray_single_level.to_dataset(dim="vars")
#         ])
        
#         pressure_interp = full_state_pressure_interpolation(
#             ds_merged,
#             self.surface_geopotential,
#             **self.config["predict"]["interp_pressure"]
#         )
        
#         mslp = torch.from_numpy(
#             pressure_interp['mean_sea_level_pressure'].values
#         ).unsqueeze(1).unsqueeze(2)
        
#         return torch.cat([y_pred_phys, mslp], dim=1)
    
#     def _compute_restart_datetime(self, batch):
#         """Compute datetime for restarting from this forecast step."""
#         original_ic_time = datetime.fromtimestamp(batch["datetime"][0].item())
#         restart_time = original_ic_time + timedelta(hours=6)
#         return restart_time

#     def _generate_config_name(self, interface_idx: int, mode: str = 'flux') -> str:
#         """Generate unique config name based on existing files."""
#         random_suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
        
#         if mode == 'stateB':
#             # Count existing stateB configs
#             existing = len(list(self.stateB_dir.glob('stateB_config_*.pkl')))
#             return f"stateB_config_{existing+1:04d}_{random_suffix}"
        
#         # interface_idx directly corresponds to lambda index: interfaces[1] = λ₁
#         lambda_label = interface_idx
        
#         if mode == 'flux':
#             # Count existing flux configs
#             existing = len(list(self.flux_dir.glob(f'lambda{lambda_label}_config_*.pkl')))
#             return f"lambda{lambda_label}_config_{existing+1:04d}_{random_suffix}"
#         else:
#             # Count existing shooting configs
#             existing = len(list(self.shoot_dir.glob(f'lambda{lambda_label}_config_*.pkl')))
#             return f"lambda{lambda_label}_config_{existing+1:04d}_{random_suffix}"
    
#     def _save_config(self, config: InterfaceConfig, mode: str = 'flux'):
#         """Save config to disk."""
#         save_dir = self.flux_dir if mode == 'flux' else self.shoot_dir
        
#         # Save pickle
#         config_path = save_dir / f"{config.config_name}.pkl"
#         with open(config_path, 'wb') as f:
#             pickle.dump(config, f)
        
#         return config_path
    
#     def _save_mslp_figure(self, y_phys_with_mslp: torch.Tensor, 
#                           config_name: str, save_dir: Path,
#                           datetime_obj: Optional[datetime] = None,
#                           feature_idx: Optional[int] = None,
#                           location: Optional[Tuple[float, float]] = None):
#         """Save MSLP figure at interface crossing."""
#         mslp_channel_idx = 71
#         mslp_pa = y_phys_with_mslp[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         basin_mask = self.get_basin_mask()
#         mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
#         # Prepare coordinates
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)

#         # --- 1. Find min location for marker ---
#         if location is not None:
#             min_lat, min_lon = location
#             lat_idx = np.argmin(np.abs(lats - min_lat))
#             lon_idx = np.argmin(np.abs(lons_180 - min_lon))
#         else:
#             # fallback to raw min in basin
#             mslp_masked = np.where(basin_mask, mslp_hpa, np.nan)
#             min_idx = np.nanargmin(mslp_masked)
#             lat_idx, lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
#             min_lat = lats[lat_idx]
#             min_lon = lons_180[lon_idx]

#         # --- 2. Calculate Basin Bounds ---
#         basin_rows, basin_cols = np.where(basin_mask)
#         if len(basin_rows) > 0:
#             pad = 10
#             row_min = max(0, basin_rows.min() - pad)
#             row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
#             col_min = max(0, basin_cols.min() - pad)
#             col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
#         else:
#             row_min, row_max = 0, mslp_smooth.shape[0]
#             col_min, col_max = 0, mslp_smooth.shape[1]

#         # --- 3. Crop Data ---
#         mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
#         lat_crop = lats[row_min:row_max]
#         lon_crop = lons_180[col_min:col_max]

#         # Create figure
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
#         # --- 4. Set Extent ---
#         ax.set_extent([lon_crop.min(), lon_crop.max(), 
#                     lat_crop.min(), lat_crop.max()], 
#                     crs=ccrs.PlateCarree())

#         # --- 5. Fixed Levels ---
#         levels = np.arange(960, 1030, 4)
        
#         pcm = ax.contourf(lon_crop, lat_crop, mslp_crop, levels=levels,
#                         cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=1)
        
#         contour_levels = np.arange(960, 1030, 8)
#         cs = ax.contour(lon_crop, lat_crop, mslp_crop,
#                         levels=contour_levels,
#                         colors='k', linewidths=0.5, transform=ccrs.PlateCarree(), zorder=2)
#         ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

#         # Features
#         ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
#         ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
#         ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)
        
#         # Plot marker
#         ax.plot(min_lon, min_lat, 'k*', markersize=20, 
#                 markeredgewidth=2, markeredgecolor='yellow',
#                 transform=ccrs.PlateCarree(), zorder=5)
        
#         # Gridlines
#         gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)
#         gl.top_labels = False
#         gl.right_labels = False
        
#         # Title - CLEAN VERSION
#         datetime_str = None
#         if datetime_obj is not None:
#             datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
#         min_val = mslp_hpa[lat_idx, lon_idx]
#         title = f'MSLP (hPa) - Basin Min: {min_val:.1f} hPa @ ({min_lat:.1f}°N, {min_lon:.1f}°W)'
#         if datetime_str is not None:
#             title = f'{datetime_str}\n{title}'
#         title += f'\nConfig: {config_name}'
#         ax.set_title(title, fontsize=12, fontweight='bold')
        
#         plt.colorbar(pcm, ax=ax, label='MSLP (hPa)', shrink=0.8)
#         plt.tight_layout()
        
#         # Save
#         fig_path = save_dir / f"{config_name}.png"
#         plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         return fig_path

#     def _save_failure_figure(self, y_phys_with_mslp: torch.Tensor, 
#                             failure_name: str,
#                             datetime_obj: Optional[datetime] = None,
#                             feature_idx: Optional[int] = None,
#                             location: Optional[Tuple[float, float]] = None,
#                             parent_config: Optional[InterfaceConfig] = None,
#                             diagnostic_info: Optional[Dict] = None):
#         """Save 3-panel MSLP figure for failed trajectory with diagnostic info.
        
#         Panel 1: Parent config (before)
#         Panel 2: Current config (after - failed)
#         Panel 3: Both overlaid showing displacement
#         """
#         mslp_channel_idx = 71
        
#         # Extract current MSLP
#         mslp_pa = y_phys_with_mslp[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         basin_mask = self.get_basin_mask()
#         mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
        
#         # Prepare coordinates
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)

#         # Find current location
#         if location is not None:
#             curr_lat, curr_lon = location
#             curr_lat_idx = np.argmin(np.abs(lats - curr_lat))
#             curr_lon_idx = np.argmin(np.abs(lons_180 - curr_lon))
#         else:
#             mslp_masked = np.where(basin_mask, mslp_hpa, np.nan)
#             min_idx = np.nanargmin(mslp_masked)
#             curr_lat_idx, curr_lon_idx = np.unravel_index(min_idx, mslp_masked.shape)
#             curr_lat = lats[curr_lat_idx]
#             curr_lon = lons_180[curr_lon_idx]

#         curr_mslp = mslp_hpa[curr_lat_idx, curr_lon_idx]

#         # Extract parent MSLP if available
#         parent_mslp_hpa = None
#         parent_mslp_smooth = None
#         parent_lat, parent_lon = None, None
#         parent_mslp_val = None
        
#         if parent_config and hasattr(parent_config, '_y_phys'):
#             parent_mslp_pa = parent_config._y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#             parent_mslp_hpa = parent_mslp_pa / 100.0
#             parent_mslp_smooth = gaussian_filter(parent_mslp_hpa, sigma=1.5)
            
#             if parent_config.feature_location:
#                 parent_lat, parent_lon = parent_config.feature_location
#                 parent_lat_idx = np.argmin(np.abs(lats - parent_lat))
#                 parent_lon_idx = np.argmin(np.abs(lons_180 - parent_lon))
#                 parent_mslp_val = parent_mslp_hpa[parent_lat_idx, parent_lon_idx]

#         # Calculate basin bounds for consistent view
#         basin_rows, basin_cols = np.where(basin_mask)
#         if len(basin_rows) > 0:
#             pad = 10
#             row_min = max(0, basin_rows.min() - pad)
#             row_max = min(mslp_smooth.shape[0], basin_rows.max() + pad)
#             col_min = max(0, basin_cols.min() - pad)
#             col_max = min(mslp_smooth.shape[1], basin_cols.max() + pad)
#         else:
#             row_min, row_max = 0, mslp_smooth.shape[0]
#             col_min, col_max = 0, mslp_smooth.shape[1]

#         # Crop data
#         mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
#         lat_crop = lats[row_min:row_max]
#         lon_crop = lons_180[col_min:col_max]
        
#         if parent_mslp_smooth is not None:
#             parent_mslp_crop = parent_mslp_smooth[row_min:row_max, col_min:col_max]

#         # Create 3-panel figure
#         fig = plt.figure(figsize=(20, 7))
        
#         # Shared plotting function
#         def plot_mslp_panel(ax, mslp_data, title, marker_configs):
#             """Helper to plot MSLP panel with markers.
            
#             marker_configs: list of (lon, lat, marker, size, color, edgecolor, label)
#             """
#             ax.set_extent([lon_crop.min(), lon_crop.max(), 
#                         lat_crop.min(), lat_crop.max()], 
#                         crs=ccrs.PlateCarree())

#             levels = np.arange(960, 1030, 4)
#             pcm = ax.contourf(lon_crop, lat_crop, mslp_data, levels=levels,
#                             cmap='RdBu_r', extend='both', 
#                             transform=ccrs.PlateCarree(), zorder=1)
            
#             contour_levels = np.arange(960, 1030, 8)
#             cs = ax.contour(lon_crop, lat_crop, mslp_data,
#                         levels=contour_levels, colors='k', 
#                         linewidths=0.5, transform=ccrs.PlateCarree(), zorder=2)
#             ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

#             ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0, zorder=3)
#             ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, alpha=0.6, zorder=3)
#             ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5, zorder=3)
            
#             # Plot markers
#             for marker_cfg in marker_configs:
#                 lon, lat, marker, size, color, edgecolor, label = marker_cfg
#                 ax.plot(lon, lat, marker, markersize=size, 
#                     color=color, markeredgewidth=2 if marker == '*' else 3,
#                     markeredgecolor=edgecolor,
#                     transform=ccrs.PlateCarree(), zorder=5, label=label)
            
#             gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, 
#                             linestyle='--', zorder=2)
#             gl.top_labels = False
#             gl.right_labels = False
            
#             ax.set_title(title, fontsize=11, fontweight='bold')
            
#             return pcm
        
#         # PANEL 1: Parent config (if available)
#         if parent_mslp_smooth is not None and parent_lat is not None:
#             ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
            
#             parent_datetime_str = ""
#             if hasattr(parent_config, '_datetime_obj'):
#                 parent_datetime_str = parent_config._datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
            
#             title1 = f"PARENT CONFIG\n{parent_datetime_str}\n"
#             title1 += f"MSLP: {parent_mslp_val:.1f} hPa @ ({parent_lat:.1f}°N, {parent_lon:.1f}°W)"
            
#             markers1 = [(parent_lon, parent_lat, '*', 20, 'k', 'yellow', 'Parent')]
#             _ = plot_mslp_panel(ax1, parent_mslp_crop, title1, markers1)
#             ax1.legend(loc='upper right')
#         else:
#             # No parent data - show text
#             ax1 = fig.add_subplot(1, 3, 1)
#             ax1.text(0.5, 0.5, 'Parent config\ndata not available', 
#                     ha='center', va='center', fontsize=14, color='gray')
#             ax1.axis('off')
#             _ = None
        
#         # PANEL 2: Current (failed) config
#         ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
        
#         curr_datetime_str = ""
#         if datetime_obj is not None:
#             curr_datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
        
#         title2 = f"CURRENT CONFIG (LOST TRACK)\n{curr_datetime_str}\n"
#         title2 += f"MSLP: {curr_mslp:.1f} hPa @ ({curr_lat:.1f}°N, {curr_lon:.1f}°W)"
        
#         markers2 = [(curr_lon, curr_lat, 'X', 20, 'r', 'r', 'Lost track')]
#         pcm2 = plot_mslp_panel(ax2, mslp_crop, title2, markers2)
#         ax2.legend(loc='upper right')
        
#         # PANEL 3: Overlay with both locations
#         ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
        
#         if diagnostic_info:
#             lat_diff, lon_diff = diagnostic_info['distance']
#             title3 = f"DISPLACEMENT\n{curr_datetime_str}\n"
#             title3 += f"Distance: Δlat={lat_diff:.1f}°, Δlon={lon_diff:.1f}°"
#         else:
#             title3 = f"OVERLAY\n{curr_datetime_str}"
        
#         markers3 = [(curr_lon, curr_lat, 'X', 20, 'r', 'r', 'Lost track')]
#         if parent_lat is not None:
#             markers3.append((parent_lon, parent_lat, '*', 20, 'k', 'yellow', 'Parent'))
        
#         pcm3 = plot_mslp_panel(ax3, mslp_crop, title3, markers3)
        
#         # Draw connecting line if both locations available
#         if parent_lat is not None:
#             ax3.plot([parent_lon, curr_lon], [parent_lat, curr_lat], 
#                     'r--', linewidth=2, transform=ccrs.PlateCarree(), zorder=4)
        
#         ax3.legend(loc='upper right')
        
#         # Add single colorbar for all panels
#         fig.subplots_adjust(right=0.92, wspace=0.3)
#         cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
#         _ = fig.colorbar(pcm2 if pcm2 else pcm3, cax=cbar_ax, label='MSLP (hPa)')
        
#         # Overall title
#         fig.suptitle(f'❌ LOST TRACK DIAGNOSTIC: {failure_name}', 
#                     fontsize=14, fontweight='bold', color='red', y=0.98)
        
#         # Save
#         fig_path = self.failed_dir / f"{failure_name}.png"
#         plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         return fig_path
    
#     def _load_parent_config(self, parent_config_name):
#         """Load parent config from disk."""
#         # Try shooting directory first
#         parent_path = self.shoot_dir / f'{parent_config_name}.pkl'
#         if parent_path.exists():
#             with open(parent_path, 'rb') as f:
#                 return pickle.load(f)
        
#         # Fall back to flux directory
#         parent_path = self.flux_dir / f'{parent_config_name}.pkl'
#         if parent_path.exists():
#             with open(parent_path, 'rb') as f:
#                 return pickle.load(f)
        
#         return None

#     def _filter_crossings(self, crossings: List[InterfaceConfig], 
#                          mode: str, 
#                          ic_has_preexisting: bool,
#                          parent_config_name: Optional[str]) -> List[InterfaceConfig]:
#         """
#         Filter crossings based on mode and tracking criteria.
#         Override in subclass for tracking-based filtering.
#         """
#         if mode == 'flux':
#             # Base class: no filtering
#             return crossings
#         elif mode == 'shoot':
#             # Base class: no filtering
#             return crossings
#         return crossings

#     def rollout_with_monitoring(self, data_loader, ensemble_size: int = 1,
#                             start_interface: int = -1,
#                             capture_crossings: bool = True,
#                             initial_state_override: Optional[torch.Tensor] = None,
#                             mode: str = 'flux',
#                             parent_config_name: Optional[str] = None) -> Dict:
#         """
#         Single trajectory rollout with interface monitoring.
#         """
#         # Track location from PREVIOUS timestep for jump detection
#         previous_location = None
#         if mode == 'shoot' and parent_config_name:
#             parent_config = self._load_parent_config(parent_config_name)
#             if parent_config and parent_config.feature_location:
#                 previous_location = parent_config.feature_location  # Start with parent
        
#         trajectory_mslp = []
#         trajectory_features = []
#         crossings = []  # Collect in memory first
#         current_interface = start_interface
#         trajectory_status = 'ongoing'
#         can_save_crossing = True
#         saved_states = {}
        
#         # Track if IC has pre-existing system
#         ic_has_preexisting_system = False
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(data_loader):
#                 forecast_step = batch["forecast_step"].item()
                
#                 # Initial input processing
#                 if forecast_step == 1:
#                     if initial_state_override is not None:
#                         x = initial_state_override.to(self.device).float()
#                     else:
#                         if "x_surf" in batch:
#                             x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device).float()
#                         else:
#                             x = reshape_only(batch["x"]).to(self.device).float()
                        
#                         if ensemble_size > 1:
#                             x = torch.repeat_interleave(x, ensemble_size, 0)
                
#                 # Add forcing and static
#                 if "x_forcing_static" in batch:
#                     x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4).float()
#                     if ensemble_size > 1:
#                         x_forcing_batch = torch.repeat_interleave(x_forcing_batch, ensemble_size, 0)
#                     x = torch.cat((x, x_forcing_batch), dim=1)
                
#                 # Model prediction
#                 y_pred = self.model(x, forecast_step=forecast_step - 1)
#                 y_pred_phys = self.state_transformer.inverse_transform(y_pred.cpu())
#                 y_pred_phys_with_mslp = self.calculate_mslp_wrapper(y_pred_phys, batch)
                
#                 # Extract MSLP - compare to PREVIOUS timestep location
#                 mslp, feature_idx, location = self.extract_mslp(
#                     y_pred_phys_with_mslp, batch, forecast_step, previous_location
#                 )
#                 trajectory_mslp.append(mslp)
#                 trajectory_features.append((forecast_step, feature_idx, location))
                
#                 # CHECK FOR LOCATION JUMP (only in shoot mode with tracking)
#                 if mode == 'shoot' and previous_location and location:
#                     prev_lat, prev_lon = previous_location
#                     curr_lat, curr_lon = location
#                     lat_diff = abs(curr_lat - prev_lat)
#                     lon_diff = abs(curr_lon - prev_lon)
                    
#                     # If jumped >10° in one timestep → lost track!
#                     if lat_diff > 10 or lon_diff > 10:
#                         trajectory_status = 'lost_track'
#                         print(f" LOST TRACK (Δlat={lat_diff:.1f}°, Δlon={lon_diff:.1f}° in 1 timestep)")
#                         break
                
#                 # UPDATE previous location for next timestep
#                 if location:
#                     previous_location = location
                
#                 # CHECK IC FOR PRE-EXISTING SYSTEM (only during flux generation)
#                 if forecast_step == 1 and mode == 'flux':
#                     if mslp < self.interfaces[0]:  # λ₀
#                         ic_has_preexisting_system = True
                
#                 # Normalize for saving
#                 y_pred_norm = self.state_transformer.transform_array(y_pred_phys).to(self.device)
                
#                 # Save state
#                 if capture_crossings:
#                     if batch.get("y_diag") is not None:
#                         varnum_diag = batch["y_diag"].shape[1]
#                         saved_states[forecast_step] = y_pred_norm[:, :-varnum_diag, ...].cpu().clone()
#                     else:
#                         saved_states[forecast_step] = y_pred_norm.cpu().clone()
                
#                 # Check trajectory status
#                 status, interface_idx = self.check_trajectory_status(mslp, current_interface, mode)
                
#                 if status == 'crossed_forward':
#                     if mode == 'flux':
#                         current_interface = interface_idx
                        
#                         # Save λ₀ crossings if decorrelated
#                         if interface_idx == 0 and can_save_crossing:
#                             config_name = self._generate_config_name(interface_idx, mode='flux')
                            
#                             # Create crossing WITHOUT saving yet
#                             crossing = InterfaceConfig(
#                                 input_state=saved_states[forecast_step],
#                                 latents=None,
#                                 forecast_step=forecast_step,
#                                 mslp_value=mslp,
#                                 interface_idx=interface_idx,
#                                 timestamp=datetime.now().isoformat(),
#                                 restart_datetime=self._compute_restart_datetime(batch),
#                                 config_name=config_name,
#                                 parent_config=parent_config_name,
#                                 track_id=None,
#                                 feature_location=location
#                             )
                            
#                             # Store metadata for later saving - COPY datetime
#                             crossing._feature_idx = feature_idx
#                             crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
#                             crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                            
#                             crossings.append(crossing)
#                             can_save_crossing = False
                    
#                     elif mode == 'shoot':
#                         # CRITICAL: Save at NEXT interface (current + 1), not furthest crossed
#                         # This enables instant success when shooting from the saved interface
#                         next_interface = current_interface + 1
#                         config_name = self._generate_config_name(next_interface, mode='shoot')
                        
#                         crossing = InterfaceConfig(
#                             input_state=saved_states[forecast_step],
#                             latents=None,
#                             forecast_step=forecast_step,
#                             mslp_value=mslp,
#                             interface_idx=next_interface,  # <-- FIXED: save at target interface
#                             timestamp=datetime.now().isoformat(),
#                             restart_datetime=self._compute_restart_datetime(batch),
#                             config_name=config_name,
#                             parent_config=parent_config_name,
#                             track_id=None,
#                             feature_location=location
#                         )
                        
#                         crossing._feature_idx = feature_idx
#                         crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
#                         crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                        
#                         crossings.append(crossing)
#                         trajectory_status = 'success'
#                         break
                
#                 elif status == 'reached_B':
#                     config_name = self._generate_config_name(interface_idx=-1, mode='stateB')
                    
#                     crossing = InterfaceConfig(
#                         input_state=saved_states[forecast_step],
#                         latents=None,
#                         forecast_step=forecast_step,
#                         mslp_value=mslp,
#                         interface_idx=-1,
#                         timestamp=datetime.now().isoformat(),
#                         restart_datetime=self._compute_restart_datetime(batch),
#                         config_name=config_name,
#                         parent_config=parent_config_name,
#                         track_id=None,
#                         feature_location=location
#                     )
                    
#                     crossing._feature_idx = feature_idx
#                     crossing._datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
#                     crossing._y_phys = y_pred_phys_with_mslp.cpu().clone()
                    
#                     crossings.append(crossing)
#                     trajectory_status = 'reached_B'
#                     print(f" → STATE B ({mslp:.1f} hPa)")
#                     break
                
#                 elif status == 'returned_A':
#                     trajectory_status = 'failure'
#                     print(f" → STATE A ({mslp:.1f} hPa)")
#                     break
                
#                 elif status == 'returned_backward':
#                     # FLUX MODE: check decorrelation condition
#                     if mode == 'flux':
#                         if self.decorrelation_interface is not None:
#                             # Use λ₋₁ for decorrelation
#                             if mslp >= self.decorrelation_interface:
#                                 can_save_crossing = True
#                         else:
#                             # Require return to state A
#                             if mslp >= self.state_A_threshold:
#                                 can_save_crossing = True
                    
#                     current_interface = interface_idx if interface_idx is not None else -1
                
#                 # Prepare next step
#                 if batch.get("y_diag") is not None:
#                     varnum_diag = batch["y_diag"].shape[1]
#                     x = y_pred_norm[:, :-varnum_diag, ...].detach()
#                 else:
#                     x = y_pred_norm.detach()
                
#                 if batch.get("stop_forecast", torch.tensor(False)).item():
#                     if trajectory_status == 'ongoing':
#                         trajectory_status = 'completed' if mode == 'flux' else 'failure'
#                     break
        
#         # Filter crossings (subclass can override)
#         valid_crossings = self._filter_crossings(
#             crossings, mode, ic_has_preexisting_system, parent_config_name
#         )
        
#         # Handle lost_track status - save diagnostic figure
#         if trajectory_status == 'lost_track' and len(crossings) == 0:
#             # No crossing was saved before losing track
#             # Create diagnostic with last known location
#             if len(trajectory_features) > 0:
#                 last_step, last_feature_idx, last_location = trajectory_features[-1]
                
#                 parent_config = self._load_parent_config(parent_config_name) if parent_config_name else None
#                 diagnostic_info = None
                
#                 if parent_config and parent_config.feature_location and last_location:
#                     parent_lat, parent_lon = parent_config.feature_location
#                     last_lat, last_lon = last_location
#                     lat_diff = abs(parent_lat - last_lat)
#                     lon_diff = abs(parent_lon - last_lon)
#                     diagnostic_info = {
#                         'parent_location': (parent_lat, parent_lon),
#                         'crossing_location': (last_lat, last_lon),
#                         'distance': (lat_diff, lon_diff)
#                     }
                
#                 failure_name = f"LOST_TRACK_step{last_step}"
                
#                 # Need to get the y_phys for the last timestep - we don't have it saved
#                 # So we'll skip the figure for now - just note the lost track
#                 print(" (diagnostic figure skipped - no crossing saved)")
        
#         if mode == 'shoot' and len(crossings) > 0 and len(valid_crossings) == 0:
#             print(" LOST TRACK (filtered out)")
            
#             # Save diagnostic figure for lost track
#             for crossing in crossings:
#                 failure_name = f"LOST_TRACK_{crossing.config_name}"
                
#                 # Add diagnostic info to figure
#                 parent_config = self._load_parent_config(parent_config_name) if parent_config_name else None
#                 diagnostic_info = None
#                 if parent_config and parent_config.feature_location and crossing.feature_location:
#                     parent_lat, parent_lon = parent_config.feature_location
#                     cross_lat, cross_lon = crossing.feature_location
#                     lat_diff = abs(parent_lat - cross_lat)
#                     lon_diff = abs(parent_lon - cross_lon)
#                     diagnostic_info = {
#                         'parent_location': (parent_lat, parent_lon),
#                         'crossing_location': (cross_lat, cross_lon),
#                         'distance': (lat_diff, lon_diff)
#                     }
                
#                 self._save_failure_figure(
#                     crossing._y_phys,
#                     failure_name=failure_name,
#                     datetime_obj=crossing._datetime_obj,
#                     feature_idx=crossing._feature_idx,
#                     location=crossing.feature_location,
#                     parent_config=parent_config,
#                     diagnostic_info=diagnostic_info
#                 )
        
#         # Save valid crossings to disk
#         for crossing in valid_crossings:
#             # Determine save directory
#             if crossing.interface_idx == -1:
#                 save_dir = self.stateB_dir
#                 mode_str = 'stateB'
#             elif mode == 'flux':
#                 save_dir = self.flux_dir
#                 mode_str = 'flux'
#             else:
#                 save_dir = self.shoot_dir
#                 mode_str = 'shoot'
            
#             # Save figure AND config
#             self._save_mslp_figure(
#                 crossing._y_phys,
#                 config_name=crossing.config_name,
#                 save_dir=save_dir,
#                 datetime_obj=crossing._datetime_obj,
#                 feature_idx=crossing._feature_idx,
#                 location=crossing.feature_location
#             )
            
#             self._save_config(crossing, mode=mode_str)
#             print(f"  → SAVED: {crossing.config_name} (MSLP: {crossing.mslp_value:.1f} hPa)")
        
#         return {
#             'status': trajectory_status,
#             'mslp_trajectory': trajectory_mslp,
#             'crossings': valid_crossings,
#             'final_mslp': trajectory_mslp[-1] if trajectory_mslp else None,
#             'ic_had_preexisting': ic_has_preexisting_system if mode == 'flux' else False
#         }

#     def generate_flux_at_lambda0(self, initial_data_loader, n_trials: int = 100):
#         """Phase 0: Flux generation - collect decorrelated crossings at λ₀."""
#         print(f"\n{'='*70}")
#         print(f"PHASE 0: FLUX GENERATION at λ₀ = {self.interfaces[0]} hPa")
#         if self.decorrelation_interface is not None:
#             print(f"Decorrelation interface: λ₋₁ = {self.decorrelation_interface} hPa")
#         else:
#             print(f"Decorrelation: Return to state A (MSLP > {self.state_A_threshold} hPa)")
#         print(f"Target: {n_trials} decorrelated crossings")
#         print(f"{'='*70}\n")
        
#         trajectory_count = 0
#         total_time_days = 0.0
#         worker_crossings = 0
#         direct_B_formations = 0  # Count direct A→B events
        
#         while True:
#             # Check GLOBAL count
#             global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
            
#             if global_count >= n_trials:
#                 print(f"✓ Target reached: {global_count}/{n_trials}, stopping")
#                 break
            
#             trajectory_count += 1
#             print(f"Trajectory {trajectory_count} (global: {global_count}/{n_trials}): ", end='')
            
#             data_loader = copy.deepcopy(initial_data_loader)
            
#             result = self.rollout_with_monitoring(
#                 data_loader,
#                 start_interface=-1,
#                 capture_crossings=True,
#                 mode='flux'
#             )
            
#             configs_from_this_traj = len(result['crossings'])
#             worker_crossings += configs_from_this_traj
            
#             # Track direct B formations (reached B without crossing λ₀)
#             if result['status'] == 'reached_B' and configs_from_this_traj == 0:
#                 direct_B_formations += 1
            
#             # Get actual config names
#             config_names = [c.config_name for c in result['crossings']]
            
#             # Log flux trajectory
#             self.logger.log_flux_trajectory(trajectory_count, result, config_names)
            
#             if configs_from_this_traj > 0:
#                 print(f" ({configs_from_this_traj} crossings)")
#             elif result['status'] == 'reached_B':
#                 print(" (direct B formation)")
#             else:
#                 print(" (no crossings)")
            
#             num_timesteps = len(result['mslp_trajectory'])
#             elapsed_days = num_timesteps * 6.0 / 24.0
#             total_time_days += elapsed_days
        
#         # Calculate FFS flux rate (λ₀ crossings)
#         self.flux_estimate = worker_crossings / total_time_days if total_time_days > 0 else 0.0
        
#         # Calculate direct formation rate (brute force estimate)
#         self.direct_B_count = direct_B_formations
#         self.direct_B_rate = direct_B_formations / total_time_days if total_time_days > 0 else 0.0
        
#         final_global_count = len(list(self.flux_dir.glob('lambda0_config_*.pkl')))
        
#         print(f"\n{'='*70}")
#         print("✓ FLUX GENERATION COMPLETE")
#         print(f"{'='*70}")
#         print(f"Trajectories run: {trajectory_count}")
#         print(f"Worker λ₀ crossings: {worker_crossings}")
#         print(f"Worker direct B formations: {direct_B_formations}")
#         print(f"Total time: {total_time_days:.1f} days")
#         print("\nFFS Flux Rate:")
#         print(f"  Φ₀ = {worker_crossings}/{total_time_days:.1f} days = {self.flux_estimate:.6f} crossings/day")
#         print("\nDirect Formation Rate (brute force):")
#         print(f"  Φ_direct = {direct_B_formations}/{total_time_days:.1f} days = {self.direct_B_rate:.6e} formations/day")
#         print(f"\nGlobal configs at λ₀: {final_global_count}")
#         print(f"{'='*70}\n")
    
#     def shoot_single_trajectory(self, config: InterfaceConfig, 
#                            current_interface: int) -> Dict:
#         """Shoot single trajectory from saved config.
        
#         Args:
#             current_interface: The TARGET interface we're shooting TO
#             config: Config FROM the previous interface
#         """
#         # Check if config already satisfies the TARGET interface
#         if config.mslp_value < self.interfaces[current_interface]:
#             # Already at target - immediate success
#             print(f"   Config {config.config_name} already at {config.mslp_value:.1f} hPa < λ_{current_interface} ({self.interfaces[current_interface]} hPa) - INSTANT SUCCESS")
            
#             # Create new config AT TARGET interface (not next!)
#             new_config_name = self._generate_config_name(current_interface, mode='shoot')
#             new_config = InterfaceConfig(
#                 input_state=config.input_state,
#                 latents=config.latents,
#                 forecast_step=config.forecast_step,
#                 mslp_value=config.mslp_value,
#                 interface_idx=current_interface,  # Save at TARGET
#                 timestamp=datetime.now().isoformat(),
#                 restart_datetime=config.restart_datetime,
#                 config_name=new_config_name,
#                 parent_config=config.config_name,
#                 track_id=config.track_id,
#                 feature_location=config.feature_location
#             )
            
#             # Save figure and config
#             if hasattr(config, '_y_phys'):
#                 self._save_mslp_figure(
#                     config._y_phys,
#                     config_name=new_config_name,
#                     save_dir=self.shoot_dir,
#                     datetime_obj=config._datetime_obj if hasattr(config, '_datetime_obj') else None,
#                     feature_idx=config._feature_idx if hasattr(config, '_feature_idx') else None,
#                     location=config.feature_location
#                 )
            
#             self._save_config(new_config, mode='shoot')

#             # Log instant success
#             self.logger.log_instant_success(
#                 current_interface,  # Log at TARGET
#                 config.config_name, 
#                 new_config_name, 
#                 config.mslp_value
#             )
            
#             return {
#                 'status': 'success',
#                 'mslp_trajectory': [config.mslp_value],
#                 'crossings': [new_config],
#                 'final_mslp': config.mslp_value
#             }
        
#         restart_time = config.restart_datetime
#         forecast_times = [[
#             (restart_time + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
#             (restart_time + timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
#         ]]
        
#         restart_dataset = Predict_Dataset_Batcher(
#             **self.dataset_params,
#             fcst_datetime=forecast_times
#         )
#         restart_loader = BatchForecastLenDataLoader(restart_dataset)
        
#         print(f"   Using config: {config.config_name} (MSLP: {config.mslp_value:.1f} hPa, {config.restart_datetime})", end='')

#         # Configs are AT interface_idx, but check_trajectory_status expects current_interface 
#         # to be BELOW the interface we're at. So use interface_idx - 1.
#         result = self.rollout_with_monitoring(
#             restart_loader,
#             start_interface=current_interface - 1,
#             capture_crossings=True,
#             initial_state_override=config.input_state,
#             mode='shoot',
#             parent_config_name=config.config_name
#         )
        
#         return result
    
#     def shoot_from_interface(self, interface_idx: int, n_trials: int = 100):
#         """Shoot trajectories from interface λᵢ until GLOBAL target reached."""
#         if len(self.interface_configs[interface_idx]) == 0:
#             raise ValueError(f"No configs available for interface {interface_idx}")
        
#         # Match old convention for display: interface_idx=1 → lambda_label=0
#         lambda_label = interface_idx - 1
#         next_interface_idx = interface_idx + 1
#         next_lambda_label = next_interface_idx - 1
        
#         if self.rank == 0 and self.worker_id == 0:
#             print(f"\n{'='*70}")
#             print(f"SHOOTING from λ_{lambda_label} = {self.interfaces[interface_idx]} hPa")
#             print(f"Available configs: {len(self.interface_configs[interface_idx])}")
#             print(f"Target configs: {n_trials}")
#             print(f"{'='*70}\n")
        
#         successes = 0
#         failures = 0
#         total_attempts = 0
        
#         # Count configs at NEXT interface before starting
#         initial_count = len(list(self.shoot_dir.glob(f'lambda{next_lambda_label}_config_*.pkl')))
        
#         while True:
#             # Check GLOBAL count - FIX BUG 1
#             global_count = len(list(self.shoot_dir.glob(f'lambda{next_lambda_label}_config_*.pkl')))
            
#             if global_count >= n_trials:
#                 print(f"\n✓ Target reached: {global_count}/{n_trials} configs exist, stopping")
#                 break
            
#             total_attempts += 1
#             config = np.random.choice(self.interface_configs[interface_idx])
            
#             print(f"Attempt {total_attempts} (global: {global_count}/{n_trials}):", end=' ')
            
#             result = self.shoot_single_trajectory(config, interface_idx)

#             child_config = None
#             if result['crossings']:
#                 child_config = result['crossings'][0].config_name
            
#             self.logger.log_shooting_attempt(
#                 interface_idx, 
#                 total_attempts, 
#                 config.config_name, 
#                 result, 
#                 child_config
#             )
            
#             if result['status'] == 'success' or result['status'] == 'reached_B':
#                 successes += 1

#                 # Check AGAIN after saving - FIX BUG 1
#                 global_count = len(list(self.shoot_dir.glob(f'lambda{next_lambda_label}_config_*.pkl')))
#                 if global_count >= n_trials:
#                     print(f"✓ Target reached: {global_count}/{n_trials}, stopping")
#                     break
#             elif result['status'] == 'failure':
#                 failures += 1
        
#         P_forward = successes / total_attempts if total_attempts > 0 else 0.0
#         self.transition_probs.append(P_forward)

#         # Final count from disk - FIX BUG 1
#         final_count = len(list(self.shoot_dir.glob(f'lambda{next_lambda_label}_config_*.pkl')))
#         configs_saved = final_count - initial_count

#         self.logger.log_interface_summary(
#             interface_idx, 
#             successes, 
#             failures, 
#             total_attempts, 
#             P_forward, 
#             configs_saved
#         )
        
#         print(f"\n{'='*70}")
#         print(f"✓ SHOOTING from λ_{lambda_label} COMPLETE")
#         print(f"{'='*70}")
#         print(f"Total attempts: {total_attempts}")
#         print(f"Successes: {successes}")
#         print(f"Failures: {failures}")
#         print(f"P(λ_{lambda_label} → λ_{lambda_label + 1}) = {successes}/{total_attempts} = {P_forward:.4f}")
#         print(f"Configs saved at λ_{lambda_label + 1}: {configs_saved}")
#         print(f"{'='*80}\n")
        
#         # Load the actual saved configs from disk for next interface - FIX BUG 3
#         if next_interface_idx < len(self.interfaces):
#             saved_config_files = list(self.shoot_dir.glob(f'lambda{next_lambda_label}_config_*.pkl'))
            
#             # Build set of existing names to avoid duplicates
#             existing_names = {c.config_name for c in self.interface_configs[next_interface_idx]}
            
#             # Load each config file
#             for config_file in saved_config_files:
#                 with open(config_file, 'rb') as f:
#                     config = pickle.load(f)
#                     if config.config_name not in existing_names:
#                         self.interface_configs[next_interface_idx].append(config)
#                         existing_names.add(config.config_name)
        
#         return P_forward
    
#     def run_ffs(self, initial_data_loader, n_flux_trials: int = 10, n_shoot_trials: int = 10):
#         """
#         Full FFS algorithm:
#         1. Flux generation at λ₀
#         2. Shoot from each interface
#         3. Estimate total P(IC → B)
#         """
#         # Phase 0: Flux generation at λ₀ (interfaces[0])
#         self.generate_flux_at_lambda0(initial_data_loader, n_trials=n_flux_trials)
        
#         # Phase 1-N: Shoot from λ₀, λ₁, λ₂, ...
#         for i in range(0, len(self.interfaces) - 1):
#             self.shoot_from_interface(i, n_trials=n_shoot_trials)
        
#         # Compute FFS total probability
#         ffs_total_prob = self.flux_estimate * np.prod(self.transition_probs)
        
#         print(f"\n{'='*70}")
#         print("FINAL FFS RESULTS")
#         print(f"{'='*70}")
#         print("\nFFS Enhanced Estimate:")
#         print(f"  Φ₀ (flux at λ₀ = {self.interfaces[0]} hPa) = {self.flux_estimate:.6f} crossings/day")
#         for i, p in enumerate(self.transition_probs):
#             print(f"  P(λ_{i} → λ_{i+1}) = {p:.4f}")
#         print("\n  P_FFS(IC → Hurricane) = Φ₀ × Π P(λᵢ→λᵢ₊₁)")
#         print(f"  P_FFS(IC → Hurricane) = {ffs_total_prob:.2e} per day")
#         if ffs_total_prob > 0:
#             print(f"  Enhancement factor: {1/ffs_total_prob:.1f}x over direct sampling")
        
#         print("\nDirect Formation Rate (brute force baseline):")
#         print(f"  Direct B formations: {self.direct_B_count}")
#         print(f"  P_direct(IC → Hurricane) = {self.direct_B_rate:.2e} per day")
        
#         if self.direct_B_rate > 0 and ffs_total_prob > 0:
#             ratio = ffs_total_prob / self.direct_B_rate
#             print(f"  FFS/Direct ratio: {ratio:.2f}")
        
#         print(f"{'='*70}\n")

#         # Log final results
#         self.logger.log_final_results(
#             self.flux_estimate, 
#             self.transition_probs, 
#             ffs_total_prob,
#             self.direct_B_count,
#             self.direct_B_rate
#         )
    
#         return ffs_total_prob


# class HurricaneGenesisFFS_Tracked(HurricaneGenesisFFS):
#     """
#     Extended FFS class with feature tracking to distinguish new genesis from pre-existing systems.
    
#     Uses tobac's tracking capabilities to link features across timesteps and filter out
#     systems that existed at initial conditions.
#     """
    
#     def __init__(self, *args, **kwargs):
#         """Initialize with tracking state variables."""
#         super().__init__(*args, **kwargs)
        
#         # Tracking state
#         self.feature_history = []  # Store all detected features across trajectory
#         self.tracks = None  # Linked tracks from tobac
#         self.initial_track_ids = set()  # Track IDs present at IC (t=0)
#         self.feature_id_map = {}  # Map forecast_step -> feature_idx -> track_id
        
#         # Tracking parameters
#         self.tracking_params = {
#             'v_max': 50,  # Maximum velocity between frames (km/h)
#             'dt': 6 * 3600,  # Time step in seconds (6 hours)
#             'dxy': 111000,  # Spatial resolution (111 km per degree)
#             'memory': 2,  # Allow 2 timestep gaps in tracking
#             'stubs': 1  # Minimum detections for valid track
#         }
    
#     def enable_visualization(self):
#         """Enable real-time MSLP visualization during rollout."""
#         self.visualize_mslp = True
#         print("✓ MSLP visualization enabled")
    
#     def disable_visualization(self):
#         """Disable MSLP visualization and close plot."""
#         self.visualize_mslp = False
#         if self.fig is not None:
#             plt.close(self.fig)
#             self.fig = None
#             self.ax = None
#             self._colorbar_added = False
#         print("✓ MSLP visualization disabled")

#     def extract_mslp(self, y_phys: torch.Tensor, batch: Dict = None,
#                      forecast_step: int = None,
#                      parent_location: Optional[Tuple[float, float]] = None) -> Tuple:
#         """
#         Extract MSLP with feature detection and tracking metadata.
        
#         Returns:
#             Tuple of (min_mslp, feature_position, (lat, lon))
#         """
#         try:
#             import tobac
#             import iris
#         except ImportError:
#             mslp = self._extract_mslp_fallback(y_phys, batch)
#             return mslp, None, None
        
#         mslp_channel_idx = 71
#         mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         # Get lat/lon
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)
        
#         # Basin mask
#         lat_mask = (lats >= self.basin['lat_min']) & (lats <= self.basin['lat_max'])
#         lon_mask = (lons_180 >= self.basin['lon_min']) & (lons_180 <= self.basin['lon_max'])
#         basin_mask = lat_mask[:, None] & lon_mask[None, :]
        
#         mslp_basin = np.where(basin_mask, mslp_hpa, np.nan)
        
#         # Sort for tobac
#         lons_tobac = lons_180.copy()
#         mslp_tobac = mslp_basin.copy()
        
#         if not np.all(np.diff(lons_tobac) > 0):
#             sort_idx = np.argsort(lons_tobac)
#             lons_tobac = lons_tobac[sort_idx]
#             mslp_tobac = mslp_tobac[:, sort_idx]
        
#         # Create iris cube
#         try:
#             lat_coord = iris.coords.DimCoord(lats, standard_name='latitude', units='degrees')
#             lon_coord = iris.coords.DimCoord(lons_tobac, standard_name='longitude', units='degrees')
#             time_coord = iris.coords.DimCoord([forecast_step], standard_name='time', units='hours since 2000-01-01 00:00:00')
            
#             cube = iris.cube.Cube(
#                 mslp_tobac[np.newaxis, :, :],
#                 standard_name='air_pressure_at_mean_sea_level',
#                 units='hPa',
#                 dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)]
#             )
            
#             basin_min = np.nanmin(mslp_tobac)
#             basin_max = np.nanmax(mslp_tobac)
#             thresholds = np.arange(max(basin_min - 5, 950), min(basin_max + 5, 1020), 2)
#             thresholds = sorted(thresholds, reverse=False)
        
#             features = tobac.feature_detection_multithreshold(
#                 field_in=cube,
#                 dxy=self.tracking_params['dxy'],
#                 threshold=thresholds,
#                 target='minimum',
#                 position_threshold='weighted_diff',
#                 sigma_threshold=1.5,
#                 n_min_threshold=3
#             )
            
#             if features is not None and len(features) > 0:
#                 # If parent location provided, prioritize nearby features
#                 if parent_location is not None:
#                     parent_lat, parent_lon = parent_location
                    
#                     # Calculate distance for each feature
#                     features['lat_diff'] = abs(features['latitude'] - parent_lat)
#                     features['lon_diff'] = abs(features['longitude'] - parent_lon)
                    
#                     # Filter to features within 10 degrees
#                     nearby = features[(features['lat_diff'] < 10) & (features['lon_diff'] < 10)]
                    
#                     if len(nearby) > 0:
#                         features_to_use = nearby
#                     else:
#                         features_to_use = features
#                 else:
#                     features_to_use = features
                
#                 # Get strongest feature
#                 strongest_idx = features_to_use['threshold_value'].idxmin()
#                 strongest = features_to_use.loc[strongest_idx]
                
#                 # Get position in ORIGINAL features list
#                 feature_position = features.index.get_loc(strongest_idx)
                
#                 # Get feature location
#                 feature_lat = float(strongest['latitude'])
#                 feature_lon = float(strongest['longitude'])
                
#                 # Extract ACTUAL MSLP at this location
#                 lat_idx = np.argmin(np.abs(lats - feature_lat))
#                 lon_idx = np.argmin(np.abs(lons_180 - feature_lon))
                
#                 min_mslp = float(mslp_basin[lat_idx, lon_idx])
                
#                 # Add to history for tracking
#                 features['forecast_step'] = forecast_step
#                 self.feature_history.append(features.copy())
                
#                 # VISUALIZATION
#                 if self.visualize_mslp:
#                     datetime_str = None
#                     if batch is not None and "datetime" in batch:
#                         try:
#                             datetime_obj = datetime.fromtimestamp(batch["datetime"][0].item())
#                             datetime_str = datetime_obj.strftime('%Y-%m-%d %H:%M UTC')
#                         except Exception:
#                             pass
                    
#                     mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
#                     self._plot_mslp(
#                         mslp_field=mslp_hpa,
#                         mslp_smooth=mslp_smooth,
#                         basin_mask=basin_mask,
#                         datetime_str=datetime_str,
#                         center_lat=feature_lat,
#                         center_lon=feature_lon,
#                         center_val=min_mslp
#                     )
                
#                 return min_mslp, feature_position, (feature_lat, feature_lon)
#             else:
#                 mslp = self._extract_mslp_fallback(y_phys, batch)
#                 return mslp, None, None
        
#         except Exception as e:
#             print(f"\n⚠ tobac error: {e}, using fallback")
#             mslp = self._extract_mslp_fallback(y_phys, batch)
#             return mslp, None, None
    
#     def _extract_mslp_fallback(self, y_phys: torch.Tensor, batch: Dict = None) -> float:
#         """Fallback MSLP extraction using simple minimum."""
#         mslp_channel_idx = 71
#         mslp_pa = y_phys[0, mslp_channel_idx, 0, :, :].cpu().numpy()
#         mslp_hpa = mslp_pa / 100.0
        
#         basin_mask = self.get_basin_mask()
        
#         # Smooth for location, raw for value
#         mslp_smooth = gaussian_filter(mslp_hpa, sigma=1.5)
#         mslp_smooth_masked = np.where(basin_mask, mslp_smooth, np.nan)
        
#         min_idx = np.nanargmin(mslp_smooth_masked)
#         min_row, min_col = np.unravel_index(min_idx, mslp_smooth_masked.shape)
#         min_mslp = float(mslp_hpa[min_row, min_col])
        
#         if self.visualize_mslp:
#             datetime_str = None
#             if batch is not None and "datetime" in batch:
#                 try:
#                     dt = datetime.fromtimestamp(batch["datetime"][0].item())
#                     datetime_str = dt.strftime('%Y-%m-%d %H:%M UTC')
#                 except Exception:
#                     pass
        
#             min_lat = self.latlons.latitude.values[min_row]
#             lons_180 = np.where(self.latlons.longitude.values > 180,
#                                 self.latlons.longitude.values - 360,
#                                 self.latlons.longitude.values)
#             min_lon = lons_180[min_col]
        
#             self._plot_mslp(
#                 mslp_field=mslp_hpa,
#                 mslp_smooth=mslp_smooth,
#                 basin_mask=basin_mask,
#                 datetime_str=datetime_str,
#                 center_lat=float(min_lat),
#                 center_lon=float(min_lon),
#                 center_val=min_mslp
#             )

#         return min_mslp
    
#     def _plot_mslp(self, mslp_field: np.ndarray, mslp_smooth: np.ndarray,
#                    basin_mask: np.ndarray, datetime_str: str = None,
#                    center_lat: float = None, center_lon: float = None,
#                    center_val: float = None):
#         """Plot MSLP field centered on the storm position."""
#         from IPython.display import display, clear_output
    
#         # init fig/ax
#         if self.fig is None:
#             plt.ion()
#             self.fig = plt.figure(figsize=(14, 10))
#             self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
#         self.ax.clear()
    
#         # coords
#         lats = self.latlons.latitude.values
#         lons = self.latlons.longitude.values
#         lons_180 = np.where(lons > 180, lons - 360, lons)
    
#         # convert center lat/lon to row/col
#         # row_idx = np.argmin(np.abs(lats - center_lat))
#         # col_idx = np.argmin(np.abs(lons_180 - center_lon))
    
#         # basin crop logic
#         basin_rows, basin_cols = np.where(basin_mask)
#         if len(basin_rows) > 0:
#             row_min = max(0, basin_rows.min() - 10)
#             row_max = min(mslp_smooth.shape[0], basin_rows.max() + 10)
#             col_min = max(0, basin_cols.min() - 10)
#             col_max = min(mslp_smooth.shape[1], basin_cols.max() + 10)
#         else:
#             row_min, row_max = 0, mslp_smooth.shape[0]
#             col_min, col_max = 0, mslp_smooth.shape[1]
    
#         # crop arrays
#         mslp_crop = mslp_smooth[row_min:row_max, col_min:col_max]
#         lat_crop = lats[row_min:row_max]
#         lon_crop = lons_180[col_min:col_max]
    
#         # set map bounds
#         self.ax.set_extent(
#             [lon_crop.min(), lon_crop.max(), lat_crop.min(), lat_crop.max()],
#             crs=ccrs.PlateCarree()
#         )
    
#         # filled contours
#         levels = np.arange(960, 1030, 4)
#         pcm = self.ax.contourf(
#             lon_crop, lat_crop, mslp_crop,
#             levels=levels, cmap='RdBu_r', extend='both',
#             transform=ccrs.PlateCarree(), zorder=1
#         )
    
#         # contour lines
#         cs = self.ax.contour(
#             lon_crop, lat_crop, mslp_crop,
#             levels=np.arange(960, 1030, 8),
#             colors='k', linewidths=0.5,
#             transform=ccrs.PlateCarree(), zorder=4
#         )
#         self.ax.clabel(cs, inline=True, fontsize=8)
    
#         # geo features
#         self.ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.0)
#         self.ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5, alpha=0.6)
#         self.ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, alpha=0.5)
    
#         # plot storm center
#         self.ax.plot(
#             center_lon, center_lat,
#             marker='*', markersize=20,
#             markeredgecolor='yellow', markeredgewidth=2,
#             color='k', transform=ccrs.PlateCarree(), zorder=10
#         )
    
#         # gridlines
#         gl = self.ax.gridlines(draw_labels=True, linewidth=0.6, alpha=0.5, linestyle='--')
#         gl.top_labels = False
#         gl.right_labels = False
    
#         title = f"MSLP (hPa) - Min: {center_val:.1f} hPa @ ({center_lat:.1f}°, {center_lon:.1f}°)"
#         if datetime_str is not None:
#             title = f"{datetime_str}\n{title}"
    
#         self.ax.set_title(title, fontsize=14, fontweight="bold")
    
#         if not self._colorbar_added:
#             plt.colorbar(pcm, ax=self.ax, label="MSLP (hPa)", shrink=0.8)
#             self._colorbar_added = True
    
#         clear_output(wait=True)
#         display(self.fig)
#         plt.pause(0.01)
    
#     def perform_tracking(self):
#         """Link features across timesteps using tobac tracking."""
#         if len(self.feature_history) == 0:
#             return
        
#         try:
#             import tobac
#             import pandas as pd
#         except ImportError:
#             return
        
#         # Combine all features
#         all_features = pd.concat(self.feature_history, ignore_index=False)
        
#         if len(all_features) == 0:
#             return
        
#         # Add unique identifier before tracking
#         all_features = all_features.reset_index(drop=False)
#         all_features = all_features.rename(columns={'index': 'original_idx'})
        
#         # Perform tracking
#         self.tracks = tobac.linking_trackpy(
#             all_features,
#             field_in=None,
#             dt=self.tracking_params['dt'],
#             dxy=self.tracking_params['dxy'],
#             v_max=self.tracking_params['v_max'],
#             memory=self.tracking_params['memory'],
#             stubs=self.tracking_params['stubs']
#         )
        
#         # Identify tracks present at IC
#         initial_features = self.tracks[self.tracks['forecast_step'] == 1]
#         if len(initial_features) > 0:
#             self.initial_track_ids = set(initial_features['cell'].unique())
#         else:
#             self.initial_track_ids = set()
        
#         # Build mapping using ORIGINAL indices
#         for _, row in self.tracks.iterrows():
#             step = int(row['forecast_step'])
#             original_idx = int(row['original_idx'])
#             track_id = row['cell']
            
#             if step not in self.feature_id_map:
#                 self.feature_id_map[step] = {}
#             self.feature_id_map[step][original_idx] = track_id
        
#     def is_new_genesis(self, forecast_step: int, feature_idx: Optional[int]) -> bool:
#         """Check if feature represents NEW genesis (not present at IC)."""
#         if feature_idx is None:
#             return False
        
#         if self.tracks is None or len(self.feature_id_map) == 0:
#             return False
        
#         if forecast_step not in self.feature_id_map:
#             return False
        
#         if feature_idx not in self.feature_id_map[forecast_step]:
#             return False
        
#         track_id = self.feature_id_map[forecast_step][feature_idx]
#         is_new = track_id not in self.initial_track_ids
        
#         return is_new
    
#     def get_track_info(self, feature_idx: Optional[int], forecast_step: int) -> Dict:
#         """Get detailed tracking information for a feature."""
#         info = {
#             'tracked': False,
#             'track_id': None,
#             'is_new': False,
#             'track_length': 0,
#             'first_detection': None
#         }
        
#         if feature_idx is None or self.tracks is None:
#             return info
        
#         if forecast_step not in self.feature_id_map:
#             return info
        
#         if feature_idx not in self.feature_id_map[forecast_step]:
#             return info
        
#         track_id = self.feature_id_map[forecast_step][feature_idx]
#         track_data = self.tracks[self.tracks['cell'] == track_id]
        
#         info['tracked'] = True
#         info['track_id'] = track_id
#         info['is_new'] = track_id not in self.initial_track_ids
#         info['track_length'] = len(track_data)
#         info['first_detection'] = int(track_data['forecast_step'].min())
        
#         return info
    
#     def reset_tracking(self):
#         """Reset tracking state for a new trajectory."""
#         self.feature_history = []
#         self.tracks = None
#         self.initial_track_ids = set()
#         self.feature_id_map = {}

#     def _filter_crossings(self, crossings: List[InterfaceConfig], 
#                         mode: str, 
#                         ic_has_preexisting: bool,
#                         parent_config_name: Optional[str]) -> List[InterfaceConfig]:
#         """
#         Filter crossings based on tracking criteria.
#         Override base class method with tracking-specific logic.
        
#         NOTE: Location-based jump detection now happens DURING rollout,
#         so we only need to handle flux mode filtering here.
#         """
#         # Perform tracking to get track IDs
#         self.perform_tracking()
        
#         # Add track IDs to all crossings
#         for crossing in crossings:
#             feature_idx = crossing._feature_idx
#             forecast_step = crossing.forecast_step
            
#             track_info = self.get_track_info(feature_idx, forecast_step)
#             crossing.track_id = track_info['track_id'] if track_info['tracked'] else None
        
#         # FILTERING LOGIC
#         valid_crossings = []
        
#         if mode == 'flux':
#             # During flux: filter out trajectories with pre-existing systems
#             if ic_has_preexisting:
#                 pass  # Filter out all crossings
#             else:
#                 valid_crossings = crossings
        
#         elif mode == 'shoot':
#             # During shooting: jumps already detected during rollout
#             # Accept all crossings that made it here
#             valid_crossings = crossings
        
#         return valid_crossings


# # Keep the main execution block from original
# if __name__ == "__main__":
#     filepath = "/glade/derecho/scratch/schreck/CREDIT_runs/ensemble/scheduler/"
#     device = "cuda"

#     with open(os.path.join(filepath, "model.yml"), "r") as f:
#         conf = yaml.safe_load(f)

#     conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
#     data_config = setup_data_loading(conf)

#     ensemble_size = 1
#     conf["trainer"]["ensemble_size"] = ensemble_size
#     conf["predict"]["ensemble_size"] = ensemble_size

#     model = load_model(conf, load_weights=True).to("cuda")
#     model = model.eval()

#     forecast_times = [['2022-08-28 00:00:00', '2022-09-10 00:00:00']]

#     dataset_params = {
#         'varname_upper_air': data_config["varname_upper_air"],
#         'varname_surface': data_config["varname_surface"],
#         'varname_dyn_forcing': data_config["varname_dyn_forcing"],
#         'varname_forcing': data_config["varname_forcing"],
#         'varname_static': data_config["varname_static"],
#         'varname_diagnostic': data_config["varname_diagnostic"],
#         'filenames': data_config["all_ERA_files"],
#         'filename_surface': data_config["surface_files"],
#         'filename_dyn_forcing': data_config["dyn_forcing_files"],
#         'filename_forcing': data_config["forcing_files"],
#         'filename_static': data_config["static_files"],
#         'filename_diagnostic': data_config["diagnostic_files"],
#         'lead_time_periods': 6,
#         'history_len': data_config["history_len"],
#         'skip_periods': data_config["skip_periods"],
#         'transform': load_transforms(conf),
#         'sst_forcing': data_config["sst_forcing"],
#         'batch_size': 1,
#         'rank': 0,
#         'world_size': 1,
#     }

#     initial_dataset = Predict_Dataset_Batcher(
#         **dataset_params,
#         fcst_datetime=forecast_times
#     )

#     initial_loader = BatchForecastLenDataLoader(initial_dataset)

#     ffs = HurricaneGenesisFFS(
#         model=model,
#         state_transformer=Normalize_ERA5_and_Forcing(conf),
#         config=conf,
#         initial_dataset=initial_dataset,
#         device='cuda'
#     )

#     ffs.run_ffs(initial_loader, n_flux_trials=20, n_shoot_trials=10)