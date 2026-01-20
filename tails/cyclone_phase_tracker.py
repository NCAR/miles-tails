import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class StormState:
    """Storm state with MSLP + CPS."""
    location: Tuple[float, float]
    mslp: float
    B: float
    VTL: float
    VTU: float
    is_tropical: bool
    phase: str


class CyclonePhaseTracker:
    """Compute CPS parameters from pressure interpolation.
        https://moe.met.fsu.edu/cyclonephase/help.html
    """
    
    def __init__(self, latlons, radius_km=500):
        self.lats = latlons.latitude.values
        self.lons = np.where(
            latlons.longitude.values > 180,
            latlons.longitude.values - 360,
            latlons.longitude.values
        )
        self.lons_2d, self.lats_2d = np.meshgrid(self.lons, self.lats, indexing='xy')
        self.radius_km = radius_km
        self.B_threshold = 10.0
    
    def get_mask(self, center_lon, center_lat, hemisphere='right', motion_dir=45.0):
        """Get semicircle mask for B parameter."""
        R = 6371.0
        lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
        lat2, lon2 = np.radians(self.lats_2d), np.radians(self.lons_2d)
        
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        dist_km = 2 * R * np.arcsin(np.sqrt(a))
        within = dist_km <= self.radius_km
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        relative = (bearing - motion_dir + 360) % 360
        
        if hemisphere == 'right':
            in_semi = (relative >= 270) | (relative <= 90)
        else:
            in_semi = (relative > 90) & (relative < 270)
        
        return within & in_semi
    
    def compute_CPS(self, pressure_interp, center_lon, center_lat, motion_dir=45.0):
        """
        Compute B, VTL, VTU from pressure_interp.
        
        Parameters
        ----------
        pressure_interp : xarray.Dataset
            Output from full_state_pressure_interpolation
            Has Z_PRES(time, pressure, lat, lon)
        center_lon, center_lat : float
            Storm center
        motion_dir : float
            Direction of motion (degrees, 0=N, 90=E)
        
        Returns
        -------
        dict with B, VTL, VTU, is_tropical, phase
        """
        # Extract Z at 300, 600, 900 hPa
        z300 = pressure_interp['Z_PRES'].sel(pressure=300).isel(time=0).values
        z600 = pressure_interp['Z_PRES'].sel(pressure=600).isel(time=0).values
        z900 = pressure_interp['Z_PRES'].sel(pressure=900).isel(time=0).values
        
        # B parameter (900-600 hPa thickness asymmetry)
        thickness = z600 - z900
        mask_r = self.get_mask(center_lon, center_lat, 'right', motion_dir)
        mask_l = self.get_mask(center_lon, center_lat, 'left', motion_dir)
        B = float(np.nanmean(thickness[mask_r]) - np.nanmean(thickness[mask_l]))
        
        # VTL (lower thermal wind: 900-600 hPa)
        R = 6371.0
        lat_rad = np.radians(self.lats_2d - center_lat)
        lon_rad = np.radians(self.lons_2d - center_lon)
        a = np.sin(lat_rad/2)**2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(self.lats_2d)) * np.sin(lon_rad/2)**2
        dist = 2 * R * np.arcsin(np.sqrt(a))
        mask = dist <= self.radius_km
        
        thick_lower = z600 - z900
        thick_lower[~mask] = np.nan
        VTL = -(np.nanmax(thick_lower) - np.nanmin(thick_lower))
        
        # VTU (upper thermal wind: 600-300 hPa)
        thick_upper = z300 - z600
        thick_upper[~mask] = np.nan
        VTU = -(np.nanmax(thick_upper) - np.nanmin(thick_upper))
        
        # Classify
        is_symmetric = abs(B) < self.B_threshold
        has_warm_core_lower = VTL < 0
        has_warm_core_upper = VTU < 0
        is_tropical = is_symmetric and has_warm_core_lower and has_warm_core_upper
        
        # Phase name
        if is_symmetric and has_warm_core_lower and has_warm_core_upper:
            phase = "Tropical"
        elif is_symmetric and has_warm_core_lower:
            phase = "Subtropical"
        elif not is_symmetric and not has_warm_core_lower:
            phase = "Extratropical"
        elif not is_symmetric and has_warm_core_lower:
            phase = "Hybrid"
        else:
            phase = "Cold Core"
        
        return {
            'B': B,
            'VTL': VTL,
            'VTU': VTU,
            'is_tropical': is_tropical,
            'phase': phase
        }
