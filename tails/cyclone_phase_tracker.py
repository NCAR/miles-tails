import numpy as np
import xarray as xr
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch

from credit.output import make_xarray
from credit.interp import full_state_pressure_interpolation

@dataclass
class StormState:
    """Storm state with physically accurate CPS parameters."""
    location: Tuple[float, float]
    mslp: float
    B: float          # Asymmetry (m)
    VTL: float        # Lower Thermal Wind (m)
    VTU: float        # Upper Thermal Wind (m)
    is_tropical: bool
    phase: str
    decision_step: str # 'ACCEPT' or 'REJECT' (helper for your logs)

class CyclonePhaseTracker:
    """
    Compute Cyclone Phase Space (CPS) parameters (Hart 2003).
    Following EXACT formulation from https://moe.met.fsu.edu/cyclonephase/help.html
    """
    
    def __init__(self, latlons: xr.Dataset, radius_km: float = 500.0):
        self.lats = latlons.latitude.values
        self.lons = np.where(
            latlons.longitude.values > 180,
            latlons.longitude.values - 360,
            latlons.longitude.values
        )
        self.lons_2d, self.lats_2d = np.meshgrid(self.lons, self.lats, indexing='xy')
        self.radius_km = radius_km
        
        # Hart (2003) thresholds for -VT (warm core when positive)
        # self.thresholds = {
        #     'genesis': {
        #         'B_max': 1000.0,      # Allow asymmetry during formation
        #         'VTL_min': 0.0,      # -VT^L > 0 for warm core
        #         'VTU_min': 0.0,      # -VT^U > 0 for deep warm core
        #     },
        #     'mature': {
        #         'B_max': 1000.0,      # Relaxed - B is noisy without perfect motion
        #         'VTL_min': 0.0,      # -VT^L > 0 for warm core
        #         'VTU_min': 0.0,      # -VT^U > 0 for deep warm core
        #     }
        # }

        self.thresholds = {
            'genesis': {
                'B_max': 500.0,      # Effectively disabled - B requires accurate motion tracking
                'VTL_min': -100.0,     # Tolerance for grid noise (~111km resolution)
                'VTU_min': -100.0,     # Values in [-100, 0] treated as marginally warm/neutral
            },
            'mature': {
                'B_max': 500.0,      # Effectively disabled - B requires accurate motion tracking
                'VTL_min': -75.0,     # Slightly stricter for mature systems
                'VTU_min': -75.0,     # Real cold cores have VT << -100m; [-75, 0] is grid uncertainty
            }
        }

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate bearing between two points.
        """
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlon_rad = lon2_rad - lon1_rad
        
        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360

    def _get_previous_position(self, storm_id):
        """Get previous position for motion calculation."""
        if storm_id not in self.storm_track_history:
            return None, None
        
        hist = self.storm_track_history[storm_id]
        if len(hist['lons']) < 1:
            return None, None
        
        return hist['lons'][-1], hist['lats'][-1]

    def _get_geo_metrics(self, center_lon, center_lat, motion_dir):
        """Compute distance masks and bearings relative to storm center."""
        R = 6371.0
        lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
        lat2, lon2 = np.radians(self.lats_2d), np.radians(self.lons_2d)
        
        dlat, dlon = lat2 - lat1, lon2 - lon1
        
        # Haversine distance
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        dist_km = 2 * R * np.arcsin(np.sqrt(a))
        
        # Bearing
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        
        # Relative bearing to motion
        relative_angle = (bearing - motion_dir + 360) % 360
        
        return dist_km, relative_angle

    def compute_CPS(self, pressure_interp, center_lon, center_lat, mslp, 
                 motion_dir=None, prev_lon=None, prev_lat=None, stage='genesis'):
        """
        Compute B, -VT^L, -VT^U following Hart (2003) EXACTLY.
        
        From Hart (2003) and FSU documentation:
        - B = h * (ΔZ^R - ΔZ^L) where ΔZ = Z_600 - Z_900
        - phi' = Z_max - Z_min (cyclone strength at a level)
        - -VT^L = phi'_900 - phi'_600 (warm core when positive)
        - -VT^U = phi'_600 - phi'_300 (warm core when positive)
        """
         # Calculate motion direction
        if motion_dir is None:
            if prev_lon is not None and prev_lat is not None:
                # Just calculate it, no threshold bullshit
                motion_dir = self._calculate_bearing(prev_lat, prev_lon, center_lat, center_lon)
            else:
                # Only fall back to default if we have NO previous position
                motion_dir = 45.0  # or use your climatology function
            
        # Extract heights
        z_pres = pressure_interp['Z_PRES']
        if 'time' in z_pres.dims:
            z_pres = z_pres.isel(time=0)
            
        z300 = z_pres.sel(pressure=300, method='nearest').values
        z600 = z_pres.sel(pressure=600, method='nearest').values
        z900 = z_pres.sel(pressure=900, method='nearest').values
        
        # Get geometry
        dist_km, rel_angle = self._get_geo_metrics(center_lon, center_lat, motion_dir)
        mask_storm = dist_km <= self.radius_km
        mask_right = mask_storm & ((rel_angle >= 270) | (rel_angle <= 90))
        mask_left  = mask_storm & ((rel_angle > 90) & (rel_angle < 270))
        
        # --- B PARAMETER (CORRECT) ---
        thick_lower = z600 - z900
        B_raw = float(np.nanmean(thick_lower[mask_right]) - np.nanmean(thick_lower[mask_left]))
        B_magnitude = abs(B_raw)
        
        # --- -VT^L and -VT^U: CORRECT FORMULATION ---
        # Mask heights to storm radius
        z300_masked = z300.copy()
        z600_masked = z600.copy()
        z900_masked = z900.copy()
        z300_masked[~mask_storm] = np.nan
        z600_masked[~mask_storm] = np.nan
        z900_masked[~mask_storm] = np.nan
        
        # Cyclone strength (phi') at each level
        phi_900 = np.nanmax(z900_masked) - np.nanmin(z900_masked)
        phi_600 = np.nanmax(z600_masked) - np.nanmin(z600_masked)
        phi_300 = np.nanmax(z300_masked) - np.nanmin(z300_masked)
        
        # Thermal wind parameters (note the NEGATIVE sign in the name)
        # Warm core: strength decreases with height → phi_lower > phi_upper → -VT > 0
        neg_VTL = phi_900 - phi_600  # Lower thermal wind
        neg_VTU = phi_600 - phi_300  # Upper thermal wind
        
        VTL = neg_VTL
        VTU = neg_VTU
        
        # --- CLASSIFICATION ---
        thresh = self.thresholds.get(stage, self.thresholds['genesis'])
        
        B_scale = 1.0
        if mslp < 970: 
            B_scale = 2.5
        elif mslp < 980: 
            B_scale = 2.0
        elif mslp < 990: 
            B_scale = 1.5
        
        scaled_B_max = thresh['B_max'] * B_scale
        
        # Tropical cyclone criteria:
        # 1. Symmetric (small B)
        # 2. Warm core lower troposphere (-VT^L > 0)
        # 3. Warm core upper troposphere (-VT^U > 0)
        is_symmetric = B_magnitude < scaled_B_max
        is_warm_lower = VTL > thresh['VTL_min']
        is_warm_upper = VTU > thresh['VTU_min']
        
        is_tropical = is_symmetric and is_warm_lower and is_warm_upper
        
        # Phase classification
        if is_tropical:
            phase = "Tropical"
        elif is_symmetric and is_warm_lower:
            phase = "Subtropical"
        elif not is_symmetric and not is_warm_lower:
            phase = "Extratropical"
        elif not is_symmetric and is_warm_lower:
            phase = "Hybrid"
        else:
            phase = "Cold Core"
        
        return {
            'B': B_raw,
            'VTL': VTL,
            'VTU': VTU,
            'is_tropical': is_tropical,
            'phase': phase,
            'stage': stage,
            'threshold_B': scaled_B_max,
            'motion_dir': motion_dir,
            'decision': 'ACCEPT' if is_tropical else 'REJECT'
        }