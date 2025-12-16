
import rasterio
import numpy as np
import pysheds
from pysheds.grid import Grid
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
import geopandas as gpd
from scipy.spatial import cKDTree
import config
import logging
from typing import Dict, Optional, Tuple
import os

# Setup Logging
logger = logging.getLogger(__name__)

class SpatialFeatureExtractor:
    def __init__(self):
        self.dem_path = config.DEM_PATH
        self.grid = None
        self.dem = None
        self.slope_map = None
        self.acc = None
        self.twi = None
        self.transform = None
        
        # Approximate River Geometry (Mahakam & Karang Mumus) if shapefile missing
        mahakam = LineString([(117.05, -0.55), (117.13, -0.52), (117.16, -0.50), (117.22, -0.53), (117.30, -0.55)])
        karang_mumus = LineString([(117.16, -0.40), (117.165, -0.45), (117.155, -0.48), (117.15, -0.50)])
        
        self.rivers = MultiLineString([mahakam, karang_mumus])
        self._load_dem()

    def _load_dem(self):
        try:
            logger.info(f"Loading DEM from {self.dem_path}...")
            if not os.path.exists(self.dem_path):
                 logger.error(f"DEM file not found: {self.dem_path}")
                 return

            with rasterio.open(self.dem_path) as src:
                self.dem = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.bounds = src.bounds
                logger.info(f"DEM Bounds: {self.bounds}")
            
            # Pysheds Grid
            try:
                self.grid = Grid.from_raster(self.dem_path)
                self.dem_grid = self.grid.read_raster(self.dem_path)
            except Exception as e:
                logger.warning(f"Pysheds load failed (likely CRS issue): {e}. Continuing with basic features.")
                self.grid = None

            # --- Slope Calculation (Numpy) ---
            try:
                dy, dx = np.gradient(self.dem)
                # Correction for Lat/Lon to meters approx (1 deg ~ 111km)
                # This is a Rough Proxy
                self.slope_map = np.arctan(np.sqrt(dx**2 + dy**2)) 
            except Exception as e:
                logger.error(f"Slope calc error: {e}")
                self.slope_map = np.zeros_like(self.dem)

            # TWI Calculation (Requires pysheds)
            if self.grid is not None:
                try:
                    logger.info("Calculating Flow/TWI (Pysheds)...")
                    dem_inf = self.grid.resolve_flats(self.dem_grid)
                    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
                    fdir = self.grid.flowdir(dem_inf, dirmap=dirmap)
                    self.acc = self.grid.accumulation(fdir, dirmap=dirmap)
                    
                    slope_tan = np.tan(self.slope_map + 0.001) # Avoid zero
                    self.twi = np.log((self.acc + 1) / slope_tan)
                    logger.info("TWI and Flow Accumulation Calculated.")
                except Exception as e:
                    logger.error(f"TWI calc failed: {e}")
                    self.twi = np.zeros_like(self.dem)
            else:
                self.twi = np.zeros_like(self.dem)
            
        except Exception as e:
            logger.error(f"Error loading/processing DEM: {e}")

    def get_features(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Extract Elevation, Slope, River Distance, TWI for a point.
        """
        elev = 0.0
        twi_val = 0.0
        slope_val = 0.0
        
        # 1. Elevation & TWI from Raster
        try:
            with rasterio.open(self.dem_path) as src:
                vals = list(src.sample([(lon, lat)]))
                if vals and len(vals[0]) > 0:
                     elev = float(vals[0][0])
                
                # Grid Lookup for TWI/Slope
                row, col = src.index(lon, lat)
                
                if self.twi is not None:
                    if 0 <= row < self.twi.shape[0] and 0 <= col < self.twi.shape[1]:
                        twi_val = float(self.twi[row, col])
                        
                if self.slope_map is not None:
                    if 0 <= row < self.slope_map.shape[0] and 0 <= col < self.slope_map.shape[1]:
                        slope_val = float(self.slope_map[row, col])
                
        except Exception as e:
            logger.error(f"Error extracting raster features: {e}")
            
        # 2. Distance to River
        point = Point(lon, lat)
        dist_deg = point.distance(self.rivers)
        dist_m = dist_deg * 111139
        
        # 3. Flow Accumulation (New)
        acc_val = 0.0
        if self.acc is not None:
             try:
                 # Grid Lookup for Acc
                row, col = src.index(lon, lat)
                if 0 <= row < self.acc.shape[0] and 0 <= col < self.acc.shape[1]:
                    acc_val = float(self.acc[row, col])
             except:
                 pass

        return {
            "elevation": elev,
            "slope": slope_val,
            "twi": twi_val,
            "flow_accumulation": acc_val,
            "river_dist_m": dist_m
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = SpatialFeatureExtractor()
    # Test Point (Within Bounds)
    lat, lon = -0.6, 117.15
    feats = extractor.get_features(lat, lon)
    print(f"Features for ({lat}, {lon}):")
    print(feats)
