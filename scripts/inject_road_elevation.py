import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import os

DEM_PATH = "data/dem/DEMNAS_1915-13_v1.0.tif"
ROADS_PATH = "data/samarinda_roads.parquet"

def add_elevation():
    if not os.path.exists(ROADS_PATH):
        print(f"Error: {ROADS_PATH} not found.")
        return
        
    if not os.path.exists(DEM_PATH):
        print(f"Error: DEM not found at {DEM_PATH}")
        return

    print(f"Loading existing roads from {ROADS_PATH}...")
    gdf = gpd.read_parquet(ROADS_PATH)
    
    # Prepare vertices for sampling
    print("Preparing vertices for sampling...")
    all_points = []
    road_point_indices = []
    current_idx_pointer = 0
    
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            # Handle MultiLineString if necessary
            coords = []
            for g in geom.geoms:
                coords.extend(list(g.coords))
        else:
            coords = []
            
        all_points.extend(coords)
        count = len(coords)
        road_point_indices.append((current_idx_pointer, current_idx_pointer + count))
        current_idx_pointer += count

    print(f"Sampling elevation for {len(all_points)} vertices...")
    try:
        with rasterio.open(DEM_PATH) as src:
            elevation_gen = src.sample(all_points)
            elevations = np.array([x[0] for x in elevation_gen])
            
            mean_elevs = []
            min_elevs = []
            max_elevs = []
            
            for start, end in road_point_indices:
                if start == end:
                    mean_elevs.append(5.0)
                    min_elevs.append(5.0)
                    max_elevs.append(5.0)
                    continue
                    
                road_vals = elevations[start:end]
                valid_vals = road_vals[road_vals > -500] 
                
                if len(valid_vals) > 0:
                    mean_elevs.append(float(np.mean(valid_vals)))
                    min_elevs.append(float(np.min(valid_vals)))
                    max_elevs.append(float(np.max(valid_vals)))
                else:
                    mean_elevs.append(5.0)
                    min_elevs.append(5.0)
                    max_elevs.append(5.0)
            
            gdf['mean_elev'] = mean_elevs
            gdf['min_elev'] = min_elevs
            gdf['max_elev'] = max_elevs
            
            # Save back
            gdf.to_parquet(ROADS_PATH)
            print(f"✅ Success! Updated {ROADS_PATH} with elevation data.")
            print(gdf[['name', 'mean_elev']].head(10))
            
    except Exception as e:
        print(f"Failed to sample elevation: {e}")

if __name__ == "__main__":
    add_elevation()
