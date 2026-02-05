import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
import argparse
import os
import rasterio
import numpy as np

# Overpass API URL
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
DEM_PATH = "data/dem/DEMNAS_1915-13_v1.0.tif"

def get_roads_query(bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    return query

def fetch_roads_with_elevation_optimized(bbox=( -0.730, 117.030, -0.300, 117.330 )): 
    print(f"Fetching roads for bbox: {bbox}...")
    
    query = get_roads_query(bbox)
    try:
        response = requests.get(OVERPASS_URL, params={'data': query})
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            return None
        data = response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    nodes = {n['id']: (n['lon'], n['lat']) for n in data['elements'] if n['type'] == 'node'}
    
    roads = []
    all_points = []
    road_point_indices = [] # Stores (start_idx, end_idx) for each road
    
    current_idx_pointer = 0
    
    for el in data['elements']:
        if el['type'] == 'way' and 'tags' in el:
            road_nodes = el.get('nodes', [])
            coords = [nodes[nid] for nid in road_nodes if nid in nodes]
            
            if len(coords) < 2:
                continue
                
            geom = LineString(coords)
            
            # Prepare for Batch Sampling
            # Only sample vertices (coords)
            points_for_this_road = coords # [(lon, lat), ...]
            all_points.extend(points_for_this_road)
            
            count = len(points_for_this_road)
            road_point_indices.append((current_idx_pointer, current_idx_pointer + count))
            current_idx_pointer += count
            
            tags = el['tags']
            roads.append({
                'osm_id': el['id'],
                'name': tags.get('name', 'Unnamed Road'),
                'highway': tags.get('highway', 'unknown'),
                'geometry': geom
            })
            
    if not roads:
        print("No roads found.")
        return None
        
    gdf = gpd.GeoDataFrame(roads, crs="EPSG:4326")
    print(f"Fetched {len(gdf)} road segments. \nTotal vertices to sample: {len(all_points)}")
    
    # Check DEM existence
    if not os.path.exists(DEM_PATH):
        print(f"⚠️ GENERIC MODE: DEM file not found at {DEM_PATH}. Skipping elevation.")
        gdf['mean_elev'] = 5.0
        return gdf

    try:
        print("Sampling elevation (Vectorized)...")
        with rasterio.open(DEM_PATH) as src:
            # Batch Sample
            elevation_gen = src.sample(all_points)
            elevations = np.array([x[0] for x in elevation_gen])
            
            # Re-assign to roads
            mean_elevs = []
            min_elevs = []
            max_elevs = []
            
            for start, end in road_point_indices:
                road_vals = elevations[start:end]
                # Filter NoData (-9999 or huge neg)
                valid_vals = road_vals[road_vals > -500] 
                
                if len(valid_vals) > 0:
                    mean_elevs.append(np.mean(valid_vals))
                    min_elevs.append(np.min(valid_vals))
                    max_elevs.append(np.max(valid_vals))
                else:
                    mean_elevs.append(np.nan)
                    min_elevs.append(np.nan)
                    max_elevs.append(np.nan)
            
            gdf['mean_elev'] = mean_elevs
            gdf['min_elev'] = min_elevs
            gdf['max_elev'] = max_elevs
            
            # Fill NaNs with reasonable default (e.g. 5m)
            gdf['mean_elev'] = gdf['mean_elev'].fillna(5.0)
            gdf['min_elev'] = gdf['min_elev'].fillna(5.0)
            gdf['max_elev'] = gdf['max_elev'].fillna(5.0)
            
            print("Elevation sampling complete.")
            
    except Exception as e:
        print(f"Elevation sampling failed: {e}")
        gdf['mean_elev'] = 5.0
        gdf['min_elev'] = 0.0
        gdf['max_elev'] = 10.0
        
    return gdf

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Samarinda Approximate Bbox
    bbox = (-0.65, 117.05, -0.35, 117.30)
    
    gdf = fetch_roads_with_elevation_optimized(bbox)
    
    if gdf is not None:
        output_path = "data/samarinda_roads.parquet"
        gdf.to_parquet(output_path)
        
        # Verify
        print(f"✅ Saved road network to {output_path}")
        print(gdf[['name', 'mean_elev', 'min_elev', 'max_elev']].head(10))
