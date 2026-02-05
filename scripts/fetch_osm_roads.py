import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import argparse
import os

# Overpass API URL
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

def get_roads_query(bbox):
    """
    Constructs Overpass QL query for major roads within bbox.
    """
    # Order: South, West, North, East
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    return query

def fetch_roads(bbox=( -0.730, 117.030, -0.300, 117.330 )): # Default Samarinda Bbox
    print(f"Fetching roads for bbox: {bbox}...")
    
    query = get_roads_query(bbox)
    response = requests.get(OVERPASS_URL, params={'data': query})
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    data = response.json()
    
    # Process Nodes
    nodes = {n['id']: (n['lon'], n['lat']) for n in data['elements'] if n['type'] == 'node'}
    
    # Process Ways (Roads)
    roads = []
    for el in data['elements']:
        if el['type'] == 'way' and 'tags' in el:
            road_nodes = el.get('nodes', [])
            coords = [nodes[nid] for nid in road_nodes if nid in nodes]
            
            if len(coords) < 2:
                continue
                
            geom = LineString(coords)
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
    print(f"fetched {len(gdf)} road segments.")
    return gdf

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Samarinda Approximate Bbox
    # South, West, North, East
    bbox = (-0.65, 117.05, -0.35, 117.30)
    
    gdf = fetch_roads(bbox)
    
    if gdf is not None:
        output_path = "data/samarinda_roads.parquet"
        gdf.to_parquet(output_path)
        print(f"✅ Saved road network to {output_path}")
        print(gdf.head())
