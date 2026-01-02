import osmnx as ox
import geopandas as gpd
import os
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CACHE_DIR = "data/osm_cache"
BUILDINGS_FILE = os.path.join(CACHE_DIR, "samarinda_buildings.parquet")
ROADS_FILE = os.path.join(CACHE_DIR, "samarinda_roads.parquet")
PLACE_NAME = "Samarinda, East Kalimantan, Indonesia"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_osm_data():
    ensure_dir(CACHE_DIR)
    
    # 1. Fetch Buildings
    if os.path.exists(BUILDINGS_FILE):
        logging.info("♻️  Using cached buildings data.")
    else:
        logging.info(f"🏗️  Fetching buildings for {PLACE_NAME}...")
        try:
            # Fetch building footprints
            tags = {"building": True}
            gdf_buildings = ox.features_from_place(PLACE_NAME, tags)
            
            # Simple cleanup: keep only relevant columns
            cols_to_keep = ['geometry', 'building', 'name', 'amenity']
            # Filter columns that exist
            cols = [c for c in cols_to_keep if c in gdf_buildings.columns]
            gdf_buildings = gdf_buildings[cols]
            
            # Save to Parquet (faster than GeoJSON/Shapefile)
            gdf_buildings.to_parquet(BUILDINGS_FILE)
            logging.info(f"✅ Saved {len(gdf_buildings)} buildings to {BUILDINGS_FILE}")
            
        except Exception as e:
            logging.error(f"❌ Failed to fetch buildings: {e}")

    # 2. Fetch Critical Infrastructure (Hospitals, Schools) specifically if needed
    # (Note: 'amenity' tag in buildings usually covers this, but we can be specific)

if __name__ == "__main__":
    fetch_osm_data()
