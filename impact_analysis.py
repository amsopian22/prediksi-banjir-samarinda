import geopandas as gpd
import pandas as pd
import os
import logging

# Constants
CACHE_DIR = "data/osm_cache"
BUILDINGS_FILE = os.path.join(CACHE_DIR, "samarinda_buildings.parquet")

logger = logging.getLogger(__name__)

def load_buildings():
    """Load buildings from Parquet cache."""
    if not os.path.exists(BUILDINGS_FILE):
        return None
    try:
        return gpd.read_parquet(BUILDINGS_FILE)
    except Exception as e:
        logger.error(f"Error loading buildings: {e}")
        return None

def analyze_impact(risk_gdf):
    """
    Calculate affected buildings based on risk polygons.
    risk_gdf: GeoDataFrame containing risk polygons (e.g. from flood model).
    """
    buildings = load_buildings()
    
    if buildings is None or risk_gdf is None or risk_gdf.empty:
        return {
            "total_affected": 0,
            "schools_affected": 0,
            "hospitals_affected": 0,
            "details": []
        }

    # Ensure CRS match (OSMnx usually 4326, make sure risk_gdf is too)
    if buildings.crs != risk_gdf.crs:
        risk_gdf = risk_gdf.to_crs(buildings.crs)

    # Filter for high risk only (assuming 'risk_level' column or similar)
    # If not present, assume all input polygons are risk areas
    high_risk_areas = risk_gdf  

    # Spatial Join: Find buildings within risk areas
    affected = gpd.sjoin(buildings, high_risk_areas, how="inner", predicate="intersects")
    
    # Analysis
    total_affected = len(affected)
    
    # Identify critical infrastructure
    schools = affected[affected['amenity'].isin(['school', 'university', 'college', 'kindergarten'])]
    hospitals = affected[affected['amenity'].isin(['hospital', 'clinic', 'pharmacy'])]
    
    return {
        "total_affected": total_affected,
        "schools_affected": len(schools),
        "hospitals_affected": len(hospitals),
        "details": affected[['name', 'amenity']].dropna().head(10).to_dict('records')
    }
