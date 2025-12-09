
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import json
import config
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_zonal_stats():
    """
    Calculates P10 and P50 elevation for each polygon in the risk map
    and saves the updated GeoJSON.
    """
    input_file = "data-refactored/samarinda_risk_map_calculated.geojson"
    output_file = "data-refactored/samarinda_risk_map_percentiles.geojson"
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info("Loading polygons...")
    gdf = gpd.read_file(input_file)
    
    logger.info(f"Loading DEM from {config.DEM_PATH}...")
    try:
        with rasterio.open(config.DEM_PATH) as src:
            
            p10_list = []
            p50_list = []
            
            for idx, row in gdf.iterrows():
                geom = [row['geometry']]
                try:
                    out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                    valid_data = out_image[0][out_image[0] > -100] # Filter NoData
                    
                    if valid_data.size > 0:
                        p10 = np.percentile(valid_data, 10)
                        p50 = np.median(valid_data)
                    else:
                        p10 = 0.0
                        p50 = 0.0
                        
                except Exception as e:
                    logger.warning(f"Error processing {row.get('NAMOBJ', idx)}: {e}")
                    p10 = 0.0
                    p50 = 0.0
                
                p10_list.append(p10)
                p50_list.append(p50)
            
            gdf['p10_elev'] = p10_list
            gdf['p50_elev'] = p50_list
            
            # Save updated GeoJSON
            logger.info(f"Saving updated map to {output_file}...")
            gdf.to_file(output_file, driver='GeoJSON')
            logger.info("Done.")
            
    except Exception as e:
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    calculate_zonal_stats()
