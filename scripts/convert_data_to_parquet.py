import pandas as pd
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_parquet():
    """
    Converts the main CSV dataset to Parquet for performance optimization.
    Parquet is faster to read and smaller in size, ideal for Streamlit usage.
    """
    csv_path = os.path.join(config.BASE_DIR, 'data', 'dataset_banjir_v8_10years.csv')
    parquet_path = os.path.join(config.BASE_DIR, 'data', 'dataset_banjir_v8_10years.parquet')
    
    if not os.path.exists(csv_path):
        logger.error(f"Source CSV not found at: {csv_path}")
        return
        
    logger.info(f"Reading CSV from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        
        # Ensure 'date' column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Saving to Parquet: {parquet_path}...")
        df.to_parquet(parquet_path, index=False)
        
        size_csv = os.path.getsize(csv_path) / (1024 * 1024)
        size_pq = os.path.getsize(parquet_path) / (1024 * 1024)
        
        logger.info(f"✅ Conversion Success!")
        logger.info(f"   CSV Size:     {size_csv:.2f} MB")
        logger.info(f"   Parquet Size: {size_pq:.2f} MB")
        logger.info(f"   Reduction:    {(1 - size_pq/size_csv)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Failed to convert: {e}")

if __name__ == "__main__":
    convert_to_parquet()
