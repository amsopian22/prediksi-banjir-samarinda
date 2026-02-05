
"""
fetch_historical_10yr.py - Fetch 10 years of daily rainfall data for Samarinda.
Uses Open-Meteo Archive API.
"""
import pandas as pd
import requests
import os
import sys
import logging
from datetime import datetime, timedelta

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_historical_daily(lat, lon, start_date, end_date, prefix=""):
    """
    Fetch historical daily weather data from Open-Meteo Archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum", "rain_sum", "precipitation_hours"],
        "timezone": config.TIMEZONE
    }
    
    try:
        logger.info(f"Fetching {prefix} historical data from {start_date} to {end_date}...")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        daily = data['daily']
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            f'{prefix}precipitation_sum': daily['precipitation_sum'],
            f'{prefix}rain_sum': daily['rain_sum'],
            f'{prefix}precipitation_hours': daily['precipitation_hours']
        })
        
        logger.info(f"Fetched {len(df)} daily records for {prefix}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching historical weather ({prefix}): {e}")
        return pd.DataFrame()

def main():
    # Define time range: 10 years back from today
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d') # Archive usually has 2 days lag
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    logger.info(f"Targeting range: {start_date} to {end_date}")
    
    # 1. Fetch Samarinda Local Data
    df_local = fetch_historical_daily(config.LATITUDE, config.LONGITUDE, start_date, end_date, prefix="local_")
    
    # 2. Fetch Upstream Data
    upstream_coord = config.UPSTREAM_LOCATIONS.get("Hulu Karang Mumus (Badak Baru)")
    if upstream_coord:
        df_upstream = fetch_historical_daily(upstream_coord[0], upstream_coord[1], start_date, end_date, prefix="upstream_")
        # Merge
        if not df_local.empty and not df_upstream.empty:
            df_final = pd.merge(df_local, df_upstream, on='date', how='outer')
        else:
            df_final = df_local
    else:
        df_final = df_local
        
    if df_final.empty:
        logger.error("No data fetched. Check internet connection or API limits.")
        return
    
    # Save to data directory
    output_path = os.path.join(config.BASE_DIR, "data", "rainfall_history_10y.csv")
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"✅ Successfully saved 10-year rainfall history to {output_path}")
    logger.info(f"Total records: {len(df_final)}")
    logger.info(f"Memory coverage: {df_final['date'].min()} to {df_final['date'].max()}")

if __name__ == "__main__":
    main()
