"""
Headless data fetching script for CI/CD environments.
Fetches weather data from Open-Meteo and saves to data directory.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
WEATHER_FILE = os.path.join(DATA_DIR, "weather_latest.csv")
SAMARINDA_LAT = -0.4948
SAMARINDA_LON = 117.1436

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_openmeteo_data():
    """Fetch weather data from Open-Meteo API."""
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    
    # Setup cached session
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": SAMARINDA_LAT,
        "longitude": SAMARINDA_LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                   "rain", "weather_code", "surface_pressure", "wind_speed_10m"],
        "timezone": "Asia/Makassar",
        "past_days": 7,
        "forecast_days": 7
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        hourly_data = {
            "timestamp": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation": hourly.Variables(2).ValuesAsNumpy(),
            "rain": hourly.Variables(3).ValuesAsNumpy(),
            "weather_code": hourly.Variables(4).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(5).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(6).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Makassar')
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo data: {e}")
        return None

def main():
    """Main function to fetch and save data."""
    ensure_dir(DATA_DIR)
    
    logger.info("🌤️ Fetching weather data from Open-Meteo...")
    weather_df = fetch_openmeteo_data()
    
    if weather_df is not None:
        weather_df.to_csv(WEATHER_FILE, index=False)
        logger.info(f"✅ Saved weather data to {WEATHER_FILE} ({len(weather_df)} rows)")
    else:
        logger.error("❌ Failed to fetch weather data")
        return 1
    
    logger.info("✅ Data update complete!")
    return 0

if __name__ == "__main__":
    exit(main())
