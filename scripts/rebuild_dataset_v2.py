
import pandas as pd
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import config
import logging
from datetime import timedelta
import time
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Open-Meteo Client
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_historical_features(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["precipitation", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"],
        "timezone": "Asia/Singapore"
    }
    
    logger.info(f"Fetching Archive Data: {start_date} to {end_date}")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process Hourly
    hourly = response.Hourly()
    hourly_precip = hourly.Variables(0).ValuesAsNumpy()
    hourly_soil0 = hourly.Variables(1).ValuesAsNumpy()
    hourly_soil7 = hourly.Variables(2).ValuesAsNumpy()
    
    hourly_data = {
        "date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )
    }
    hourly_data["precipitation"] = hourly_precip
    hourly_data["soil_moisture_0_7cm"] = hourly_soil0
    hourly_data["soil_moisture_7_28cm"] = hourly_soil7
    
    df_hourly = pd.DataFrame(data = hourly_data)
    
    # Convert to Timezone
    df_hourly['date'] = df_hourly['date'].dt.tz_convert(config.TIMEZONE)
    df_hourly['date_only'] = df_hourly['date'].dt.date
    
    # Aggregate Daily
    df_daily = df_hourly.groupby('date_only').agg({
        'precipitation': ['sum', 'max'], # Total Rain, Max Intensity
        'soil_moisture_0_7cm': 'mean',
        'soil_moisture_7_28cm': 'mean'
    })
    
    # Flatten columns
    df_daily.columns = ['rain_sum_imputed', 'rain_intensity_max', 'soil_moisture_surface_mean', 'soil_moisture_root_mean']
    return df_daily

def rebuild_dataset():
    # 1. Load Existing Labels
    df_labels = pd.read_csv("dataset_banjir_samarinda_final.csv")
    df_labels['tanggal'] = pd.to_datetime(df_labels['tanggal']).dt.date
    
    start_date = df_labels['tanggal'].min()
    end_date = df_labels['tanggal'].max()
    
    # 2. Fetch Hydrology Features
    logger.info("Fetching Hydrology Features (Soil Moisture, etc)...")
    try:
        # Split into yearly chunks if needed, but Open-Meteo Archive can handle multi-year
        df_hydro = fetch_historical_features(
            config.LATITUDE, config.LONGITUDE, 
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        
        # 3. Merge
        logger.info("Merging Data...")
        # Reset index to make date_only a column
        df_hydro = df_hydro.reset_index()
        df_hydro['date_only'] = pd.to_datetime(df_hydro['date_only']).dt.date

        df_merged = pd.merge(df_labels, df_hydro, left_on='tanggal', right_on='date_only', how='left')
        
        # Drop redundant
        df_merged.drop(columns=['date_only'], inplace=True)
        
        # Save
        outfile = "dataset_banjir_v2_advanced.csv"
        df_merged.to_csv(outfile, index=False)
        logger.info(f"Success! Saved to {outfile}")
        logger.info(df_merged.head())
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")

if __name__ == "__main__":
    rebuild_dataset()
