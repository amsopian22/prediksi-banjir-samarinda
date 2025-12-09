
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import tide_utils
import numpy as np
from datetime import datetime
import logging
import config

# Setup Logging
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self):
        # Setup Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = config.OPENMETEO_URL

    def fetch_hourly_data(self, lat: float = -0.4851, lon: float = 117.2536) -> pd.DataFrame:
        """
        Fetch hourly weather data (precipitation, rain, showers).
        Default coordinates: Near APT Pranoto (Samarinda Utara / Hulu).
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["precipitation", "rain", "showers"],
            "timezone": config.TIMEZONE, 
            "past_days": 1,
            "forecast_days": 7
        }
        
        try:
            logger.info(f"Fetching weather data for lat={lat}, lon={lon}...")
            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
            hourly_rain = hourly.Variables(1).ValuesAsNumpy()
            hourly_showers = hourly.Variables(2).ValuesAsNumpy()
            
            hourly_data = {"date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}
            
            hourly_data["precipitation"] = hourly_precipitation
            hourly_data["rain"] = hourly_rain
            hourly_data["showers"] = hourly_showers
            
            hourly_df = pd.DataFrame(data = hourly_data)
            
            # Convert UTC to local
            hourly_df["date"] = hourly_df["date"].dt.tz_convert(config.TIMEZONE)
            
            logger.info(f"Weather data fetched successfully. Rows: {len(hourly_df)}")
            return hourly_df
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()

class TidePredictor:
    def __init__(self):
        self.model = tide_utils.load_tide_model()
        
    def predict_hourly(self, dates: pd.Series) -> np.ndarray:
        """
        Predict hourly tide levels using Utide model.
        dates: list or array of datetime objects.
        """
        if not self.model:
            logger.warning("Tide model not loaded! Returning zeros.")
            return np.zeros(len(dates))
            
        return tide_utils.predict_tide(self.model, dates)

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    wf = WeatherFetcher()
    df = wf.fetch_hourly_data()
    print("Weather Data Head:")
    print(df.head())
    
    tp = TidePredictor()
    if not df.empty:
        tides = tp.predict_hourly(df['date'])
        df['tide_level'] = tides
        print("\nWith Tide:")
        print(df.head())
