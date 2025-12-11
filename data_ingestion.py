
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

    @staticmethod
    def fetch_weather_data(lat: float = config.LATITUDE, lon: float = config.LONGITUDE):
        """
        Fetches hourly weather forecast from Open-Meteo.
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["precipitation", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm"],
                "timezone": config.TIMEZONE,
                "past_days": 3,
                "forecast_days": 3
            }
            
            responses = openmeteo_requests.Client().weather_api(config.OPENMETEO_URL, params=params) # Use a new client for static method
            response = responses[0]
            
            # Process Hourly
            hourly = response.Hourly()
            hourly_precip = hourly.Variables(0).ValuesAsNumpy()
            hourly_soil0 = hourly.Variables(1).ValuesAsNumpy()
            hourly_soil1 = hourly.Variables(2).ValuesAsNumpy()
            
            hourly_data = {
                "date": pd.date_range(
                    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                    end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                    freq = pd.Timedelta(seconds = hourly.Interval()),
                    inclusive = "left"
                )
            }
            hourly_data["precipitation"] = hourly_precip
            hourly_data["soil_moisture_surface"] = hourly_soil0
            hourly_data["soil_moisture_root"] = hourly_soil1
            
            df = pd.DataFrame(data = hourly_data)
            
            # Feature Engineering on the fly (Rolling, etc)
            df['rain_rolling_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
            df['rain_rolling_3h'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
            
            # Convert timezone
            df['date'] = df['date'].dt.tz_convert(config.TIMEZONE)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
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
