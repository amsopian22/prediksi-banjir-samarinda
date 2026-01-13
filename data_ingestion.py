
import openmeteo_requests
import requests
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
        # Setup Open-Meteo API client with retry on error
        # REMOVED requests_cache to avoid ReadOnly database errors in Cloud
        session = requests.Session()
        retry_session = retry(session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = config.OPENMETEO_URL

    def fetch_weather_data(self, lat: float = None, lon: float = None, location_label: str = None):
        """
        Fetches hourly weather forecast from Open-Meteo.
        If location_label is provided and exists in UPSTREAM_LOCATIONS, uses that coord.
        Otherwise uses provided lat/lon or defaults.
        """
        # Determine Coordinates
        if location_label and location_label in config.UPSTREAM_LOCATIONS:
            lat, lon = config.UPSTREAM_LOCATIONS[location_label]
        elif lat is None or lon is None:
            lat = config.LATITUDE
            lon = config.LONGITUDE
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["precipitation", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "wind_speed_10m"],
                "timezone": config.TIMEZONE,
                "past_days": 3,
                "forecast_days": 14
            }
            
            # Use the cached session client initialized in __init__
            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]
            
            # Process Hourly
            hourly = response.Hourly()
            hourly_precip = hourly.Variables(0).ValuesAsNumpy()
            hourly_soil0 = hourly.Variables(1).ValuesAsNumpy()
            hourly_soil1 = hourly.Variables(2).ValuesAsNumpy()
            hourly_wind = hourly.Variables(3).ValuesAsNumpy()
            
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
            hourly_data["wind_speed"] = hourly_wind
            
            df = pd.DataFrame(data = hourly_data)
            
            # Feature Engineering on the fly (Rolling, etc)
            df['rain_rolling_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
            df['rain_rolling_3h'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
            df['wind_speed_max_24h'] = df['wind_speed'].rolling(window=24, min_periods=1).max()
            
            
            # Convert timezone
            df['date'] = df['date'].dt.tz_convert(config.TIMEZONE)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            logging.warning("⚠️ OPTION-METEO ERROR: Failed to connect. Switching to Mock Data mode.")
            
            # --- FALLBACK: GENERATE MOCK DATA ---
            # To ensure the dashboard doesn't crash offline, we generate a synthetic dataset.
            try:
                now = pd.Timestamp.now(tz=config.TIMEZONE)
                start_date = now - pd.Timedelta(days=3)
                end_date = now + pd.Timedelta(days=14)
                
                # Create hourly index
                date_rng = pd.date_range(start=start_date, end=end_date, freq='h')
                
                # Mock Values (Zero rain, safe default soil)
                length = len(date_rng)
                mock_data = {
                    "date": date_rng,
                    "precipitation": [0.0] * length, 
                    "soil_moisture_surface": [0.4] * length,
                    "soil_moisture_root": [0.4] * length,
                    "wind_speed": [2.0] * length
                }
                
                df = pd.DataFrame(mock_data)
                
                # Feature Engineering for Mock Data
                df['rain_rolling_24h'] = 0.0
                df['rain_rolling_3h'] = 0.0
                df['wind_speed_max_24h'] = 2.0
                
                return df
                
            except Exception as e_mock:
                logging.error(f"Failed to generate mock data: {e_mock}")
                return pd.DataFrame()


class BMKGFetcher:
    """Fetcher for BMKG (Indonesian Meteorological Agency) weather data."""
    
    def __init__(self):
        self.base_url = config.BMKG_API_URL
        self.session = requests.Session()
        self.openmeteo_fallback = WeatherFetcher()
    
    def fetch_weather_data(self, location_name: str = None, adm4_code: str = None):
        """
        Fetch weather forecast from BMKG API.
        
        Args:
            location_name: Name of location (must exist in config.BMKG_LOCATIONS)
            adm4_code: Direct ADM4 code (overrides location_name)
        
        Returns:
            DataFrame with hourly weather data
        """
        # Determine ADM4 code
        if adm4_code is None:
            if location_name and location_name in config.BMKG_LOCATIONS:
                adm4_code = config.BMKG_LOCATIONS[location_name]
            else:
                adm4_code = config.BMKG_DEFAULT_ADM4
        
        try:
            url = f"{self.base_url}?adm4={adm4_code}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"BMKG API returned {response.status_code}, falling back to Open-Meteo")
                return self._fallback_to_openmeteo(location_name)
            
            data = response.json()
            return self._parse_bmkg_response(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"BMKG API request failed: {e}")
            return self._fallback_to_openmeteo(location_name)
        except Exception as e:
            logger.error(f"BMKG parsing error: {e}")
            return self._fallback_to_openmeteo(location_name)
    
    def _parse_bmkg_response(self, data: dict) -> pd.DataFrame:
        """Parse BMKG JSON response into DataFrame."""
        try:
            # Navigate to weather data
            # Structure: data -> [location] -> cuaca -> [[day1_hours], [day2_hours], ...]
            if 'data' not in data or len(data['data']) == 0:
                logger.warning("BMKG response has no data")
                return pd.DataFrame()
            
            location_data = data['data'][0]
            cuaca_data = location_data.get('cuaca', [])
            
            # Flatten all hourly data
            all_hours = []
            for day_data in cuaca_data:
                if isinstance(day_data, list):
                    all_hours.extend(day_data)
            
            if not all_hours:
                logger.warning("No hourly data in BMKG response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for hour in all_hours:
                records.append({
                    'date': pd.to_datetime(hour.get('local_datetime')),
                    'precipitation': float(hour.get('tp', 0)),  # Total precipitation
                    'temperature': float(hour.get('t', 27)),    # Temperature
                    'humidity': float(hour.get('hu', 80)),      # Humidity
                    'wind_speed': float(hour.get('ws', 0)),     # Wind speed m/s
                    'wind_direction': hour.get('wd', 'N'),
                    'weather_code': int(hour.get('weather', 0)),
                    'weather_desc': hour.get('weather_desc', ''),
                    'cloud_cover': float(hour.get('tcc', 0)),   # Total cloud cover
                    'source': 'BMKG'
                })
            
            df = pd.DataFrame(records)
            
            # Localize timezone
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(config.TIMEZONE)
            
            # Add rolling features to match existing format
            df['rain_rolling_24h'] = df['precipitation'].rolling(window=24, min_periods=1).sum()
            df['rain_rolling_3h'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
            df['wind_speed_max_24h'] = df['wind_speed'].rolling(window=24, min_periods=1).max()
            
            # Default soil moisture (BMKG doesn't provide this)
            df['soil_moisture_surface'] = 0.4
            df['soil_moisture_root'] = 0.4
            
            logger.info(f"BMKG data fetched: {len(df)} hours from {df['date'].min()} to {df['date'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing BMKG response: {e}")
            return pd.DataFrame()
    
    def _fallback_to_openmeteo(self, location_name: str = None) -> pd.DataFrame:
        """Fallback to Open-Meteo when BMKG fails."""
        logger.info("Falling back to Open-Meteo API")
        
        # Get coordinates for the location
        lat, lon = None, None
        if location_name and location_name in config.LOCATIONS:
            loc_data = config.LOCATIONS[location_name]
            lat, lon = loc_data[0], loc_data[1]
        
        df = self.openmeteo_fallback.fetch_weather_data(lat=lat, lon=lon)
        if not df.empty:
            df['source'] = 'OpenMeteo'
        return df

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
