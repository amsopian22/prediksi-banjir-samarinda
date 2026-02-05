"""
DuckDB Database Manager for Flood Prediction System.
Provides lightweight, fast OLAP storage for historical predictions and weather data.
"""
import duckdb
import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
DB_PATH = "data/flood_data.duckdb"

class FloodDatabase:
    """Manager class for DuckDB operations."""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._ensure_dir()
        self._init_schema()
    
    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _get_conn(self):
        try:
            return duckdb.connect(self.db_path)
        except duckdb.SerializationException as e:
            logger.error(f"Database corrupted or version mismatch: {e}. Renaming and starting fresh.")
            old_path = self.db_path + ".corrupted"
            if os.path.exists(self.db_path):
                import shutil
                shutil.move(self.db_path, old_path)
            return duckdb.connect(self.db_path)

    def _init_schema(self):
        """Initialize database schema."""
        try:
            with self._get_conn() as conn:
                # Historical predictions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP,
                        latitude DOUBLE,
                        longitude DOUBLE,
                        rain_24h DOUBLE,
                        tide_level DOUBLE,
                        probability DOUBLE,
                        risk_level VARCHAR,
                        actual_outcome VARCHAR,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Weather data table (Open-Meteo)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS weather_data (
                        id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP,
                        temperature DOUBLE,
                        humidity DOUBLE,
                        precipitation DOUBLE,
                        rain DOUBLE,
                        pressure DOUBLE,
                        wind_speed DOUBLE,
                        source VARCHAR DEFAULT 'open-meteo'
                    )
                """)
                
                # BMKG Weather data table (per kelurahan)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS bmkg_weather (
                        id INTEGER PRIMARY KEY,
                        kelurahan_code VARCHAR,
                        kelurahan_name VARCHAR,
                        kecamatan_name VARCHAR,
                        timestamp TIMESTAMP,
                        local_datetime TIMESTAMP,
                        temperature DOUBLE,
                        humidity DOUBLE,
                        precipitation DOUBLE,
                        weather_code INTEGER,
                        weather_desc VARCHAR,
                        wind_speed DOUBLE,
                        wind_direction VARCHAR,
                        cloud_cover DOUBLE,
                        visibility DOUBLE,
                        fetched_at TIMESTAMP DEFAULT CAST(CURRENT_TIMESTAMP AS TIMESTAMP)
                    )
                """)
                
                # Create sequences for auto-increment
                conn.execute("CREATE SEQUENCE IF NOT EXISTS pred_seq START 1")
                conn.execute("CREATE SEQUENCE IF NOT EXISTS weather_seq START 1")
                conn.execute("CREATE SEQUENCE IF NOT EXISTS bmkg_seq START 1")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            # Potentially the file is read-only or locked

    
    def log_prediction(self, lat, lon, rain_24h, tide_level, probability, risk_level):
        """Log a prediction to the database."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO predictions (id, timestamp, latitude, longitude, rain_24h, tide_level, probability, risk_level)
                VALUES (nextval('pred_seq'), CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            """, [lat, lon, rain_24h, tide_level, probability, risk_level])
    
    def log_weather(self, df: pd.DataFrame):
        """Log weather data from DataFrame."""
        with self._get_conn() as conn:
            # Insert from DataFrame
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT INTO weather_data (id, timestamp, temperature, humidity, precipitation, rain, pressure, wind_speed)
                    VALUES (nextval('weather_seq'), ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row.get('timestamp'), 
                    row.get('temperature_2m'),
                    row.get('relative_humidity_2m'),
                    row.get('precipitation'),
                    row.get('rain'),
                    row.get('surface_pressure'),
                    row.get('wind_speed_10m')
                ])
    
    def get_prediction_history(self, limit=100):
        """Get recent prediction history."""
        with self._get_conn() as conn:
            return conn.execute(f"""
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """).fetchdf()
    
    def get_weather_history(self, hours=72):
        """Get recent weather data."""
        with self._get_conn() as conn:
            return conn.execute(f"""
                SELECT * FROM weather_data 
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '{hours} hours'
                ORDER BY timestamp DESC
            """).fetchdf()
    
    def get_stats(self):
        """Get database statistics."""
        try:
            with self._get_conn() as conn:
                pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                weather_count = conn.execute("SELECT COUNT(*) FROM weather_data").fetchone()[0]
                try:
                    bmkg_count = conn.execute("SELECT COUNT(*) FROM bmkg_weather").fetchone()[0]
                except duckdb.CatalogException:
                    bmkg_count = 0
                return {
                    "predictions": pred_count,
                    "weather_records": weather_count,
                    "bmkg_records": bmkg_count
                }
        except duckdb.CatalogException as e:
            logger.warning(f"Error getting stats: {e}")
            return {"predictions": 0, "weather_records": 0, "bmkg_records": 0}
    
    def log_bmkg_weather(self, df: pd.DataFrame, kelurahan_code: str, kelurahan_name: str, kecamatan_name: str = ""):
        """Log BMKG weather data for a kelurahan."""
        with self._get_conn() as conn:
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT INTO bmkg_weather (
                        id, kelurahan_code, kelurahan_name, kecamatan_name, 
                        timestamp, local_datetime, temperature, humidity, 
                        precipitation, weather_code, weather_desc, 
                        wind_speed, wind_direction, cloud_cover, visibility
                    )
                    VALUES (nextval('bmkg_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    kelurahan_code,
                    kelurahan_name,
                    kecamatan_name,
                    row.get('date'),
                    row.get('date'),
                    row.get('temperature', 0),
                    row.get('humidity', 0),
                    row.get('precipitation', 0),
                    row.get('weather_code', 0),
                    row.get('weather_desc', ''),
                    row.get('wind_speed', 0),
                    row.get('wind_direction', ''),
                    row.get('cloud_cover', 0),
                    row.get('visibility', 0)
                ])
    
    def get_bmkg_weather(self, kelurahan_code: str = None, hours: int = 72):
        """Get BMKG weather data, optionally filtered by kelurahan."""
        try:
            with self._get_conn() as conn:
                if kelurahan_code:
                    return conn.execute(f"""
                        SELECT * FROM bmkg_weather 
                        WHERE kelurahan_code = ?
                        AND fetched_at >= CAST(CURRENT_TIMESTAMP AS TIMESTAMP) - INTERVAL '{hours} hours'
                        ORDER BY timestamp DESC
                    """, [kelurahan_code]).fetchdf()
                else:
                    return conn.execute(f"""
                        SELECT * FROM bmkg_weather 
                        WHERE fetched_at >= CAST(CURRENT_TIMESTAMP AS TIMESTAMP) - INTERVAL '{hours} hours'
                        ORDER BY timestamp DESC
                    """).fetchdf()
        except duckdb.CatalogException as e:
            # Table doesn't exist yet - return empty DataFrame
            logger.warning(f"bmkg_weather table not found: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    
    def get_latest_bmkg_by_kelurahan(self):
        """Get latest weather data for each kelurahan."""
        try:
            with self._get_conn() as conn:
                return conn.execute("""
                    SELECT DISTINCT ON (kelurahan_code) 
                        kelurahan_code, kelurahan_name, kecamatan_name,
                        timestamp, temperature, humidity, precipitation, 
                        weather_desc, wind_speed
                    FROM bmkg_weather
                    ORDER BY kelurahan_code, fetched_at DESC
                """).fetchdf()
        except duckdb.CatalogException as e:
            logger.warning(f"bmkg_weather table not found: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

# Singleton instance
_db_instance = None

def get_db():
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FloodDatabase()
    return _db_instance
