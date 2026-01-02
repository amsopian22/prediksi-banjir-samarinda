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
        return duckdb.connect(self.db_path)
    
    def _init_schema(self):
        """Initialize database schema."""
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
            
            # Weather data table
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
            
            # Create sequence for auto-increment
            conn.execute("CREATE SEQUENCE IF NOT EXISTS pred_seq START 1")
            conn.execute("CREATE SEQUENCE IF NOT EXISTS weather_seq START 1")
    
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
        with self._get_conn() as conn:
            pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            weather_count = conn.execute("SELECT COUNT(*) FROM weather_data").fetchone()[0]
            return {
                "predictions": pred_count,
                "weather_records": weather_count
            }

# Singleton instance
_db_instance = None

def get_db():
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FloodDatabase()
    return _db_instance
