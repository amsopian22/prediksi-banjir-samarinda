"""
Feature Store - Centralized Feature Engineering Module
Consolidates all feature engineering logic for the flood prediction model.

This module extracts and centralizes the feature computation logic from
model_utils.py and dashboard.py for easier maintenance and testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import config
import logging

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Centralized feature engineering for flood prediction model.
    
    Consolidates all feature computation logic in one place:
    - Daily features for single-point predictions
    - Hourly features for time series predictions
    - Feature name registry for model compatibility
    """
    
    # Feature groups for documentation and validation
    CORE_FEATURES = [
        "rain_sum_imputed",
        "rain_intensity_max",
        "soil_moisture_surface_mean",
        "soil_moisture_root_mean",
        "pasut_msl_max"
    ]
    
    LAG_FEATURES = [f"rain_lag{i}" for i in range(1, 8)]
    
    DERIVED_FEATURES = [
        "soil_saturation_index",
        "rain_cumsum_3d",
        "rain_cumsum_7d",
        "tide_rain_interaction",
        "is_high_tide",
        "is_heavy_rain",
        "api_7day"
    ]
    
    TEMPORAL_FEATURES = [
        "month_sin",
        "month_cos",
        "is_rainy_season",
        "is_weekend"
    ]
    
    HISTORICAL_FEATURES = [
        "prev_flood_30d",
        "prev_meluap_30d"
    ]
    
    V6_FEATURES = [
        "rain_intensity_3h",
        "rain_burst_count",
        "soil_saturation_trend",
        "tide_rain_sync",
        "consecutive_rain_days",
        "hour_risk_factor",
        "drain_capacity_index",
        "upstream_rain_6h",
        "wind_speed_max",
        "rainfall_acceleration"
    ]
    
    REGRESSION_FEATURES = [
        "elevation_m",
        "drainage_capacity"
    ]
    
    # Decay factor for API calculation
    API_DECAY_FACTOR = config.API_DECAY_FACTOR
    
    @classmethod
    def get_all_feature_names(cls) -> List[str]:
        """Get list of all feature names used by the model."""
        return (
            cls.CORE_FEATURES +
            cls.LAG_FEATURES +
            cls.DERIVED_FEATURES +
            cls.TEMPORAL_FEATURES +
            cls.HISTORICAL_FEATURES +
            cls.V6_FEATURES +
            cls.REGRESSION_FEATURES
        )
    
    @classmethod
    def compute_api(cls, rain_values: List[float]) -> float:
        """
        Compute Antecedent Precipitation Index (API).
        
        API = P_0 + k*P_1 + k^2*P_2 + ... + k^n*P_n
        
        Args:
            rain_values: List of rainfall values [today, lag1, lag2, ..., lag7]
            
        Returns:
            API value
        """
        k = cls.API_DECAY_FACTOR
        api = 0.0
        for i, rain in enumerate(rain_values):
            api += (k ** i) * rain
        return api
    
    @classmethod
    def compute_daily_features(cls, input_data: Dict[str, float]) -> Dict[str, float]:
        """
        Compute all features for a single prediction point.
        Used by predict_flood() for manual simulation.
        
        Args:
            input_data: Raw input data dictionary with basic measurements
            
        Returns:
            Dictionary with all computed features
        """
        # Extract raw values with sensible defaults
        rain_today = input_data.get("rain_sum_imputed", 
                                    input_data.get("rain_rolling_24h", 0))
        rain_intensity = input_data.get("rain_intensity_max", 0)
        pasut_max = input_data.get("pasut_msl_max", 0)
        soil_surface = input_data.get("soil_moisture_surface_mean", 0.45)
        soil_root = input_data.get("soil_moisture_root_mean", 0.45)
        
        # Extract lag values
        lags = []
        for i in range(1, 8):
            lag_val = input_data.get(f"rain_lag{i}", 
                                     input_data.get(f"hujan_lag{i}", 0))
            lags.append(lag_val)
        
        # Compute API
        rain_sequence = [rain_today] + lags
        api_7day = cls.compute_api(rain_sequence)
        
        # Build feature dictionary
        features = {
            # Core features
            "rain_sum_imputed": rain_today,
            "rain_intensity_max": rain_intensity,
            "soil_moisture_surface_mean": soil_surface,
            "soil_moisture_root_mean": soil_root,
            "soil_saturation_index": (soil_surface + soil_root) / 2,
            "pasut_msl_max": pasut_max,
            
            # Lag features
            "rain_lag1": lags[0],
            "rain_lag2": lags[1],
            "rain_lag3": lags[2],
            "rain_lag4": lags[3],
            "rain_lag5": lags[4],
            "rain_lag6": lags[5],
            "rain_lag7": lags[6],
            
            # Derived features
            "rain_cumsum_3d": input_data.get("rain_cumsum_3d", 
                                              rain_today + lags[0] + lags[1]),
            "rain_cumsum_7d": input_data.get("rain_cumsum_7d", 
                                              sum([rain_today] + lags[:6])),
            "tide_rain_interaction": pasut_max * rain_today,
            "is_high_tide": 1 if pasut_max > 2.5 else 0,
            "is_heavy_rain": 1 if rain_today > 50 else 0,
            "api_7day": api_7day,
            
            # Temporal features
            "month_sin": input_data.get("month_sin", 0),
            "month_cos": input_data.get("month_cos", 1),
            "is_rainy_season": input_data.get("is_rainy_season", 1),
            "is_weekend": input_data.get("is_weekend", 0),
            
            # Historical features
            "prev_flood_30d": input_data.get("prev_flood_30d", 0),
            "prev_meluap_30d": input_data.get("prev_meluap_30d", 0),
            
            # V6 features
            "rain_intensity_3h": input_data.get("rain_intensity_3h", 
                                                 rain_intensity * 3),
            "rain_burst_count": input_data.get("rain_burst_count", 0),
            "soil_saturation_trend": input_data.get("soil_saturation_trend", 0),
            "tide_rain_sync": 1 if (pasut_max > 2.5 and rain_today > 50) else 0,
            "consecutive_rain_days": input_data.get("consecutive_rain_days", 0),
            "hour_risk_factor": input_data.get("hour_risk_factor", 1.0),
            "upstream_rain_6h": input_data.get("upstream_rain_6h", 0),
            "wind_speed_max": input_data.get("wind_speed_max", 0),
            "rainfall_acceleration": input_data.get("rainfall_acceleration", 0),
            
            # Regression features
            "elevation_m": input_data.get("elevation_m", 2.0),
            "drainage_capacity": input_data.get("drainage_capacity", 50.0),
        }
        
        # Compute drain_capacity_index after rain_cumsum_7d is available
        features["drain_capacity_index"] = input_data.get(
            "drain_capacity_index", 
            features["rain_cumsum_7d"] / 200.0
        )
        
        return features
    
    @classmethod
    def compute_hourly_features(
        cls, 
        hourly_df: pd.DataFrame, 
        daily_lags_lookup: Dict
    ) -> pd.DataFrame:
        """
        Compute all features for hourly time series prediction.
        
        Args:
            hourly_df: DataFrame with columns: time, precipitation, est, soil_moisture_*
            daily_lags_lookup: Dict mapping date -> {hujan_lag1, hujan_lag2, ...}
            
        Returns:
            DataFrame with all computed features
        """
        df = hourly_df.copy()
        
        # Rolling rain features
        df['rain_sum_imputed'] = df['precipitation'].rolling(
            window=24, min_periods=1
        ).sum()
        df['rain_intensity_max'] = df['precipitation']
        df['rain_rolling_3h'] = df['precipitation'].rolling(
            window=3, min_periods=1
        ).sum()
        
        # Tide features
        df['pasut_msl_max'] = df['est']
        
        # Soil moisture (use from df or default)
        df['soil_moisture_surface_mean'] = df.get('soil_moisture_surface', 0.4)
        df['soil_moisture_root_mean'] = df.get('soil_moisture_root', 0.4)
        
        # Apply daily lags
        df['date'] = df['time'].dt.date
        
        for lag_num in range(1, 8):
            col_name = f'rain_lag{lag_num}'
            df[col_name] = df['date'].apply(
                lambda d: daily_lags_lookup.get(d, {}).get(f'hujan_lag{lag_num}', 0)
            )
        
        # Compute API
        k = cls.API_DECAY_FACTOR
        df['api_7day'] = (
            df['rain_sum_imputed'] +
            k * df['rain_lag1'] +
            (k**2) * df['rain_lag2'] +
            (k**3) * df['rain_lag3'] +
            (k**4) * df['rain_lag4'] +
            (k**5) * df['rain_lag5'] +
            (k**6) * df['rain_lag6'] +
            (k**7) * df['rain_lag7']
        )
        
        # Derived features
        df['soil_saturation_index'] = (
            df['soil_moisture_surface_mean'] + df['soil_moisture_root_mean']
        ) / 2
        df['rain_cumsum_3d'] = df['precipitation'].rolling(
            window=72, min_periods=1
        ).sum()
        df['rain_cumsum_7d'] = df['precipitation'].rolling(
            window=168, min_periods=1
        ).sum()
        df['tide_rain_interaction'] = df['pasut_msl_max'] * df['rain_sum_imputed']
        df['is_high_tide'] = (df['pasut_msl_max'] > 2.5).astype(int)
        df['is_heavy_rain'] = (df['rain_sum_imputed'] > 50).astype(int)
        
        # Temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
        df['is_rainy_season'] = df['time'].dt.month.isin([11, 12, 1, 2, 3]).astype(int)
        df['is_weekend'] = (df['time'].dt.dayofweek >= 5).astype(int)
        
        # Historical features (default to 0 for forecast)
        df['prev_flood_30d'] = 0
        df['prev_meluap_30d'] = 0
        
        # V6 features
        df['rain_intensity_3h'] = df['rain_rolling_3h']
        df['rain_burst_count'] = (
            df['precipitation'] > 10
        ).rolling(window=24, min_periods=1).sum()
        df['soil_saturation_trend'] = df['soil_saturation_index'].diff(periods=72).fillna(0)
        df['tide_rain_sync'] = (
            (df['pasut_msl_max'] > 2.5) & (df['rain_sum_imputed'] > 50)
        ).astype(int)
        
        # Consecutive rain days
        has_rain = (df['precipitation'] > 0).astype(int)
        rain_streaks = has_rain.groupby(
            (has_rain != has_rain.shift()).cumsum()
        ).cumsum()
        df['consecutive_rain_days'] = (rain_streaks / 24.0).fillna(0)
        
        # Hour risk factor
        hour = df['time'].dt.hour
        is_night = ((hour >= 22) | (hour <= 6)).astype(int)
        df['hour_risk_factor'] = 1.0 + (is_night * 0.2)
        
        # Drain capacity index
        df['drain_capacity_index'] = (df['rain_cumsum_7d'] / 200.0).clip(0, 2.0)
        
        # Upstream rain (default to 0 if not available)
        df['upstream_rain_6h'] = df.get('upstream_precipitation', 0)
        
        # Wind speed max
        if 'wind_speed' in df.columns:
            df['wind_speed_max'] = df['wind_speed'].rolling(
                window=24, min_periods=1
            ).max()
        else:
            df['wind_speed_max'] = 0
        
        # Rainfall acceleration
        rain_smooth = df['precipitation'].rolling(window=3, min_periods=1).mean()
        df['rainfall_acceleration'] = rain_smooth.diff().fillna(0)
        
        # Regression features (defaults)
        df['elevation_m'] = 2.0
        df['drainage_capacity'] = 50.0
        
        return df
    
    @classmethod
    def select_model_features(
        cls, 
        df: pd.DataFrame, 
        features_needed: List[str]
    ) -> pd.DataFrame:
        """
        Select only the features required by the model.
        
        Args:
            df: DataFrame with all computed features
            features_needed: List of feature names the model expects
            
        Returns:
            DataFrame with only model-required columns
        """
        available_cols = [f for f in features_needed if f in df.columns]
        missing_cols = [f for f in features_needed if f not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing features for model: {missing_cols}")
            
        return df[available_cols].fillna(0)
