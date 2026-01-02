"""
rebuild_dataset_v4.py - Data Preprocessing for Flood Prediction Model V4

Key improvements:
1. Uses ALL 2,161 rows with status_siaga labels
2. 3-class classification: Aman (0), Air Meluap (1), Banjir (2)
3. Removes duplicate dates
4. Enhanced feature engineering
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_class_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert status_siaga to numeric labels."""
    label_map = {
        'Aman': 0,
        'Air Meluap': 1,
        'Banjir': 2
    }
    df['label'] = df['status_siaga'].map(label_map)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """Add lag features for a given column."""
    for lag in lags:
        df[f'{col}_lag{lag}'] = df[col].shift(lag).fillna(0)
    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative rain features."""
    # Cumulative rain 3 days
    df['rain_cumsum_3d'] = df['rain_sum_imputed'].rolling(window=3, min_periods=1).sum()
    # Cumulative rain 7 days
    df['rain_cumsum_7d'] = df['rain_sum_imputed'].rolling(window=7, min_periods=1).sum()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions."""
    # Tide x Rain interaction
    df['tide_rain_interaction'] = df['pasut_msl_max'] * df['rain_sum_imputed']
    
    # Soil saturation index (combined surface + root)
    df['soil_saturation_index'] = (df['soil_moisture_surface_mean'] + df['soil_moisture_root_mean']) / 2
    
    # Boolean flags
    df['is_high_tide'] = (df['pasut_msl_max'] > 2.5).astype(int)
    df['is_heavy_rain'] = (df['rain_sum_imputed'] > 50).astype(int)
    
    return df


def add_api_feature(df: pd.DataFrame, k: float = 0.85) -> pd.DataFrame:
    """Add Antecedent Precipitation Index (API) - 7 day weighted sum."""
    # Create lag columns first if not exist
    for i in range(1, 8):
        if f'rain_lag{i}' not in df.columns:
            df[f'rain_lag{i}'] = df['rain_sum_imputed'].shift(i).fillna(0)
    
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
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features (seasonality)."""
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    
    # Month as cyclical feature (sin/cos encoding)
    df['month'] = df['tanggal'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Is rainy season (Nov-Mar)
    df['is_rainy_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
    
    # Day of week (weekday vs weekend)
    df['day_of_week'] = df['tanggal'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def add_flood_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on recent flood history."""
    # Number of flood events in last 30 days
    df['prev_flood_30d'] = df['label'].apply(lambda x: 1 if x == 2 else 0).rolling(window=30, min_periods=1).sum()
    
    # Number of air meluap events in last 30 days
    df['prev_meluap_30d'] = df['label'].apply(lambda x: 1 if x >= 1 else 0).rolling(window=30, min_periods=1).sum()
    
    return df


def rebuild_dataset_v4():
    """Main function to rebuild dataset with enhanced features."""
    
    # 1. Load Data
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v2_advanced.csv")
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Status siaga distribution:\n{df['status_siaga'].value_counts()}")
    
    # 2. Remove duplicates (keep first occurrence)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.sort_values('tanggal').drop_duplicates(subset=['tanggal'], keep='first')
    logger.info(f"After removing duplicates: {df.shape}")
    
    # 3. Create class labels from status_siaga
    df = create_class_labels(df)
    
    # Check for missing labels
    missing_labels = df['label'].isna().sum()
    if missing_labels > 0:
        logger.warning(f"Found {missing_labels} rows with missing labels. Dropping...")
        df = df.dropna(subset=['label'])
    
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    # 4. Sort by date for time-series features
    df = df.sort_values('tanggal').reset_index(drop=True)
    
    # 5. Feature Engineering
    logger.info("Creating lag features...")
    df = add_lag_features(df, 'rain_sum_imputed', list(range(1, 8)))
    
    logger.info("Creating cumulative features...")
    df = add_cumulative_features(df)
    
    logger.info("Creating interaction features...")
    df = add_interaction_features(df)
    
    logger.info("Creating API feature...")
    df = add_api_feature(df, k=config.API_DECAY_FACTOR)
    
    logger.info("Creating time features...")
    df = add_time_features(df)
    
    logger.info("Creating flood history features...")
    df = add_flood_history_features(df)
    
    # 6. Define final feature set
    features = [
        # Current day weather
        'rain_sum_imputed',
        'rain_intensity_max',
        
        # Soil moisture
        'soil_moisture_surface_mean',
        'soil_moisture_root_mean',
        'soil_saturation_index',
        
        # Tide
        'pasut_msl_max',
        
        # Lag features
        'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4', 'rain_lag5', 'rain_lag6', 'rain_lag7',
        
        # Cumulative
        'rain_cumsum_3d',
        'rain_cumsum_7d',
        
        # Interactions
        'tide_rain_interaction',
        'is_high_tide',
        'is_heavy_rain',
        
        # Derived
        'api_7day',
        
        # Time features
        'month_sin',
        'month_cos',
        'is_rainy_season',
        'is_weekend',
        
        # Flood history
        'prev_flood_30d',
        'prev_meluap_30d',
    ]
    
    # Ensure all features exist
    for f in features:
        if f not in df.columns:
            logger.warning(f"Feature {f} not found. Setting to 0.")
            df[f] = 0
    
    # 7. Select final columns
    final_cols = ['tanggal'] + features + ['label', 'status_siaga']
    df_final = df[final_cols].copy()
    
    # 8. Save processed dataset
    output_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v4_processed.csv")
    df_final.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")
    
    # 9. Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET V4 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(df_final)}")
    logger.info(f"Features: {len(features)}")
    logger.info(f"\nClass distribution:")
    logger.info(f"  Aman (0):       {(df_final['label'] == 0).sum()}")
    logger.info(f"  Air Meluap (1): {(df_final['label'] == 1).sum()}")
    logger.info(f"  Banjir (2):     {(df_final['label'] == 2).sum()}")
    logger.info(f"\nDate range: {df_final['tanggal'].min()} to {df_final['tanggal'].max()}")
    
    return df_final, features


if __name__ == "__main__":
    rebuild_dataset_v4()
