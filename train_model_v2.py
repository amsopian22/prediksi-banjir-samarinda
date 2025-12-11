
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_v2():
    # 1. Load Data
    data_path = "dataset_banjir_v2_advanced.csv"
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    
    # 2. Features Engineering
    # Calculate Lags dynamically from the new 'rain_sum_imputed' column
    df['hujan_lag1'] = df['rain_sum_imputed'].shift(1).fillna(0)
    df['hujan_lag2'] = df['rain_sum_imputed'].shift(2).fillna(0)
    
    # Target: is_banjir (0 or 1)
    df = df.dropna(subset=['is_banjir'])
    
    features = [
        'rain_sum_imputed',        # Total Hujan Harian (from Hourly Sum)
        'rain_intensity_max',      # Intensitas Terderas (mm/hour)
        'soil_moisture_surface_mean', # 0-7cm
        'soil_moisture_root_mean',    # 7-28cm
        'pasut_msl_max',           # Tide Max
        'hujan_lag1',              # History
        'hujan_lag2'
    ]
    
    target = 'is_banjir'
    
    X = df[features]
    y = df[target]
    
    # Impute missing values (if any from API fails)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Model Training (Random Forest)
    # Using 'class_weight' to handle imbalance if present
    logger.info("Training Random Forest V2...")
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # 5. Evaluation
    y_pred = rf.predict(X_test)
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Feature Importance:")
    logger.info("\n" + str(importances))
    
    # 6. Save Model
    model_pack = {
        "model": rf,
        "features": features,
        "threshold": 0.5 # Default threshold, can be optimized
    }
    
    model_out = "model_banjir_v2_advanced.pkl"
    joblib.dump(model_pack, model_out)
    logger.info(f"Model saved to {model_out}")

if __name__ == "__main__":
    train_model_v2()
