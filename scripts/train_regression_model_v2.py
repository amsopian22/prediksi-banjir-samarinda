
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import logging
import os

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/dataset_banjir_v7_regression.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_banjir_v7_regression.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_banjir_v7_metadata.json")

def train_model():
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Feature Selection
    # Drop non-feature columns
    exclude_cols = ['tanggal', 'label', 'water_depth_cm', 'status_siaga']
    features = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features]
    y = df['water_depth_cm']
    
    logger.info(f"Features ({len(features)}): {features}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    logger.info("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model Performance:")
    logger.info(f"MAE: {mae:.2f} cm")
    logger.info(f"RMSE: {rmse:.2f} cm")
    logger.info(f"R2 Score: {r2:.3f}")
    
    # Save Model
    logger.info(f"Saving model to {MODEL_PATH}...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, MODEL_PATH)
    
    # Save Metadata
    metadata = {
        "features": features,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        },
        "model_type": "GradientBoostingRegressor",
        "description": "Water Depth Prediction (cm)"
    }
    
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Training Complete.")

if __name__ == "__main__":
    train_model()
