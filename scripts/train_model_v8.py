
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from datetime import datetime

# (Remove imblearn imports)

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_v8():
    logger.info("🧠 Training Model V8 (10-Year Knowledge)...")
    
    # 1. Load Data
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v8_10years.csv")
    df = pd.read_csv(data_path)
    
    # 2. Select Features
    features = [
        'rain_sum_imputed', 'rain_intensity_max',
        'soil_moisture_surface_mean', 'soil_moisture_root_mean', 'soil_saturation_index',
        'pasut_msl_max',
        'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4', 'rain_lag5', 'rain_lag6', 'rain_lag7',
        'rain_cumsum_3d', 'rain_cumsum_7d',
        'tide_rain_interaction', 'is_high_tide', 'is_heavy_rain',
        'api_7day',
        'month_sin', 'month_cos', 'is_rainy_season', 'is_weekend',
        'prev_flood_30d', 'prev_meluap_30d', 'drain_capacity_index', 'upstream_rain_6h'
    ]
    
    X = df[features]
    y = df['label']
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Define Pipeline with Random Forest (using class_weight='balanced' for imbalance)
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    # 5. Hyperparameter Tuning (Simplified for speed)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    logger.info("🔎 Tuning hyperparameters...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"✅ Best parameters: {grid_search.best_params_}")
    
    # 6. Evaluation
    y_pred = best_model.predict(X_test)
    logger.info("\n" + "=" * 40)
    logger.info("MODEL V8 PERFORMANCE REPORT")
    logger.info("=" * 40)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Aman', 'Air Meluap', 'Banjir']))
    
    # 7. Importance
    rf = best_model.named_steps['classifier']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    logger.info("\n🔑 Feature Importance Top 10:")
    for i in range(10):
        logger.info(f"   {features[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # 8. Save Model
    model_save_path = os.path.join(config.MODELS_DIR, "model_banjir_v8_10years.pkl")
    
    # Prepare model pack for model_utils compatibility
    model_pack = {
        'model': best_model,
        'features': features,
        'version': 'V8_10YR',
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'metrics': {
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
    }
    
    joblib.dump(model_pack, model_save_path)
    logger.info(f"💾 Model V8 saved to: {model_save_path}")
    
    # Update config.py to use this model
    # (Optional: Usually better to let user decide, but I'll prepare the swap)
    logger.info("💡 Recommendation: Update config.MODEL_PATH to point to this new V8 model.")

if __name__ == "__main__":
    train_v8()
