
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_v3():
    """
    Train Model V3 with Enhanced Features:
    - Lag features t-1 to t-7 (7 days history)
    - Rolling 3h rain intensity
    - SMOTE for imbalanced data
    - 10-Fold Cross Validation
    - Model comparison to find the best
    """
    
    # 1. Load Data
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v2_advanced.csv")
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Features Engineering
    logger.info("Creating lag features t-1 to t-7...")
    
    # Lag Features (t-1 to t-7)
    for lag in range(1, 8):
        df[f'hujan_lag{lag}'] = df['rain_sum_imputed'].shift(lag).fillna(0)
    
    # Rolling 3h Rain Intensity (simulated from daily data)
    df['rain_rolling_3h'] = df['rain_intensity_max'] * 3  # Approximation for daily data
    
    # API (Antecedent Precipitation Index) - 7 day weighted sum
    k = config.API_DECAY_FACTOR  # 0.85
    df['api_7day'] = (
        df['rain_sum_imputed'] +
        k * df['hujan_lag1'] +
        (k**2) * df['hujan_lag2'] +
        (k**3) * df['hujan_lag3'] +
        (k**4) * df['hujan_lag4'] +
        (k**5) * df['hujan_lag5'] +
        (k**6) * df['hujan_lag6'] +
        (k**7) * df['hujan_lag7']
    )
    
    # Target: is_banjir (0 or 1)
    df = df.dropna(subset=['is_banjir'])
    
    # 3. Define Features (V3 - Enhanced)
    features = [
        # Current Day Features
        'rain_sum_imputed',           # Total Hujan Harian
        'rain_intensity_max',         # Intensitas Terderas (mm/hour)
        'rain_rolling_3h',            # NEW: Rolling 3 Jam
        
        # Soil Condition
        'soil_moisture_surface_mean', # 0-7cm
        'soil_moisture_root_mean',    # 7-28cm
        
        # Tide
        'pasut_msl_max',              # Tide Max
        
        # Lag Features (7 Days History)
        'hujan_lag1',
        'hujan_lag2',
        'hujan_lag3',
        'hujan_lag4',
        'hujan_lag5',
        'hujan_lag6',
        'hujan_lag7',
        
        # Derived
        'api_7day'
    ]
    
    target = 'is_banjir'
    
    # Check which features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0
    
    X = df[features].copy()
    y = df[target].copy()
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # 4. Train/Test Split (for final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Apply SMOTE to training data
    logger.info("Applying SMOTE for imbalanced data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE - Training set distribution:\n{pd.Series(y_train_resampled).value_counts()}")
    
    # 6. Define Models to Compare
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    # 7. 10-Fold Cross Validation for each model
    logger.info("=" * 60)
    logger.info("10-FOLD CROSS VALIDATION")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_model_name = None
    best_f1_mean = 0
    cv_results = {}
    
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        
        # Scale data for LogisticRegression
        if name == 'LogisticRegression':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_resampled)
        else:
            X_scaled = X_train_resampled
        
        # Cross validation scores
        f1_scores = cross_val_score(model, X_scaled, y_train_resampled, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X_scaled, y_train_resampled, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(model, X_scaled, y_train_resampled, cv=cv, scoring='accuracy')
        
        cv_results[name] = {
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'auc_mean': auc_scores.mean(),
            'acc_mean': acc_scores.mean()
        }
        
        logger.info(f"  F1 Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        logger.info(f"  ROC-AUC:   {auc_scores.mean():.4f} (+/- {auc_scores.std() * 2:.4f})")
        logger.info(f"  Accuracy:  {acc_scores.mean():.4f} (+/- {acc_scores.std() * 2:.4f})")
        
        if f1_scores.mean() > best_f1_mean:
            best_f1_mean = f1_scores.mean()
            best_model_name = name
    
    logger.info("\n" + "=" * 60)
    logger.info(f"BEST MODEL: {best_model_name} (F1 = {best_f1_mean:.4f})")
    logger.info("=" * 60)
    
    # 8. Train Best Model on Full SMOTE Data
    best_model = models[best_model_name]
    
    if best_model_name == 'LogisticRegression':
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_resampled)
        X_test_final = scaler.transform(X_test)
    else:
        X_train_final = X_train_resampled
        X_test_final = X_test
        scaler = None
    
    logger.info(f"\nTraining final {best_model_name} model...")
    best_model.fit(X_train_final, y_train_resampled)
    
    # 9. Final Evaluation on Test Set (unseen data)
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
    
    # Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importance:")
        logger.info("\n" + str(importances))
        
        # Save feature importance
        importance_out = os.path.join(config.BASE_DIR, "docs", "feature_importance_v3.csv")
        importances.to_csv(importance_out, index=False)
        logger.info(f"Feature importance saved to {importance_out}")
    
    # 10. Save Model Package
    model_pack = {
        "model": best_model,
        "features": features,
        "threshold": 0.5,
        "version": "3.0",
        "description": f"V3 with lag t-1 to t-7, rolling 3h, SMOTE, 10-fold CV. Best: {best_model_name}",
        "scaler": scaler,  # Include scaler if used
        "cv_results": cv_results
    }
    
    model_out = os.path.join(config.MODELS_DIR, "model_banjir_v3_enhanced.pkl")
    joblib.dump(model_pack, model_out)
    logger.info(f"\nModel saved to {model_out}")
    
    return model_pack

if __name__ == "__main__":
    train_model_v3()
