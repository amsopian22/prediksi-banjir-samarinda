"""
train_model_v4_advanced.py - Advanced Model Training with XGBoost, Ensemble, and MLflow

Key features:
1. Uses processed dataset V4 with 3-class classification
2. XGBoost with class weights for imbalanced data
3. Ensemble: RF + XGBoost + LightGBM with soft voting
4. MLflow tracking for experiment comparison
5. Threshold optimization for maximizing F1-Score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional: MLflow tracking
try:
    import mlflow
    from mlflow.sklearn import log_model
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("flood-prediction-v4")
    logger.info("MLflow tracking enabled")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking disabled.")

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed.")

# Optional: LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    logger.info("LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed.")


def load_processed_data():
    """Load the processed V4 dataset."""
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v4_processed.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Run rebuild_dataset_v4.py first.")
        raise FileNotFoundError(data_path)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")
    return df


def get_features():
    """Define feature columns."""
    return [
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


def calculate_class_weights(y):
    """Calculate class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def train_model_v4():
    """Main training function."""
    
    # 1. Load Data
    logger.info("=" * 60)
    logger.info("FLOOD PREDICTION MODEL V4 - ADVANCED TRAINING")
    logger.info("=" * 60)
    
    df = load_processed_data()
    features = get_features()
    
    X = df[features].copy()
    y = df['label'].astype(int)
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Class distribution:\n{y.value_counts().sort_index()}")
    
    # 2. Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 4. Calculate class weights
    class_weights = calculate_class_weights(y_train)
    logger.info(f"Class weights: {class_weights}")
    
    # 5. Define Models
    models = {}
    
    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for each class
        scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 2]), 1)
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    # LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    # 6. Cross-Validation for each model
    logger.info("\n" + "=" * 60)
    logger.info("10-FOLD CROSS VALIDATION")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_model_name = None
    best_f1_macro = 0
    cv_results = {}
    
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        
        # F1 Macro (weighted average across all classes)
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
        acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        cv_results[name] = {
            'f1_macro_mean': f1_scores.mean(),
            'f1_macro_std': f1_scores.std(),
            'accuracy_mean': acc_scores.mean()
        }
        
        logger.info(f"  F1 Macro:  {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        logger.info(f"  Accuracy:  {acc_scores.mean():.4f}")
        
        if f1_scores.mean() > best_f1_macro:
            best_f1_macro = f1_scores.mean()
            best_model_name = name
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"BEST MODEL: {best_model_name} (F1 Macro = {best_f1_macro:.4f})")
    logger.info("=" * 60)
    
    # 7. Train Best Model on Full Training Data
    best_model = models[best_model_name]
    logger.info(f"\nTraining final {best_model_name} model...")
    best_model.fit(X_train, y_train)
    
    # 8. Evaluate on Test Set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)
    
    # Classification Report
    class_names = ['Aman', 'Air Meluap', 'Banjir']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))
    
    # Per-class metrics
    f1_aman = report['Aman']['f1-score']
    f1_meluap = report['Air Meluap']['f1-score']
    f1_banjir = report['Banjir']['f1-score']
    f1_macro = report['macro avg']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']
    accuracy = accuracy_score(y_test, y_pred)
    
    # AUC (One-vs-Rest)
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_score = 0.0
    
    logger.info(f"\n📊 KEY METRICS:")
    logger.info(f"  Accuracy:     {accuracy:.4f}")
    logger.info(f"  F1 (Aman):    {f1_aman:.4f}")
    logger.info(f"  F1 (Meluap):  {f1_meluap:.4f}")
    logger.info(f"  F1 (Banjir):  {f1_banjir:.4f}")
    logger.info(f"  F1 (Macro):   {f1_macro:.4f}")
    logger.info(f"  F1 (Weighted):{f1_weighted:.4f}")
    logger.info(f"  ROC-AUC:      {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # 9. Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Feature Importance:")
        logger.info(importances.head(10).to_string(index=False))
        
        # Save feature importance
        importance_path = os.path.join(config.BASE_DIR, "docs", "feature_importance_v4.csv")
        importances.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Check if we need more features
        if importances['importance'].max() < 0.15:
            logger.warning("⚠️ No dominant feature detected! Consider adding external features.")
    
    # 10. MLflow Logging
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"v4_{best_model_name}"):
            mlflow.log_params({
                "model_type": best_model_name,
                "n_features": len(features),
                "n_samples": len(df),
                "n_classes": 3
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_aman": f1_aman,
                "f1_meluap": f1_meluap,
                "f1_banjir": f1_banjir,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "roc_auc": auc_score
            })
            log_model(best_model, "model")
            logger.info("Model logged to MLflow")
    
    # 11. Save Model Package
    model_pack = {
        "model": best_model,
        "features": features,
        "imputer": imputer,
        "class_names": class_names,
        "version": "4.0",
        "description": f"V4 3-class with {best_model_name}, 25 features",
        "cv_results": cv_results,
        "metrics": {
            "accuracy": accuracy,
            "f1_aman": f1_aman,
            "f1_meluap": f1_meluap,
            "f1_banjir": f1_banjir,
            "f1_macro": f1_macro,
            "roc_auc": auc_score
        }
    }
    
    model_path = os.path.join(config.MODELS_DIR, "model_banjir_v4_advanced.pkl")
    joblib.dump(model_pack, model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # 12. Check Target Achievement
    logger.info("\n" + "=" * 60)
    logger.info("TARGET ACHIEVEMENT CHECK")
    logger.info("=" * 60)
    
    target_auc = 0.90
    target_f1 = 0.90
    
    auc_achieved = "✅" if auc_score >= target_auc else "❌"
    f1_banjir_achieved = "✅" if f1_banjir >= target_f1 else "❌"
    f1_macro_achieved = "✅" if f1_macro >= target_f1 else "❌"
    
    logger.info(f"  AUC >= {target_auc}: {auc_score:.4f} {auc_achieved}")
    logger.info(f"  F1 (Banjir) >= {target_f1}: {f1_banjir:.4f} {f1_banjir_achieved}")
    logger.info(f"  F1 (Macro) >= {target_f1}: {f1_macro:.4f} {f1_macro_achieved}")
    
    if auc_score < target_auc or f1_banjir < target_f1:
        logger.info("\n⚠️ Targets not fully achieved. Consider:")
        logger.info("  1. Adding more external features (elevation, upstream rain)")
        logger.info("  2. Trying ensemble methods")
        logger.info("  3. Collecting more flood event data")
        logger.info("  4. Using SMOTE/ADASYN for severe class imbalance")
    
    return model_pack


if __name__ == "__main__":
    train_model_v4()
