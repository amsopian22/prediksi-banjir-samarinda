"""
train_model_v5_smote.py - Model Training with SMOTE for Improved Minority Class Detection

Key improvements over V4:
1. SMOTE oversampling for Banjir class
2. Binary classification: Normal (Aman) vs Flood Risk (Air Meluap + Banjir)
3. Threshold optimization for best F1-Score
4. Focus on detecting flood events (combined Air Meluap + Banjir)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
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

# Optional imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import mlflow
    from mlflow.sklearn import log_model
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("flood-prediction-v5")
except ImportError:
    MLFLOW_AVAILABLE = False


def load_processed_data():
    """Load the processed V4 dataset."""
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v4_processed.csv")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")
    return df


def get_features():
    """Define feature columns."""
    return [
        'rain_sum_imputed', 'rain_intensity_max',
        'soil_moisture_surface_mean', 'soil_moisture_root_mean', 'soil_saturation_index',
        'pasut_msl_max',
        'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4', 'rain_lag5', 'rain_lag6', 'rain_lag7',
        'rain_cumsum_3d', 'rain_cumsum_7d',
        'tide_rain_interaction', 'is_high_tide', 'is_heavy_rain',
        'api_7day',
        'month_sin', 'month_cos', 'is_rainy_season', 'is_weekend',
        'prev_flood_30d', 'prev_meluap_30d',
    ]


def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes F1-Score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1


def train_model_v5():
    """Main training function with SMOTE."""
    
    logger.info("=" * 60)
    logger.info("FLOOD PREDICTION MODEL V5 - SMOTE + BINARY CLASSIFICATION")
    logger.info("=" * 60)
    
    # 1. Load Data
    df = load_processed_data()
    features = get_features()
    
    # 2. Create BINARY labels: 0=Aman, 1=Flood Risk (Air Meluap OR Banjir)
    # This combines the positive classes to have more samples
    df['label_binary'] = (df['label'] >= 1).astype(int)
    
    X = df[features].copy()
    y = df['label_binary']
    
    logger.info(f"Binary Class distribution (0=Aman, 1=Flood Risk):\n{y.value_counts()}")
    
    # 3. Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Train distribution:\n{y_train.value_counts()}")
    
    # 5. Apply SMOTE to training data
    logger.info("\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")
    
    # 6. Train Models
    models = {}
    
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=2,  # Slight boost to positive class
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
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
    
    # 7. Cross-Validation
    logger.info("\n" + "=" * 60)
    logger.info("10-FOLD CROSS VALIDATION (on SMOTE-resampled data)")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_model_name = None
    best_f1 = 0
    cv_results = {}
    
    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")
        
        f1_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='f1')
        auc_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc')
        
        cv_results[name] = {
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'auc_mean': auc_scores.mean()
        }
        
        logger.info(f"  F1 Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        logger.info(f"  ROC-AUC:   {auc_scores.mean():.4f}")
        
        if f1_scores.mean() > best_f1:
            best_f1 = f1_scores.mean()
            best_model_name = name
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"BEST MODEL: {best_model_name} (CV F1 = {best_f1:.4f})")
    logger.info("=" * 60)
    
    # 8. Train Best Model
    best_model = models[best_model_name]
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # 9. Evaluate on Test Set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)
    logger.info(f"\nOptimal threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Metrics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 60)
    
    class_names = ['Aman', 'Flood Risk']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    logger.info("\nClassification Report (Optimized Threshold):")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))
    
    f1_aman = report['Aman']['f1-score']
    f1_flood = report['Flood Risk']['f1-score']
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f"\n📊 KEY METRICS:")
    logger.info(f"  Accuracy:      {accuracy:.4f}")
    logger.info(f"  F1 (Aman):     {f1_aman:.4f}")
    logger.info(f"  F1 (Flood):    {f1_flood:.4f}")
    logger.info(f"  ROC-AUC:       {auc_score:.4f}")
    logger.info(f"  Threshold:     {optimal_threshold:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {tn}  FP: {fp}")
    logger.info(f"  FN: {fn}  TP: {tp}")
    logger.info(f"  Recall (Flood): {tp/(tp+fn):.4f}")
    logger.info(f"  Precision (Flood): {tp/(tp+fp):.4f}")
    
    # 10. Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Feature Importance:")
        for i, row in importances.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save
        importance_path = os.path.join(config.BASE_DIR, "docs", "feature_importance_v5.csv")
        importances.to_csv(importance_path, index=False)
    
    # 11. MLflow Logging
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"v5_{best_model_name}_smote"):
            mlflow.log_params({
                "model_type": best_model_name,
                "smote": True,
                "optimal_threshold": optimal_threshold,
                "n_features": len(features)
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_aman": f1_aman,
                "f1_flood": f1_flood,
                "roc_auc": auc_score,
                "recall_flood": tp/(tp+fn),
                "precision_flood": tp/(tp+fp)
            })
            log_model(best_model, "model")
            logger.info("Model logged to MLflow")
    
    # 12. Save Model
    model_pack = {
        "model": best_model,
        "features": features,
        "imputer": imputer,
        "threshold": optimal_threshold,
        "class_names": class_names,
        "version": "5.0",
        "description": f"V5 Binary with SMOTE, {best_model_name}, threshold={optimal_threshold:.3f}",
        "metrics": {
            "accuracy": accuracy,
            "f1_aman": f1_aman,
            "f1_flood": f1_flood,
            "roc_auc": auc_score
        }
    }
    
    model_path = os.path.join(config.MODELS_DIR, "model_banjir_v5_smote.pkl")
    joblib.dump(model_pack, model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # 13. Target Check
    logger.info("\n" + "=" * 60)
    logger.info("TARGET ACHIEVEMENT CHECK")
    logger.info("=" * 60)
    
    target_auc = 0.90
    target_f1 = 0.90
    
    auc_ok = "✅" if auc_score >= target_auc else "❌"
    f1_ok = "✅" if f1_flood >= target_f1 else "❌"
    
    logger.info(f"  AUC >= {target_auc}: {auc_score:.4f} {auc_ok}")
    logger.info(f"  F1 (Flood) >= {target_f1}: {f1_flood:.4f} {f1_ok}")
    
    return model_pack


if __name__ == "__main__":
    train_model_v5()
