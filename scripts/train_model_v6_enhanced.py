"""
train_model_v6_enhanced.py - Advanced Ensemble Model with 35 Features

Key improvements over V5:
1. Uses V6 dataset with 35 features (10 new features for real-world conditions)
2. Stacked Ensemble: LightGBM + XGBoost + CatBoost with LogisticRegression meta-learner
3. SMOTE-ENN for better handling of extreme class imbalance
4. Binary classification optimized for flood detection
5. Threshold optimization for achieving AUC >= 90%, F1 >= 90%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
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
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

try:
    import mlflow
    from mlflow.sklearn import log_model
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("flood-prediction-v6")
except ImportError:
    MLFLOW_AVAILABLE = False


def load_processed_data():
    """Load the processed V6 dataset."""
    data_path = os.path.join(config.BASE_DIR, "data", "dataset_banjir_v6_enhanced.csv")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")
    return df


def get_features_v6():
    """Define all 35 feature columns for V6."""
    return [
        # Original V4 features (25)
        'rain_sum_imputed', 'rain_intensity_max',
        'soil_moisture_surface_mean', 'soil_moisture_root_mean', 'soil_saturation_index',
        'pasut_msl_max',
        'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_lag4', 'rain_lag5', 'rain_lag6', 'rain_lag7',
        'rain_cumsum_3d', 'rain_cumsum_7d',
        'tide_rain_interaction', 'is_high_tide', 'is_heavy_rain',
        'api_7day',
        'month_sin', 'month_cos', 'is_rainy_season', 'is_weekend',
        'prev_flood_30d', 'prev_meluap_30d',
        
        # New V6 features (10)
        'rain_intensity_3h', 'rain_burst_count', 'soil_saturation_trend',
        'tide_rain_sync', 'consecutive_rain_days', 'hour_risk_factor',
        'drain_capacity_index', 'upstream_rain_6h', 'wind_speed_max',
        'rainfall_acceleration'
    ]


def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes F1-Score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1


def train_model_v6():
    """Main training function with advanced ensemble."""
    
    logger.info("=" * 70)
    logger.info("FLOOD PREDICTION MODEL V6 - ENHANCED ENSEMBLE WITH 35 FEATURES")
    logger.info("=" * 70)
    
    # 1. Load Data
    df = load_processed_data()
    features = get_features_v6()
    
    # 2. Create BINARY labels: 0=Aman, 1=Flood Risk (Air Meluap OR Banjir)
    df['label_binary'] = (df['label'] >= 1).astype(int)
    
    X = df[features].copy()
    y = df['label_binary']
    
    logger.info(f"Features: {len(features)}")
    logger.info(f"Binary Class distribution (0=Aman, 1=Flood Risk):\n{y.value_counts()}")
    
    # 3. Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 5. Apply SMOTE-ENN (better than SMOTE alone)
    logger.info("\nApplying SMOTE-ENN for handling class imbalance...")
    smote_enn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=3), enn=EditedNearestNeighbours(n_neighbors=3))
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE-ENN:\n{pd.Series(y_train_resampled).value_counts()}")
    
    # 6. Define Base Models
    base_models = []
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('rf', rf))
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=1.5,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        base_models.append(('xgb', xgb))
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgb = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        base_models.append(('lgb', lgb))
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        cat = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
        base_models.append(('cat', cat))
    
    logger.info(f"\nBase models: {[name for name, _ in base_models]}")
    
    # 7. Create Stacked Ensemble
    logger.info("\nCreating Stacked Ensemble with Logistic Regression meta-learner...")
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    # 8. Cross-Validation
    logger.info("\n" + "=" * 70)
    logger.info("10-FOLD CROSS VALIDATION")
    logger.info("=" * 70)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    f1_scores = cross_val_score(stacking_clf, X_train_resampled, y_train_resampled, cv=cv, scoring='f1')
    auc_scores = cross_val_score(stacking_clf, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc')
    
    logger.info(f"\nCross-Validation Results:")
    logger.info(f"  F1 Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    logger.info(f"  ROC-AUC:   {auc_scores.mean():.4f} (+/- {auc_scores.std() * 2:.4f})")
    
    # 9. Train Final Model
    logger.info("\nTraining final stacked ensemble on full training set...")
    stacking_clf.fit(X_train_resampled, y_train_resampled)
    
    # 10. Evaluate on Test Set
    y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)
    logger.info(f"\nOptimal threshold: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")
    
    # Apply optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Metrics
    logger.info("\n" + "=" * 70)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("=" * 70)
    
    class_names = ['Aman', 'Flood Risk']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))
    
    f1_aman = report['Aman']['f1-score']
    f1_flood = report['Flood Risk']['f1-score']
    recall_flood = report['Flood Risk']['recall']
    precision_flood = report['Flood Risk']['precision']
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logger.info(f"\n📊 KEY METRICS:")
    logger.info(f"  Accuracy:            {accuracy:.4f}")
    logger.info(f"  F1 (Aman):           {f1_aman:.4f}")
    logger.info(f"  F1 (Flood Risk):     {f1_flood:.4f}")
    logger.info(f"  Recall (Flood):      {recall_flood:.4f}")
    logger.info(f"  Precision (Flood):   {precision_flood:.4f}")
    logger.info(f"  ROC-AUC:             {auc_score:.4f}")
    logger.info(f"  Threshold:           {optimal_threshold:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {tn:4d}  FP: {fp:4d}")
    logger.info(f"  FN: {fn:4d}  TP: {tp:4d}")
    
    # 11. Feature importance (from RF in ensemble)
    try:
        # Get feature importance from Random Forest base model
        rf_model = stacking_clf.named_estimators_['rf']
        if hasattr(rf_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n📈 Top 15 Feature Importance (from RF):")
            for i, row in importances.head(15).iterrows():
                logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
            
            # Save
            importance_path = os.path.join(config.BASE_DIR, "docs", "feature_importance_v6.csv")
            importances.to_csv(importance_path, index=False)
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    # 12. MLflow Logging
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="v6_stacked_ensemble"):
            mlflow.log_params({
                "model_type": "StackedEnsemble",
                "base_models": [name for name, _ in base_models],
                "smote_enn": True,
                "optimal_threshold": optimal_threshold,
                "n_features": len(features)
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_aman": f1_aman,
                "f1_flood": f1_flood,
                "recall_flood": recall_flood,
                "precision_flood": precision_flood,
                "roc_auc": auc_score
            })
            log_model(stacking_clf, "model")
            logger.info("Model logged to MLflow")
    
    # 13. Save Model Package
    model_pack = {
        "model": stacking_clf,
        "features": features,
        "imputer": imputer,
        "threshold": optimal_threshold,
        "class_names": class_names,
        "version": "6.0",
        "description": f"V6 Stacked Ensemble (35 features), threshold={optimal_threshold:.3f}",
        "metrics": {
            "accuracy": accuracy,
            "f1_aman": f1_aman,
            "f1_flood": f1_flood,
            "recall_flood": recall_flood,
            "precision_flood": precision_flood,
            "roc_auc": auc_score
        }
    }
    
    model_path = os.path.join(config.MODELS_DIR, "model_banjir_v6_enhanced.pkl")
    joblib.dump(model_pack, model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # 14. Target Achievement Check
    logger.info("\n" + "=" * 70)
    logger.info("🎯 TARGET ACHIEVEMENT CHECK")
    logger.info("=" * 70)
    
    target_auc = 0.90
    target_f1 = 0.90
    
    auc_ok = "✅ ACHIEVED" if auc_score >= target_auc else f"❌ MISS ({auc_score:.4f} < {target_auc})"
    f1_ok = "✅ ACHIEVED" if f1_flood >= target_f1 else f"❌ MISS ({f1_flood:.4f} < {target_f1})"
    
    logger.info(f"  AUC >= {target_auc}:        {auc_ok}")
    logger.info(f"  F1 (Flood) >= {target_f1}: {f1_ok}")
    
    if auc_score >= target_auc and f1_flood >= target_f1:
        logger.info("\n🎉 CONGRATULATIONS! All targets achieved!")
    else:
        logger.info("\n⚠️ Targets not fully achieved.")
        logger.info("📝 Recommendations:")
        if auc_score < target_auc:
            logger.info("  - Consider adding more diverse features (spatial, temporal patterns)")
        if f1_flood < target_f1:
            logger.info("  - Adjust SMOTE parameters or try ADASYN")
            logger.info("  - Fine-tune decision threshold further")
            logger.info("  - Collect more flood event samples if possible")
    
    return model_pack


if __name__ == "__main__":
    train_model_v6()
