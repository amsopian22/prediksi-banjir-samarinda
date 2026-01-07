"""
SHAP Explainability Module for Flood Prediction Model.
Provides explanations for individual predictions using SHAP values.
"""
import numpy as np
import pandas as pd
import logging
import warnings

# Suppress SHAP warnings about TreeExplainer and StackingClassifier
warnings.filterwarnings('ignore', category=FutureWarning, module='shap')
warnings.filterwarnings('ignore', message='.*TreeExplainer.*')

logger = logging.getLogger(__name__)

def explain_prediction(model_pack, input_data: dict):
    """
    Generate SHAP explanation for a prediction.
    
    Args:
        model_pack: The loaded model package containing model and preprocessor
        input_data: Dictionary of input features
        
    Returns:
        Dictionary with SHAP values and feature contributions
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None
    
    if model_pack is None:
        logger.warning("Model pack is None")
        return {"error": "Model pack is None"}
    
    try:
        model = model_pack.get('model')
        # Handle different model pack key names - try multiple keys
        feature_names = model_pack.get('feature_names') or model_pack.get('features', [])
        scaler = model_pack.get('preprocessor') or model_pack.get('scaler')
        
        if model is None:
            logger.error("Model is None")
            return {"error": "Model is None"}
        
        # CRITICAL: Validate feature_names before proceeding
        if not feature_names or len(feature_names) == 0:
            logger.warning(f"No feature names found in model pack. Keys available: {list(model_pack.keys())}")
            # Try to get feature names from model if available
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
                logger.info(f"Retrieved feature names from model.feature_names_in_: {feature_names}")
            elif hasattr(model, 'n_features_in_'):
                # Create generic feature names
                feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
                logger.warning(f"Using generic feature names for {model.n_features_in_} features")
            else:
                logger.error("Cannot determine feature names from model")
                return {"error": "No feature names available"}
        
        logger.info(f"Using {len(feature_names)} features: {feature_names}")
        
        # Prepare input DataFrame with all required features
        df = pd.DataFrame([input_data])
        
        # Add missing features with default values
        missing_features = []
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
                missing_features.append(f)
        
        if missing_features:
            logger.debug(f"Added {len(missing_features)} missing features with value 0: {missing_features[:5]}...")
        
        # Reorder columns to match training order
        df = df[feature_names]
        
        # Final validation
        if df.shape[1] == 0:
            logger.error(f"DataFrame has 0 columns after filtering. Input keys: {list(input_data.keys())}")
            return {"error": "No matching features between input and model"}
        
        logger.info(f"Input shape: {df.shape}, Columns: {list(df.columns)}")
        
        # Apply scaling if available
        if scaler is not None:
            try:
                X = scaler.transform(df)
                logger.info(f"Applied scaler, output shape: {X.shape}")
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}, using raw values")
                X = df.values
        else:
            X = df.values
        
        # Create SHAP explainer with check_additivity=False to avoid strict checks
        try:
            # Check if model is StackingClassifier (not supported by TreeExplainer)
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            
            is_stacking = isinstance(model, (StackingClassifier, StackingRegressor))
            
            if is_stacking:
                # Skip TreeExplainer for stacking models - raise exception to trigger fallback
                logger.info("StackingClassifier detected, skipping TreeExplainer (not supported)")
                raise ValueError("StackingClassifier not supported by TreeExplainer")
            
            # Use feature_perturbation='interventional' for better compatibility
            explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X, check_additivity=False)
            
            # For multi-class classification, shap_values is a list
            if isinstance(shap_values, list):
                # Get SHAP values for 'Banjir' class (usually class 1 or 2)
                # Try to find the flood class index
                if hasattr(model, 'classes_'):
                    classes = list(model.classes_)
                    if 'Banjir' in classes:
                        flood_idx = classes.index('Banjir')
                    else:
                        flood_idx = len(classes) - 1  # Assume last class is highest risk
                else:
                    flood_idx = 1
                
                shap_values = shap_values[flood_idx]
                base_value = explainer.expected_value[flood_idx]
            else:
                base_value = explainer.expected_value
            
            # Get single sample values
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Create feature contribution dict
            contributions = {}
            for i, name in enumerate(feature_names):
                if i < len(values):
                    contributions[name] = float(values[i])
                else:
                    contributions[name] = 0.0
            
            # Sort by absolute contribution
            sorted_contributions = dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            # Get top contributors
            top_positive = [(k, v) for k, v in sorted_contributions.items() if v > 0][:5]
            top_negative = [(k, v) for k, v in sorted_contributions.items() if v < 0][:5]
            
            logger.info(f"SHAP explanation generated successfully. Top feature: {list(sorted_contributions.keys())[0]}")
            
            return {
                "contributions": contributions,
                "sorted_contributions": sorted_contributions,
                "top_positive": top_positive,
                "top_negative": top_negative,
                "base_value": float(base_value) if isinstance(base_value, (int, float, np.number)) else 0.0,
                "feature_names": feature_names
            }
            
        except Exception as e:
            # Silently skip TreeExplainer for StackingClassifier (expected behavior)
            # Use debug level instead of warning since this is expected for StackingClassifier
            logger.debug(f"TreeExplainer not compatible: {e}")
            # Return error to trigger fallback to model feature importance
            return {"error": str(e)}
            
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {"error": str(e)}


def get_feature_importance_chart_data(explanation):
    """
    Convert SHAP explanation to chart-ready data.
    
    Returns:
        List of dicts for plotting
    """
    if explanation is None or "contributions" not in explanation:
        return []
    
    sorted_contrib = explanation.get("sorted_contributions", {})
    
    chart_data = []
    for feature, value in list(sorted_contrib.items())[:10]:
        # Format feature names for display
        display_name = feature.replace("_", " ").title()
        chart_data.append({
            "feature": display_name,
            "contribution": value,
            "direction": "Increase Risk" if value > 0 else "Decrease Risk"
        })
    
    return chart_data
