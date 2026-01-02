"""
SHAP Explainability Module for Flood Prediction Model.
Provides explanations for individual predictions using SHAP values.
"""
import numpy as np
import pandas as pd
import logging

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
        return None
    
    try:
        model = model_pack.get('model')
        preprocessor = model_pack.get('preprocessor')
        feature_names = model_pack.get('feature_names', [])
        
        if model is None:
            return None
        
        # Prepare input DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        
        df = df[feature_names]
        
        # Apply preprocessing if available
        if preprocessor is not None:
            X = preprocessor.transform(df)
        else:
            X = df.values
        
        # Create SHAP explainer
        # Use TreeExplainer for tree-based models (Random Forest, XGBoost, etc.)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For binary classification, get SHAP for positive class (flood)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 = Flood
            
            # Get single sample values
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Create feature contribution dict
            contributions = {}
            for i, name in enumerate(feature_names):
                contributions[name] = float(values[i]) if i < len(values) else 0.0
            
            # Sort by absolute contribution
            sorted_contributions = dict(sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            # Get top contributors
            top_positive = [(k, v) for k, v in sorted_contributions.items() if v > 0][:5]
            top_negative = [(k, v) for k, v in sorted_contributions.items() if v < 0][:5]
            
            return {
                "contributions": contributions,
                "sorted_contributions": sorted_contributions,
                "top_positive": top_positive,
                "top_negative": top_negative,
                "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value),
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.warning(f"TreeExplainer failed, trying KernelExplainer: {e}")
            # Fallback to KernelExplainer (slower but more general)
            explainer = shap.KernelExplainer(model.predict_proba, X)
            shap_values = explainer.shap_values(X)
            
            return {
                "contributions": {},
                "error": str(e)
            }
            
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
