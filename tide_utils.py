import utide
import pandas as pd
import numpy as np
import pickle
import os

MODEL_FILE = 'tide_model_urs.pkl'

def train_tide_model(df, time_col='t', value_col='est', lat=-0.5022):
    """
    Train a harmonic analysis model using Utide.
    
    Args:
        df (pd.DataFrame): Dataframe containing tide data.
        time_col (str): Column name for datetime objects.
        value_col (str): Column name for water level (MSL/EST).
        lat (float): Latitude for nodal corrections.
        
    Returns:
        dict: The trained model coefficients (coef).
    """
    print(f"🌊 Training Utide model on {len(df)} observations...")
    
    # Ensure time is datetime
    time = pd.to_datetime(df[time_col]).values
    # Check for NaNs
    mask = ~np.isnan(df[value_col].values)
    
    coef = utide.solve(
        time[mask], 
        df[value_col].values[mask],
        lat=lat,
        nodal=True,
        trend=False,
        method='ols',
        conf_int='linear',
        verbose=False
    )
    
    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(coef, f)
        
    print(f"✅ Tide Model trained and saved to {MODEL_FILE}")
    return coef

def load_tide_model():
    """Load the trained tide model."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def predict_tide(model_coef, dates):
    """
    Predict tide levels for specific dates using Utide.
    
    Args:
        model_coef (dict): Trained Utide model coefficients.
        dates (array-like): List/Array of datetime objects (pandas or python datetime).
        
    Returns:
        np.array: Predicted tide levels.
    """
    
    # utide.reconstruct expects just the coef and time
    # It returns a bunch of stuff, we want 'h' (height)
    
    # Convert dates to matplotlib num2date format logic internal to utide? 
    # Actually utide.reconstruct accepts standard array of datetimes or matplotlib dates.
    # Best to pass as numpy array of datetimes.
    
    t_pred = pd.to_datetime(dates).values
    
    prediction = utide.reconstruct(t_pred, model_coef, verbose=False)
    
    return prediction['h']

def get_tide_slope(current_level, prev_level):
    """Calculate slope (positive = rising, negative = falling)."""
    return current_level - prev_level
