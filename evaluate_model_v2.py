import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import config

# Load Data
print("📊 Loading Data...")
df = pd.read_csv("dataset_banjir_v2_advanced.csv")
print(f"   Rows: {len(df)}")
print(f"   Columns: {df.columns.tolist()}")

# Generate Lags
df = df.sort_values('tanggal').reset_index(drop=True)
df['hujan_lag1'] = df['rain_sum_imputed'].shift(1).fillna(0)
df['hujan_lag2'] = df['rain_sum_imputed'].shift(2).fillna(0)

features_needed = [
    'rain_sum_imputed', 
    'rain_intensity_max',
    'soil_moisture_surface_mean', 
    'soil_moisture_root_mean',
    'pasut_msl_max', 
    'hujan_lag1', 
    'hujan_lag2'
]
target = 'is_banjir'

# CLEAN DATA
print(f"   Original Rows: {len(df)}")
df = df.dropna(subset=[target])
df[target] = df[target].astype(int)
print(f"   Cleaned Rows: {len(df)}")

X = df[features_needed]
y = df[target]

# Split (Same seed as training ideally, or just valid set)
# Since we saved the model trained on ALL data probably, this is a "Training Recall" check mostly.
# But for realism, we want to see how it separates classes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Model
print("\n🤖 Loading Model V2...")
model_pack = joblib.load(config.MODEL_PATH)
print(f"   Model Pack Type: {type(model_pack)}")
if isinstance(model_pack, dict):
    print(f"   Keys: {model_pack.keys()}")
    model = model_pack.get('model') or model_pack.get('model_v2') or list(model_pack.values())[0]
else:
    model = model_pack # It is the model itself

if not model:
    raise ValueError("Could not find model in model_pack")

# Predict
print("\n🔮 Predicting on Test Set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

# APPLY THRESHOLD FROM CONFIG
threshold = config.THRESHOLD_FLOOD_PROBABILITY
y_pred = (y_pred_proba > threshold).astype(int)

# Metrics
print(f"\n📈 EVALUATION REPORT (Threshold: {threshold})")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=['Aman', 'Banjir']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"TN: {cm[0][0]}\tFP: {cm[0][1]}")
print(f"FN: {cm[1][0]}\tTP: {cm[1][1]}")

try:
    roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n⭐ ROC-AUC Score: {roc:.4f}")
except:
    pass

# FEATURE IMPORTANCE
print("\n🔑 Feature Importance:")
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print(f"   {features_needed[indices[f]]}: {importances[indices[f]]:.4f}")
    elif hasattr(model, 'steps'): # Pipeline
        # Assuming random forest is last step named 'classifier' or accessed via index
        rf = model.named_steps.get('classifier') or model[-1]
        if hasattr(rf, 'feature_importances_'):
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(X.shape[1]):
                print(f"   {features_needed[indices[f]]}: {importances[indices[f]]:.4f}")
except Exception as e:
    print(f"   Could not retrieve importance: {e}")

# REALITY CHECKS (SCENARIOS)
print("\n🌍 REALITY CHECKS (Synthetic Scenarios)")
print("-" * 40)
scenarios = [
    # 1. Hujan Deras, Pasang Normal, Tanah Jenuh (BANJIR)
    {"name": "Hujan Ekstrem + Tanah Jenuh", "data": [150, 50, 0.6, 0.6, 1.5, 20, 10]},
    # 2. Tidak Hujan, Pasang Tinggi (ROB)
    {"name": "Pasang Tinggi (Rob) + Kering", "data": [0, 0, 0.3, 0.3, 3.1, 0, 0]},
    # 3. Hujan Sedang, Pasang Normal, Tanah Kering (AMAN)
    {"name": "Hujan Biasa + Tanah Kering", "data": [20, 5, 0.3, 0.3, 1.0, 0, 0]},
    # 4. FALSE POSITIVE CHECK (User Case): Hujan 0, Pasang 2.0 (Aman)
    {"name": "User Case: Cerah, Pasang Normal", "data": [0, 0, 0.4, 0.4, 2.0, 0, 0]}
]

for sc in scenarios:
    fname = sc['name']
    vals = sc['data']
    # Create DF
    sdf = pd.DataFrame([vals], columns=features_needed)
    prob = model.predict_proba(sdf)[0][1]
    status = "BAHAYA 🚨" if prob > threshold else "AMAN ✅"
    print(f"Scenario: {fname:<30} | Prob: {prob:.4f} | Result: {status}")

print(f"\n✅ Evaluation Complete. Threshold used: {threshold}")
