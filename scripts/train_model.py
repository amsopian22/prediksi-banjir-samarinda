import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import config

def train_model():
    print("🚀 MEMULAI TRAINING MODEL...")
    
    # 1. Load Data
    data_file = 'dataset_banjir_samarinda_final.csv'
    df = pd.read_csv(data_file)
    print(f"📂 Dataset loaded: {len(df)} baris")
    
    # 2. Feature Engineering
    # Ensure sorted by date
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.sort_values('tanggal')
    
    # Handle Missing Values
    # Fill numerical NaNs with 0 (assuming rain/duration NaN means 0)
    # Be careful with Pasut NaNs - we covered 100% in merge, but just in case
    df['curah_hujan_mm'] = df['curah_hujan_mm'].fillna(0)
    df['durasi_hujan_jam'] = df['durasi_hujan_jam'].fillna(0)
    
    # Create Lag Features (URS Requirement)
    print("🛠️  Creating lag features (URS: R_t-1...3, Tide Slope)...")
    df['hujan_lag1'] = df['curah_hujan_mm'].shift(1).fillna(0)
    df['hujan_lag2'] = df['curah_hujan_mm'].shift(2).fillna(0)
    df['hujan_lag3'] = df['curah_hujan_mm'].shift(3).fillna(0) # URS R(t-3)
    
    # Accumulation (URS R(24h) handled by daily R_t, we keep 3days as extra context)
    # --- UPGRADE: Antecedent Precipitation Index (API) ---
    print("✨ Calculating Antecedent Precipitation Index (API)...")
    k = config.API_DECAY_FACTOR
    
    # Vectorized / Recursive Loop for API
    # Since it's time-ordered, we can do a loop
    api_values = []
    current_api = 0
    
    for rain in df['curah_hujan_mm'].values:
        current_api = rain + (k * current_api)
        api_values.append(current_api)
        
    df['hujan_3days'] = api_values # Mapping API to the same feature name 'hujan_3days' for compatibility
    
    # Tide Slope (URS: Tide_t - Tide_t-1)
    # Using Daily Max Tide Slope
    df['pasut_slope'] = df['pasut_msl_max'] - df['pasut_msl_max'].shift(1)
    df['pasut_slope'] = df['pasut_slope'].fillna(0)
    
    df['pasut_max_lag1'] = df['pasut_msl_max'].shift(1).fillna(method='bfill')
    
    # Select Features and Target
    features = ['curah_hujan_mm', 'durasi_hujan_jam', 'pasut_msl_max', 'pasut_slope',
                'hujan_lag1', 'hujan_lag2', 'hujan_lag3', 'hujan_3days']
    target = 'status_siaga'
    
    X = df[features]
    y = df[target]
    
    print(f"   Features: {features}")
    print(f"   Target: {target}")
    
    # 3. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"   Classes: {le.classes_}")
    
    # 4. Split Data
    # Use temporal split (Train on past, Test on recent) or random?
    # Random split is okay for generic evaluation, but temporal is more realistic.
    # Let's use standard random split for now to ensure class distribution in test set 
    # (since 'Banjir' is rare).
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f"   Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 4.1 Manual Oversampling for Minority Class ('Banjir')
    from sklearn.utils import resample
    
    # Combine X and y for resampling
    train_data = pd.concat([X_train, pd.DataFrame(y_train, columns=['target'], index=X_train.index)], axis=1)
    
    # Separate classes
    # Assuming standard encoding: 0=Air Meluap, 1=Aman, 2=Banjir (Need to verify encoder mapping!)
    # Better to filter by value found in y_train corresponding to 'Banjir'
    banjir_code = le.transform(['Banjir'])[0]
    aman_code = le.transform(['Aman'])[0]
    air_meluap_code = le.transform(['Air Meluap'])[0]
    
    df_banjir = train_data[train_data.target == banjir_code]
    df_aman = train_data[train_data.target == aman_code]
    df_meluap = train_data[train_data.target == air_meluap_code]
    
    print(f"   Original Train Counts: Aman={len(df_aman)}, Air Meluap={len(df_meluap)}, Banjir={len(df_banjir)}")
    
    # Upsample Banjir
    df_banjir_upsampled = resample(df_banjir, 
                                   replace=True,     # sample with replacement
                                   n_samples=len(df_aman),    # to match majority class
                                   random_state=42) 
    
    # Upsample Air Meluap (if needed, it's also minority compared to Aman)
    df_meluap_upsampled = resample(df_meluap,
                                   replace=True,
                                   n_samples=len(df_aman),
                                   random_state=42)
                                   
    # Combine back
    train_upsampled = pd.concat([df_aman, df_meluap_upsampled, df_banjir_upsampled])
    
    X_train_res = train_upsampled.drop('target', axis=1)
    y_train_res = train_upsampled.target
    
    print(f"   Upsampled Train Size: {len(X_train_res)}")
    
    # 5. Train Model
    print("🔄 Training Random Forest Classifier (on Upsampled Data)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42) # Removed class_weight as we balanced manually
    model.fit(X_train_res, y_train_res)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    
    print("\n📊 EVALUATION RESULTS:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("-" * 60)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Feature Importance
    feature_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n🌟 Feature Importance:")
    print(feature_imp)
    
    # --- 6.1 Threshold Optimization (Precision-Recall Curve) ---
    print("\n⚖️  Optimizing Threshold for 'Banjir' class...")
    
    # Get probability for 'Banjir'
    # Check index of 'Banjir' class
    banjir_idx = list(le.classes_).index('Banjir')
    y_scores = model.predict_proba(X_test)[:, banjir_idx]
    
    # Binarize y_test for 'Banjir' (1 if Banjir, 0 otherwise)
    y_test_banjir = (y_test == le.transform(['Banjir'])[0]).astype(int)
    
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test_banjir, y_scores)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # Handle division by zero
    
    # Locate maximum F1
    # thresholds array is 1 element shorter than precisions/recalls
    # We ignore the last precision/recall value (which is usually 1.0/0.0)
    best_idx = np.argmax(f1_scores[:-1]) 
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"   🏆 Optimal Threshold: {best_threshold:.4f} (Max F1: {best_f1:.4f})")
    
    # Plot Curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score")
    plt.axvline(best_threshold, color='k', linestyle=':', label=f'Optimum ({best_threshold:.2f})')
    plt.title(f"Threshold Optimization Curve (Target: Banjir)\nBest Threshold: {best_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("threshold_optimization_curve.png")
    print("   📉 Curve saved to 'threshold_optimization_curve.png'")
    
    # 7. Save Model & Artifacts
    print("\n💾 Saving model...")
    
    model_pack = {
        "model": model,
        "features": features,
        "threshold": float(best_threshold) # Save optimized threshold
    }
    
    with open('model_banjir.pkl', 'wb') as f:
        pickle.dump(model_pack, f)
        
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print(f"✅ Model saved as 'model_banjir.pkl' (Pack format). Threshold: {best_threshold:.4f}")

if __name__ == "__main__":
    train_model()
