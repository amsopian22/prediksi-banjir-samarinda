
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_regression_dataset():
    input_path = "data/dataset_banjir_v6_enhanced.csv"
    output_path = "data/dataset_banjir_v7_regression.csv"
    
    logger.info(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Base Depth from Label
    # Aman -> 0
    # Air Meluap -> 20 (Range 10-30)
    # Banjir -> 80 (Range 50-100)
    
    np.random.seed(42) # For reproducibility
    
    def get_base_depth(row):
        label = row['label']
        if label == 'Aman':
            return 0.0
        elif label == 'Air Meluap':
            return float(np.random.randint(10, 31))
        elif label == 'Banjir':
            return float(np.random.randint(50, 101))
        return 0.0
        
    df['water_depth_cm'] = df.apply(get_base_depth, axis=1)
    
    # 2. Physics Boost
    # Rain Intensity impact: + (Intensity * 2) cm
    df['water_depth_cm'] += (df['rain_intensity_max'] * 2.0)
    
    # Tide Impact: If Tide > 2.5m, add excess
    # (Tide - 2.5) * 20 cm
    tide_excess = (df['pasut_msl_max'] - 2.5).clip(lower=0)
    df['water_depth_cm'] += (tide_excess * 20.0)
    
    # 3. Add Noise (-5 to +5 cm) to 'Meluap'/'Banjir' only to avoid negative Aman
    mask_wet = df['water_depth_cm'] > 0
    noise = np.random.uniform(-5, 5, size=len(df))
    df.loc[mask_wet, 'water_depth_cm'] += noise[mask_wet]
    
    # Ensure non-negative
    df['water_depth_cm'] = df['water_depth_cm'].clip(lower=0).round(1)
    
    # 4. Add Constraints / Logical Features
    # Elevation: Constant 2.0m for now (City Center)
    # In a real app, this would be dynamic per point.
    df['elevation_m'] = 2.0
    
    # Drainage Capacity (0-100)
    # Assume 50 is standard.
    df['drainage_capacity'] = 50.0
    
    # 5. Save
    logger.info(f"Saving to {output_path}...")
    cols = ['tanggal', 'water_depth_cm', 'elevation_m', 'drainage_capacity'] + [c for c in df.columns if c not in ['water_depth_cm', 'elevation_m', 'drainage_capacity']]
    df[cols].to_csv(output_path, index=False)
    logger.info("Done.")
    
    # Preview
    print(df[['tanggal', 'label', 'rain_sum_imputed', 'pasut_msl_max', 'water_depth_cm']].sample(10))

if __name__ == "__main__":
    generate_regression_dataset()
