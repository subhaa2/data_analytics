from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

RANDOM_STATE = 42
TEST_SIZE = 0.2

def extract_second_peak_features(df, cir_cols):
    """Signal processing to find the 2nd dominant path."""
    peak2_pos = []
    peak2_amp = []
    
    # Convert CIR to numpy for speed
    cir_matrix = df[cir_cols].values
    fp_indices = df['FP_IDX'].values
    
    for i in range(len(df)):
        # Start searching AFTER the first path index
        start_idx = int(fp_indices[i])
        signal = cir_matrix[i, start_idx:]
        
        peaks, props = find_peaks(signal, height=0)
        
        if len(peaks) > 0:
            # Get index of the highest peak in the remaining signal
            highest_peak_idx = np.argmax(props['peak_heights'])
            peak2_pos.append(peaks[highest_peak_idx] + start_idx)
            peak2_amp.append(props['peak_heights'][highest_peak_idx])
        else:
            peak2_pos.append(0)
            peak2_amp.append(0)
            
    return np.array(peak2_pos), np.array(peak2_amp)

def main():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "processed" / "uwb_combined_scaled.csv"
    output_dir = project_root / "data_prep_output"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["NLOS", "RANGE"])
    y_class = df["NLOS"]
    y_reg = df["RANGE"]

    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )

    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    cir_cols = [c for c in X.columns if c.startswith("CIR")]
    
    # --- NEW: EXTRACT SECOND PATH FEATURES ---
    print("Extracting Second Dominant Path features...")
    X_train_p2_pos, X_train_p2_amp = extract_second_peak_features(X_train, cir_cols)
    X_test_p2_pos, X_test_p2_amp = extract_second_peak_features(X_test, cir_cols)
    
    # Add them to the non-CIR columns list
    non_cir_cols = [c for c in X.columns if not c.startswith("CIR")]
    
    # --- SCALE NON-CIR (Including FP_IDX now and the new PEAK2 features) ---
    scaler_non_cir = StandardScaler()
    # We combine them into a temporary array for scaling
    X_train_struct = np.column_stack([X_train[non_cir_cols], X_train_p2_pos, X_train_p2_amp])
    X_test_struct = np.column_stack([X_test[non_cir_cols], X_test_p2_pos, X_test_p2_amp])
    
    X_train_non_cir_scaled = scaler_non_cir.fit_transform(X_train_struct)
    X_test_non_cir_scaled = scaler_non_cir.transform(X_test_struct)

    # --- PCA ON CIR ---
    scaler_cir = StandardScaler()
    X_train_cir_scaled = scaler_cir.fit_transform(X_train[cir_cols])
    X_test_cir_scaled = scaler_cir.transform(X_test[cir_cols])

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_cir_pca = pca.fit_transform(X_train_cir_scaled)
    X_test_cir_pca = pca.transform(X_test_cir_scaled)

    # --- COMBINE EVERYTHING ---
    X_train_final = np.hstack([X_train_non_cir_scaled, X_train_cir_pca])
    X_test_final = np.hstack([X_test_non_cir_scaled, X_test_cir_pca])
    
    # Save outputs (identical to your original saving logic)
    np.save(output_dir / "X_train.npy", X_train_final.astype(np.float32))
    np.save(output_dir / "X_test.npy", X_test_final.astype(np.float32))
    np.save(output_dir / "y_train_class.npy", y_train_class.to_numpy())
    np.save(output_dir / "y_test_class.npy", y_test_class.to_numpy())
    np.save(output_dir / "y_train_reg.npy", y_train_reg.to_numpy())
    np.save(output_dir / "y_test_reg.npy", y_test_reg.to_numpy())

    print(f"Preprocessing completed. Final Feature Count: {X_train_final.shape[1]}")

if __name__ == "__main__":
    main()