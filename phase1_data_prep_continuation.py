from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


RANDOM_STATE = 42
TEST_SIZE = 0.2     # 80:20 train-test

# split data and PCA

def main():

    '''
     Load processed dataset
    '''
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "processed" / "uwb_combined_scaled.csv"
    output_dir = project_root / "data_prep_output"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    print(f"Loaded dataset shape: {df.shape}")

    '''
     Separate targets
     - Structured radio features --> keep as-is
    '''
    X = df.drop(columns=["NLOS", "RANGE"])
    y_class = df["NLOS"]
    y_reg = df["RANGE"]

    '''
     Train-test split (stratified for classification)
    '''
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class
    )

    # Regression target aligned with same indices
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    '''
     Separate CIR and non-CIR features
    '''
    cir_cols = [c for c in X.columns if c.startswith("CIR")]
    non_cir_cols = [c for c in X.columns if not c.startswith("CIR")]

    '''
     Scale non-CIR features (fit on train only)
    '''
    scaler_non_cir = StandardScaler()
    X_train_non_cir = scaler_non_cir.fit_transform(X_train[non_cir_cols])
    X_test_non_cir = scaler_non_cir.transform(X_test[non_cir_cols])

    '''
     Scale CIR features before PCA
    '''
    scaler_cir = StandardScaler()
    X_train_cir_scaled = scaler_cir.fit_transform(X_train[cir_cols])
    X_test_cir_scaled = scaler_cir.transform(X_test[cir_cols])

    '''
     PCA on CIR (retain 95% variance)
     - PCA on the 1016 time samples (raw waveforms)
    '''
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)

    X_train_cir_pca = pca.fit_transform(X_train_cir_scaled)
    X_test_cir_pca = pca.transform(X_test_cir_scaled)

    print(f"Original CIR dim: {len(cir_cols)}")
    print(f"PCA CIR dim: {X_train_cir_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    '''
     Combine final feature matrix
    '''
    X_train_final = np.hstack([X_train_non_cir, X_train_cir_pca])
    X_test_final = np.hstack([X_test_non_cir, X_test_cir_pca])

    '''
    Save outputs 
    '''
    np.save(output_dir / "X_train.npy", X_train_final.astype(np.float32))
    np.save(output_dir / "X_test.npy", X_test_final.astype(np.float32))

    np.save(output_dir / "y_train_class.npy", y_train_class.to_numpy())
    np.save(output_dir / "y_test_class.npy", y_test_class.to_numpy())

    np.save(output_dir / "y_train_reg.npy", y_train_reg.to_numpy())
    np.save(output_dir / "y_test_reg.npy", y_test_reg.to_numpy())

    # if you want to export CSV for easier viewing (optional)
#    pd.DataFrame(X_train_final).to_csv(output_dir / "X_train.csv", index=False) 
    
    # Save metadata
    metadata = {
        "train_samples": int(X_train_final.shape[0]),
        "test_samples": int(X_test_final.shape[0]),
        "original_cir_dim": len(cir_cols),
        "pca_cir_dim": int(X_train_cir_pca.shape[1]),
        "explained_variance": float(pca.explained_variance_ratio_.sum())
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("preprocessing completed.")


if __name__ == "__main__":
    main()