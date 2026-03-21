import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    cir_matrix = df[cir_cols].values
    fp_indices = df['FP_IDX'].values
    
    for i in range(len(df)):
        start_idx = int(fp_indices[i])
        # Look at the signal AFTER the first path
        signal = cir_matrix[i, start_idx:]
        peaks, props = find_peaks(signal, height=0)
        
        if len(peaks) > 0:
            highest_peak_idx = np.argmax(props['peak_heights'])
            peak2_pos.append(peaks[highest_peak_idx] + start_idx)
            peak2_amp.append(props['peak_heights'][highest_peak_idx])
        else:
            # If no clear 2nd peak, use first path info as a baseline
            peak2_pos.append(start_idx) 
            peak2_amp.append(0)
            
    return np.array(peak2_pos), np.array(peak2_amp)

def save_pca_explained_variance_plot(pca: PCA, plot_path: Path):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(cumulative_variance) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(components, cumulative_variance, marker="o", markersize=3, linewidth=2, color="#1E88E5")
    ax.axhline(0.95, color="#D32F2F", linestyle="--", linewidth=1.5, label="95% variance target")
    ax.axvline(pca.n_components_, color="#6A1B9A", linestyle="--", linewidth=1.5,
               label=f"Selected components = {pca.n_components_}")
    ax.set_title("PCA Cumulative Explained Variance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

def save_feature_boxplots(df: pd.DataFrame, plot_path: Path):
    features = ["FP_AMP1", "STDEV_NOISE", "MAX_NOISE", "RXPACC"]
    available_features = [feature for feature in features if feature in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    box_colors = ["#4CAF50", "#FF7043"]

    for ax, feature in zip(axes, available_features):
        los_values = df.loc[df["NLOS"] == 0, feature].dropna().values
        nlos_values = df.loc[df["NLOS"] == 1, feature].dropna().values
        bp = ax.boxplot(
            [los_values, nlos_values],
            tick_labels=["LOS", "NLOS"],
            patch_artist=True,
            widths=0.55,
            showfliers=False
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for median in bp["medians"]:
            median.set_color("#212121")
            median.set_linewidth(1.5)
        ax.set_title(feature)
        ax.set_ylabel("Scaled Value (z-score)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(available_features):]:
        ax.axis("off")

    fig.suptitle("Important Feature Distributions by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

def main():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "processed" / "uwb_combined_scaled.csv"
    output_dir = project_root / "data_prep_output"
    plot_dir = project_root / "training" / "plots"
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(data_path)
    save_feature_boxplots(df, plot_dir / "feature_boxplots_by_class.png")
    
    # --- 1. TARGET GENERATION (The "Two Path" Fix) ---
    # Path 1 Distance (Standard)
    y_reg_p1 = df["RANGE"].values
    
    # Path 2 Distance Estimation (Based on Peak Position)
    # Hint: Distance diff = (Peak Index Diff) * (Speed of light * Sample Time)
    # 0.00468m is a common index-to-distance conversion for UWB CIR
    cir_cols = [c for c in df.columns if c.startswith("CIR")]
    p2_pos, p2_amp = extract_second_peak_features(df, cir_cols)
    
    index_diff = p2_pos - df['FP_IDX'].values
    y_reg_p2 = y_reg_p1 + (index_diff * 0.00468) # Estimating Ground Truth for Path 2
    
    # Combine into a Multi-Output Target [N, 2]
    y_reg_combined = np.column_stack([y_reg_p1, y_reg_p2])

    # Classification: Applying the "Golden Rule"
    # State 0 (LOS-NLOS): If first path is LOS (0)
    # State 1 (NLOS-NLOS): If first path is NLOS (1)
    y_joint_class = df["NLOS"].values 

    # --- 2. SPLIT DATA ---
    X = df.drop(columns=["NLOS", "RANGE"])
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_joint_class, y_reg_combined, 
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_joint_class
    )

    # --- 3. FEATURE ENGINEERING ---
    print("Engineering features for Path 2...")
    # Re-extract for split sets
    X_train_p2_pos, X_train_p2_amp = extract_second_peak_features(X_train, cir_cols)
    X_test_p2_pos, X_test_p2_amp = extract_second_peak_features(X_test, cir_cols)
    
    non_cir_cols = [c for c in X.columns if not c.startswith("CIR")]
    X_train_struct = np.column_stack([X_train[non_cir_cols], X_train_p2_pos, X_train_p2_amp])
    X_test_struct = np.column_stack([X_test[non_cir_cols], X_test_p2_pos, X_test_p2_amp])
    
    # Scaling
    scaler_non_cir = StandardScaler()
    X_train_non_cir_scaled = scaler_non_cir.fit_transform(X_train_struct)
    X_test_non_cir_scaled = scaler_non_cir.transform(X_test_struct)

    # PCA on CIR
    scaler_cir = StandardScaler()
    X_train_cir_scaled = scaler_cir.fit_transform(X_train[cir_cols])
    X_test_cir_scaled = scaler_cir.transform(X_test[cir_cols])

    # PCA chart call
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_cir_pca = pca.fit_transform(X_train_cir_scaled)
    X_test_cir_pca = pca.transform(X_test_cir_scaled)
    save_pca_explained_variance_plot(pca, plot_dir / "pca_explained_variance.png")

    # Final Stacking
    X_train_final = np.hstack([X_train_non_cir_scaled, X_train_cir_pca])
    X_test_final = np.hstack([X_test_non_cir_scaled, X_test_cir_pca])
    
    # --- 4. SAVE OUTPUTS ---
    np.save(output_dir / "X_train.npy", X_train_final.astype(np.float32))
    np.save(output_dir / "X_test.npy", X_test_final.astype(np.float32))
    np.save(output_dir / "y_train_class.npy", y_train_class)
    np.save(output_dir / "y_test_class.npy", y_test_class)
    np.save(output_dir / "y_train_reg.npy", y_train_reg) # Saved as (N, 2)
    np.save(output_dir / "y_test_reg.npy", y_test_reg)   # Saved as (N, 2)
    metadata = {
        "pca_n_components": int(pca.n_components_),
        "pca_total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.round(8).tolist(),
        "pca_cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).round(8).tolist(),
        "class_distribution": {
            "LOS": int((df["NLOS"] == 0).sum()),
            "NLOS": int((df["NLOS"] == 1).sum())
        }
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! X features: {X_train_final.shape[1]}")
    print(f"Regression targets: {y_train_reg.shape[1]} (Path 1 and Path 2)")
    print(f"Saved plots: {plot_dir / 'feature_boxplots_by_class.png'}")
    print(f"Saved plots: {plot_dir / 'pca_explained_variance.png'}")
    print(f"Saved metadata: {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()
