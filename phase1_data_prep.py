from pathlib import Path
import numpy as np
import pandas as pd

DATASET_GLOB = "uwb_dataset_part*.csv"
TARGET_COLS = ["NLOS", "RANGE"]
# IMPORTANT: Keep FP_IDX raw so we can use it for signal processing in 1.2
DONT_SCALE = ["FP_IDX"] + TARGET_COLS 

def load_and_combine(dataset_dir: Path) -> pd.DataFrame:
    files = sorted(dataset_dir.glob(DATASET_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found with pattern {DATASET_GLOB}")
    frames = [pd.read_csv(file) for file in files]
    return pd.concat(frames, ignore_index=True)

def summarize_missing_values(df: pd.DataFrame):
    return df.isna().sum(), int(df.isna().sum().sum())

def summarize_outliers_iqr(df: pd.DataFrame, feature_cols: list[str]):
    q1, q3 = df[feature_cols].quantile(0.25), df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df[feature_cols] < (q1 - 1.5 * iqr)) | (df[feature_cols] > (q3 + 1.5 * iqr))
    return outlier_mask.sum(), int(outlier_mask.sum().sum())

def zscore_scale_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Only scale features that aren't in our DONT_SCALE list
    to_scale = [c for c in feature_cols if c not in DONT_SCALE]
    means = df[to_scale].mean()
    stds = df[to_scale].std(ddof=0).replace(0, 1.0)
    df[to_scale] = ((df[to_scale] - means) / stds).astype(np.float32)
    return df

def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset"
    output_dir = project_root / "processed"
    output_dir.mkdir(exist_ok=True)

    df = load_and_combine(dataset_dir)
    _, missing_total = summarize_missing_values(df)
    
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    _, outlier_total = summarize_outliers_iqr(df, feature_cols)

    # Scaling Step (Skips FP_IDX)
    df = zscore_scale_features(df, feature_cols)

    combined_csv_path = output_dir / "uwb_combined_scaled.csv"
    df.to_csv(combined_csv_path, index=False)
    np.save(output_dir / "uwb_combined_scaled.npy", df.to_numpy(dtype=np.float32))
    print("Phase 1.1 Completed: FP_IDX preserved for processing.")

if __name__ == "__main__":
    main()