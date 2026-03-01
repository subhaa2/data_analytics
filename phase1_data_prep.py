from pathlib import Path

import numpy as np
import pandas as pd


DATASET_GLOB = "uwb_dataset_part*.csv"
TARGET_COLS = ["NLOS", "RANGE"]


def load_and_combine(dataset_dir: Path) -> pd.DataFrame:
    """Read all dataset parts and stack them into one table."""
    files = sorted(dataset_dir.glob(DATASET_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found with pattern {DATASET_GLOB} in {dataset_dir}")

    frames = [pd.read_csv(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined


def summarize_missing_values(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Return missing-value counts per column and the total count."""
    missing_per_col = df.isna().sum()
    missing_total = int(missing_per_col.sum())
    return missing_per_col, missing_total


def summarize_outliers_iqr(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.Series, int]:
    """
    Detect outliers using IQR rule without modifying data.
    Outlier rule per feature: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
    """
    q1 = df[feature_cols].quantile(0.25)
    q3 = df[feature_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (df[feature_cols] < lower) | (df[feature_cols] > upper)
    outlier_per_feature = outlier_mask.sum()
    outlier_total = int(outlier_per_feature.sum())
    return outlier_per_feature, outlier_total


def zscore_scale_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Normalize feature columns with z-score scaling.
    Keeps target columns (NLOS, RANGE) unchanged.
    """
    means = df[feature_cols].mean()
    stds = df[feature_cols].std(ddof=0).replace(0, 1.0)
    df[feature_cols] = ((df[feature_cols] - means) / stds).astype(np.float32)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset"
    output_dir = project_root / "processed"
    output_dir.mkdir(exist_ok=True)

    # 1) Combine all 7 dataset parts.
    df = load_and_combine(dataset_dir)
    print(f"Combined shape: {df.shape}")

    # 2) Check missing values on the combined dataset.
    _, missing_total = summarize_missing_values(df)
    print(f"Total missing values: {missing_total}")

    # 3) Check outliers on input features only.
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    outlier_per_feature, outlier_total = summarize_outliers_iqr(df, feature_cols)
    print(f"Features with >=1 outlier: {(outlier_per_feature > 0).sum()}")
    print(f"Total outlier values: {outlier_total}")

    # 4) Scale input features; targets remain in original units.
    df = zscore_scale_features(df, feature_cols)

    # Save one CSV and one NumPy array as requested.
    combined_csv_path = output_dir / "uwb_combined_scaled.csv"
    combined_npy_path = output_dir / "uwb_combined_scaled.npy"

    df.to_csv(combined_csv_path, index=False)
    np.save(combined_npy_path, df.to_numpy(dtype=np.float32))

    print(f"Saved CSV: {combined_csv_path}")
    print(f"Saved NumPy array: {combined_npy_path}")


if __name__ == "__main__":
    main()
