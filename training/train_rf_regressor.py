import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"  RMSE : {rmse:.4f} m")
    print(f"  MAE  : {mae:.4f} m")
    print(f"  R²   : {r2:.4f}")
    return rmse, mae, r2

def plot_actual_vs_predicted(y_test, y_pred, rmse, plot_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color="blue", s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title(f"Actual vs Predicted (RMSE: {rmse:.3f} m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "regression_performance.png", dpi=150)
    plt.close()
    print("  [Plot saved] regression_performance.png")

def plot_feature_importance(model, n_features, plot_dir, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    # Feature Map including our new "Injected" Classifier Label
    # Since we added it as the LAST column, its index is (n_features - 1)
    feature_map = {
        0: "FP_IDX", 1: "STDEV_NOISE", 2: "PEAK2_POS", 3: "PEAK2_AMP",
        (n_features - 1): "PRED_NLOS_STATE"  # Our injected feature!
    }
    
    top_labels = [feature_map.get(i, f"PCA_{i-4}") for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=top_labels, hue=top_labels, palette="magma", legend=False)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances (Distance Prediction)")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance_reg.png", dpi=150)
    plt.close()
    print("  [Plot saved] feature_importance_reg.png")

def plot_residuals(y_test, y_pred, plot_dir):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, color="purple", s=10)
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Predicted Distance (m)")
    plt.ylabel("Residual (m)")
    plt.title("Residuals vs Predicted (Error Distribution)")
    plt.tight_layout()
    plt.savefig(plot_dir / "residuals.png", dpi=150)
    plt.close()
    print("  [Plot saved] residuals.png")

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data_prep_output"
    plot_dir = root / "training" / "plots"
    model_dir = root / "training" / "models"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and Augment data ──────────────────────────────────────────────
    try:
        # Load raw features and regression targets
        X_train_raw = np.load(data_dir / "X_train.npy")
        y_train_reg = np.load(data_dir / "y_train_reg.npy")
        X_test_raw  = np.load(data_dir / "X_test.npy")
        y_test_reg  = np.load(data_dir / "y_test_reg.npy")

        # LOAD CLASSIFIER OUTPUTS
        # For training, we use the ground truth class labels
        # For testing, we MUST use the predictions from our Classifier
        y_train_class = np.load(data_dir / "y_train_class.npy").reshape(-1, 1)
        y_test_pred_class = np.load(data_dir / "y_test_pred_class.npy").reshape(-1, 1)

        # INJECTION: Horizontal stack the classification labels as a new feature
        X_train = np.hstack((X_train_raw, y_train_class))
        X_test  = np.hstack((X_test_raw, y_test_pred_class))

    except FileNotFoundError as e:
        print(f"Error: Required files missing in {data_dir}. Ensure Classifier ran first!")
        print(f"Specific missing file: {e.filename}")
        return

    print(f"Dataset Augmented: {X_train.shape[0]} samples.")
    print(f"Features count increased to {X_train.shape[1]} (Classifier label injected).")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n--- Training Robust Distance Regressor with NLOS Awareness ---")
    
    reg = RandomForestRegressor(
        n_estimators=150, 
        max_depth=8,        # Slightly higher depth to process the new logic
        min_samples_leaf=15, 
        random_state=42, 
        n_jobs=-1, 
        verbose=1
    )
    
    reg.fit(X_train, y_train_reg)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("       TERMINAL REPORT: REGRESSION")
    print("=" * 45)

    y_pred_train = reg.predict(X_train)
    y_pred_test  = reg.predict(X_test)

    train_rmse, _, _ = evaluate(y_train_reg, y_pred_train, label="Train Results")
    test_rmse,  _, _ = evaluate(y_test_reg,  y_pred_test,  label="Test Results (Unseen Data)")

    overfit_gap = test_rmse - train_rmse
    print(f"\n  Generalization Error (Test - Train RMSE): {overfit_gap:.4f} m")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n--- Generating Visualization Files ---")
    plot_actual_vs_predicted(y_test_reg, y_pred_test, test_rmse, plot_dir)
    plot_feature_importance(reg, X_train.shape[1], plot_dir)
    plot_residuals(y_test_reg, y_pred_test, plot_dir)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = model_dir / "rf_distance_regressor_augmented.joblib"
    joblib.dump(reg, model_path)
    print(f"\n[Saved] Augmented Model Weights → {model_path}")

    # ── Save predictions ──────────────────────────────────────────────────
    np.save(data_dir / "y_test_pred_reg.npy", y_pred_test)
    print("[Saved] Y_Pred → y_test_pred_reg.npy")

    print("\n[PROCESS COMPLETE]")

if __name__ == "__main__":
    main()