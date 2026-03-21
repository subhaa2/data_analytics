import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(y_true, y_pred, path_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  Results for {path_name}:")
    print(f"  RMSE : {rmse:.4f} m")
    print(f"  MAE  : {mae:.4f} m")
    print(f"  R²   : {r2:.4f}")
    return rmse, mae, r2


def plot_two_paths(y_test, y_pred, plot_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Path 1
    rmse1 = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    mae1  = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    r2_1  = r2_score(y_test[:, 0], y_pred[:, 0])

    ax1.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.3, color='blue', s=5)
    lims1 = [min(y_test[:, 0].min(), y_pred[:, 0].min()),
             max(y_test[:, 0].max(), y_pred[:, 0].max())]
    ax1.plot(lims1, lims1, 'r--', lw=2, label='Perfect prediction')
    ax1.set_title("Path 1 (Direct): Actual vs Predicted")
    ax1.set_xlabel("Actual Range (m)")
    ax1.set_ylabel("Predicted Range (m)")
    ax1.legend()
    # add metrics box
    ax1.text(0.04, 0.97, f"RMSE : {rmse1:.4f} m\nMAE  : {mae1:.4f} m\nR²     : {r2_1:.4f}",
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#CCCCCC', alpha=0.9))

    # Path 2
    rmse2 = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    mae2  = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    r2_2  = r2_score(y_test[:, 1], y_pred[:, 1])

    ax2.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.3, color='green', s=5)
    lims2 = [min(y_test[:, 1].min(), y_pred[:, 1].min()),
             max(y_test[:, 1].max(), y_pred[:, 1].max())]
    ax2.plot(lims2, lims2, 'r--', lw=2, label='Perfect prediction')
    ax2.set_title("Path 2 (Reflection): Actual vs Predicted")
    ax2.set_xlabel("Actual Range (m)")
    ax2.set_ylabel("Predicted Range (m)")
    ax2.legend()
    # add metrics box
    ax2.text(0.04, 0.97, f"RMSE : {rmse2:.4f} m\nMAE  : {mae2:.4f} m\nR²     : {r2_2:.4f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    plt.savefig(plot_dir / "(v2)regression_two_paths.png", dpi=150)
    plt.close()
    print("  [Plot saved] regression_two_paths.png")

def plot_residuals(y_test, y_pred, plot_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for ax, col, color, label in [
        (ax1, 0, 'blue',  'Path 1 (Direct)'),
        (ax2, 1, 'green', 'Path 2 (Reflection)')
    ]:
        residuals = y_test[:, col] - y_pred[:, col]
        rmse = np.sqrt(mean_squared_error(y_test[:, col], y_pred[:, col]))
        mae = mean_absolute_error(y_test[:, col], y_pred[:, col])
        ax.scatter(y_pred[:, col], residuals, alpha=0.3, color=color, s=5)
        ax.axhline(0, color='red', linestyle='--', lw=2)
        ax.set_title(f"Residuals: {label}")
        ax.set_xlabel("Predicted Range (m)")
        ax.set_ylabel("Residual (m)")
        # add metrics box
        ax.text(0.04, 0.97, f"RMSE : {rmse:.4f} m\nMAE  : {mae:.4f} m",   # ADD
                transform=ax.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='#CCCCCC', alpha=0.9))

    plt.tight_layout()
    plt.savefig(plot_dir / "(v2)residuals_two_paths.png", dpi=150)
    plt.close()
    print("  [Plot saved] residuals_two_paths.png")


def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    root      = Path(__file__).resolve().parent.parent
    data_dir  = root / "data_prep_output"
    plot_dir  = root / "training" / "plots"
    model_dir = root / "training" / "models"
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        X_train      = np.load(data_dir / "X_train.npy")
        X_test       = np.load(data_dir / "X_test.npy")
        y_train_reg  = np.load(data_dir / "y_train_reg.npy")   # shape (N, 2)
        y_test_reg   = np.load(data_dir / "y_test_reg.npy")    # shape (N, 2)
    except FileNotFoundError as e:
        print(f"Error: Missing file → {e.filename}")
        return

    print(f"X_train : {X_train.shape}  |  y_train_reg : {y_train_reg.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test_reg  : {y_test_reg.shape}")

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n--- Training Multi-Output Regressor (Path 1 & Path 2) ---")
    # MultiOutputRegressor trains one RF per output column (Path 1 and Path 2)
    # Start with n_estimators=100 to verify it runs, then increase to 200
    base_rf = RandomForestRegressor(
        n_estimators=100,    # increase to 200 for final submission
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    reg = MultiOutputRegressor(base_rf)
    reg.fit(X_train, y_train_reg)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("      TERMINAL REPORT: TWO-PATH REGRESSION")
    print("=" * 50)

    y_pred_train: np.ndarray = np.asarray(reg.predict(X_train))
    y_pred_test: np.ndarray  = np.asarray(reg.predict(X_test))

    paths = ["Path 1 (Direct)", "Path 2 (Reflection)"]

    print("\n  [Train Results]")
    for i, name in enumerate(paths):
        evaluate(y_train_reg[:, i], y_pred_train[:, i], name)

    print("\n  [Test Results (Unseen Data)]")
    test_rmses = []
    for i, name in enumerate(paths):
        rmse, _, _ = evaluate(y_test_reg[:, i], y_pred_test[:, i], name)
        test_rmses.append(rmse)

    print(f"\n  Average Test RMSE across both paths: {np.mean(test_rmses):.4f} m")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n--- Generating Visualization Files ---")
    plot_two_paths(y_test_reg, y_pred_test, plot_dir)
    plot_residuals(y_test_reg, y_pred_test, plot_dir)

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = model_dir / "rf_multioutput_regressor.joblib"
    joblib.dump(reg, model_path)
    print(f"\n[Saved] Model → {model_path}")

    # ── Save predictions ──────────────────────────────────────────────────
    np.save(data_dir / "y_test_pred_reg.npy", y_pred_test)
    print("[Saved] Predictions → y_test_pred_reg.npy")

    print("\n[PROCESS COMPLETE]")


if __name__ == "__main__":
    main()