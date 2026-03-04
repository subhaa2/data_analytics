from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Setup paths
root = Path(__file__).resolve().parent.parent
data_dir = root / "data_prep_output"

# 2. Load the data your previous scripts made
X_train = np.load(data_dir / "X_train.npy")
y_train = np.load(data_dir / "y_train_reg.npy")
X_test = np.load(data_dir / "X_test.npy")
y_test = np.load(data_dir / "y_test_reg.npy")

print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

