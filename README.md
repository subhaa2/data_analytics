# data_analytics

## Phase 1 (Combine + Check + Scale)

### Required input files

Place all 7 dataset parts inside `dataset/`:
- `uwb_dataset_part1.csv`
- `uwb_dataset_part2.csv`
- `uwb_dataset_part3.csv`
- `uwb_dataset_part4.csv`
- `uwb_dataset_part5.csv`
- `uwb_dataset_part6.csv`
- `uwb_dataset_part7.csv`

### Generate outputs

Run:

```powershell
python phase1_data_prep.py
```

This script does:
1. Combine all 7 dataset parts from `dataset/`.
2. Check missing values after combining (no filling is done).
3. Check outliers using IQR rule (summary only, no capping).
4. Normalize feature columns using z-score scaling.

Outputs are saved in `processed/`:
- `uwb_combined_scaled.csv` (scaled full table)
- `uwb_combined_scaled.npy` (same table as one NumPy array)

## Phase 1 continuation (PCA + Train-Test Split)

### Required library installation
In terminal:
```Bash
pip install scikit-learn
```

### Generate outputs

Run:

```powershell
python phase1_data_prep_continuation.py
```

This script does:
1. Reads  `uwb_combined_scaled.csv` from `processed/`.
2. Separates input features and targets:
    - `NLOS` -> classification target
    - `RANGE` -> regression target
3. 80:20 stratified train-test split (balanced LOS/NLOS distribution)
4. Separates:
    - CIR features (`CIR<i>`)
    - Non-CIR radio features (FP_AMP, noise, RXPACC, etc.)
5. Standardizes features using training data only (prevents data leakage)
6. Applies PCA to CIR features only (95% of total variance)
7. Combines Scaled non-CIR features and PCA-reduced CIR features

Outputs are saved in `data_prep_output/`:
- Feature Matrices: `X_train.npy` and `X_test.npy`
- Classification targets: `y_train_class.npy` and `y_test_class.npy`
- Regression Targets: `y_train_reg.npy` and `y_test_reg.npy`
- Metadata: `metadata.json` -> Contains PCA dimension and explained variance info
* (optional) If you want to see the output as a csv, look at line 107 inside `phase1_data_prep_continuation.py` and change the filenames accordingly.
* didn't load csv cause large and slow, with floating point issue
