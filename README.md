# CSC3105 Data Analytics Project — Group 11

A machine learning pipeline for UWB (Ultra-Wideband) signal classification and distance estimation in LOS/NLOS warehouse environments. The system classifies whether a signal travels via a direct (LOS) or reflected (NLOS) path, and estimates both direct-path and reflected-path distances using Random Forest models.

---

## Project Structure

```
DATA ANALYTICS PROJECT/
├── dataset/                         # Raw dataset files (not included — see Setup)
├── processed/                       # Output of phase1_data_prep.py
├── data_prep_output/                # Output of phase1_data_prep_continuation.py
├── training/
│   ├── models/                      # Saved trained models (.joblib)
│   └── plots/                       # Generated plots (.png)
├── phase1_data_prep.py              # Step 1: Load, clean, scale raw data
├── phase1_data_prep_continuation.py # Step 2: Feature engineering, PCA, save splits
├── train_rf_classifier.py           # Step 3: Train and evaluate RF classifier
└── train_rf_regressor.py            # Step 4: Train and evaluate RF regressor
```

---

## Installation

Install the required Python library:

```bash
pip install scikit-learn
```

Other libraries used (`numpy`, `pandas`, `matplotlib`, `scipy`) are part of the standard scientific Python stack. If you do not have them:

```bash
pip install numpy pandas matplotlib scipy
```

---

## Setup

Before running anything, place all 7 dataset parts inside the `dataset/` folder:

```
dataset/
├── uwb_dataset_part1.csv
├── uwb_dataset_part2.csv
├── uwb_dataset_part3.csv
├── uwb_dataset_part4.csv
├── uwb_dataset_part5.csv
├── uwb_dataset_part6.csv
└── uwb_dataset_part7.csv
```

---

## Running the Pipeline

Run the scripts **in order**. Each step depends on the output of the previous one.

### Step 1 — Data Preparation
```bash
python phase1_data_prep.py
```
Combines all 7 dataset parts, checks for missing values and outliers (IQR), applies z-score scaling, and saves the cleaned dataset to `processed/uwb_combined_scaled.csv`.

### Step 2 — Feature Engineering & PCA
```bash
python phase1_data_prep_continuation.py
```
Extracts second-peak features from the CIR signal, applies PCA to reduce CIR dimensionality to 95% variance (860 components), and saves the final train/test splits to `data_prep_output/` as `.npy` files. Also generates `feature_boxplots_by_class.png` and `pca_explained_variance.png` in `training/plots/`.

### Step 3 — Train Classifier
```bash
python train_rf_classifier.py
```
Trains a Random Forest Classifier (300 trees) to predict LOS/NLOS signal state. Outputs accuracy, AUC, confusion matrix, and feature importance plots to `training/plots/`.

### Step 4 — Train Regressor
```bash
python train_rf_regressor.py
```
Trains a Multi-Output Random Forest Regressor to predict direct-path (Path 1) and reflected-path (Path 2) distances. Outputs residual and actual-vs-predicted plots to `training/plots/`, and saves the trained model to `training/models/`.

---

## Output Files

| File | Description |
|---|---|
| `processed/uwb_combined_scaled.csv` | Cleaned and scaled full dataset |
| `data_prep_output/X_train.npy` | Training feature matrix |
| `data_prep_output/X_test.npy` | Test feature matrix |
| `data_prep_output/y_train_class.npy` | Classification labels (train) |
| `data_prep_output/y_test_class.npy` | Classification labels (test) |
| `data_prep_output/y_train_reg.npy` | Regression targets — Path 1 & 2 (train) |
| `data_prep_output/y_test_reg.npy` | Regression targets — Path 1 & 2 (test) |
| `data_prep_output/metadata.json` | PCA config and class distribution info |
| `training/models/rf_multioutput_regressor.joblib` | Saved regressor model |
| `training/plots/` | All generated plots |

---

## Results Summary

| Model | Metric | Value |
|---|---|---|
| RF Classifier | Accuracy | 85% |
| RF Classifier | AUC | 0.8941 |
| RF Regressor | Test RMSE — Path 1 (Direct) | 1.41 m |
| RF Regressor | Test RMSE — Path 2 (Reflected) | 1.42 m |

## Github link
[Github link](https://github.com/subhaa2/data_analytics)