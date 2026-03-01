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
