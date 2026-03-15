import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, accuracy_score, roc_auc_score
)

def main():
    # 1. Setup paths
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data_prep_output"
    plot_dir = root / "training" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the 80/20 split data
    try:
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train_class.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_test = np.load(data_dir / "y_test_class.npy")
    except FileNotFoundError:
        print("Error: Files not found. Run your prep scripts with 80/20 split first!")
        return

    print(f"--- Training Joint-Class Gatekeeper (80/20 Split) ---")

    # 3. A-GRADE SETTINGS: 
    # max_features=None ensures the model looks at PEAK2_AMP at every split.
    # class_weight='balanced' handles the LOS/NLOS sample distribution.
    clf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=25, 
        max_features='log2', 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 4. Generate Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # 5. TERMINAL REPORT (Joint-Class Specification)
    target_names = ['State 0 (LOS-NLOS)', 'State 1 (NLOS-NLOS)']
    
    print("\n" + "="*50)
    print("      TERMINAL REPORT: TWO-PATH CLASSIFICATION")
    print("="*50)
    
    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc_val:.4f}")
    
    print("\nConfusion Matrix Breakdown:")
    print(f"True State 0 (Correct):  {cm[0][0]}")
    print(f"False State 1 (Mistake): {cm[0][1]}")
    print(f"False State 0 (Mistake): {cm[1][0]} <-- False LOS (Critical Weakness)")
    print(f"True State 1 (Correct):  {cm[1][1]}")
    
    print("\nDetailed Performance (Brief Requirement):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature Importance Terminal Print (Proving Signal Processing)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Mapping indices based on stacking in data_prep_continuation:
    # 0: FP_IDX, 1: FP_AMP, 2: RXPACC, 3: STDEV_NOISE, 4: PEAK2_POS, 5: PEAK2_AMP, 6+: PCA
    feature_map = {
        0: "FP_IDX", 
        1: "FP_AMP", 
        2: "RXPACC", 
        3: "STDEV_NOISE",
        4: "PEAK2_POS", 
        5: "PEAK2_AMP"
    }
    
    print("\nTop 10 Feature Importance Ranking:")
    for i in range(10):
        idx = indices[i]
        name = feature_map.get(idx, f"PCA_Comp_{idx-6}")
        print(f"{i+1:2}. {name:20} : {importances[idx]:.4f}")

    # 6. PLOT GENERATION (For Overleaf Report)
    print("\n--- Generating Visualization Files ---")

    # Chart 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Joint-Class Confusion Matrix: Two-Path States")
    plt.ylabel('Actual State')
    plt.xlabel('Predicted State')
    plt.savefig(plot_dir / "confusion_matrix.png")
    plt.close()

    # Chart 2: Feature Importance
    top_n = 10
    top_indices = indices[:top_n]
    top_names = [feature_map.get(i, f"PCA_{i-6}") for i in top_indices]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[top_indices], y=top_names, palette="viridis", hue=top_names, legend=False)
    plt.title("Top 10 Feature Importance (Gatekeeper)")
    plt.savefig(plot_dir / "feature_importance.png")
    plt.close()

    # Chart 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir / "roc_curve.png")
    plt.close()

    # 7. Final Save for Regressor
    np.save(data_dir / "y_test_pred_class.npy", y_pred)
    print(f"\n[SUCCESS] Terminal results printed and plots saved to: {plot_dir}")

if __name__ == "__main__":
    main()