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

    print(f"--- Training Balanced-Robust Classifier ---")

    # 3. BALANCED SETTINGS: 
    # using 'balanced' weights and a depth of 25 to capture PEAK2 features 
    # without forcing the model to guess NLOS for everything.
    clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=25,        
        max_features='log2', 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 4. THRESHOLD TUNING: 
    # 0.45 is slightly "cautious." It catches more NLOS than a standard 0.5 threshold,
    # but it won't "break" the model like 0.35 did in your last run.
    y_prob = clf.predict_proba(X_test)[:, 1] 
    y_pred = (y_prob > 0.45).astype(int)      

    # 5. TERMINAL OUTPUT
    print("\n" + "="*45)
    print("      TERMINAL REPORT: CLASSIFICATION")
    print("="*45)
    
    acc = accuracy_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc_val:.4f}")
    
    print("\nConfusion Matrix (Balanced-Cautious Threshold):")
    print(f"True LOS (Correct):   {cm[0][0]}")
    print(f"False NLOS (Mistake): {cm[0][1]}")
    print(f"False LOS (Mistake):  {cm[1][0]}")
    print(f"True NLOS (Correct):  {cm[1][1]}")
    
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['LOS', 'NLOS']))

    # Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_map = {0: "FP_IDX", 1: "STDEV_NOISE", 2: "PEAK2_POS", 3: "PEAK2_AMP"}
    
    print("\nTop 10 Feature Importances:")
    for i in range(10):
        idx = indices[i]
        name = feature_map.get(idx, f"PCA_Comp_{idx-4}")
        print(f"{i+1:2}. {name:15} : {importances[idx]:.4f}")

    # 6. PLOT GENERATION
    print("\n--- Generating Visualization Files ---")

    # Chart 1: Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred LOS', 'Pred NLOS'], yticklabels=['Actual LOS', 'Actual NLOS'])
    plt.title("Confusion Matrix (Threshold: 0.45)")
    plt.savefig(plot_dir / "confusion_matrix.png")
    plt.close()

    # Chart 2: Feature Importance
    top_n = 10
    top_indices = indices[:top_n]
    top_names = [feature_map.get(i, f"PCA_{i-4}") for i in top_indices]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[top_indices], y=top_names, hue=top_names, palette="magma", legend=False)
    plt.title("Top 10 Feature Importance")
    plt.savefig(plot_dir / "feature_importance.png")
    plt.close()

    # Chart 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(plot_dir / "roc_curve.png")
    plt.close()

    # 7. Final Save for Regressor
    np.save(data_dir / "y_test_pred_class.npy", y_pred)
    print(f"[SUCCESS] Balanced labels saved. Run your Regressor now!")

if __name__ == "__main__":
    main()