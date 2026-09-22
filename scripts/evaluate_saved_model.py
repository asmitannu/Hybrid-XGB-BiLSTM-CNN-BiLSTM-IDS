import os
import argparse
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so we can import src
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import _prepare_dataset
from apps.inference import model_predict_proba

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model without retraining.")
    parser.add_argument("--model", required=True, help="Path to the saved model (.h5 or .pkl)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., cicids, unsw, nslkdd, wsnds)")
    parser.add_argument("--scaler", required=True, help="Path to the saved scaler.pkl")
    parser.add_argument("--le", required=True, help="Path to the saved label_encoder.pkl")
    parser.add_argument("--feature_meta", default=None, help="Path to feature_meta.pkl (if xgb pipeline was used)")
    parser.add_argument("--out_dir", default="results/eval_saved_model", help="Output directory for results")
    args = parser.parse_args()

    print(f"[INFO] Loading dataset '{args.dataset}' via existing preprocessing pipeline...")
    # This uses the existing logic to load, split, and encode the data
    X_train, X_test, y_train, y_test, new_scaler, new_le, feature_names, results_key = _prepare_dataset(args.dataset)

    # The new preprocess.py automatically scales X_test using the freshly fitted scaler.
    # To evaluate exactly as the model was trained, we inverse transform with the new scaler,
    # then transform with the SAVED scaler.
    print(f"[INFO] Applying saved scaler from '{args.scaler}'...")
    saved_scaler = joblib.load(args.scaler)
    
    # Check if the new_scaler has an inverse_transform method (it's a StandardScaler)
    if hasattr(new_scaler, 'inverse_transform'):
        X_test_raw = new_scaler.inverse_transform(X_test)
    else:
        # Fallback if somehow not scaled
        X_test_raw = X_test
        
    X_test_scaled = saved_scaler.transform(X_test_raw)

    print(f"[INFO] Applying saved label encoder from '{args.le}'...")
    saved_le = joblib.load(args.le)
    
    # If XGBoost pipeline was used, slice the top features
    if args.feature_meta:
        print(f"[INFO] Applying feature selection from '{args.feature_meta}'...")
        meta = joblib.load(args.feature_meta)
        idx = np.array(meta["selected_indices"], dtype=int)
        X_test_scaled = X_test_scaled[:, idx]

    print(f"[INFO] Generating predictions using model '{args.model}'...")
    # Use inference helper to handle SavedModel, Keras, or XGBoost formats robustly
    probs = model_predict_proba(args.model, X_test_scaled)
    
    if probs.ndim > 1 and probs.shape[1] > 1:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = (probs.ravel() > 0.5).astype(int)

    # Ensure y_test is 1D
    if y_test.ndim > 1:
        y_test_1d = np.argmax(y_test, axis=1)
    else:
        y_test_1d = y_test

    # Calculate metrics
    acc = accuracy_score(y_test_1d, y_pred)
    prec = precision_score(y_test_1d, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_1d, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_1d, y_pred, average='weighted', zero_division=0)
    
    present_classes = np.unique(np.concatenate([y_test_1d, y_pred]))
    target_names = saved_le.inverse_transform(present_classes)
    
    report = classification_report(y_test_1d, y_pred, labels=present_classes, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test_1d, y_pred, labels=present_classes)

    print("\n" + "="*50)
    print("                 EVALUATION RESULTS                 ")
    print("="*50)
    print(f"Accuracy:           {acc:.4f}")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall:    {rec:.4f}")
    print(f"Weighted F1 Score:  {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("Confusion Matrix:\n")
    print(cm)
    print("="*50)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save confusion matrix image
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix heatmap -> {cm_path}")

    # Save report
    report_path = os.path.join(args.out_dir, "evaluation_metrics.txt")
    with open(report_path, "w") as f:
        f.write("=== EVALUATION METRICS ===\n")
        f.write(f"Accuracy:           {acc:.4f}\n")
        f.write(f"Weighted Precision: {prec:.4f}\n")
        f.write(f"Weighted Recall:    {rec:.4f}\n")
        f.write(f"Weighted F1 Score:  {f1:.4f}\n\n")
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report)
        f.write("\n=== CONFUSION MATRIX ===\n")
        f.write(np.array2string(cm))
    print(f"[INFO] Saved metrics report -> {report_path}")


if __name__ == "__main__":
    main()
