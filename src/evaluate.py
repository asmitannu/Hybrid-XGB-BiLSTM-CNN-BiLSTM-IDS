import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, label_encoder, dataset_name="results"):
    """
    Evaluate a model and save:
      - classification_report (text)
      - confusion_matrix.png (heatmap)
      - confusion_matrix.csv (numeric matrix)
      - confusion_matrix.npy (numpy array)
      - metrics_fpr.json (accuracy + binary FPR + per-class FPRs)
    """
    print("\n--- Evaluating model ---")

    # Predict (supports one-hot output)
    y_prob = model.predict(X_test)
    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
        y_pred = np.argmax(y_prob, axis=1)
    else:
        # binary output
        y_pred = (y_prob.ravel() > 0.5).astype(int)

    # Ensure y_test is integer labels (if one-hot -> argmax)
    if getattr(y_test, "ndim", 1) > 1 and y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = np.array(y_test)

    # Determine classes present and label names
    present_classes = np.unique(np.concatenate([y_true, y_pred]))
    target_names = label_encoder.inverse_transform(present_classes)

    # Classification report (dict)
    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Results directory
    results_dir = os.path.join("results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save classification report (readable)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        for label in report:
            f.write(f"{label}: {report[label]}\n")
    print(f"Saved classification report → {report_path}")

    # Confusion matrix (numeric)
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    # Save numeric CM (csv + npy)
    cm_csv = os.path.join(results_dir, "confusion_matrix.csv")
    np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)
    print(f"Saved numeric confusion matrix → {cm_csv} and .npy")

    # Save heatmap (visual) as before
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix image → {cm_path}")

    # --- Compute binary FPR (FAR) treating "Normal" as negative ---
    # Try common normal labels in order (adjust list if your dataset uses different string)
    possible_normal_labels = ["Normal", "normal", "Normal Traffic", "normal traffic"]
    normal_label_name = None
    # find exact matching label among target_names
    for candidate in possible_normal_labels:
        if candidate in target_names:
            normal_label_name = candidate
            break
    # fallback: assume the class with largest support is normal
    if normal_label_name is None:
        # pick label with largest support from report (if present)
        supports = {lab: report[lab].get("support", 0) for lab in report if lab not in ("accuracy","macro avg","weighted avg")}
        if supports:
            normal_label_name = max(supports, key=supports.get)

    # Map label name -> numeric index in present_classes
    try:
        normal_index = np.where(target_names == normal_label_name)[0][0]
    except Exception:
        normal_index = None

    metrics_out = {
        "dataset_name": dataset_name,
        "present_classes": target_names.tolist(),
        "classification_report": report
    }

    if normal_index is not None:
        # Build binary confusion matrix: 0 = normal, 1 = attack
        # Sum all columns/rows: normal is normal_index; everything else -> attack
        # For binary counts:
        # TN = cm[normal_index, normal_index]
        # FP = sum(cm[:, normal_index]) - TN  (others predicted as normal)
        # but careful with orientation: cm[row=true, col=pred]
        TN = int(cm[normal_index, normal_index])
        # FP = number of actual non-normal predicted as normal:
        FP = int(cm[:, normal_index].sum() - TN)
        # FN = number of actual normal predicted as non-normal:
        FN = int(cm[normal_index, :].sum() - TN)
        # TP = all other correct: total non-normal predicted non-normal correctly
        TP = int(cm.sum() - (TN + FP + FN))

        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        metrics_out.update({
            "binary_counts": {"tn": TN, "fp": FP, "fn": FN, "tp": TP},
            "binary_fpr": float(fpr),
            "binary_normal_label": normal_label_name
        })

        # also compute per-class "FPR" = FP_for_class / (FP_for_class + TN_for_class)
        per_class_fpr = {}
        total = cm.sum()
        for idx, cname in enumerate(target_names):
            # For class 'c':
            # FP_for_c = sum of predictions == c for true != c
            FP_c = int(cm[:, idx].sum() - cm[idx, idx])
            # TN_for_c = total - (TP_c + FP_c + FN_c)
            TP_c = int(cm[idx, idx])
            FN_c = int(cm[idx, :].sum() - TP_c)
            TN_c = int(total - (TP_c + FP_c + FN_c))
            per_class_fpr[cname] = {"fp": FP_c, "tn": TN_c,
                                     "fpr": (FP_c / (FP_c + TN_c) if (FP_c + TN_c) > 0 else 0.0)}
        metrics_out["per_class_fpr"] = per_class_fpr

    else:
        metrics_out["binary_fpr"] = None
        metrics_out["binary_counts"] = None
        metrics_out["binary_normal_label"] = None
        metrics_out["per_class_fpr"] = {}

    # Save metrics + FPR
    metrics_path = os.path.join(results_dir, "metrics_fpr.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics + FPR → {metrics_path}")

    print("Evaluation completed.")
