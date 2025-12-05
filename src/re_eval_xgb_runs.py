import os
import joblib
import glob
import importlib
from tensorflow.keras.models import load_model

# adjust import path if your project structure differs
from src.evaluate import evaluate_model
from src.preprocess import load_and_preprocess, load_nslkdd_preprocessed, load_wsnds_preprocessed

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

def find_xgb_runs(results_root):
    runs = []
    # results/<dataset>/<run_name (contains (xgb))>
    for dataset_dir in os.listdir(results_root):
        dataset_path = os.path.join(results_root, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
        for run in os.listdir(dataset_path):
            if "(xgb)" in run:
                runs.append(os.path.join(dataset_path, run))
    return runs

def load_testset_for_run(run_dir):
    # run_dir typically contains run metadata: run_meta.pkl, label_encoder.pkl or saved copies
    # train.py saves label_encoder/scaler into run_dir, but original loaders require file paths
    meta_le_path = os.path.join(run_dir, "label_encoder.pkl")
    if os.path.exists(meta_le_path):
        le = joblib.load(meta_le_path)
    else:
        # fallback: try to load from dataset folder (older behavior)
        le = None

    # decide dataset name from run_dir path: results/<dataset>/<runname>
    components = run_dir.split(os.sep)
    dataset_key = components[-2]  # e.g., 'nslkdd' or 'wsnds' etc.

    # Use same preprocess functions as train.py to get X_test, y_test, scaler, le, feature_names
    if dataset_key == "nslkdd":
        # requires that your data files exist in data/ paths defined in train.py
        X_train, X_test, y_train, y_test, scaler, le2, feature_names = load_nslkdd_preprocessed(None, None)
        # The function signature in your code expects train/test paths; adjust if necessary
    elif dataset_key == "wsnds":
        # modify as your function signature expects
        X_train, X_test, y_train, y_test, scaler, le2, feature_names = load_wsnds_preprocessed(None)
    else:
        # generic CSV loader
        csv_path = os.path.join(PROJECT_ROOT, "data", dataset_key + ".csv")
        X_train, X_test, y_train, y_test, scaler, le2, feature_names = load_and_preprocess(csv_path, test_size=0.2)

    # prefer label encoder saved in run_dir if exists
    if le is None and le2 is not None:
        le = le2

    return X_test, y_test, le

def main():
    runs = find_xgb_runs(RESULTS_ROOT)
    print(f"Found {len(runs)} xgb runs")
    for r in runs:
        print("Processing:", r)
        # find final model file
        model_file = None
        for candidate in ["final_xgb_model.h5", "final_xgb_model.h5", "final_cnn_model.h5", "final_xgb_model.h5"]:
            path = os.path.join(r, candidate)
            if os.path.exists(path):
                model_file = path
                break
        # fallback: search for .h5
        if model_file is None:
            h5s = glob.glob(os.path.join(r, "*.h5"))
            if h5s:
                model_file = h5s[-1]
        if model_file is None:
            print("  No model .h5 found in", r)
            continue

        model = load_model(model_file)
        X_test, y_test, le = load_testset_for_run(r)
        if X_test is None or y_test is None or le is None:
            print("  Could not reconstruct test set for run:", r)
            continue

        # call evaluate_model which now saves CSV, PNG, numpy and metrics_fpr.json
        evaluate_model(model, X_test, y_test, label_encoder=le, dataset_name=os.path.join(os.path.basename(os.path.dirname(r)), os.path.basename(r)))

if __name__ == "__main__":
    main()
