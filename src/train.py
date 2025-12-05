"""
Training script with pipeline-specific timestamped run directories, optional SMOTE,
and metrics.json written per run.

Usage examples:
  python -m src.train --dataset cicids --pipeline cnn --epochs 70 --batch_size 256
  python -m src.train --dataset nslkdd --pipeline xgb --num_features 40 --epochs 70 --batch_size 256 --use_smote
"""
import os
import argparse
import joblib
import json
import time
import numpy as np
from datetime import datetime

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from sklearn.metrics import accuracy_score, f1_score, classification_report
try:
    from imblearn.over_sampling import SMOTE
    _has_smote = True
except Exception:
    _has_smote = False

from .utils import get_dataset_results_dir, save_scaler, save_label_encoder
from .preprocess import load_and_preprocess, load_nslkdd_preprocessed, load_wsnds_preprocessed
from .xgboost_features import select_top_features
from .cnn_bilstm_model import build_cnn_bilstm
from .bilstm_model import build_bilstm
from .evaluate import evaluate_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATASET_MAP = {
    "cicids": os.path.join(DATA_DIR, "CICIDS2017_cleaned.csv"),
    "unsw": os.path.join(DATA_DIR, "UNSW-NB15.csv"),
    "nslkdd_train": os.path.join(DATA_DIR, "KDDTrain+.txt"),
    "nslkdd_test": os.path.join(DATA_DIR, "KDDTest+.txt"),
    "wsnds": os.path.join(DATA_DIR, "WSN-DS.csv"),
}

def _timestamped_run_dir(dataset_key, pipeline_name):
    base = get_dataset_results_dir(dataset_key)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{dataset_key}_{ts} ({pipeline_name})"
    run_dir = os.path.join(base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _prepare_dataset(key, test_size=0.2, sample_frac=None):
    key = key.lower()
    if key == "nslkdd":
        train_path = DATASET_MAP.get("nslkdd_train")
        test_path  = DATASET_MAP.get("nslkdd_test")
        if not train_path or not test_path:
            raise FileNotFoundError("NSL-KDD train/test paths missing in DATA_DIR")
        X_train, X_test, y_train, y_test, scaler, le, feature_names = load_nslkdd_preprocessed(train_path, test_path)
        results_key = "nslkdd"
    elif key == "wsnds":
        csv_path = DATASET_MAP.get("wsnds")
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        X_train, X_test, y_train, y_test, scaler, le, feature_names = load_wsnds_preprocessed(csv_path, test_size=test_size)
        results_key = "wsnds"
    else:
        csv_path = DATASET_MAP.get(key)
        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found for dataset key: {key} -> {csv_path}")
        X_train, X_test, y_train, y_test, scaler, le, feature_names = load_and_preprocess(csv_path, test_size=test_size, sample_frac=sample_frac)
        results_key = key
    return X_train, X_test, y_train, y_test, scaler, le, feature_names, results_key

def _make_callbacks(run_dir, initial_epoch):
    ckpt_pattern = os.path.join(run_dir, "ckpt_epoch-{epoch:02d}_valacc-{val_accuracy:.4f}.h5")
    best_path = os.path.join(run_dir, "best_model.h5")
    csvlog_path = os.path.join(run_dir, "training_log.csv")

    cp = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_pattern, monitor='val_accuracy', save_best_only=False, verbose=1)
    best = tf.keras.callbacks.ModelCheckpoint(filepath=best_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    csvlog = tf.keras.callbacks.CSVLogger(csvlog_path, append=(initial_epoch > 0))
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    return [cp, best, csvlog, es]

def train(dataset, pipeline="cnn", epochs=25, batch_size=256, num_features=40, test_size=0.2, sample_frac=None, resume=None, use_smote=False):
    # load dataset
    X_train, X_test, y_train, y_test, scaler, le, feature_names, results_key = _prepare_dataset(dataset, test_size=test_size, sample_frac=sample_frac)

    # create run dir with pipeline suffix
    run_dir = _timestamped_run_dir(results_key, pipeline)
    print(f"[INFO] Run directory: {run_dir}")

    # save scaler/label encoder to legacy dataset folder and to run dir
    try:
        save_scaler(scaler, results_key)
        save_label_encoder(le, results_key)
    except Exception:
        pass
    joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(run_dir, "label_encoder.pkl"))

    # select pipeline
    pipeline = pipeline.lower()
    if pipeline == "cnn":
        print("[PIPELINE] CNN + BiLSTM (full features)")
        input_len = X_train.shape[1]
        model = build_cnn_bilstm(num_timesteps=input_len, num_classes=int(len(np.unique(y_train))))
        X_train_in, X_test_in = X_train, X_test

    elif pipeline == "xgb":
        print("[PIPELINE] XGBoost -> BiLSTM (top-k features)")
        X_train_sel, X_test_sel, idx = select_top_features(X_train, y_train, X_test, num_features=num_features)
        input_len = X_train_sel.shape[1]
        model = build_bilstm(num_timesteps=input_len, num_classes=int(len(np.unique(y_train))))
        X_train_in, X_test_in = X_train_sel, X_test_sel
        meta = {"selected_indices": idx.tolist(), "feature_names": [feature_names[i] for i in idx]}
        joblib.dump(meta, os.path.join(run_dir, "feature_meta.pkl"))
    else:
        raise ValueError("Unknown pipeline: choose 'cnn' or 'xgb'")

    # Optionally apply SMOTE to training set only
    if use_smote:
        if not _has_smote:
            raise RuntimeError("imblearn not installed. pip install imbalanced-learn to use --use_smote")
        print("[INFO] Applying SMOTE to training data...")
        sm = SMOTE(random_state=SEED)
        X_train_in, y_train = sm.fit_resample(X_train_in, y_train)
        print(f"[INFO] After SMOTE, training set size = {X_train_in.shape[0]}")

    # resume logic
    initial_epoch = 0
    if resume:
        if os.path.isdir(resume):
            resume_run_dir = resume
        elif os.path.isfile(resume):
            resume_run_dir = os.path.dirname(resume)
        else:
            resume_run_dir = None

        if resume_run_dir:
            possible_log = os.path.join(resume_run_dir, "training_log.csv")
            if os.path.exists(possible_log):
                try:
                    import pandas as pd
                    df = pd.read_csv(possible_log)
                    initial_epoch = len(df)
                    print(f"[RESUME] Found training_log.csv in {resume_run_dir}, setting initial_epoch={initial_epoch}")
                except Exception as e:
                    print("[RESUME] Could not parse training_log.csv:", e)
            if os.path.isfile(resume) and resume.lower().endswith((".h5", ".keras")):
                print(f"[RESUME] Loading model from file: {resume}")
                try:
                    model = tf.keras.models.load_model(resume)
                    print("[RESUME] Model loaded successfully.")
                except Exception as e:
                    print("[RESUME] Failed to load model file:", e)

    callbacks = _make_callbacks(run_dir, initial_epoch)

    # train & time it
    t0 = time.time()
    history = model.fit(
        X_train_in, y_train,
        validation_data=(X_test_in, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=initial_epoch
    )
    train_time_seconds = time.time() - t0

    # save final model into run dir
    final_name = f"final_{pipeline}_model.h5"
    final_path = os.path.join(run_dir, final_name)
    model.save(final_path)
    print(f"[DONE] Saved final model to {final_path}")

    # save run metadata
    try:
        joblib.dump({"run_dir": run_dir, "final_model": final_name, "pipeline": pipeline, "dataset": results_key}, os.path.join(run_dir, "run_meta.pkl"))
    except Exception:
        pass

    # Evaluate and explicitly save outputs inside the run_dir
    run_basename = os.path.basename(run_dir)
    eval_dataset_name = f"{results_key}/{run_basename}"
    try:
        evaluate_model(model, X_test_in, y_test, label_encoder=le, dataset_name=eval_dataset_name)
    except Exception as e:
        print("[WARN] Evaluation failed:", e)

    # compute numeric metrics and save metrics.json
    y_pred = np.argmax(model.predict(X_test_in), axis=1)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": report,
        "train_time_seconds": float(train_time_seconds),
        "pipeline": pipeline,
        "dataset": results_key,
        "num_features_used": int(X_train_in.shape[1])
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics.json -> {os.path.join(run_dir, 'metrics.json')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cicids", help="cicids | unsw | nslkdd | wsnds")
    parser.add_argument("--pipeline", type=str, default="cnn", help="cnn | xgb")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_features", type=int, default=40, help="top-k features for xgb pipeline")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None, help="path to run directory or model file to resume from")
    parser.add_argument("--sample_frac", type=float, default=None, help="fraction of dataset to sample for quick runs")
    parser.add_argument("--use_smote", action="store_true", help="apply SMOTE to training set (imbalanced datasets)")
    args = parser.parse_args()

    train(dataset=args.dataset, pipeline=args.pipeline, epochs=args.epochs, batch_size=args.batch_size,
          num_features=args.num_features, test_size=args.test_size, sample_frac=args.sample_frac,
          resume=args.resume, use_smote=args.use_smote)

if __name__ == "__main__":
    main()
