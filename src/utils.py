import os
import joblib
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GLOBAL_RESULTS = os.path.join(PROJECT_ROOT, "results")

def get_dataset_results_dir(dataset_name):
    d = os.path.join(GLOBAL_RESULTS, dataset_name.lower())
    os.makedirs(d, exist_ok=True)
    return d

def save_scaler(scaler, dataset_name, fname="scaler.pkl"):
    d = get_dataset_results_dir(dataset_name)
    path = os.path.join(d, fname)
    joblib.dump(scaler, path)
    return path

def load_scaler(dataset_name, fname="scaler.pkl"):
    path = os.path.join(get_dataset_results_dir(dataset_name), fname)
    return joblib.load(path)

def save_label_encoder(le, dataset_name, fname="label_encoder.pkl"):
    d = get_dataset_results_dir(dataset_name)
    p = os.path.join(d, fname)
    joblib.dump(le, p)
    return p

def load_label_encoder(dataset_name, fname="label_encoder.pkl"):
    path = os.path.join(get_dataset_results_dir(dataset_name), fname)
    return joblib.load(path)

def save_json(obj, dataset_name, fname):
    d = get_dataset_results_dir(dataset_name)
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path

def write_text(s, dataset_name, fname):
    d = get_dataset_results_dir(dataset_name)
    path = os.path.join(d, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    return path

def list_saved_checkpoints(dataset_name):
    d = get_dataset_results_dir(dataset_name)
    return [os.path.join(d, f) for f in os.listdir(d) if f.startswith("ckpt_epoch")]
