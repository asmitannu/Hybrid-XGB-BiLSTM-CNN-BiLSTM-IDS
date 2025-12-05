# src/streamlit_app.py
# Drop-in replacement. Save and restart streamlit.

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import os
import importlib
import traceback
from pathlib import Path

st.set_page_config(page_title="Hybrid Intrusion Detection System", layout="wide")

ROOT = Path.cwd()
APP_MODELS = ROOT / "app_models"

# simple styling
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    body { background-color: #0b0b0b; color: #fff; }
    h1, h2, h3, label { color: #ffdddd; }
    .stButton>button { background-color: #a40d0d; color: white; border: none; padding: 6px 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Hybrid CNN-XGB BILSTM IDS")
st.write("Upload a CSV. App will auto-select a matching model and predict.")

uploaded = st.file_uploader("Upload CSV (raw features)", type=["csv"])
threshold = st.slider("Binary attack threshold (p_attack >= threshold)", 0.01, 0.99, 0.5, 0.01)
show_match_details = st.checkbox("Show model match details", value=True)

@st.cache_data(ttl=300)
def discover_app_models(models_dir: Path):
    """
    Discover model variant folders under app_models/.
    Accepts two patterns:
      - app_models/UNSW_CNN/
      - app_models/UNSW/UNSW_CNN/
    Returns list of meta dicts for each variant folder.
    """
    out = []
    if not models_dir.exists():
        return out

    def make_meta(sub: Path):
        meta = {"name": sub.name, "path": sub, "mtime": sub.stat().st_mtime}
        model_file = None
        for fname in ["best_model.h5","final_model.h5","best_model.pkl","final_model.pkl","model.h5","model.pkl"]:
            p = sub / fname
            if p.exists():
                model_file = p
                break
        if model_file is None:
            candidates = list(sub.glob("*.h5")) + list(sub.glob("*.pkl"))
            if candidates:
                model_file = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
        meta["model_file"] = model_file
        meta["scaler"] = (sub / "scaler.pkl") if (sub / "scaler.pkl").exists() else None
        meta["label_enc"] = (sub / "label_encoder.pkl") if (sub / "label_encoder.pkl").exists() else ((sub / "label_enc.pkl") if (sub / "label_enc.pkl").exists() else None)
        meta["feature_meta"] = (sub / "feature_meta.pkl") if (sub / "feature_meta.pkl").exists() else None
        return meta

    # first, check direct children
    for sub in sorted(models_dir.iterdir()):
        if sub.is_dir():
            # if this folder itself looks like a variant (has model/scaler/feature_meta)
            has_any = any((sub / f).exists() for f in ["best_model.h5","model.h5","scaler.pkl","feature_meta.pkl"])
            if has_any:
                out.append(make_meta(sub))
            else:
                # check one level deeper for variant folders
                for sub2 in sorted(sub.iterdir()):
                    if sub2.is_dir():
                        # treat sub2 as potential variant
                        has_any2 = any((sub2 / f).exists() for f in ["best_model.h5","model.h5","scaler.pkl","feature_meta.pkl"])
                        if has_any2:
                            out.append(make_meta(sub2))
    return out

models_list = discover_app_models(APP_MODELS)

def score_model_match(meta, df: pd.DataFrame):
    # 0..1 score of match quality
    # 1. if feature_meta present: fraction of required features present
    if meta["feature_meta"]:
        try:
            fm = joblib.load(meta["feature_meta"])
            feat_names = fm.get("feature_names") if isinstance(fm, dict) else None
            if feat_names:
                present = sum(1 for f in feat_names if f in df.columns)
                return present / len(feat_names)
        except Exception:
            pass
    # 2. fallback: scaler.n_features_in_
    if meta["scaler"]:
        try:
            sc = joblib.load(meta["scaler"])
            n_in = getattr(sc, "n_features_in_", None)
            if n_in is not None:
                return 1.0 if int(n_in) == df.shape[1] else 0.0
        except Exception:
            pass
    return 0.0

def pick_best_model(df: pd.DataFrame, models_meta):
    scored = []
    for m in models_meta:
        s = score_model_match(m, df)
        scored.append((s, m))
    if not scored:
        return None, 0.0
    scored = sorted(scored, key=lambda x: (x[0], x[1]["mtime"]), reverse=True)
    return scored[0][1], float(scored[0][0])

def load_artifacts(meta):
    out = {}
    out["model_file"] = meta["model_file"]
    out["scaler"] = joblib.load(meta["scaler"]) if meta["scaler"] else None
    out["le"] = joblib.load(meta["label_enc"]) if meta["label_enc"] else None
    try:
        out["feature_meta"] = joblib.load(meta["feature_meta"]) if meta["feature_meta"] else None
    except Exception:
        out["feature_meta"] = None
    return out

def safe_prepare_input(df, scaler, feature_meta):
    """
    - Use feature_meta['feature_names'] ordering if available.
    - Fill missing features with zeros.
    - Truncate extra columns.
    - Pad with zeros if scaler expects more features.
    Returns X_scaled and a message list.
    """
    msgs = []
    # build X_raw in expected order
    if feature_meta and isinstance(feature_meta, dict) and feature_meta.get("feature_names"):
        expected = list(feature_meta["feature_names"])
        missing = [f for f in expected if f not in df.columns]
        if missing:
            msgs.append(f"Missing {len(missing)} features; filling them with 0. Example: {missing[:6]}{'...' if len(missing)>6 else ''}")
        # create numeric array
        X_raw = np.zeros((len(df), len(expected)), dtype=float)
        for i, fname in enumerate(expected):
            if fname in df.columns:
                vals = pd.to_numeric(df[fname], errors="coerce").fillna(0).values
                X_raw[:, i] = vals
            else:
                X_raw[:, i] = 0.0
    else:
        # no feature_meta, use uploaded df columns order
        X_raw = df.values.astype(float)
        msgs.append("No feature_meta available; using CSV columns order for inference.")

    # ensure correct shape for scaler
    if scaler is not None:
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in is not None:
            n_in = int(n_in)
            if X_raw.shape[1] < n_in:
                pad_n = n_in - X_raw.shape[1]
                X_raw = np.hstack([X_raw, np.zeros((X_raw.shape[0], pad_n), dtype=float)])
                msgs.append(f"Padded input with {pad_n} zero columns to match scaler.n_features_in_={n_in}.")
            elif X_raw.shape[1] > n_in:
                X_raw = X_raw[:, :n_in]
                msgs.append(f"Truncated input to first {n_in} columns to match scaler.n_features_in_={n_in}.")
    else:
        msgs.append("Warning: no scaler provided; skipping scaling.")

    # scale if scaler present
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_raw)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")
    else:
        X_scaled = X_raw
    return X_scaled, msgs

# Keep existing tf/keras fallback loaders available
def _try_tf_load_with_custom(path):
    """Attempt to load Keras model using tf.keras with a helpful custom_objects map."""
    # map common layer names to tf.keras implementations
    import tensorflow as tf
    custom = {
        "LSTM": tf.keras.layers.LSTM,
        "GRU": tf.keras.layers.GRU,
        "Bidirectional": tf.keras.layers.Bidirectional,
        "TimeDistributed": tf.keras.layers.TimeDistributed,
        "Masking": tf.keras.layers.Masking,
        "LayerNormalization": tf.keras.layers.LayerNormalization,
        "Dropout": tf.keras.layers.Dropout,
        "Dense": tf.keras.layers.Dense,
        "Conv1D": tf.keras.layers.Conv1D,
        "Conv2D": tf.keras.layers.Conv2D,
        "Flatten": tf.keras.layers.Flatten,
        "Embedding": tf.keras.layers.Embedding,
        # add more if you used custom layers
    }
    # try normal load first (works in most cases)
    try:
        model = tf.keras.models.load_model(str(path))
        return model
    except Exception as e:
        # try with custom_objects
        try:
            model = tf.keras.models.load_model(str(path), custom_objects=custom, compile=False)
            return model
        except Exception:
            # bubble up last exception for debugging
            raise

def _try_keras_load(path):
    """If standalone keras is installed, try loading with it (some models saved with keras need this)."""
    try:
        keras = importlib.import_module("keras")
    except Exception:
        return None
    try:
        model = keras.models.load_model(str(path), compile=False)
        return model
    except Exception:
        # try with custom_objects mapping to keras.layers
        try:
            custom = {
                "LSTM": keras.layers.LSTM,
                "GRU": keras.layers.GRU,
                "Bidirectional": keras.layers.Bidirectional,
                "TimeDistributed": keras.layers.TimeDistributed,
                "Masking": keras.layers.Masking,
                "LayerNormalization": keras.layers.LayerNormalization,
            }
            model = keras.models.load_model(str(path), custom_objects=custom, compile=False)
            return model
        except Exception:
            return None

@tf.keras.utils.register_keras_serializable(package="custom")
class LSTMCompat(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        # silently drop keys that older/newer tf.keras doesn't accept
        config.pop("time_major", None)
        config.pop("time_major", None)  # safe no-op if not present
        return super().from_config(config)


# ----------------- REPLACED model_predict_proba -----------------
def model_predict_proba(model_file, X):
    """
    Robust model loader/predictor.
    - model_file may be:
        * path to a SavedModel directory -> use tf.saved_model.load + signature
        * path to a Keras .h5/.keras -> tf.keras.models.load_model
        * path to a joblib/sklearn/xgboost .pkl -> joblib.load
    - X: numpy array (n_samples, n_features)
    Returns: probs (n_samples, n_classes) as numpy array
    """
    model_path = Path(model_file)

    # 1) SavedModel directory
    if model_path.is_dir():
        try:
            loaded = tf.saved_model.load(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to tf.saved_model.load('{model_path}'): {e}")

        # Prefer 'serving_default' signature when present
        sig = None
        try:
            if hasattr(loaded, "signatures") and loaded.signatures:
                if "serving_default" in loaded.signatures:
                    sig = loaded.signatures["serving_default"]
                else:
                    sig = list(loaded.signatures.values())[0]
        except Exception:
            sig = None

        # If no signature found, try calling the loaded object directly if callable
        if sig is None:
            if callable(loaded):
                try:
                    tf_in = tf.convert_to_tensor(X, dtype=tf.float32)
                    out = loaded(tf_in)
                    if isinstance(out, dict):
                        arr = list(out.values())[0].numpy()
                    else:
                        arr = out.numpy()
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        arr = np.vstack([1 - arr, arr]).T
                    return arr
                except Exception as e:
                    raise RuntimeError(f"SavedModel has no usable signature and direct call failed: {e}")
            raise RuntimeError("SavedModel has no signatures. Could not find 'serving_default' or other callable entrypoint.")

        # Call signature robustly (positional or named)
        try:
            tf_in = tf.convert_to_tensor(X, dtype=tf.float32)
            try:
                result = sig(tf_in)
            except Exception:
                # try calling with detected input name
                structured = sig.structured_input_signature
                kwargs = structured[1] if isinstance(structured, tuple) and len(structured) > 1 else {}
                if kwargs:
                    input_name = list(kwargs.keys())[0]
                    result = sig(**{input_name: tf_in})
                else:
                    result = sig(tf_in)
        except Exception as e:
            raise RuntimeError(f"Failed to call SavedModel signature: {e}")

        # Extract numpy array from result
        if isinstance(result, dict):
            val = list(result.values())[0]
            probs = val.numpy()
        else:
            probs = result.numpy()

        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        return probs

    # 2) Keras H5 or native .keras file
    suf = model_path.suffix.lower()
    if suf in [".h5", ".keras"]:
        try:
            model = tf.keras.models.load_model(str(model_path))
        except Exception as e:
            try:
                custom = {
                    "LSTM": LSTMCompat,
                    "Bidirectional": tf.keras.layers.Bidirectional,
                    "GRU": tf.keras.layers.GRU,
                    "RNN": tf.keras.layers.RNN,
                    # add more if your model uses them
                }

                model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects=custom)
                # model = tf.keras.models.load_model(str(model_path), compile=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load Keras model: {e2}")
        preds = model.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = np.vstack([1 - preds, preds]).T
        return preds

    # 3) joblib / sklearn / xgboost
    try:
        mdl = joblib.load(str(model_path))
    except Exception:
        raise RuntimeError(f"Model file not found or unsupported format: {model_file}")

    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        return probs

    if hasattr(mdl, "predict"):
        preds = mdl.predict(X)
        preds = np.asarray(preds)
        # convert discrete preds to one-hot-ish probabilities
        classes_unique = np.unique(preds)
        probs = np.zeros((len(preds), len(classes_unique)), dtype=float)
        for i, p in enumerate(preds):
            idx = int(np.where(classes_unique == p)[0][0])
            probs[i, idx] = 1.0
        return probs

    raise RuntimeError("Loaded model has no predict_proba or predict method.")
# ----------------- END replaced function -----------------

# UI
if uploaded is None:
    st.info("Upload a CSV file to run inference against models in app_models/.")
    if not models_list:
        st.warning("No model folders found under app_models/. Place your model variant folders (one per variant) there.")
else:
    # read uploaded CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error("Failed to read CSV: " + str(e))
        st.stop()

    st.write(f"Uploaded {len(df)} rows, {df.shape[1]} columns. Preview:")
    st.dataframe(df.head())

    if not models_list:
        st.error("No model folders found in app_models/.")
        st.stop()

    with st.spinner("Matching uploaded CSV to available model variants..."):
        best_meta, score = pick_best_model(df, models_list)
        time.sleep(0.15)

    if best_meta is None or score == 0.0:
        st.error("No matching model found. Make sure each model folder contains scaler.pkl and feature_meta.pkl (or scaler with n_features_in_).")
        # show available models and why they didn't match (debug)
        if show_match_details:
            st.write("Available model folders and quick info:")
            for m in models_list:
                st.write({ "name": m["name"], "has_model_file": bool(m["model_file"]), "has_scaler": bool(m["scaler"]), "has_feature_meta": bool(m["feature_meta"]) })
        st.stop()

    st.markdown(f"**Auto-selected model folder:** `{best_meta['name']}`  — match score: **{score:.3f}**")
    if show_match_details:
        st.write("Artifacts present:", {k: bool(best_meta.get(k)) for k in ["model_file","scaler","label_enc","feature_meta"]})

    artifacts = load_artifacts(best_meta)
    if artifacts["scaler"] is None or artifacts["model_file"] is None:
        st.error("Selected folder missing required artifacts (scaler or model file).")
        st.stop()

    # prepare input safely
    try:
        X_scaled, prep_msgs = safe_prepare_input(df, artifacts["scaler"], artifacts["feature_meta"])
    except Exception as e:
        st.error("Failed to prepare input: " + str(e))
        st.stop()

    for m in prep_msgs:
        st.info(m)

    # if feature selection indices exist, apply them AFTER scaling
    if artifacts["feature_meta"] and isinstance(artifacts["feature_meta"], dict) and artifacts["feature_meta"].get("selected_indices"):
        idx = np.array(artifacts["feature_meta"]["selected_indices"], dtype=int)
        try:
            X_in = X_scaled[:, idx]
        except Exception as e:
            st.error("Failed to apply selected_indices: " + str(e))
            st.stop()
    else:
        X_in = X_scaled

    # predict
    print(artifacts)
    print(artifacts["model_file"])
    with st.spinner("Running model inference..."):
        try:
            probs = model_predict_proba(artifacts["model_file"], X_in)
        except Exception as e:
            st.error("Model inference failed: " + str(e))
            st.stop()

    # determine class labels
    try:
        classes = list(artifacts["le"].classes_) if artifacts.get("le") is not None else None
    except Exception:
        classes = None

    if classes is None:
        # try to infer labels from model outputs (if one-hot turned into label indices)
        # create default class names 0..n-1
        n_classes = probs.shape[1]
        classes = [str(i) for i in range(n_classes)]

    # probs -> dataframe
    prob_df = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [classes[i] for i in pred_idx]

    out_df = df.reset_index(drop=True).copy()
    out_df["pred_label_multiclass"] = pred_labels
    out_df = pd.concat([out_df, prob_df.reset_index(drop=True)], axis=1)

    # determine normal_label
    normal_candidates = {"normal", "Normal", "NORMAL", "benign", "BENIGN", "Benign", "Normal Traffic"}
    normal_label = None
    for c in classes:
        if c in normal_candidates:
            normal_label = c
            break
    if normal_label is None:
        avg_probs = probs.mean(axis=0)
        normal_label = classes[int(np.argmax(avg_probs))]

    out_df["p_attack"] = 1.0 - out_df[f"p_{normal_label}"]
    out_df["pred_label_binary"] = np.where(out_df["p_attack"] >= threshold, "attack", "normal")

    st.success("Prediction completed.")
    st.markdown(f"**Detected normal label:** `{normal_label}`")
    st.subheader("Sample predictions")
    st.dataframe(out_df.head(10))
    st.download_button("Download predictions CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

    # quick eval if ground truth present
    true_col = None
    for c in ["label","Label","label_true","ground_truth"]:
        if c in df.columns:
            true_col = c
            break
    if true_col:
        from sklearn.metrics import accuracy_score, classification_report
        try:
            acc = accuracy_score(df[true_col].values, out_df["pred_label_multiclass"].values)
            st.write("Multiclass accuracy on uploaded file:", float(acc))
            st.text(classification_report(df[true_col].values, out_df["pred_label_multiclass"].values, zero_division=0))
        except Exception:
            pass

st.markdown("---")
st.caption("app_models/ auto-match: prefers feature_meta name matches then scaler feature count.")
