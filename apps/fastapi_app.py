# apps/fastapi_app.py
# Thin FastAPI REST adapter over the existing IDS inference pipeline.
# All ML logic is delegated to apps/inference.py (extracted from the
# original Streamlit application).  No new ML code is introduced.

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import (
    discover_app_models,
    load_artifacts,
    model_predict_proba,
    safe_prepare_input,
    determine_class_labels,
    determine_normal_label,
)

# ── Pydantic request / response schemas ──────────────────────────────

class SinglePredictRequest(BaseModel):
    model_name: Optional[str] = Field(
        None,
        description=(
            "Name of the model variant folder to use "
            "(e.g. 'CICIDS_CNN').  If omitted, the best-matching "
            "variant is selected automatically based on the supplied "
            "feature names."
        ),
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Mapping of feature-name → numeric value for one network flow.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Binary attack threshold (p_attack >= threshold → 'attack').",
    )


class BatchPredictRequest(BaseModel):
    model_name: Optional[str] = Field(
        None,
        description="Model variant name (same semantics as single predict).",
    )
    samples: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of feature dicts — one per network flow.",
    )
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Binary attack threshold.",
    )


class PredictionResult(BaseModel):
    predicted_label: str
    binary_label: str
    class_probabilities: Dict[str, float]
    preprocessing_messages: List[str] = []


class SinglePredictResponse(BaseModel):
    model_used: str
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    model_used: str
    predictions: List[PredictionResult]


# ── Application setup ────────────────────────────────────────────────

APP_MODELS_DIR = Path(os.environ.get("APP_MODELS_DIR", Path(__file__).resolve().parent.parent / "app_models"))

app = FastAPI(
    title="Hybrid IDS — Real-Time Intrusion Detection Service",
    description=(
        "Production REST API for the Hybrid CNN→BiLSTM / XGBoost→BiLSTM "
        "intrusion detection system.  Supports single-flow and batch "
        "prediction across CICIDS 2017, UNSW-NB15, NSL-KDD, and WSN-DS "
        "model variants."
    ),
    version="1.0.0",
)

# In-memory cache populated at startup
_models_meta: list = []
_loaded_artifacts: dict = {}  # name -> artifacts dict


@app.on_event("startup")
def _startup_load_models():
    """Discover and pre-load every model variant on startup."""
    global _models_meta, _loaded_artifacts
    _models_meta = discover_app_models(APP_MODELS_DIR)
    for meta in _models_meta:
        try:
            _loaded_artifacts[meta["name"]] = load_artifacts(meta)
        except Exception as exc:
            print(f"[WARN] Could not pre-load model '{meta['name']}': {exc}")
    print(f"[INFO] Loaded {len(_loaded_artifacts)} model variant(s) from {APP_MODELS_DIR}")


# ── Helper: resolve model variant ────────────────────────────────────

def _resolve_model(model_name: Optional[str], df: pd.DataFrame):
    """Return (artifacts_dict, meta_name).  Raises HTTPException on failure."""
    if model_name:
        if model_name not in _loaded_artifacts:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found.  Available: {list(_loaded_artifacts.keys())}",
            )
        return _loaded_artifacts[model_name], model_name

    # Auto-select: use feature-name overlap scoring (same as Streamlit)
    from .inference import score_model_match

    best, best_score, best_name = None, -1.0, None
    for meta in _models_meta:
        s = score_model_match(meta, df)
        if s > best_score:
            best_score = s
            best_name = meta["name"]
    if best_name is None or best_score == 0.0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not auto-select a model.  Either specify 'model_name' "
                "or ensure feature names match one of the available variants: "
                f"{list(_loaded_artifacts.keys())}"
            ),
        )
    return _loaded_artifacts[best_name], best_name


# ── Helper: run inference pipeline (shared by single & batch) ────────

def _run_inference(artifacts: dict, df: pd.DataFrame, threshold: float):
    """
    Execute the full inference pipeline (preprocessing → model → post-process).
    Returns a list of PredictionResult (one per row) and a list of messages.
    """
    scaler = artifacts["scaler"]
    feature_meta = artifacts["feature_meta"]

    if scaler is None or artifacts["model_file"] is None:
        raise HTTPException(status_code=500, detail="Model variant is missing required artifacts (scaler or model file).")

    # 1. Prepare input (same as Streamlit)
    X_scaled, prep_msgs = safe_prepare_input(df, scaler, feature_meta)

    # 2. Apply feature selection indices if present (XGB pipeline)
    if feature_meta and isinstance(feature_meta, dict) and feature_meta.get("selected_indices"):
        idx = np.array(feature_meta["selected_indices"], dtype=int)
        try:
            X_in = X_scaled[:, idx]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to apply selected_indices: {e}")
    else:
        X_in = X_scaled

    # 3. Predict
    try:
        probs = model_predict_proba(artifacts["model_file"], X_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # 4. Determine class labels
    classes = determine_class_labels(artifacts.get("le"))
    if classes is None:
        n_classes = probs.shape[1]
        classes = [str(i) for i in range(n_classes)]

    # 5. Normal label detection (same as Streamlit)
    normal_label = determine_normal_label(classes, probs)
    normal_idx = classes.index(normal_label) if normal_label in classes else 0

    # 6. Build per-row results
    results: List[PredictionResult] = []
    for row_idx in range(probs.shape[0]):
        row_probs = probs[row_idx]
        pred_idx = int(np.argmax(row_probs))
        pred_label = classes[pred_idx]
        p_attack = float(1.0 - row_probs[normal_idx])
        binary_label = "attack" if p_attack >= threshold else "normal"
        class_probs = {classes[j]: float(row_probs[j]) for j in range(len(classes))}
        results.append(
            PredictionResult(
                predicted_label=pred_label,
                binary_label=binary_label,
                class_probabilities=class_probs,
                preprocessing_messages=prep_msgs if row_idx == 0 else [],
            )
        )

    return results


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Service health check."""
    return {
        "status": "ok",
        "service": "Hybrid IDS — Real-Time Intrusion Detection Service",
        "models_loaded": len(_loaded_artifacts),
        "available_models": list(_loaded_artifacts.keys()),
    }


@app.post("/predict", response_model=SinglePredictResponse, tags=["Prediction"])
def predict_single(req: SinglePredictRequest):
    """
    Single-flow prediction.

    Submit one network-flow sample as a dict of feature-name → value.
    Returns the multiclass label, binary (normal/attack) label,
    and per-class probabilities.
    """
    if not req.features:
        raise HTTPException(status_code=422, detail="'features' must be a non-empty dict.")

    # Build a 1-row DataFrame so the existing pipeline can process it
    df = pd.DataFrame([req.features])
    artifacts, model_name = _resolve_model(req.model_name, df)
    results = _run_inference(artifacts, df, req.threshold)
    return SinglePredictResponse(model_used=model_name, prediction=results[0])


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """
    Batch prediction.

    Submit multiple network-flow samples.  Each element in 'samples'
    is a dict of feature-name → value.  Returns one prediction per
    input row, preserving input order.
    """
    if not req.samples:
        raise HTTPException(status_code=422, detail="'samples' must be a non-empty list.")

    df = pd.DataFrame(req.samples)
    artifacts, model_name = _resolve_model(req.model_name, df)
    results = _run_inference(artifacts, df, req.threshold)
    return BatchPredictResponse(model_used=model_name, predictions=results)
