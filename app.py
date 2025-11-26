# app.py
import os
import logging
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- CONFIG ----------
DEFAULT_THRESHOLD = 0.5
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-api")
# ----------------------------

# When model.pkl is next to app.py, the simplest deployment-friendly path is just "model.pkl"
MODEL_PATH = "model.pkl"

# Globals populated on startup
artifact: Optional[dict] = None
pipeline = None
FEATURES: List[str] = []
FEATURE_MEDIANS: Dict[str, float] = {}
MODEL_NAME = "model"

def try_load_model() -> bool:
    """
    Attempt to load model.pkl from MODEL_PATH.
    Populate artifact, pipeline, FEATURES, FEATURE_MEDIANS.
    Return True on success, False otherwise (no exceptions propagated).
    """
    global artifact, pipeline, FEATURES, FEATURE_MEDIANS, MODEL_NAME
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        return False

    try:
        logger.info("Loading model artifact from %s", MODEL_PATH)
        artifact = joblib.load(MODEL_PATH)

        pipeline = artifact.get("pipeline") or artifact.get("model")
        if pipeline is None:
            logger.error("Loaded artifact does not contain 'pipeline' or 'model'.")
            artifact = None
            pipeline = None
            return False

        FEATURES = artifact.get("features") or []
        FEATURE_MEDIANS = artifact.get("feature_medians", {}) or {}
        MODEL_NAME = artifact.get("model_name", MODEL_NAME)

        logger.info("Model loaded successfully. model_name=%s features_count=%d", MODEL_NAME, len(FEATURES))
        return True
    except Exception as e:
        logger.exception("Failed to load model from %s: %s", MODEL_PATH, e)
        artifact = None
        pipeline = None
        FEATURES = []
        FEATURE_MEDIANS = {}
        return False

# ---------- FastAPI app ----------
app = FastAPI(title="Credit Risk Scoring - Minimal", version="1.0")

@app.on_event("startup")
def startup_event():
    loaded = try_load_model()
    if not loaded:
        logger.error("Model not loaded on startup. /score will return 503 until model.pkl is placed next to app.py")

@app.get("/", summary="Health / model status")
def root():
    return {"status": "ok", "model_loaded": pipeline is not None, "model_name": MODEL_NAME, "features_loaded": len(FEATURES)}

# ---------- Request / Response models ----------
class ScoreRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description='Mapping feature name -> value, e.g. {"age":45, "DebtRatio":0.3, ...}')

class ScoreResponse(BaseModel):
    probability_default: float
    prediction: int
    model: str

# ---------- Input preparation ----------
def _prepare_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the exact FEATURES order and impute missing values
    using FEATURE_MEDIANS. Coerce values to numeric where possible.
    """
    if not FEATURES:
        # If model didn't provide features metadata, try to use keys from payload (best-effort)
        df = pd.DataFrame([payload])
        df = df.astype(float, errors="ignore")
        return df

    # Start with DataFrame from payload
    df = pd.DataFrame([payload])

    # Ensure all expected feature columns exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = pd.NA

    # Reorder to FEATURES
    df = df[FEATURES]

    # Normalize missing markers and coerce to numeric where possible
    df = df.replace({pd.NA: np.nan, None: np.nan, "": np.nan})
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing values using FEATURE_MEDIANS (fallback to 0.0 if not present)
    impute_map = {col: float(FEATURE_MEDIANS.get(col, 0.0)) for col in df.columns}
    df = df.fillna(value=impute_map)

    # Final cast to float for model consumption
    try:
        df = df.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert input features to float: {e}")

    return df

# ---------- Scoring endpoint ----------
@app.post("/score", response_model=ScoreResponse, summary="Score a single record")
def score(req: ScoreRequest):
    # Ensure model is loaded
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place model.pkl next to app.py and redeploy.")

    payload = req.data
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="'data' must be a JSON object mapping feature names to values.")

    # Prepare input
    try:
        df = _prepare_input_df(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error preparing input")
        raise HTTPException(status_code=500, detail=f"Input preparation error: {e}")

    # Make prediction
    try:
        # Prefer predict_proba when available
        if hasattr(pipeline, "predict_proba"):
            proba_arr = pipeline.predict_proba(df)
            # handle 1-d and 2-d outputs
            if getattr(proba_arr, "ndim", 2) == 1:
                proba = float(proba_arr[0])
            else:
                # assume class 1 is positive / default
                proba = float(proba_arr[:, 1][0])
            prediction = int(proba >= DEFAULT_THRESHOLD)
        else:
            # Fallback to predict (some models might output probability-like numbers)
            pred = pipeline.predict(df)[0]
            # If predict returns probability-like float in [0,1], use it; otherwise treat as class label 0/1
            try:
                proba = float(pred)
                prediction = int(proba >= DEFAULT_THRESHOLD)
            except Exception:
                proba = float(pred)
                prediction = int(pred)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return ScoreResponse(
        probability_default=proba,
        prediction=prediction,
        model=MODEL_NAME
    )

# ---------- CLI run for local dev (uvicorn) ----------
if __name__ == "__main__":
    import uvicorn
    # Use PORT env var if set (helps local testing with Azure-like behavior)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
