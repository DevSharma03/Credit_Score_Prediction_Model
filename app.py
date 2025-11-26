# src/app.py
import os
import logging
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------- CONFIG -------------
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
# Resolve project root (src is inside the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-api")

# ------------- LOAD ARTIFACT -------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run training to produce model/model.pkl")

artifact = joblib.load(MODEL_PATH)

pipeline = artifact.get("pipeline") or artifact.get("model")
if pipeline is None:
    raise RuntimeError("Loaded artifact does not contain 'pipeline' or 'model'.")

FEATURES: List[str] = artifact.get("features")
if not FEATURES:
    raise RuntimeError("Loaded artifact must include a 'features' list (order matters).")

FEATURE_MEDIANS = artifact.get("feature_medians", None) or {f: 0.0 for f in FEATURES}
# ---------------------------------------

app = FastAPI(title="Credit Risk Scoring - Minimal", version="1.0")

class ScoreRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Mapping of feature name -> value. Example: {\"age\": 45, \"DebtRatio\": 0.3, ...}")

class ScoreResponse(BaseModel):
    probability_default: float
    prediction: int
    model: str

def _prepare_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the exact FEATURES order and impute missing values
    using FEATURE_MEDIANS. Coerce values to numeric where possible.
    """
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

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
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
        proba_arr = pipeline.predict_proba(df)
        # prefer probability of positive class in column 1, but handle edge cases
        if getattr(proba_arr, "ndim", 2) == 1:
            proba = float(proba_arr[0])
        else:
            proba = float(proba_arr[:, 1][0])
        prediction = int(proba >= DEFAULT_THRESHOLD)
    except AttributeError:
        # pipeline has no predict_proba; fallback to predict
        try:
            pred = pipeline.predict(df)[0]
            proba = float(pred)
            prediction = int(pred)
        except Exception as e:
            logger.exception("Prediction fallback failed")
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return ScoreResponse(
        probability_default=proba,
        prediction=prediction,
        model=artifact.get("model_name", "model")
    )


