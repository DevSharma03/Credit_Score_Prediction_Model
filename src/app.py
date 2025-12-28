# src/app.py
import os
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ======================================================
# CONFIG
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/model.pkl")

LOW_RISK_THRESHOLD = 0.25
HIGH_RISK_THRESHOLD = 0.75

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-risk-api")

# ======================================================
# LOAD MODEL
# ======================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model artifact not found. Train the model first.")

artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
FEATURES = artifact["features"]
MODEL_NAME = artifact.get("model_name", "LightGBM-Credit-Risk")

# ======================================================
# FEATURE NAME MAP (CRITICAL FIX)
# ======================================================
FEATURE_NAME_MAP = {
    "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
}

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(
    title="Credit Default Risk API",
    version="2.1",
    description="Minimal-input credit risk scoring API"
)

# ======================================================
# INPUT SCHEMA (MINIMAL)
# ======================================================
class CreditApplication(BaseModel):
    age: int
    MonthlyIncome: float
    DebtRatio: float
    RevolvingUtilizationOfUnsecuredLines: float
    NumberOfOpenCreditLinesAndLoans: int

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        0, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        0, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfTimes90DaysLate: int = 0

    NumberRealEstateLoansOrLines: int = 0
    NumberOfDependents: int | None = None

    class Config:
        populate_by_name = True


# ======================================================
# RESPONSE SCHEMA
# ======================================================
class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ScoreResponse(BaseModel):
    probability_default: float
    decision: int
    decision_label: str
    low_risk_threshold: float
    high_risk_threshold: float
    model_features: int
    top_features: List[FeatureImportance]


# ======================================================
# FEATURE ENGINEERING (MATCHES TRAINING)
# ======================================================
def build_features(app: CreditApplication) -> pd.DataFrame:
    df = pd.DataFrame([{
        "age": app.age,
        "MonthlyIncome": app.MonthlyIncome,
        "DebtRatio": app.DebtRatio,
        "RevolvingUtilizationOfUnsecuredLines": app.RevolvingUtilizationOfUnsecuredLines,
        "NumberOfOpenCreditLinesAndLoans": app.NumberOfOpenCreditLinesAndLoans,
        "NumberRealEstateLoansOrLines": app.NumberRealEstateLoansOrLines,
        "NumberOfDependents": app.NumberOfDependents,
        "NumberOfTime30_59DaysPastDueNotWorse": app.NumberOfTime30_59DaysPastDueNotWorse,
        "NumberOfTime60_89DaysPastDueNotWorse": app.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfTimes90DaysLate": app.NumberOfTimes90DaysLate,
    }])

    # -------- SAFE IMPUTATION --------
    df["NumberOfDependents"] = (
        df["NumberOfDependents"]
        .fillna(0)
        .infer_objects(copy=False)
    )

    # -------- FEATURE ENGINEERING --------
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=10)

    df["MonthlyIncome_clipped"] = df["MonthlyIncome"].clip(lower=0)
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome_clipped"])
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)

    df["num_delinquencies"] = (
        df["NumberOfTime30_59DaysPastDueNotWorse"]
        + df["NumberOfTime60_89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    df["severe_delinquency"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)
    df["high_debt_ratio"] = (df["DebtRatio"] > 1).astype(int)

    df["util_x_open_accounts"] = (
        df["RevolvingUtilizationOfUnsecuredLines"]
        * df["NumberOfOpenCreditLinesAndLoans"]
    )

    df["delinq_per_account"] = df["num_delinquencies"] / (
        df["NumberOfOpenCreditLinesAndLoans"].replace(0, 1)
    )

    df["debt_to_income"] = df["DebtRatio"] / (df["MonthlyIncome_clipped"] + 1)
    df["credit_per_account"] = df["MonthlyIncome_clipped"] / (
        df["NumberOfOpenCreditLinesAndLoans"] + 1
    )

    df["age_risk"] = pd.cut(
        df["age"], bins=[0, 25, 35, 50, 65, 100], labels=False
    )

    # -------- CRITICAL FIX --------
    df = df.rename(columns=FEATURE_NAME_MAP)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df[FEATURES].astype(float)

    return df


# ======================================================
# ENDPOINTS
# ======================================================
@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/score", response_model=ScoreResponse)
def score(application: CreditApplication):
    try:
        X = build_features(application)
        proba = pipeline.predict_proba(X)[:, 1][0]

        if proba >= HIGH_RISK_THRESHOLD:
            decision, label = 1, "REJECT"
        elif proba <= LOW_RISK_THRESHOLD:
            decision, label = 0, "APPROVE"
        else:
            decision, label = -1, "MANUAL_REVIEW"

        model = pipeline.named_steps["model"]
        importances = model.feature_importances_

        top_features = sorted(
            zip(FEATURES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return ScoreResponse(
            probability_default=round(float(proba), 4),
            decision=decision,
            decision_label=label,
            low_risk_threshold=LOW_RISK_THRESHOLD,
            high_risk_threshold=HIGH_RISK_THRESHOLD,
            model_features=len(FEATURES),
            top_features=[
                FeatureImportance(feature=f, importance=float(i))
                for f, i in top_features
            ],
        )

    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(e))