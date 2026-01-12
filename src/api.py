from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import numpy as np
import shap

from src.predictor import StockMovementPredictor

def load_model_metrics(
    model,
    test_csv_path: str,
    train_csv_path: str,
):
    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)

    # Encode symbols exactly like training
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(train_df["symbol"].astype(str))
    test_df["symbol_enc"] = le.transform(test_df["symbol"].astype(str))

    # Features must match training
    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    # Predict
    probs = model.predict(X_test)
    preds = (probs >= predictor.threshold).astype(int)

    # Metrics
    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, preds))
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    # Class distribution
    class_dist = y_test.value_counts(normalize=True).round(4).to_dict()

    return {
        "test_samples": int(len(test_df)),
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": {
            "labels": ["down(0)", "up(1)"],
            "matrix": cm
        },
        "classification_report": report,
        "class_distribution": class_dist,
    }

# ---------- Load feature metadata from JSON ----------
FEATURE_METADATA_PATH = Path("data/feature_metadata.json")

with open(FEATURE_METADATA_PATH, "r") as f:
    FEATURE_METADATA = json.load(f)

FEATURES = [f["name"] for f in FEATURE_METADATA]

MIN_HISTORY_ROWS = 10
MODEL_TYPE = "LightGBM (GBDT binary classifier)"
MODEL_FILE = "models/lightgbm_stock.txt"

app = FastAPI(title="CSE Stock Movement API", version="1.0")

# Load once at startup
predictor = StockMovementPredictor.load(
    model_path="models/lightgbm_stock.txt",
    train_csv_path_for_encoder="data/train.csv",
    threshold=0.5
)

# Load available symbols (once)
TRAIN_CSV_PATH = "data/train.csv"
_symbols_df = pd.read_csv(TRAIN_CSV_PATH)
AVAILABLE_SYMBOLS = sorted(_symbols_df["symbol"].astype(str).unique().tolist())

MODEL_METRICS = load_model_metrics(
    model=predictor.model,
    test_csv_path="data/test.csv",
    train_csv_path="data/train.csv"
)

# ---------- SHAP explainer (loaded once) ----------
SHAP_EXPLAINER = shap.TreeExplainer(predictor.model)

# ---------- Request / Response Schemas ----------

class OhlcvRow(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD recommended")
    low: float
    high: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol, e.g., ABL")
    history: List[OhlcvRow] = Field(..., description="Last N trading days (>= 10). Oldest -> newest preferred.")

class PredictResponse(BaseModel):
    symbol: str
    prob_up: float
    prediction: int
    threshold: float

# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.history) < 10:
        raise HTTPException(status_code=400, detail="history must contain at least 10 rows")

    # Convert to DataFrame
    df = pd.DataFrame([r.model_dump() for r in req.history])
    df["symbol"] = req.symbol

    try:
        result = predictor.predict_from_history(df, symbol=req.symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        symbol=req.symbol,
        prob_up=result["prob_up"],
        prediction=result["prediction"],
        threshold=result["threshold"]
    )

@app.post("/explain")
def explain(req: PredictRequest):
    if len(req.history) < MIN_HISTORY_ROWS:
        raise HTTPException(status_code=400, detail=f"history must contain at least {MIN_HISTORY_ROWS} rows")

    df = pd.DataFrame([r.model_dump() for r in req.history])
    df["symbol"] = req.symbol

    try:
        # Build features
        X = predictor.feature_builder.build_features_from_history(df, symbol=req.symbol)
        X = X[FEATURES]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prediction
    prob_up = float(predictor.model.predict(X)[0])
    pred = 1 if prob_up >= predictor.threshold else 0

    # SHAP values
    shap_values = SHAP_EXPLAINER.shap_values(X)[0]   # (n_features,)
    base_value = float(SHAP_EXPLAINER.expected_value)

    # Build contribution list
    contribs = []
    for i, col in enumerate(X.columns):
        contribs.append({
            "feature": col,
            "value": float(X.iloc[0][col]),
            "shap_value": float(shap_values[i])
        })

    # Sort by absolute impact
    contribs = sorted(contribs, key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "symbol": req.symbol,
        "prob_up": prob_up,
        "prediction": pred,
        "threshold": predictor.threshold,
        "base_value": base_value,
        "top_contributions": contribs[:10],  # top 10 features
        "all_contributions": contribs        # full list (optional for frontend)
    }

@app.get("/symbols")
def get_symbols():
    return {
        "count": len(AVAILABLE_SYMBOLS),
        "symbols": AVAILABLE_SYMBOLS
    }

@app.get("/model/info")
def model_info():
    return {
        "model_type": MODEL_TYPE,
        "model_file": MODEL_FILE,
        "threshold": predictor.threshold,
        "min_history_rows": MIN_HISTORY_ROWS,
        "feature_count": len(FEATURES),
        "features": FEATURE_METADATA,
        "symbols_count": len(AVAILABLE_SYMBOLS),
        "symbols": AVAILABLE_SYMBOLS,
    }

@app.get("/model/metrics")
def model_metrics():
    return MODEL_METRICS
