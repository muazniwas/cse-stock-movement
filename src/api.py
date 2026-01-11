from __future__ import annotations

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.predictor import StockMovementPredictor

app = FastAPI(title="CSE Stock Movement API", version="1.0")

# Load once at startup
predictor = StockMovementPredictor.load(
    model_path="models/lightgbm_stock.txt",
    train_csv_path_for_encoder="data/train.csv",
    threshold=0.5
)

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
