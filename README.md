# CSE Stock Movement Prediction (LightGBM + FastAPI + Next.js)

A simple ML project that predicts **next-day stock movement** for Colombo Stock Exchange (CSE) using **LightGBM**. Includes a FastAPI backend and a minimal Next.js frontend.

## Backend (FastAPI)

### Setup (Python)
Create and activate venv, then install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install fastapi uvicorn lightgbm shap scikit-learn pandas numpy httpx matplotlib
```

### Run the API

From repo root:
```bash
python -m uvicorn src.api:app --reload --port 8000
```

### Swagger docs:
```bash
http://127.0.0.1:8000/docs
```

### Backend Endpoints
```bash
GET /health → health check

GET /symbols → supported symbols

GET /model/info → model metadata + feature definitions

GET /model/metrics → cached evaluation metrics (ROC-AUC, confusion matrix, etc.)

GET /history/{symbol}?n=15&period=5 → fetch recent OHLCV from CSE

POST /predict → predict UP probability + class

POST /explain → SHAP-based top feature contributions for a prediction
```

Note: /predict and /explain require at least 10 rows of history (due to MA_10 rolling features).

## Frontend (Next.js)

See frontend/README.md.

## Training Pipeline (one-time)

If you want to rebuild the dataset and retrain the model (optional):

Update the `STOCKS` variable in `./cse_stock_data.py` as per your requirement and then run below to fetch raw CSVs from official CSE API:

```bash
python cse_stock_data.py
```

Merge raw CSVs:
```bash
python src/make_dataset.py
```

Create labels (next_close, target):
```bash
python src/label_dataset.py
```

Create engineered features (returns, moving averages, volatility, etc.):
```bash
python src/make_features.py
```

Time-based split:
```bash
python src/split_data.py
```

Train LightGBM:
```bash
python src/train_lightgbm.py
```

Model is saved to: models/lightgbm_stock.txt

### Notes

* This is a financial prediction problem. Performance is evaluated with ROC-AUC (more meaningful than accuracy for imbalanced classes).

* The system is designed so the frontend fetches recent OHLCV (/history/{symbol}) and then calls /predict and /explain.
