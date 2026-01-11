from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
import lightgbm as lgb

from src.features import FeatureBuilder, FEATURE_COLUMNS


@dataclass
class StockMovementPredictor:
    model: lgb.Booster
    feature_builder: FeatureBuilder
    threshold: float = 0.5

    @staticmethod
    def load(
        model_path: str = "models/lightgbm_stock.txt",
        train_csv_path_for_encoder: str = "data/train.csv",
        threshold: float = 0.5,
    ) -> "StockMovementPredictor":
        model = lgb.Booster(model_file=model_path)
        le = FeatureBuilder.fit_symbol_encoder_from_training_csv(train_csv_path_for_encoder)
        fb = FeatureBuilder(label_encoder=le)
        return StockMovementPredictor(model=model, feature_builder=fb, threshold=threshold)

    def predict_from_history(self, history: pd.DataFrame, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        history: last N rows (>=10) containing low/high/close/volume (+ date recommended)
        symbol: stock symbol (optional if history contains 'symbol')
        """
        X = self.feature_builder.build_features_from_history(history=history, symbol=symbol)

        # Ensure column order matches training
        X = X[FEATURE_COLUMNS]

        prob_up = float(self.model.predict(X)[0])
        pred = 1 if prob_up >= self.threshold else 0

        return {
            "prob_up": prob_up,
            "prediction": pred,
            "threshold": self.threshold,
            "features_used": X.to_dict(orient="records")[0],
        }
