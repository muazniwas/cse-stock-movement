from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS: List[str] = [
    "low", "high", "volume", "close",
    "return_1d", "return_3d", "return_5d",
    "ma_5", "ma_10", "ma_ratio_5",
    "volatility_5", "vol_chg", "vol_ma_5", "hl_range",
    "symbol_enc",
]


@dataclass
class FeatureBuilder:
    """
    Builds the exact same features used during training for a single symbol.

    Notes:
    - Requires at least 10 historical rows to compute MA_10 and other rolling features.
    - Assumes daily OHLCV (no intraday).
    - 'date' is optional for feature computation, but recommended for sorting.
    """
    label_encoder: LabelEncoder

    @staticmethod
    def fit_symbol_encoder_from_training_csv(train_csv_path: str) -> LabelEncoder:
        """
        Fit a LabelEncoder using the same symbols present during training.
        Call this once (e.g., at API startup), then reuse.
        """
        df = pd.read_csv(train_csv_path)
        le = LabelEncoder()
        le.fit(df["symbol"].astype(str).values)
        return le

    def build_features_from_history(
        self,
        history: pd.DataFrame,
        symbol: Optional[str] = None,
        require_min_rows: int = 10,
    ) -> pd.DataFrame:
        """
        history: DataFrame containing at least columns:
          - low, high, close, volume
          - date (recommended) OR already sorted oldest->newest

        symbol: if not provided, tries to use history['symbol'].iloc[-1]
        Returns: DataFrame with 1 row (latest day) with FEATURE_COLUMNS.
        """
        required = {"low", "high", "close", "volume"}
        missing = required - set(map(str.lower, history.columns))
        # Normalize columns to lowercase for safety
        df = history.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"History is missing required columns: {sorted(missing)}")

        # Determine symbol
        if symbol is None:
            if "symbol" in df.columns and len(df["symbol"]) > 0:
                symbol = str(df["symbol"].iloc[-1])
            else:
                raise ValueError("symbol not provided and history has no 'symbol' column")

        # Sort by date if present
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        else:
            # assume already sorted
            df = df.copy()

        # Ensure numeric
        for col in ["low", "high", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["low", "high", "close", "volume"]).reset_index(drop=True)

        if len(df) < require_min_rows:
            raise ValueError(
                f"Need at least {require_min_rows} valid rows to build features, got {len(df)}"
            )

        # --- Feature engineering ---
        df["return_1d"] = df["close"].pct_change(1)
        df["return_3d"] = df["close"].pct_change(3)
        df["return_5d"] = df["close"].pct_change(5)

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_ratio_5"] = df["close"] / df["ma_5"]

        df["volatility_5"] = df["return_1d"].rolling(5).std()

        df["vol_chg"] = df["volume"].pct_change(1)
        df["vol_ma_5"] = df["volume"].rolling(5).mean()

        df["hl_range"] = (df["high"] - df["low"]) / df["close"]

        # Symbol encoding (must match training encoder)
        try:
            sym_enc = int(self.label_encoder.transform([symbol])[0])
        except ValueError as e:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Encoder was fit on: {list(self.label_encoder.classes_)}"
            ) from e

        df["symbol_enc"] = sym_enc

        # Take the latest row with all features present (drop NaNs created by rolling windows)
        feat_df = df.dropna(subset=[
            "return_1d", "return_3d", "return_5d",
            "ma_5", "ma_10", "ma_ratio_5",
            "volatility_5", "vol_chg", "vol_ma_5", "hl_range"
        ])

        if feat_df.empty:
            raise ValueError("After computing rolling features, no complete rows available.")

        latest = feat_df.iloc[[-1]].copy()

        # Return only model features in correct order
        return latest[FEATURE_COLUMNS]
