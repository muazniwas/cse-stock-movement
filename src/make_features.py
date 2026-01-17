import pandas as pd
import numpy as np

IN_PATH = "data/dataset_labeled.csv"
OUT_PATH = "data/dataset_features.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

def add_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()

    # --- Returns --- Momentum - to capture short-term price direction
    g["return_1d"] = g["close"].pct_change(1)
    g["return_3d"] = g["close"].pct_change(3)
    g["return_5d"] = g["close"].pct_change(5)

    # --- Moving averages --- Trend - to assess whether a stock was trading above or below its short-term trend
    g["ma_5"] = g["close"].rolling(5).mean()
    g["ma_10"] = g["close"].rolling(10).mean()
    g["ma_ratio_5"] = g["close"] / g["ma_5"]

    # --- Volatility --- Represent recent market instability
    g["volatility_5"] = g["return_1d"].rolling(5).std()

    # --- Volume --- To capture shifts in liquidity and trading interest
    g["vol_chg"] = g["volume"].pct_change(1)
    g["vol_ma_5"] = g["volume"].rolling(5).mean()

    # --- Daily range --- To reflect intraday dispersion and trading pressure
    g["hl_range"] = (g["high"] - g["low"]) / g["close"]

    return g

df = df.groupby("symbol", group_keys=False).apply(add_features)

# Drop early rows with NaNs from rolling windows
before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)

df.to_csv(OUT_PATH, index=False)

print("Feature dataset saved:", OUT_PATH)
print("Rows before feature drop:", before)
print("Rows after feature drop :", after)
print("Dropped due to rolling windows:", before - after)
print("\nColumns:")
print(df.columns.tolist())
print("\nSample:")
print(df.head())
