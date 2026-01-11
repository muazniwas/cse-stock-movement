import glob
import os
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/dataset.csv"

required_cols = {"date", "high", "low", "close", "volume"}

all_dfs = []
csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

if not csv_paths:
    raise FileNotFoundError(f"No CSV files found in: {RAW_DIR}")

total_dropped = 0

for path in csv_paths:
    symbol = os.path.splitext(os.path.basename(path))[0].strip()
    df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df[list(required_cols)].copy()
    df["symbol"] = symbol

    # Parse date and remove time
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    before = len(df)
    df = df.dropna(subset=["date"])

    # Convert to numeric
    for col in ["high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_mask = df[["high", "low", "close", "volume"]].isna().any(axis=1)
    dropped = bad_mask.sum()

    if dropped > 0:
        print(f"{symbol}: dropping {dropped} invalid rows")

    df = df[~bad_mask]

    if df.empty:
        raise ValueError(f"{path} has no valid rows after cleaning.")

    total_dropped += dropped
    all_dfs.append(df)

dataset = pd.concat(all_dfs, ignore_index=True)
dataset = dataset.sort_values(["symbol", "date"]).reset_index(drop=True)
dataset = dataset.drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)

dataset.to_csv(OUT_PATH, index=False)

print("\nDataset created:", OUT_PATH)
print("Total rows:", len(dataset))
print("Symbols:", dataset["symbol"].nunique())
print("Total dropped rows:", total_dropped)
print("Date range:", dataset["date"].min(), "→", dataset["date"].max())
