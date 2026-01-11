import pandas as pd

IN_PATH = "data/dataset.csv"
OUT_PATH = "data/dataset_labeled.csv"

df = pd.read_csv(IN_PATH)

# Ensure correct types
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Next day's close per symbol
df["next_close"] = df.groupby("symbol")["close"].shift(-1)

# Binary target: 1 if next_close > close else 0
df["target"] = (df["next_close"] > df["close"]).astype(int)

# Drop rows where next_close is missing (last available day per symbol)
before = len(df)
df = df.dropna(subset=["next_close"]).reset_index(drop=True)
after = len(df)

df.to_csv(OUT_PATH, index=False)

print(f"Saved labeled dataset: {OUT_PATH}")
print("Rows before:", before)
print("Rows after :", after)
print("Dropped (last-day rows per symbol):", before - after)
print("\nClass balance (target):")
print(df["target"].value_counts(normalize=True).rename("proportion"))
print("\nSample:")
print(df.head(10))
