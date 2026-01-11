import pandas as pd

IN_PATH = "data/dataset_features.csv"

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])

# Sort globally by time (important)
df = df.sort_values("date").reset_index(drop=True)

# Get unique sorted dates
dates = df["date"].sort_values().unique()

n_dates = len(dates)
train_end = int(n_dates * 0.70)
val_end = int(n_dates * 0.85)

train_dates = dates[:train_end]
val_dates = dates[train_end:val_end]
test_dates = dates[val_end:]

train_df = df[df["date"].isin(train_dates)]
val_df   = df[df["date"].isin(val_dates)]
test_df  = df[df["date"].isin(test_dates)]

print("Date ranges:")
print("Train:", train_dates[0], "→", train_dates[-1])
print("Val  :", val_dates[0], "→", val_dates[-1])
print("Test :", test_dates[0], "→", test_dates[-1])

print("\nSizes:")
print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

print("\nTarget balance:")
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(name, d["target"].value_counts(normalize=True).round(3).to_dict())

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\nSaved data/train.csv, data/val.csv, data/test.csv")
