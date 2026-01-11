import pandas as pd
import json

df = pd.read_csv("data/dataset_labeled.csv")
df = df[df["symbol"] == "WATA"].sort_values("date").tail(15)

payload = {
    "symbol": "WATA",
    "history": df[["date","low","high","close","volume"]].to_dict(orient="records")
}

print(json.dumps(payload, indent=2, default=str))
