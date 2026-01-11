import pandas as pd

from src.features import FeatureBuilder

# Load training split to fit encoder (same symbols as training)
le = FeatureBuilder.fit_symbol_encoder_from_training_csv("data/train.csv")
fb = FeatureBuilder(label_encoder=le)

# Example: take last 15 days of ABL from your feature dataset source (raw-ish OHLCV)
df = pd.read_csv("data/dataset_labeled.csv")
df = df[df["symbol"] == "ABL"].sort_values("date").tail(15)

X = fb.build_features_from_history(df, symbol="ABL")
print(X)
print("\nColumns:", list(X.columns))
