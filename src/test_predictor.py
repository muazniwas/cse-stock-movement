import pandas as pd
from src.predictor import StockMovementPredictor

predictor = StockMovementPredictor.load(
    model_path="models/lightgbm_stock.txt",
    train_csv_path_for_encoder="data/train.csv",
    threshold=0.5
)

df = pd.read_csv("data/dataset_labeled.csv")
df = df[df["symbol"] == "ABL"].sort_values("date").tail(15)

result = predictor.predict_from_history(df, symbol="ABL")
print(result)
