import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Paths
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"

# Load
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

# Encode symbol
le = LabelEncoder()
train_df["symbol_enc"] = le.fit_transform(train_df["symbol"])
val_df["symbol_enc"] = le.transform(val_df["symbol"])
test_df["symbol_enc"] = le.transform(test_df["symbol"])

# Feature columns
FEATURES = [
    "low","high","volume","close",
    "return_1d","return_3d","return_5d",
    "ma_5","ma_10","ma_ratio_5",
    "volatility_5","vol_chg","vol_ma_5","hl_range",
    "symbol_enc"
]

TARGET = "target"

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val, y_val = val_df[FEATURES], val_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# Handle imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("scale_pos_weight:", round(scale_pos_weight, 3))

# LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
    "verbosity": -1
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

# ---- Evaluation ----
def evaluate(name, X, y):
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)

    print(f"\n{name} results")
    print("ROC-AUC:", round(roc_auc_score(y, probs), 4))
    print("Confusion matrix:\n", confusion_matrix(y, preds))
    print(classification_report(y, preds, digits=4))

evaluate("Validation", X_val, y_val)
evaluate("Test", X_test, y_test)

# Save model
model.save_model("models/lightgbm_stock.txt")
print("\nModel saved to models/lightgbm_stock.txt")
