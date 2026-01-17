import os
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

MODEL_PATH = "models/lightgbm_stock.txt"
TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
OUT_DIR = "reports"

FEATURES = [
    "low","high","volume","close",
    "return_1d","return_3d","return_5d",
    "ma_5","ma_10","ma_ratio_5",
    "volatility_5","vol_chg","vol_ma_5","hl_range",
    "symbol_enc"
]

THRESHOLD = 0.5

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def fit_symbol_encoder(train_df: pd.DataFrame) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(train_df["symbol"].astype(str))
    return le

def add_symbol_enc(df: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    out = df.copy()
    out["symbol_enc"] = le.transform(out["symbol"].astype(str))
    return out

def evaluate_split(model: lgb.Booster, df: pd.DataFrame, split_name: str):
    X = df[FEATURES]
    y = df["target"].astype(int)

    prob = model.predict(X)
    pred = (prob >= THRESHOLD).astype(int)

    auc = roc_auc_score(y, prob)
    acc = accuracy_score(y, pred)

    # For binary class metrics, report per class
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )

    # Return as dict
    return {
        "split": split_name,
        "samples": int(len(df)),
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "precision_0": float(prec[0]),
        "recall_0": float(rec[0]),
        "f1_0": float(f1[0]),
        "support_0": int(sup[0]),
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
        "support_1": int(sup[1]),
    }, y, prob, pred

def save_confusion_matrix(y_true, y_pred, path_png: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down(0)", "Up(1)"])

    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def save_roc_curve(y_true, y_prob, path_png: str, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=200)
    plt.close()

def df_to_markdown_table(df: pd.DataFrame) -> str:
    # simple markdown table output without extra deps
    return df.to_markdown(index=False)

def main():
    ensure_out_dir()

    # Load model
    model = lgb.Booster(model_file=MODEL_PATH)

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Encode symbol consistently with training
    le = fit_symbol_encoder(train_df)
    val_df = add_symbol_enc(val_df, le)
    test_df = add_symbol_enc(test_df, le)

    # Evaluate val + test
    val_metrics, y_val, prob_val, pred_val = evaluate_split(model, val_df, "validation")
    test_metrics, y_test, prob_test, pred_test = evaluate_split(model, test_df, "test")

    metrics_df = pd.DataFrame([val_metrics, test_metrics])

    # Round for nicer display
    display_df = metrics_df.copy()
    for col in ["roc_auc", "accuracy", "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1"]:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 4))

    # Save tables
    csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
    md_path = os.path.join(OUT_DIR, "metrics_summary.md")

    display_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df_to_markdown_table(display_df))
        f.write("\n")

    # Save confusion matrix + ROC curve for TEST
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_test.png")
    roc_path = os.path.join(OUT_DIR, "roc_curve_test.png")

    save_confusion_matrix(
        y_true=y_test,
        y_pred=pred_test,
        path_png=cm_path,
        title=f"Figure 2: Confusion Matrix (Test) @ threshold={THRESHOLD}"
    )

    save_roc_curve(
        y_true=y_test,
        y_prob=prob_test,
        path_png=roc_path,
        title="Figure 1: ROC Curve (Test)"
    )

    print("Saved evaluation artifacts:")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print(f"- {cm_path}")
    print(f"- {roc_path}")

if __name__ == "__main__":
    main()
