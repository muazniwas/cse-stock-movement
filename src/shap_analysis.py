import shap
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_PATH = "models/lightgbm_stock.txt"
TEST_PATH = "data/test.csv"
TRAIN_PATH = "data/train.csv"

# Load model and data
model = lgb.Booster(model_file=MODEL_PATH)
df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)

# ---------- Encode symbol exactly like training ----------
le = LabelEncoder()
le.fit(train_df["symbol"].astype(str))
df["symbol_enc"] = le.transform(df["symbol"].astype(str))

FEATURES = [
    "low","high","volume","close",
    "return_1d","return_3d","return_5d",
    "ma_5","ma_10","ma_ratio_5",
    "volatility_5","vol_chg","vol_ma_5","hl_range",
    "symbol_enc"
]

X = df[FEATURES]

# ---------- SHAP explainer ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP values computed. Shape:", shap_values.shape)

# ---------- Global summary ----------
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.gca().set_title("Figure 4: SHAP summary (beeswarm) plot", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Bar importance ----------
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.gca().set_title("Figure 3: Global SHAP feature importance bar chart", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("reports/shap_importance_bar.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Local explanation ----------
idx = 0
plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx],
        feature_names=X.columns
    ),
    show=False
)

plt.title("Figure 5: SHAP waterfall plot for one prediction", fontsize=11, pad=12)
plt.tight_layout()
plt.savefig("reports/shap_waterfall_example.png", dpi=200, bbox_inches="tight")
plt.close()

print("SHAP plots saved to reports/:")
print(" - shap_summary.png")
print(" - shap_importance_bar.png")
print(" - shap_waterfall_example.png")
