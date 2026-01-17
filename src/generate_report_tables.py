import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "reports"

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def save_table_image(df: pd.DataFrame, col_widths: list[float], title: str, filename: str):
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.4, len(df) * 0.6 + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    plt.title(title, pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_out_dir()

    # ---------- Dataset split summary ----------
    split_data = pd.DataFrame([
        ["Train", 2086, "2025-01-27 : 2025-09-24"],
        ["Validation", 473, "2025-09-25 : 2025-11-16"],
        ["Test", 470, "2025-11-17 : 2026-01-07"],
    ], columns=["Split", "Samples", "Date Range"])

    save_table_image(
        split_data,
        [0.2, 0.2, 0.6],
        "Table 1: Dataset Split Summary",
        "dataset_split_summary.png"
    )

    # ---------- Metrics summary ----------
    metrics_df = pd.read_csv("reports/metrics_summary.csv")

    display_cols = [
        "split", "samples", "roc_auc", "accuracy",
        "precision_1", "recall_1", "f1_1"
    ]

    metrics_df = metrics_df[display_cols].copy()
    metrics_df.columns = [
        "Split", "Samples", "ROC-AUC", "Accuracy",
        "Precision (Up)", "Recall (Up)", "F1 (Up)"
    ]

    save_table_image(
        metrics_df,
        [0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
        "Table 2: Model Performance Summary",
        "metrics_summary.png"
    )

    print("Saved report tables:")
    print("- reports/dataset_split_summary.png")
    print("- reports/metrics_summary.png")


if __name__ == "__main__":
    main()
