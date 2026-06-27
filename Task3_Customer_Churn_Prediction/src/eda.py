import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .config import OUTPUTS_DIR
    from .data_utils import detect_target_column, find_csv, load_dataset, normalize_target
except ImportError:
    from config import OUTPUTS_DIR
    from data_utils import detect_target_column, find_csv, load_dataset, normalize_target


def save_eda_plots() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = find_csv()
    df = load_dataset(csv_path).drop_duplicates()
    target = detect_target_column(df)
    df[target] = normalize_target(df[target])

    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=target)
    ax.set_title("Customer Churn Class Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customers")
    plt.tight_layout()
    class_plot = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(class_plot, dpi=160)
    plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.index, y=missing.values, color="#5B8DEF")
    plt.title("Missing Values by Feature")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing values")
    plt.tight_layout()
    missing_plot = OUTPUTS_DIR / "missing_values.png"
    plt.savefig(missing_plot, dpi=160)
    plt.close()

    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(11, 8))
        sns.heatmap(numeric_df.corr(), cmap="viridis", annot=False, linewidths=0.4)
        plt.title("Numerical Feature Correlation Heatmap")
        plt.tight_layout()
        corr_plot = OUTPUTS_DIR / "correlation_heatmap.png"
        plt.savefig(corr_plot, dpi=160)
        plt.close()
    else:
        corr_plot = None

    feature_columns = [column for column in numeric_df.columns if column != target][:8]
    if feature_columns:
        axes = df[feature_columns].hist(figsize=(13, 9), bins=30, color="#4E79A7", edgecolor="white")
        for axis in axes.flatten():
            axis.set_title(axis.get_title(), fontsize=10)
        plt.suptitle("Feature Distributions", y=1.02)
        plt.tight_layout()
        dist_plot = OUTPUTS_DIR / "feature_distributions.png"
        plt.savefig(dist_plot, dpi=160)
        plt.close()
    else:
        dist_plot = None

    summary = {
        "dataset": str(csv_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": target,
        "plots": {
            "class_distribution": str(class_plot),
            "missing_values": str(missing_plot),
            "correlation_heatmap": str(corr_plot) if corr_plot else None,
            "feature_distributions": str(dist_plot) if dist_plot else None,
        },
    }
    return summary


if __name__ == "__main__":
    print(save_eda_plots())
