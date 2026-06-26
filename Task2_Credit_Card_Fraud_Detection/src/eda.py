"""Exploratory Data Analysis (EDA) module for credit card fraud detection dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
SUMMARY_FILE = OUTPUT_DIR / "eda_summary.txt"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def _detect_fraud_column(df: pd.DataFrame) -> str:
    """Detect the fraud target column in the dataset.

    Checks for 'is_fraud' (fraudTrain/Test), 'isFraud', and 'Class' in order.

    Args:
        df: Input DataFrame.

    Returns:
        Name of the fraud column if found, otherwise 'is_fraud' as default.
    """
    for col in ["is_fraud", "isFraud", "Class"]:
        if col in df.columns:
            return col
    return "is_fraud"


def load_data(
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and combine train and test datasets.

    Args:
        train_file: Path to the training dataset. Defaults to fraudTrain.csv.
        test_file: Path to the test dataset. Defaults to fraudTest.csv.

    Returns:
        Combined DataFrame with both training and test data.

    Raises:
        FileNotFoundError: If dataset files do not exist.
        ValueError: If datasets cannot be parsed.
    """
    train_path = train_file or DATASET_DIR / "fraudTrain.csv"
    test_path = test_file or DATASET_DIR / "fraudTest.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except pd.errors.ParserError as error:
        raise ValueError("Failed to parse dataset files") from error

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df


def basic_info(df: pd.DataFrame) -> str:
    """Generate basic dataset information.

    Args:
        df: Input DataFrame.

    Returns:
        Formatted string containing dataset overview.
    """
    if df.empty:
        return "Dataset is empty."

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    missing_summary = df.isnull().sum()
    duplicates = df.duplicated().sum()

    fraud_col = _detect_fraud_column(df)
    if fraud_col in df.columns:
        fraud_count = int((df[fraud_col] == 1).sum())
        legit_count = int((df[fraud_col] == 0).sum())
        fraud_pct = (fraud_count / len(df)) * 100
    else:
        fraud_count = legit_count = fraud_pct = 0

    report = [
        "=" * 80,
        "EXPLORATORY DATA ANALYSIS (EDA) REPORT",
        "=" * 80,
        f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Total Features: {len(df.columns)}",
        f"Total Transactions: {len(df):,}",
        f"Numerical Columns: {len(numeric_cols)}",
        f"Categorical Columns: {len(categorical_cols)}",
        "",
        "Missing Values Summary:",
        str(missing_summary[missing_summary > 0]) if (missing_summary > 0).any() else "No missing values",
        "",
        f"Duplicate Rows: {int(duplicates):,}",
        "",
        f"Fraud Transactions: {fraud_count:,}",
        f"Legitimate Transactions: {legit_count:,}",
        f"Fraud Percentage: {fraud_pct:.4f}%",
        "=" * 80,
    ]

    return "\n".join(report)


def generate_plots(df: pd.DataFrame) -> None:
    """Generate and save all visualizations.

    Args:
        df: Input DataFrame for analysis.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fraud_col = _detect_fraud_column(df)

    # Plot 1: Fraud vs Legitimate Count
    plt.figure(figsize=(10, 6))
    fraud_counts = df[fraud_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    fraud_counts.plot(
        kind="bar",
        color=colors,
        title="Fraud vs Legitimate Transactions",
        xlabel="Transaction Type",
        ylabel="Count",
    )
    plt.xticks([0, 1], ["Legitimate", "Fraud"], rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_fraud_vs_legitimate.png", dpi=300)
    plt.close()

    # Plot 2: Fraud Percentage Pie Chart
    plt.figure(figsize=(10, 8))
    labels = ["Legitimate", "Fraud"]
    colors = ["#2ecc71", "#e74c3c"]
    df[fraud_col].value_counts().plot(
        kind="pie",
        labels=labels,
        autopct="%1.2f%%",
        colors=colors,
        title="Fraud Distribution (%)",
        startangle=90,
    )
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_fraud_percentage_pie.png", dpi=300)
    plt.close()

    # Plot 3: Transaction Amount Distribution
    if "amt" in df.columns or "Amount" in df.columns:
        amount_col = "amt" if "amt" in df.columns else "Amount"
        plt.figure(figsize=(12, 6))
        plt.hist(df[amount_col], bins=100, color="#3498db", edgecolor="black", alpha=0.7)
        plt.title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Amount ($)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "03_transaction_amount_dist.png", dpi=300)
        plt.close()

    # Plot 4: Fraud Amount Distribution
    if "amt" in df.columns or "Amount" in df.columns:
        amount_col = "amt" if "amt" in df.columns else "Amount"
        plt.figure(figsize=(12, 6))
        fraud_amounts = df[df[fraud_col] == 1][amount_col]
        plt.hist(fraud_amounts, bins=100, color="#e74c3c", edgecolor="black", alpha=0.7)
        plt.title("Fraudulent Transaction Amount Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Amount ($)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "04_fraud_amount_dist.png", dpi=300)
        plt.close()

    # Plot 5: Top 15 Merchant Categories
    if "category" in df.columns or "Merchant_Category" in df.columns:
        cat_col = "category" if "category" in df.columns else "Merchant_Category"
        plt.figure(figsize=(12, 8))
        top_cats = df[cat_col].value_counts().head(15)
        top_cats.plot(kind="barh", color="#9b59b6", edgecolor="black")
        plt.title("Top 15 Merchant Categories", fontsize=14, fontweight="bold")
        plt.xlabel("Transaction Count", fontsize=12)
        plt.ylabel("Category", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "05_top_merchant_categories.png", dpi=300)
        plt.close()

    # Plot 6: Top Fraud Merchant Categories
    if "category" in df.columns or "Merchant_Category" in df.columns:
        cat_col = "category" if "category" in df.columns else "Merchant_Category"
        plt.figure(figsize=(12, 8))
        fraud_cats = df[df[fraud_col] == 1][cat_col].value_counts().head(15)
        fraud_cats.plot(kind="barh", color="#e67e22", edgecolor="black")
        plt.title("Top Fraud Merchant Categories", fontsize=14, fontweight="bold")
        plt.xlabel("Fraud Count", fontsize=12)
        plt.ylabel("Category", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "06_top_fraud_merchant_categories.png", dpi=300)
        plt.close()

    # Plot 7: Top States with Fraud
    if "state" in df.columns or "State" in df.columns:
        state_col = "state" if "state" in df.columns else "State"
        plt.figure(figsize=(14, 8))
        fraud_states = df[df[fraud_col] == 1][state_col].value_counts().head(15)
        fraud_states.plot(kind="barh", color="#e74c3c", edgecolor="black")
        plt.title("Top States with Fraud", fontsize=14, fontweight="bold")
        plt.xlabel("Fraud Count", fontsize=12)
        plt.ylabel("State", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "07_top_fraud_states.png", dpi=300)
        plt.close()

    # Plot 8: Top Cities with Fraud
    if "city" in df.columns or "City" in df.columns:
        city_col = "city" if "city" in df.columns else "City"
        plt.figure(figsize=(14, 8))
        fraud_cities = df[df[fraud_col] == 1][city_col].value_counts().head(15)
        fraud_cities.plot(kind="barh", color="#c0392b", edgecolor="black")
        plt.title("Top Cities with Fraud", fontsize=14, fontweight="bold")
        plt.xlabel("Fraud Count", fontsize=12)
        plt.ylabel("City", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "08_top_fraud_cities.png", dpi=300)
        plt.close()

    # Plot 9: Fraud by Gender
    if "gender" in df.columns or "Gender" in df.columns:
        gender_col = "gender" if "gender" in df.columns else "Gender"
        plt.figure(figsize=(10, 6))
        fraud_gender = pd.crosstab(df[gender_col], df[fraud_col])
        fraud_gender.plot(
            kind="bar",
            color=["#2ecc71", "#e74c3c"],
            title="Fraud by Gender",
            xlabel="Gender",
            ylabel="Count",
        )
        plt.xticks(rotation=0)
        plt.legend(["Legitimate", "Fraud"])
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "09_fraud_by_gender.png", dpi=300)
        plt.close()

    # Plot 10: Fraud by Hour
    if "trans_date_trans_time" in df.columns:
        df_copy = df.copy()
        df_copy["trans_date_trans_time"] = pd.to_datetime(
            df_copy["trans_date_trans_time"], errors="coerce"
        )
        df_copy["hour"] = df_copy["trans_date_trans_time"].dt.hour
        plt.figure(figsize=(12, 6))
        fraud_by_hour = df_copy[df_copy[fraud_col] == 1]["hour"].value_counts().sort_index()
        fraud_by_hour.plot(kind="bar", color="#e74c3c", edgecolor="black")
        plt.title("Fraud Transactions by Hour of Day", fontsize=14, fontweight="bold")
        plt.xlabel("Hour of Day", fontsize=12)
        plt.ylabel("Fraud Count", fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "10_fraud_by_hour.png", dpi=300)
        plt.close()

    # Plot 11: Fraud by Day of Week
    if "trans_date_trans_time" in df.columns:
        df_copy = df.copy()
        df_copy["trans_date_trans_time"] = pd.to_datetime(
            df_copy["trans_date_trans_time"], errors="coerce"
        )
        df_copy["day_of_week"] = df_copy["trans_date_trans_time"].dt.day_name()
        plt.figure(figsize=(12, 6))
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fraud_by_day = df_copy[df_copy[fraud_col] == 1]["day_of_week"].value_counts()
        fraud_by_day = fraud_by_day.reindex([d for d in day_order if d in fraud_by_day.index])
        fraud_by_day.plot(kind="bar", color="#3498db", edgecolor="black")
        plt.title("Fraud Transactions by Day of Week", fontsize=14, fontweight="bold")
        plt.xlabel("Day of Week", fontsize=12)
        plt.ylabel("Fraud Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "11_fraud_by_day_of_week.png", dpi=300)
        plt.close()

    # Plot 12: Age Distribution
    if "age" in df.columns or "Age" in df.columns:
        age_col = "age" if "age" in df.columns else "Age"
        plt.figure(figsize=(12, 6))
        plt.hist(df[age_col], bins=50, color="#3498db", edgecolor="black", alpha=0.7)
        plt.title("Age Distribution of Customers", fontsize=14, fontweight="bold")
        plt.xlabel("Age (years)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "12_age_distribution.png", dpi=300)
        plt.close()

    # Plot 13: Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(14, 10))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "13_correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 14: Missing Values Heatmap
    missing_data = df.isnull()
    if missing_data.sum().sum() > 0:
        plt.figure(figsize=(14, 8))
        sns.heatmap(missing_data.iloc[:, :50], yticklabels=False, cbar=True, cmap="YlOrRd")
        plt.title("Missing Values Heatmap (First 50 Columns)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "14_missing_values_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 15: Feature Correlation with Target
    numeric_df_with_fraud = numeric_df.copy()
    if fraud_col in df.columns:
        numeric_df_with_fraud[fraud_col] = df[fraud_col]
        plt.figure(figsize=(12, 8))
        correlation_with_target = numeric_df_with_fraud.corr()[fraud_col].sort_values(ascending=False)
        correlation_with_target = correlation_with_target.drop(fraud_col, errors="ignore")
        correlation_with_target.head(20).plot(kind="barh", color="#16a085", edgecolor="black")
        plt.title("Top 20 Features Correlated with Fraud", fontsize=14, fontweight="bold")
        plt.xlabel("Correlation Coefficient", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "15_feature_correlation_with_target.png", dpi=300)
        plt.close()

    logging.info("All plots generated and saved to %s", PLOTS_DIR)


def fraud_analysis(df: pd.DataFrame) -> dict:
    """Analyze fraud patterns in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with fraud analysis metrics.
    """
    fraud_col = _detect_fraud_column(df)
    amount_col = "amt" if "amt" in df.columns else "Amount"

    total_transactions = len(df)
    fraud_count = int((df[fraud_col] == 1).sum())
    legitimate_count = int((df[fraud_col] == 0).sum())
    fraud_ratio = fraud_count / total_transactions

    avg_fraud_amount = (
        df[df[fraud_col] == 1][amount_col].mean() if amount_col in df.columns else 0
    )
    avg_legit_amount = (
        df[df[fraud_col] == 0][amount_col].mean() if amount_col in df.columns else 0
    )

    return {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_count,
        "legitimate_transactions": legitimate_count,
        "fraud_ratio": fraud_ratio,
        "fraud_percentage": fraud_ratio * 100,
        "avg_fraud_amount": avg_fraud_amount,
        "avg_legitimate_amount": avg_legit_amount,
    }


def merchant_analysis(df: pd.DataFrame) -> dict:
    """Analyze merchant-related patterns.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with merchant analysis metrics.
    """
    fraud_col = _detect_fraud_column(df)
    cat_col = "category" if "category" in df.columns else "Merchant_Category"
    merchant_col = "merchant" if "merchant" in df.columns else "Merchant"

    total_merchants = df[merchant_col].nunique() if merchant_col in df.columns else 0
    total_categories = df[cat_col].nunique() if cat_col in df.columns else 0

    top_fraud_category = (
        df[df[fraud_col] == 1][cat_col].value_counts().index[0]
        if cat_col in df.columns and len(df[df[fraud_col] == 1]) > 0
        else "N/A"
    )

    return {
        "total_merchants": total_merchants,
        "total_categories": total_categories,
        "top_fraud_category": top_fraud_category,
    }


def location_analysis(df: pd.DataFrame) -> dict:
    """Analyze location-related patterns.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with location analysis metrics.
    """
    fraud_col = _detect_fraud_column(df)
    state_col = "state" if "state" in df.columns else "State"
    city_col = "city" if "city" in df.columns else "City"

    top_fraud_state = (
        df[df[fraud_col] == 1][state_col].value_counts().index[0]
        if state_col in df.columns and len(df[df[fraud_col] == 1]) > 0
        else "N/A"
    )

    top_fraud_city = (
        df[df[fraud_col] == 1][city_col].value_counts().index[0]
        if city_col in df.columns and len(df[df[fraud_col] == 1]) > 0
        else "N/A"
    )

    return {
        "top_fraud_state": top_fraud_state,
        "top_fraud_city": top_fraud_city,
    }


def time_analysis(df: pd.DataFrame) -> dict:
    """Analyze time-related patterns.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with time analysis metrics.
    """
    fraud_col = _detect_fraud_column(df)

    if "trans_date_trans_time" in df.columns:
        df_copy = df.copy()
        df_copy["trans_date_trans_time"] = pd.to_datetime(
            df_copy["trans_date_trans_time"], errors="coerce"
        )
        fraud_by_hour = df_copy[df_copy[fraud_col] == 1]["trans_date_trans_time"].dt.hour.value_counts()
        peak_fraud_hour = int(fraud_by_hour.index[0]) if len(fraud_by_hour) > 0 else -1
    else:
        peak_fraud_hour = -1

    return {
        "peak_fraud_hour": peak_fraud_hour,
    }


def save_summary(df: pd.DataFrame) -> None:
    """Generate and save comprehensive EDA summary report.

    Args:
        df: Input DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fraud_metrics = fraud_analysis(df)
    merchant_metrics = merchant_analysis(df)
    location_metrics = location_analysis(df)
    time_metrics = time_analysis(df)

    report_lines = [
        "=" * 100,
        "CREDIT CARD FRAUD DETECTION - EDA SUMMARY REPORT",
        "=" * 100,
        "",
        "FRAUD STATISTICS",
        "-" * 100,
        f"Total Transactions: {fraud_metrics['total_transactions']:,}",
        f"Fraudulent Transactions: {fraud_metrics['fraud_transactions']:,}",
        f"Legitimate Transactions: {fraud_metrics['legitimate_transactions']:,}",
        f"Fraud Ratio: {fraud_metrics['fraud_ratio']:.6f}",
        f"Fraud Percentage: {fraud_metrics['fraud_percentage']:.4f}%",
        f"Average Fraud Amount: ${fraud_metrics['avg_fraud_amount']:.2f}",
        f"Average Legitimate Amount: ${fraud_metrics['avg_legitimate_amount']:.2f}",
        "",
        "MERCHANT ANALYSIS",
        "-" * 100,
        f"Total Merchants: {merchant_metrics['total_merchants']:,}",
        f"Total Categories: {merchant_metrics['total_categories']:,}",
        f"Top Fraud Category: {merchant_metrics['top_fraud_category']}",
        "",
        "LOCATION ANALYSIS",
        "-" * 100,
        f"Top Fraud State: {location_metrics['top_fraud_state']}",
        f"Top Fraud City: {location_metrics['top_fraud_city']}",
        "",
        "TIME ANALYSIS",
        "-" * 100,
        f"Peak Fraud Hour (24h): {time_metrics['peak_fraud_hour']}:00",
        "",
        "KEY OBSERVATIONS",
        "-" * 100,
        f"• Fraud accounts for {fraud_metrics['fraud_percentage']:.2f}% of all transactions",
        f"• {merchant_metrics['total_merchants']:,} unique merchants in the dataset",
        f"• Fraudulent transactions average ${fraud_metrics['avg_fraud_amount']:.2f} (legitimate: ${fraud_metrics['avg_legitimate_amount']:.2f})",
        f"• Highest fraud activity in {location_metrics['top_fraud_state']} state",
        f"• Peak fraud hour: {time_metrics['peak_fraud_hour']}:00",
        "",
        "=" * 100,
    ]

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(report_lines))

    logging.info("Summary report saved to %s", SUMMARY_FILE)


def main() -> None:
    """Execute the complete EDA pipeline."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        df = load_data()
        logging.info("Data loaded successfully: %s", df.shape)
    except (FileNotFoundError, ValueError) as error:
        logging.error("Failed to load data: %s", error)
        return

    logging.info(basic_info(df))

    try:
        generate_plots(df)
    except Exception as error:
        logging.error("Error generating plots: %s", error)

    try:
        save_summary(df)
    except Exception as error:
        logging.error("Error saving summary: %s", error)

    logging.info("EDA pipeline completed successfully.")


if __name__ == "__main__":
    main()
