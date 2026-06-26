"""Data preprocessing utilities for the credit card fraud detection pipeline."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT_DIR / "dataset" / "creditcard.csv"
OUTPUT_FILE = ROOT_DIR / "outputs" / "cleaned_creditcard.csv"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load credit card fraud dataset from a CSV file.

    Args:
        file_path: Optional path to the CSV file. If None, uses the default dataset location.

    Returns:
        A pandas DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset cannot be parsed.
    """
    source = file_path or DATASET_FILE

    if not source.exists():
        raise FileNotFoundError(f"Dataset file not found at: {source}")

    try:
        data_frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset file is empty: {source}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Failed to parse CSV file: {source}") from error

    return data_frame


def show_basic_info(df: pd.DataFrame) -> str:
    """Return a formatted string with dataset shape, columns, information, and memory usage."""
    if df.empty:
        return "The dataset is empty. No basic info is available."

    buffer = StringIO()
    df.info(buf=buffer, memory_usage="deep")
    info_text = buffer.getvalue().strip()

    lines = [
        "Dataset Basic Information",
        "-------------------------",
        f"Shape: {df.shape}",
        f"Columns: {list(df.columns)}",
        "",
        "DataFrame Info:",
        info_text,
    ]

    return "\n".join(lines)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing value counts and percentages."""
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    result = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage.round(4),
        }
    )
    return result.loc[result["missing_count"] > 0].sort_values(
        by="missing_count", ascending=False
    )


def check_duplicates(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows in the dataset."""
    return int(df.duplicated().sum())


def show_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a statistical summary for numeric columns in the dataset."""
    if df.empty:
        raise ValueError("Cannot compute statistics on an empty DataFrame.")

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("The dataset does not contain numerical columns.")

    statistics = numeric_df.agg(
        ["mean", "median", "std", "min", "max", "quantile"]
    )

    quartiles = numeric_df.quantile([0.25, 0.5, 0.75]).rename(index={0.25: "25%", 0.5: "50%", 0.75: "75%"})
    statistics = statistics.drop(index="quantile", errors="ignore")
    statistics = pd.concat([statistics, quartiles])

    return statistics.round(6)


def detect_features_target(
    df: pd.DataFrame, target_column: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the dataset into feature matrix and target vector."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()
    return features, target


def class_distribution(df: pd.DataFrame, target_column: str = "Class") -> Dict[str, object]:
    """Return distribution metrics for the target column."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    counts = df[target_column].value_counts(dropna=False)
    total = len(df)
    fraud_count = int(counts.get(1, 0))
    legit_count = int(counts.get(0, 0))
    fraud_percentage = (fraud_count / total) * 100 if total > 0 else 0.0

    return {
        "total_transactions": total,
        "legitimate_transactions": legit_count,
        "fraud_transactions": fraud_count,
        "fraud_percentage": round(fraud_percentage, 6),
        "distribution": counts.to_dict(),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset without modifying the target column."""
    if df.empty:
        return df.copy()

    cleaned_df = df.drop_duplicates().reset_index(drop=True)
    return cleaned_df


def save_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save the cleaned dataset to the outputs directory."""
    destination = output_path or OUTPUT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def main() -> None:
    """Run the preprocessing pipeline and print diagnostics."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    try:
        df = load_data()
    except (FileNotFoundError, ValueError) as error:
        logging.error("Failed to load dataset: %s", error)
        return

    logging.info("Dataset loaded successfully.")
    logging.info(show_basic_info(df))

    missing_report = check_missing_values(df)
    logging.info("Missing Values Report:\n%s", missing_report if not missing_report.empty else "No missing values found.")

    duplicate_count = check_duplicates(df)
    logging.info("Duplicate Rows: %d", duplicate_count)

    statistics = show_statistics(df)
    logging.info("Statistical Summary:\n%s", statistics)

    try:
        _, _ = detect_features_target(df)
    except KeyError as error:
        logging.error("Target detection failed: %s", error)
        return

    distribution = class_distribution(df)
    logging.info(
        "Class Distribution: total=%d, legitimate=%d, fraud=%d, fraud_percentage=%.6f%%",
        distribution["total_transactions"],
        distribution["legitimate_transactions"],
        distribution["fraud_transactions"],
        distribution["fraud_percentage"],
    )

    cleaned_df = clean_dataset(df)
    if len(cleaned_df) != len(df):
        logging.info("Removed %d duplicate rows.", len(df) - len(cleaned_df))
    else:
        logging.info("No duplicate rows were removed.")

    saved_path = save_dataset(cleaned_df)
    logging.info("Cleaned dataset saved to: %s", saved_path)


if __name__ == "__main__":
    main()
