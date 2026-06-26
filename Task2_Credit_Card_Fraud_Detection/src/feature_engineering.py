"""Feature Engineering module for credit card fraud detection system."""

from __future__ import annotations

import logging
import time
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
TARGET_COLUMN = "is_fraud"
F = TypeVar("F", bound=Callable[..., object])


def timed_step(func: F) -> F:
    """Log start, finish, and elapsed time for a pipeline step."""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        logging.info("Starting %s...", func.__name__)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logging.info("Finished %s in %.2f seconds", func.__name__, time.perf_counter() - start)

    return wrapper  # type: ignore[return-value]


@timed_step
def load_data(
    train_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and merge training and test datasets.

    Args:
        train_file: Path to fraudTrain.csv. Defaults to dataset/fraudTrain.csv.
        test_file: Path to fraudTest.csv. Defaults to dataset/fraudTest.csv.

    Returns:
        Combined DataFrame with all data.

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
    logging.info("Loaded training set: %s, test set: %s", train_df.shape, test_df.shape)
    logging.info("Combined dataset shape: %s", combined_df.shape)

    return combined_df


@timed_step
def detect_column_types(df: pd.DataFrame) -> Tuple[list, list, list]:
    """Automatically detect numerical, categorical, and datetime columns.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (numerical_columns, categorical_columns, datetime_columns).
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    datetime_cols = []
    for col in categorical_cols:
        try:
            pd.to_datetime(df[col], errors="coerce")
            if pd.to_datetime(df[col], errors="coerce").notna().sum() > len(df) * 0.8:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except (ValueError, TypeError):
            pass

    return numerical_cols, categorical_cols, datetime_cols


@timed_step
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from transaction datetime.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new time features.
    """
    df_copy = df.copy()

    datetime_cols = [col for col in df_copy.columns if "date" in col.lower() or "time" in col.lower()]

    if not datetime_cols:
        logging.warning("No datetime columns detected. Skipping time feature extraction.")
        return df_copy

    datetime_col = datetime_cols[0]
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col], errors="coerce")

    # Extract basic time features
    df_copy["transaction_hour"] = df_copy[datetime_col].dt.hour
    df_copy["transaction_day"] = df_copy[datetime_col].dt.day
    df_copy["transaction_month"] = df_copy[datetime_col].dt.month
    df_copy["transaction_weekday"] = df_copy[datetime_col].dt.dayofweek
    df_copy["transaction_quarter"] = df_copy[datetime_col].dt.quarter

    # Weekend flag
    df_copy["is_weekend"] = (df_copy["transaction_weekday"] >= 5).astype(int)

    # Time of day categorization
    def categorize_time_of_day(hour: int) -> int:
        """Categorize hour into time periods.

        0: Night (00:00-05:59)
        1: Morning (06:00-11:59)
        2: Afternoon (12:00-17:59)
        3: Evening (18:00-23:59)
        """
        if 0 <= hour < 6:
            return 0  # Night
        elif 6 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 18:
            return 2  # Afternoon
        else:
            return 3  # Evening

    df_copy["time_of_day"] = df_copy["transaction_hour"].apply(categorize_time_of_day)

    # Late night flag (23:00-05:59)
    df_copy["is_late_night"] = ((df_copy["transaction_hour"] >= 23) | (df_copy["transaction_hour"] < 6)).astype(int)

    logging.info("Time features created successfully.")
    return df_copy


@timed_step
def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction amount-based features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new amount features.
    """
    df_copy = df.copy()

    amount_cols = [col for col in df_copy.columns if "amt" in col.lower() or "amount" in col.lower()]

    if not amount_cols:
        logging.warning("No amount columns detected. Skipping amount feature engineering.")
        return df_copy

    amount_col = amount_cols[0]

    # Handle invalid values
    df_copy[amount_col] = pd.to_numeric(df_copy[amount_col], errors="coerce").fillna(0)

    # Log transformation
    df_copy["log_amount"] = np.log1p(df_copy[amount_col])

    # Amount quantile-based categorization
    q25, q50, q75, q95 = df_copy[amount_col].quantile([0.25, 0.5, 0.75, 0.95])

    def categorize_amount(amt: float) -> int:
        """Categorize transaction amount.

        0: Low (0-Q25)
        1: Medium (Q25-Q50)
        2: High (Q50-Q95)
        3: Very High (>Q95)
        """
        if amt <= q25:
            return 0
        elif amt <= q50:
            return 1
        elif amt <= q95:
            return 2
        else:
            return 3

    df_copy["amount_category"] = df_copy[amount_col].apply(categorize_amount)

    # High value transaction flag (top 5%)
    df_copy["is_high_value"] = (df_copy[amount_col] >= q95).astype(int)

    # Amount percentile
    df_copy["amount_percentile"] = df_copy[amount_col].rank(pct=True)

    logging.info("Amount features created successfully.")
    return df_copy


@timed_step
def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-related features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new customer features.
    """
    df_copy = df.copy()

    # Age-based features
    age_cols = [col for col in df_copy.columns if "age" in col.lower()]
    if age_cols:
        age_col = age_cols[0]
        df_copy[age_col] = pd.to_numeric(df_copy[age_col], errors="coerce").fillna(df_copy[age_col].median())

        def categorize_age(age: float) -> int:
            """Categorize age into groups.

            0: Young (0-25)
            1: Adult (26-40)
            2: Middle Age (41-55)
            3: Senior (56+)
            """
            if age <= 25:
                return 0
            elif age <= 40:
                return 1
            elif age <= 55:
                return 2
            else:
                return 3

        df_copy["age_group"] = df_copy[age_col].apply(categorize_age)

    # Gender encoding (if exists)
    gender_cols = [col for col in df_copy.columns if "gender" in col.lower()]
    if gender_cols:
        gender_col = gender_cols[0]
        df_copy["gender_encoded"] = (df_copy[gender_col].str.upper() == "M").astype(int)

    logging.info("Customer features created successfully.")
    return df_copy


@timed_step
def create_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create merchant-related features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new merchant features.
    """
    df_copy = df.copy()

    merchant_cols = [col for col in df_copy.columns if "merchant" in col.lower()]
    if not merchant_cols:
        logging.warning("No merchant column detected. Skipping merchant features.")
        return df_copy

    merchant_col = merchant_cols[0]

    # Merchant frequency (transaction count per merchant)
    df_copy["merchant_transaction_count"] = df_copy.groupby(merchant_col)[merchant_col].transform("count")

    # Merchant category frequency
    category_cols = [col for col in df_copy.columns if "category" in col.lower()]
    if category_cols:
        category_col = category_cols[0]
        df_copy["merchant_category_count"] = df_copy.groupby(category_col)[category_col].transform("count")

        # Merchant fraud rate by category
        fraud_rate_by_category = (
            df_copy.groupby(category_col)[TARGET_COLUMN].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        )
        df_copy["merchant_category_fraud_rate"] = df_copy[category_col].map(fraud_rate_by_category).fillna(0)

    # Merchant fraud rate (overall)
    merchant_fraud_rate = (
        df_copy.groupby(merchant_col)[TARGET_COLUMN].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
    )
    df_copy["merchant_fraud_rate"] = df_copy[merchant_col].map(merchant_fraud_rate).fillna(0)

    # Merchant risk score (fraud rate normalized)
    df_copy["merchant_risk_score"] = df_copy["merchant_fraud_rate"]

    logging.info("Merchant features created successfully.")
    return df_copy


@timed_step
def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create location-based features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new location features.
    """
    df_copy = df.copy()

    state_cols = [col for col in df_copy.columns if "state" in col.lower()]
    city_cols = [col for col in df_copy.columns if "city" in col.lower()]
    zip_cols = [col for col in df_copy.columns if "zip" in col.lower()]

    # State features
    if state_cols:
        state_col = state_cols[0]
        df_copy["state_transaction_count"] = df_copy.groupby(state_col)[state_col].transform("count")
        state_fraud_rate = (
            df_copy.groupby(state_col)[TARGET_COLUMN].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        )
        df_copy["state_fraud_rate"] = df_copy[state_col].map(state_fraud_rate).fillna(0)

    # City features
    if city_cols:
        city_col = city_cols[0]
        df_copy["city_transaction_count"] = df_copy.groupby(city_col)[city_col].transform("count")
        city_fraud_rate = (
            df_copy.groupby(city_col)[TARGET_COLUMN].apply(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)
        )
        df_copy["city_fraud_rate"] = df_copy[city_col].map(city_fraud_rate).fillna(0)

    # ZIP code features
    if zip_cols:
        zip_col = zip_cols[0]
        df_copy["zip_transaction_count"] = df_copy.groupby(zip_col)[zip_col].transform("count")

    logging.info("Location features created successfully.")
    return df_copy


@timed_step
def create_transaction_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction behaviour-based features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new behaviour features.
    """
    df_copy = df.copy()

    customer_cols = [col for col in df_copy.columns if "first" in col.lower() or "customer" in col.lower()]
    merchant_cols = [col for col in df_copy.columns if "merchant" in col.lower()]
    amount_cols = [col for col in df_copy.columns if "amt" in col.lower() or "amount" in col.lower()]

    # Customer transaction count
    if customer_cols:
        customer_col = customer_cols[0]
        df_copy["customer_transaction_count"] = df_copy.groupby(customer_col)[customer_col].transform("count")

        # Average transaction amount per customer
        if amount_cols:
            amount_col = amount_cols[0]
            customer_avg_amount = df_copy.groupby(customer_col)[amount_col].transform("mean")
            df_copy["customer_avg_amount"] = customer_avg_amount

    # Merchant average transaction amount
    if merchant_cols and amount_cols:
        merchant_col = merchant_cols[0]
        amount_col = amount_cols[0]
        merchant_avg_amount = df_copy.groupby(merchant_col)[amount_col].transform("mean")
        df_copy["merchant_avg_amount"] = merchant_avg_amount

    # Merchant fraud transaction count
    if merchant_cols:
        merchant_col = merchant_cols[0]
        merchant_fraud_count = df_copy.groupby(merchant_col)[TARGET_COLUMN].transform(lambda x: (x == 1).sum())
        df_copy["merchant_fraud_count"] = merchant_fraud_count

    logging.info("Transaction behaviour features created successfully.")
    return df_copy


@timed_step
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables intelligently.

    Uses LabelEncoding for ordinal features and OneHotEncoding for nominal features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with encoded categorical features.
    """
    df_copy = df.copy()

    categorical_cols = df_copy.select_dtypes(include=["object"]).columns.tolist()

    # Exclude datetime and target columns
    exclude_cols = [col for col in categorical_cols if TARGET_COLUMN in col or "date" in col.lower() or "time" in col.lower()]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    # Ordinal features (typically location/merchant info) - use LabelEncoding
    ordinal_features = ["state", "city", "merchant", "category"]
    label_encoder = LabelEncoder()

    for col in categorical_cols:
        if any(ord_feat in col.lower() for ord_feat in ordinal_features):
            try:
                df_copy[f"{col}_encoded"] = label_encoder.fit_transform(df_copy[col].astype(str))
            except Exception as error:
                logging.warning("Error encoding %s: %s", col, error)

    logging.info("Categorical features encoded successfully.")
    return df_copy


@timed_step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by handling missing values, duplicates, and invalid entries.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df_copy = df.copy()

    initial_shape = df_copy.shape

    # Remove duplicates
    df_copy.drop_duplicates(inplace=True)

    # Handle missing values
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df_copy[col].isnull().sum() > 0:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)

    # Handle infinite values
    for col in numeric_cols:
        df_copy[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_copy[col].fillna(df_copy[col].median(), inplace=True)

    # Ensure target column is numeric
    if TARGET_COLUMN in df_copy.columns:
        df_copy[TARGET_COLUMN] = pd.to_numeric(df_copy[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    final_shape = df_copy.shape
    logging.info(
        "Data cleaned. Removed %d duplicates. Initial: %s, Final: %s",
        initial_shape[0] - final_shape[0],
        initial_shape,
        final_shape,
    )

    return df_copy


@timed_step
def save_processed_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save processed dataset to outputs directory.

    Args:
        df: Processed DataFrame.
        output_path: Optional custom output path. Defaults to outputs/processed_dataset.csv.

    Returns:
        Path to saved file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = output_path or OUTPUT_DIR / "processed_dataset.csv"

    df.to_csv(destination, index=False)
    logging.info("Processed dataset saved to: %s", destination)

    return destination


def main() -> None:
    """Execute the feature engineering pipeline."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.info("Starting main...")
    start = time.perf_counter()

    try:
        # Load data
        df = load_data()
        logging.info("Dataset loaded successfully. Shape: %s", df.shape)

        # Detect column types
        numerical_cols, categorical_cols, datetime_cols = detect_column_types(df)
        logging.info("Detected - Numerical: %d, Categorical: %d, DateTime: %d", len(numerical_cols), len(categorical_cols), len(datetime_cols))

        # Create features
        df = create_time_features(df)
        df = create_amount_features(df)
        df = create_customer_features(df)
        df = create_merchant_features(df)
        df = create_location_features(df)
        df = create_transaction_behaviour_features(df)

        # Encode categorical variables
        df = encode_features(df)

        # Clean data
        df = clean_data(df)

        logging.info("Feature engineering completed. Final shape: %s", df.shape)

        # Save processed dataset
        output_file = save_processed_dataset(df)
        logging.info("Pipeline execution completed successfully. Output: %s", output_file)

    except (FileNotFoundError, ValueError) as error:
        logging.error("Pipeline execution failed: %s", error)
        return
    finally:
        logging.info("Finished main in %.2f seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
