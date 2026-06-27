import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    from .config import DATASET_DIR
except ImportError:
    from config import DATASET_DIR


TARGET_CANDIDATES = [
    "churn",
    "exited",
    "attrition",
    "attrition_flag",
    "customer_status",
    "is_churn",
    "churned",
    "tenure_status",
]

DROP_NAME_PATTERNS = ("id", "rownumber", "row_number", "surname", "name")


def ensure_archives_extracted(dataset_dir: Path = DATASET_DIR) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for archive in dataset_dir.rglob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dataset_dir)


def find_csv(dataset_dir: Path = DATASET_DIR) -> Path:
    ensure_archives_extracted(dataset_dir)
    csv_files = sorted(
        [path for path in dataset_dir.rglob("*.csv") if not any(part.startswith(".") for part in path.parts)],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV file was found inside {dataset_dir}")
    return csv_files[0]


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or find_csv()
    return pd.read_csv(csv_path)


def detect_target_column(df: pd.DataFrame) -> str:
    normalized = {column.lower().strip().replace(" ", "_"): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    binary_columns = []
    for column in df.columns:
        unique_count = df[column].dropna().nunique()
        if 1 < unique_count <= 2:
            binary_columns.append(column)

    if binary_columns:
        return binary_columns[-1]

    raise ValueError(
        "Could not detect the churn target column automatically. "
        f"Expected one of {TARGET_CANDIDATES} or a binary target column."
    )


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "churn": 1,
        "churned": 1,
        "exited": 1,
        "attrited customer": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "no churn": 0,
        "not churned": 0,
        "existing customer": 0,
        "active": 0,
    }
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(mapping)
    if mapped.notna().all():
        return mapped.astype(int)
    return series


def columns_to_drop(df: pd.DataFrame, target_column: str) -> list[str]:
    drops = []
    row_count = len(df)
    for column in df.columns:
        if column == target_column:
            continue
        column_key = column.lower().replace(" ", "_")
        if any(pattern in column_key for pattern in DROP_NAME_PATTERNS):
            unique_ratio = df[column].nunique(dropna=True) / max(row_count, 1)
            if unique_ratio > 0.75 or column_key in {"rownumber", "row_number", "surname"}:
                drops.append(column)
    return drops


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
