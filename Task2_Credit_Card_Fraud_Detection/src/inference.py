"""Inference helpers shared by the Streamlit dashboard and scripts."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from .config import Config
except ImportError:
    from config import Config


@dataclass
class ModelBundle:
    """Loaded inference artifacts."""

    model: Any
    scaler: Any
    feature_names: List[str]
    metadata: Dict[str, Any]
    threshold: float


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required model artifact not found: {path}")
    with open(path, "rb") as file:
        return pickle.load(file)


def load_model_bundle() -> ModelBundle:
    """Load model, scaler, feature names, and metadata from disk."""
    model = _load_pickle(Config.paths.BEST_MODEL)
    scaler = _load_pickle(Config.paths.SCALER)
    feature_names = _load_pickle(Config.paths.FEATURE_NAMES)

    metadata: Dict[str, Any] = {}
    if Config.paths.MODEL_METADATA.exists():
        metadata = json.loads(Config.paths.MODEL_METADATA.read_text(encoding="utf-8"))

    threshold = float(metadata.get("threshold", Config.dashboard.PREDICTION_THRESHOLD))
    return ModelBundle(model=model, scaler=scaler, feature_names=feature_names, metadata=metadata, threshold=threshold)


def validate_and_align_features(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Validate a prediction frame and align it to trained feature order."""
    if df.empty:
        raise ValueError("The prediction dataset is empty.")

    features = list(feature_names)
    aligned = pd.DataFrame(index=df.index)
    for column in features:
        aligned[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else 0.0

    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned[features]


def predict(df: pd.DataFrame, bundle: Optional[ModelBundle] = None) -> pd.DataFrame:
    """Return predictions, probabilities, confidence, and risk bands."""
    bundle = bundle or load_model_bundle()
    aligned = validate_and_align_features(df, bundle.feature_names)
    scaled = pd.DataFrame(bundle.scaler.transform(aligned), columns=bundle.feature_names, index=aligned.index)
    probability = bundle.model.predict_proba(scaled)[:, 1]
    prediction = (probability >= bundle.threshold).astype(int)

    result = df.copy()
    result["fraud_probability"] = probability
    result["prediction"] = prediction
    result["prediction_label"] = np.where(prediction == 1, "Fraud", "Legitimate")
    result["confidence"] = np.maximum(probability, 1 - probability)
    result["risk_band"] = pd.cut(
        probability,
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    return result


def append_prediction_history(predictions: pd.DataFrame) -> Path:
    """Append prediction results to the local history CSV."""
    Config.paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = Config.paths.PREDICTION_HISTORY
    export = predictions.copy()
    export.insert(0, "prediction_time_utc", pd.Timestamp.utcnow().isoformat())
    write_header = not history_path.exists()
    export.to_csv(history_path, mode="a", header=write_header, index=False)
    return history_path
