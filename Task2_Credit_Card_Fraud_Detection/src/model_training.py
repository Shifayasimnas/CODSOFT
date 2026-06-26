"""Production-oriented model training pipeline for fraud detection."""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from .config import Config
except ImportError:
    from config import Config


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
PROCESSED_DATA_PATH = OUTPUT_DIR / "processed_dataset.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "training_summary.txt"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
CURVES_PATH = OUTPUT_DIR / "model_curves.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TARGET_COLUMN = "is_fraud"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

LEAKAGE_PATTERNS = (
    "fraud_rate",
    "risk_score",
    "fraud_count",
)
IDENTIFIER_COLUMNS = {
    "Unnamed: 0",
    "cc_num",
    "trans_num",
    "unix_time",
    "zip",
}


@dataclass
class TrainingArtifacts:
    """Container for trained artifacts persisted for the dashboard."""

    model: Any
    scaler: StandardScaler
    feature_names: List[str]
    threshold: float
    best_model_name: str


def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the processed dataset from disk."""
    source = data_path or PROCESSED_DATA_PATH
    if not source.exists():
        raise FileNotFoundError(f"Processed dataset not found at {source}")
    df = pd.read_csv(source)
    logging.info("Loaded processed dataset with shape %s", df.shape)
    return df


def _is_leaky_feature(column: str) -> bool:
    """Return True for features derived directly from target leakage."""
    normalized = column.lower()
    return any(pattern in normalized for pattern in LEAKAGE_PATTERNS)


def select_model_features(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Select numeric, model-safe features and the fraud target."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    numeric = df.select_dtypes(include=[np.number]).copy()
    drop_columns = [
        col
        for col in numeric.columns
        if col == target or col in IDENTIFIER_COLUMNS or _is_leaky_feature(col)
    ]
    X = numeric.drop(columns=drop_columns, errors="ignore")
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if X.empty:
        raise ValueError("No numeric model-ready features were found after validation.")

    logging.info("Selected %d model features; excluded %d unsafe columns.", X.shape[1], len(drop_columns))
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split features into stratified train and validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=Config.model.TEST_SIZE,
        stratify=y if Config.model.STRATIFY else None,
        random_state=Config.model.RANDOM_STATE,
    )
    logging.info("Split dataset into train %s and validation %s", X_train.shape, X_val.shape)
    return X_train, y_train, X_val, y_val


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a scaler on train features and transform train/validation sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    return X_train_scaled, X_val_scaled, scaler


def balance_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Use SMOTE on the training partition only to address class imbalance."""
    fraud_count = int((y_train == 1).sum())
    normal_count = int((y_train == 0).sum())
    if fraud_count < 6:
        logging.warning("Skipping SMOTE because too few fraud samples are available.")
        return X_train, y_train

    strategy = min(Config.model.SMOTE_SAMPLING_STRATEGY, 0.25)
    smote = SMOTE(
        random_state=Config.model.SMOTE_RANDOM_STATE,
        sampling_strategy=strategy,
        k_neighbors=min(5, fraud_count - 1),
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logging.info(
        "Balanced training data from fraud/normal %d/%d to %d/%d.",
        fraud_count,
        normal_count,
        int((y_resampled == 1).sum()),
        int((y_resampled == 0).sum()),
    )
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, dtype=int)


def build_models(y_train: pd.Series) -> List[Tuple[str, Any]]:
    """Create candidate classifiers with imbalance-aware defaults."""
    normal_count = max(int((y_train == 0).sum()), 1)
    fraud_count = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = normal_count / fraud_count

    models: List[Tuple[str, Any]] = [
        (
            "Logistic Regression",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=Config.model.RANDOM_STATE,
                solver="lbfgs",
            ),
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=12,
                min_samples_leaf=20,
                random_state=Config.model.RANDOM_STATE,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                max_depth=16,
                min_samples_leaf=10,
                n_estimators=120,
                n_jobs=Config.performance.N_JOBS,
                random_state=Config.model.RANDOM_STATE,
            ),
        ),
    ]

    if XGBOOST_AVAILABLE:
        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    eval_metric="logloss",
                    learning_rate=0.08,
                    max_depth=5,
                    n_estimators=150,
                    n_jobs=Config.performance.N_JOBS,
                    random_state=Config.model.RANDOM_STATE,
                    scale_pos_weight=scale_pos_weight,
                ),
            )
        )
    else:
        logging.warning("XGBoost is not installed; skipping that candidate.")

    return models


def _probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities from classifiers with a robust fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Find the threshold that maximizes validation F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    threshold = float(thresholds[best_index])
    return threshold, {
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }


def evaluate_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """Train and evaluate a candidate model."""
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    y_prob = _probabilities(model, X_val)
    threshold, threshold_metrics = optimize_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    prediction_time = time.perf_counter() - start_pred
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_val, y_prob),
        "Average Precision": average_precision_score(y_val, y_prob),
        "Threshold": threshold,
        "Threshold Precision": threshold_metrics["precision"],
        "Threshold Recall": threshold_metrics["recall"],
        "Threshold F1": threshold_metrics["f1"],
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "Training Time": training_time,
        "Prediction Time": prediction_time,
    }
    logging.info("Evaluated %s: ROC AUC %.4f, PR AUC %.4f, F1 %.4f", name, metrics["ROC AUC"], metrics["Average Precision"], metrics["F1"])
    return metrics


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Dict[str, float]:
    """Run stratified cross-validation with fraud-focused metrics."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=Config.model.RANDOM_STATE)
    scores = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        n_jobs=Config.performance.N_JOBS,
        error_score="raise",
    )
    return {f"CV {key.replace('test_', '').title()}": float(np.mean(value)) for key, value in scores.items() if key.startswith("test_")}


def calibrate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Calibrate probabilities for better risk scores."""
    if not Config.model.ENABLE_CALIBRATION:
        return model
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=Config.model.CALIBRATION_METHOD,
        cv=3,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def feature_importance(model: Any, feature_names: Iterable[str], top_n: int = 25) -> List[Dict[str, float]]:
    """Extract feature importance from tree models or linear coefficients."""
    base_model = getattr(model, "estimator", model)
    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        values = np.abs(base_model.coef_).ravel()
    else:
        values = np.zeros(len(list(feature_names)))

    importance = pd.DataFrame({"feature": list(feature_names), "importance": values})
    importance = importance.sort_values("importance", ascending=False).head(top_n)
    return importance.to_dict(orient="records")


def save_artifacts(
    artifacts: TrainingArtifacts,
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    curves: Dict[str, Any],
) -> None:
    """Persist model artifacts, metrics, curves, and human-readable summary."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(artifacts.model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(artifacts.scaler, f)
    with open(FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(artifacts.feature_names, f)

    pd.DataFrame(results).to_csv(COMPARISON_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CURVES_PATH.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    metadata = {
        "model_name": artifacts.best_model_name,
        "model_version": "1.0.0",
        "threshold": artifacts.threshold,
        "feature_count": len(artifacts.feature_names),
        "feature_names": artifacts.feature_names,
        "target_column": TARGET_COLUMN,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "config": {
            "model": asdict(Config.model),
            "performance": asdict(Config.performance),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ranking = sorted(results, key=lambda row: (row["Average Precision"], row["ROC AUC"], row["F1"]), reverse=True)
    lines = [
        "MODEL TRAINING SUMMARY",
        "======================",
        f"Best Model: {artifacts.best_model_name}",
        f"Optimized Threshold: {artifacts.threshold:.4f}",
        f"Feature Count: {len(artifacts.feature_names)}",
        "",
        "Model Ranking:",
    ]
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"{index}. {row['Model']} - PR AUC: {row['Average Precision']:.4f}, ROC AUC: {row['ROC AUC']:.4f}, "
            f"Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}, F1: {row['F1']:.4f}"
        )
    lines.extend(
        [
            "",
            "Production Notes:",
            "- Training excludes raw identifiers and target-derived leakage features.",
            "- SMOTE is applied only after the train/validation split.",
            "- Fraud probabilities are calibrated before final threshold optimization.",
            "- Dashboard artifacts include metrics, curves, feature names, and model metadata.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_curves(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, Any]:
    """Create serializable ROC and precision-recall curve data."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "precision_recall": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def train_pipeline(data_path: Optional[Path] = None) -> TrainingArtifacts:
    """Run the full training pipeline and return persisted artifacts."""
    df = load_data(data_path)
    X, y = select_model_features(df)
    X_train, y_train, X_val, y_val = split_data(X, y)
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
    X_train_balanced, y_train_balanced = balance_training_data(X_train_scaled, y_train)

    models = build_models(y_train)
    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for model_name, model in models:
        metrics = evaluate_model(model_name, model, X_train_balanced, y_train_balanced, X_val_scaled, y_val)
        cv_folds = min(Config.model.CV_FOLDS, int(y_train_balanced.value_counts().min()))
        if cv_folds >= 2:
            cv_scores = cross_validate_model(model, X_train_balanced, y_train_balanced, folds=cv_folds)
            metrics.update(cv_scores)
        results.append(metrics)
        trained_models[model_name] = model

    best_result = max(results, key=lambda row: (row["Average Precision"], row["ROC AUC"], row["F1"]))
    best_name = best_result["Model"]
    calibrated_model = calibrate_model(trained_models[best_name], X_train_balanced, y_train_balanced)

    calibrated_prob = _probabilities(calibrated_model, X_val_scaled)
    final_threshold, _ = optimize_threshold(y_val, calibrated_prob)
    final_pred = (calibrated_prob >= final_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, final_pred, labels=[0, 1]).ravel()

    final_metrics = {
        "best_model": best_name,
        "threshold": final_threshold,
        "accuracy": accuracy_score(y_val, final_pred),
        "precision": precision_score(y_val, final_pred, zero_division=0),
        "recall": recall_score(y_val, final_pred, zero_division=0),
        "f1": f1_score(y_val, final_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, calibrated_prob),
        "average_precision": average_precision_score(y_val, calibrated_prob),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "fraud_rate_validation": float(y_val.mean()),
        "feature_importance": feature_importance(trained_models[best_name], X.columns),
    }

    artifacts = TrainingArtifacts(
        model=calibrated_model,
        scaler=scaler,
        feature_names=X.columns.tolist(),
        threshold=final_threshold,
        best_model_name=best_name,
    )
    save_artifacts(artifacts, results, final_metrics, build_curves(y_val, calibrated_prob))
    logging.info("Training pipeline completed. Best model: %s", best_name)
    return artifacts


def main() -> None:
    """Run model training as a script."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        train_pipeline()
    except Exception as error:
        logging.exception("Training pipeline failed: %s", error)


if __name__ == "__main__":
    main()
