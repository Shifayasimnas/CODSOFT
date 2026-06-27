import warnings

warnings.filterwarnings("ignore")

import importlib.util
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from .config import (
        FEATURE_IMPORTANCE_PATH,
        LABEL_ENCODER_PATH,
        METADATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        MODELS_DIR,
        OUTPUTS_DIR,
        PREPROCESSOR_PATH,
    )
    from .data_utils import columns_to_drop, detect_target_column, find_csv, load_dataset, normalize_target, write_json
    from .eda import save_eda_plots
except ImportError:
    from config import (
        FEATURE_IMPORTANCE_PATH,
        LABEL_ENCODER_PATH,
        METADATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        MODELS_DIR,
        OUTPUTS_DIR,
        PREPROCESSOR_PATH,
    )
    from data_utils import columns_to_drop, detect_target_column, find_csv, load_dataset, normalize_target, write_json
    from eda import save_eda_plots


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9),
    }
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
    return models


def predict_probability(model, x_matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_matrix)
        return 1 / (1 + np.exp(-scores))
    return model.predict(x_matrix)


def evaluate_model(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = predict_probability(model, x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else np.nan,
    }


def save_feature_importance(model, feature_names: np.ndarray) -> None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        values = np.zeros(len(feature_names))

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(30)
    )
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def train() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = find_csv()
    df = load_dataset(csv_path)
    original_shape = df.shape
    df = df.drop_duplicates().copy()

    target_column = detect_target_column(df)
    df[target_column] = normalize_target(df[target_column])

    label_encoder = None
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_column].astype(str))
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    else:
        y = df[target_column].astype(int).to_numpy()

    drop_columns = columns_to_drop(df, target_column)
    x = df.drop(columns=[target_column] + drop_columns, errors="ignore")
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in x.columns if column not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    stratify = y if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )

    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    rows = []
    trained_models = {}
    for model_name, model in get_models().items():
        metrics = evaluate_model(model, x_train_processed, y_train, x_test_processed, y_test)
        rows.append({"model": model_name, **metrics})
        trained_models[model_name] = model

    metrics_df = pd.DataFrame(rows).sort_values(["f1_score", "roc_auc", "recall"], ascending=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    feature_names = preprocessor.get_feature_names_out()
    save_feature_importance(best_model, feature_names)
    eda_summary = save_eda_plots()

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(csv_path),
        "original_rows": int(original_shape[0]),
        "rows_after_deduplication": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "dropped_columns": drop_columns,
        "feature_columns": x.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_values": {
            column: sorted([str(value) for value in x[column].dropna().unique()])[:100]
            for column in categorical_features
        },
        "numeric_ranges": {
            column: {
                "min": float(pd.to_numeric(x[column], errors="coerce").min()),
                "max": float(pd.to_numeric(x[column], errors="coerce").max()),
                "median": float(pd.to_numeric(x[column], errors="coerce").median()),
            }
            for column in numeric_features
        },
        "class_distribution": pd.Series(y).value_counts().sort_index().astype(int).to_dict(),
        "best_model": str(best_model_name),
        "best_metrics": metrics_df.iloc[0].to_dict(),
        "xgboost_available": importlib.util.find_spec("xgboost") is not None,
        "shap_available": importlib.util.find_spec("shap") is not None,
        "eda_summary": eda_summary,
    }
    write_json(METADATA_PATH, metadata)
    return metadata


if __name__ == "__main__":
    result = train()
    print(f"Training complete. Best model: {result['best_model']}")
    print(f"Artifacts saved to: {MODELS_DIR}")
