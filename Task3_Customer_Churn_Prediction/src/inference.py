import joblib
import pandas as pd

try:
    from .config import METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH
    from .data_utils import read_json
except ImportError:
    from config import METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH
    from data_utils import read_json


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = read_json(METADATA_PATH)
    return model, preprocessor, metadata


def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Medium"
    return "High"


def predict(input_data: dict | pd.DataFrame) -> dict:
    model, preprocessor, metadata = load_artifacts()
    frame = input_data if isinstance(input_data, pd.DataFrame) else pd.DataFrame([input_data])
    frame = frame.reindex(columns=metadata["feature_columns"])
    x_processed = preprocessor.transform(frame)
    probability = float(model.predict_proba(x_processed)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(x_processed)[0])
    label = int(probability >= 0.5)
    confidence = probability if label == 1 else 1 - probability
    return {
        "prediction": label,
        "probability": probability,
        "confidence": confidence,
        "risk_level": risk_level(probability),
    }
