from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODELS_DIR / "trained_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"
