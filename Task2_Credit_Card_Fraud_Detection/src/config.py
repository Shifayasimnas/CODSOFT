"""Configuration management for the fraud detection system.

Centralized configuration for paths, model parameters, and application settings.
This module provides a single source of truth for all configuration values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class Paths:
    """Centralized path management for the fraud detection system."""

    ROOT_DIR: ClassVar[Path] = Path(__file__).resolve().parents[1]
    DATASET_DIR: ClassVar[Path] = ROOT_DIR / "dataset"
    OUTPUT_DIR: ClassVar[Path] = ROOT_DIR / "outputs"
    MODEL_DIR: ClassVar[Path] = ROOT_DIR / "models"
    PLOTS_DIR: ClassVar[Path] = OUTPUT_DIR / "plots"
    REPORTS_DIR: ClassVar[Path] = ROOT_DIR / "reports"

    # Dataset files
    FRAUD_TRAIN_FILE: ClassVar[Path] = DATASET_DIR / "fraudTrain.csv"
    FRAUD_TEST_FILE: ClassVar[Path] = DATASET_DIR / "fraudTest.csv"

    # Output files
    PROCESSED_DATASET: ClassVar[Path] = OUTPUT_DIR / "processed_dataset.csv"
    EDA_SUMMARY: ClassVar[Path] = OUTPUT_DIR / "eda_summary.txt"

    # Model artifacts
    BEST_MODEL: ClassVar[Path] = MODEL_DIR / "best_model.pkl"
    SCALER: ClassVar[Path] = MODEL_DIR / "scaler.pkl"
    FEATURE_NAMES: ClassVar[Path] = MODEL_DIR / "feature_names.pkl"
    MODEL_METADATA: ClassVar[Path] = MODEL_DIR / "model_metadata.json"
    MODEL_METRICS: ClassVar[Path] = OUTPUT_DIR / "model_metrics.json"

    # Reports
    MODEL_COMPARISON: ClassVar[Path] = OUTPUT_DIR / "model_comparison.csv"
    TRAINING_SUMMARY: ClassVar[Path] = OUTPUT_DIR / "training_summary.txt"
    PREDICTION_HISTORY: ClassVar[Path] = OUTPUT_DIR / "prediction_history.csv"

    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories if they don't exist."""
        for directory in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.PLOTS_DIR, cls.REPORTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Machine learning model configuration."""

    TARGET_COLUMN: str = "is_fraud"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    STRATIFY: bool = True

    # SMOTE Configuration
    SMOTE_RANDOM_STATE: int = 42
    SMOTE_SAMPLING_STRATEGY: float = 0.5

    # Scaler Configuration
    SCALER_TYPE: str = "standard"  # "standard", "minmax", "robust"

    # Hyperparameter tuning
    ENABLE_HYPERPARAMETER_TUNING: bool = True
    TUNING_CV_FOLDS: int = 3
    N_ITER_SEARCH: int = 10

    # Threshold optimization
    ENABLE_THRESHOLD_OPTIMIZATION: bool = True
    THRESHOLD_METRIC: str = "f1"  # "precision", "recall", "f1"

    # Probability calibration
    ENABLE_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = "sigmoid"  # "sigmoid", "isotonic"

    # Model selection
    MODELS_TO_TRAIN: list = None

    def __post_init__(self):
        """Initialize default models."""
        if self.MODELS_TO_TRAIN is None:
            self.MODELS_TO_TRAIN = [
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "xgboost",
            ]


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""

    # Time features
    EXTRACT_TIME_FEATURES: bool = True
    TIME_FEATURES: list = None

    # Amount features
    EXTRACT_AMOUNT_FEATURES: bool = True
    AMOUNT_QUANTILES: list = None

    # Customer features
    EXTRACT_CUSTOMER_FEATURES: bool = True

    # Merchant features
    EXTRACT_MERCHANT_FEATURES: bool = True

    # Location features
    EXTRACT_LOCATION_FEATURES: bool = True

    # Drop original datetime columns after feature extraction
    DROP_ORIGINAL_DATETIME: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.TIME_FEATURES is None:
            self.TIME_FEATURES = [
                "transaction_hour",
                "transaction_day",
                "transaction_month",
                "transaction_weekday",
                "is_weekend",
                "time_of_day",
                "is_late_night",
            ]
        if self.AMOUNT_QUANTILES is None:
            self.AMOUNT_QUANTILES = [0.25, 0.5, 0.75, 0.95]


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration."""

    PAGE_TITLE: str = "🚨 Credit Card Fraud Detection System"
    PAGE_ICON: str = "🏦"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"

    # Theme
    THEME_PRIMARY_COLOR: str = "#EF553B"
    THEME_BACKGROUND_COLOR: str = "#F8F9FA"
    THEME_SECONDARY_BACKGROUND_COLOR: str = "#FFFFFF"
    THEME_TEXT_COLOR: str = "#262730"

    # UI Settings
    MAX_UPLOAD_SIZE_MB: int = 100
    PREDICTION_THRESHOLD: float = 0.5
    SHOW_MODEL_DETAILS: bool = True
    SHOW_FEATURE_IMPORTANCE: bool = True
    SHOW_SHAP_EXPLANATIONS: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "fraud_detection.log"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True

    # Log rotation
    ENABLE_LOG_ROTATION: bool = True
    MAX_LOG_FILE_SIZE_MB: int = 10
    BACKUP_COUNT: int = 5


@dataclass
class ExplainabilityConfig:
    """Explainability (SHAP) configuration."""

    USE_SHAP: bool = True
    SHAP_EXPLAINER_TYPE: str = "auto"  # "auto", "tree", "kernel", "linear"
    SHAP_SAMPLE_SIZE: int = 100
    TOP_FEATURES_TO_SHOW: int = 10


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    # Plotly configuration
    TEMPLATE: str = "plotly_white"
    COLOR_SCALE: str = "RdYlGn_r"
    FONT_SIZE: int = 12
    FIGURE_HEIGHT: int = 600
    FIGURE_WIDTH: int = 1000

    # Chart specific
    ROC_CURVE_STYLE: str = "plotly"
    CONFUSION_MATRIX_STYLE: str = "heatmap"
    FEATURE_IMPORTANCE_STYLE: str = "bar"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    # Caching
    ENABLE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 3600

    # Batch processing
    BATCH_SIZE: int = 1000
    N_JOBS: int = -1  # -1 uses all available cores

    # Memory management
    REDUCE_MEMORY_USAGE: bool = True


class Config:
    """Main configuration class aggregating all sub-configurations."""

    paths: Paths = Paths()
    model: ModelConfig = ModelConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()
    explainability: ExplainabilityConfig = ExplainabilityConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    performance: PerformanceConfig = PerformanceConfig()

    @classmethod
    def initialize(cls) -> None:
        """Initialize configuration and create necessary directories."""
        Paths.create_directories()

    @classmethod
    def to_dict(cls) -> dict:
        """Convert all configurations to dictionary format."""
        return {
            "paths": {k: str(v) for k, v in vars(cls.paths).items()},
            "model": vars(cls.model),
            "feature_engineering": vars(cls.feature_engineering),
            "dashboard": vars(cls.dashboard),
            "logging": vars(cls.logging),
            "explainability": vars(cls.explainability),
            "visualization": vars(cls.visualization),
            "performance": vars(cls.performance),
        }


# Environment-specific configuration override
def load_env_config() -> None:
    """Load configuration from environment variables (optional)."""
    env = os.getenv("FRAUD_DETECTION_ENV", "development")
    
    if env == "production":
        Config.logging.LOG_LEVEL = "WARNING"
        Config.performance.ENABLE_CACHING = True
        Config.model.ENABLE_HYPERPARAMETER_TUNING = False
    elif env == "development":
        Config.logging.LOG_LEVEL = "DEBUG"
        Config.performance.ENABLE_CACHING = False


# Initialize on import
Config.initialize()
load_env_config()
