# Credit Card Fraud Detection

## Project Overview

This project is a production-ready credit card fraud detection system built using Python, scikit-learn, imbalanced learning techniques, and Streamlit for visualization and scoring. It provides an end-to-end pipeline from dataset ingestion and feature engineering to model training, validation, and live scoring.

## Features

- Data preprocessing and feature engineering for transaction, customer, merchant, and location signals
- Imbalance-handling using SMOTE to augment minority fraud cases
- Multiple candidate classifiers with model comparison
- Threshold optimization and probability calibration for risk scoring
- Streamlit dashboard for single and batch prediction
- Prediction history logging and monitoring of model performance
- Responsive UI with model metadata and explainability pages

## Folder Structure

- `app.py/streamlit_app.py` - Streamlit dashboard entrypoint
- `dataset/` - Raw fraud training and test datasets
- `models/` - Persisted model artifacts created by training
- `outputs/` - Generated metrics, curves, processed dataset, and prediction history
- `src/` - Core application modules:
  - `config.py` - Configuration and path management
  - `data_preprocessing.py` - Dataset diagnostics and cleanup utilities
  - `feature_engineering.py` - Feature generation and processing pipeline
  - `model_training.py` - Training pipeline and artifact persistence
  - `inference.py` - Prediction and model loading helpers
- `scripts/` - Data inspection utilities

## Dataset

The pipeline expects the following files inside `dataset/`:

- `fraudTrain.csv` - Training transactions
- `fraudTest.csv` - Test transactions

The dataset is merged and processed for model-ready numeric features.

## Model Comparison Table

The training output includes a comparison table in `outputs/model_comparison.csv` with metrics for each candidate model:

- Model
- Accuracy
- Precision
- Recall
- F1
- ROC AUC
- Average Precision
- Threshold
- Cross-validation metrics

## Performance Metrics

Current validation metrics from `outputs/model_metrics.json`:

- Best Model: `Random Forest`
- Accuracy: `0.998151`
- Precision: `0.874774`
- Recall: `0.752850`
- F1 Score: `0.809245`
- ROC AUC: `0.997932`
- Precision-Recall AUC: `0.875621`

> These metrics reflect the trained model artifacts already present in the repository outputs.

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python src/model_training.py
```

### Run the dashboard

```bash
streamlit run app.py/streamlit_app.py
```

## Screenshots

Add screenshots of the dashboard, prediction page, and performance charts here.

## Future Improvements

- Add a full hyperparameter tuning workflow with experiment tracking
- Introduce SHAP explanations and local interpretability
- Add automated dataset drift and model drift monitoring
- Improve production deployment with Docker and CI/CD
- Add unit tests and validation checks for data contracts
