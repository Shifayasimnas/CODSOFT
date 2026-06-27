# Customer Churn Prediction Report

## Objective

The goal of this project is to predict whether a customer is likely to churn using historical customer attributes. The workflow is designed to run with different customer churn CSV datasets without manual code changes.

## Dataset Handling

The project automatically scans the `dataset/` folder for CSV files and extracts zip archives when needed. It detects common churn target names such as `Churn`, `Exited`, and `Attrition`. For the current dataset, the detected target is `Exited`.

## Preprocessing

The preprocessing pipeline removes duplicate rows, drops high-cardinality identifier columns, imputes missing numerical and categorical values, scales numerical features, and one-hot encodes categorical features. This produces a consistent model-ready feature matrix while keeping inference reusable through the saved `preprocessor.pkl` artifact.

## Exploratory Data Analysis

The pipeline generates class distribution, missing value, correlation heatmap, and feature distribution plots. These visualizations are saved inside `outputs/` for review and presentation.

## Model Training

The following models are trained and compared:

- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- KNN
- XGBoost when available

Models are evaluated using accuracy, precision, recall, F1 score, and ROC-AUC. The best model is selected using F1 score as the primary ranking metric, followed by ROC-AUC and recall.

## Deployment

The Streamlit application provides a clean interface for manual customer input, churn probability, confidence percentage, risk level, dataset summary, model comparison, and feature importance. The app uses the same saved preprocessing and model artifacts used during training.

## Key Deliverables

- `models/trained_model.pkl`
- `models/preprocessor.pkl`
- `models/model_metadata.json`
- `outputs/model_metrics.csv`
- EDA plots in `outputs/`
- Streamlit application in `app.py`
