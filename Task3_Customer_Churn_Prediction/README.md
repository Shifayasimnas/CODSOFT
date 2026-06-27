# Customer Churn Prediction

An end-to-end machine learning project for CodSoft Internship Task 3. The system automatically detects a churn dataset, builds EDA visualizations, trains and compares multiple classifiers, saves reusable model artifacts, and serves predictions through a professional Streamlit interface.

## Features

- Automatic CSV discovery from the `dataset/` folder, including zip extraction.
- Automatic churn target detection for columns such as `Churn`, `Exited`, and `Attrition`.
- Missing value handling, duplicate removal, categorical encoding, numerical scaling, and train-test split.
- EDA plots saved to `outputs/`.
- Model comparison across Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, KNN, and XGBoost when available.
- Saved artifacts in `models/`: trained model, preprocessor, metadata, and label encoder when needed.
- Streamlit app with sidebar navigation, manual customer input, confidence score, risk level, progress gauge, dataset summary, model metrics, and feature importance.

## Project Structure

```text
Task3_Customer_Churn_Prediction/
├── app.py
├── dataset/
│   ├── Churn_Modelling.csv
│   ├── archive (2).zip
│   └── src/
│       └── eda.py
├── models/
│   ├── model_metadata.json
│   ├── preprocessor.pkl
│   └── trained_model.pkl
├── notebooks/
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── feature_importance.csv
│   ├── missing_values.png
│   └── model_metrics.csv
├── report.md
├── requirements.txt
└── src/
    ├── config.py
    ├── data_utils.py
    ├── eda.py
    ├── inference.py
    └── train.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Train Models

```powershell
.\.venv\Scripts\python.exe .\src\train.py
```

## Run Streamlit App

```powershell
.\.venv\Scripts\streamlit.exe run .\app.py
```

## Outputs

- `outputs/model_metrics.csv`: model comparison table.
- `outputs/*.png`: EDA visualizations.
- `models/trained_model.pkl`: selected best model.
- `models/preprocessor.pkl`: preprocessing pipeline.
- `models/model_metadata.json`: dataset, feature, model, and UI metadata.

## Best Model Selection

The training script ranks models by F1 score, ROC-AUC, and recall. This balances general correctness with churn detection performance, which is important because missed churners are often more expensive than false alerts.
