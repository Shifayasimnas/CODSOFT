"""Enterprise Streamlit dashboard for credit card fraud detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import Config
from src.inference import append_prediction_history, load_model_bundle, predict


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "ink": "#111827",
    "muted": "#6B7280",
    "blue": "#2563EB",
    "surface": "#F8FAFC",
}


def inject_css() -> None:
    """Apply compact enterprise styling."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.65rem;}
        .risk-card {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 1rem;
            background: #FFFFFF;
        }
        .risk-title {font-size: .85rem; color: #6B7280; margin-bottom: .25rem;}
        .risk-value {font-size: 1.6rem; font-weight: 700; color: #111827;}
        .small-muted {font-size: .85rem; color: #6B7280;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=900)
def load_dataset_sample(max_rows: int = 120_000) -> pd.DataFrame:
    """Load a bounded sample for responsive dashboard charts."""
    source = Config.paths.FRAUD_TRAIN_FILE
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source, nrows=max_rows)


@st.cache_data(ttl=300)
def load_json(path: Path) -> Dict:
    """Load JSON if present."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=300)
def load_comparison() -> pd.DataFrame:
    """Load model comparison metrics."""
    if Config.paths.MODEL_COMPARISON.exists():
        return pd.read_csv(Config.paths.MODEL_COMPARISON)
    return pd.DataFrame()


@st.cache_resource
def cached_bundle():
    """Cache model artifacts for low-latency predictions."""
    return load_model_bundle()


def require_model():
    """Load model artifacts or show a helpful setup message."""
    try:
        return cached_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `python src/feature_engineering.py` and then `python src/model_training.py` before using predictions.")
        st.stop()


def fraud_column(df: pd.DataFrame) -> str:
    for column in ["is_fraud", "Class", "isFraud"]:
        if column in df.columns:
            return column
    return "is_fraud"


def amount_column(df: pd.DataFrame) -> str:
    for column in ["amt", "Amount", "amount"]:
        if column in df.columns:
            return column
    return ""


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def kpi_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-title">{label}</div>
            <div class="risk-value">{value}</div>
            <div class="small-muted">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_page() -> None:
    header("Fraud Command Center", "Portfolio-grade monitoring view for transaction risk, model readiness, and fraud patterns.")
    df = load_dataset_sample()
    metrics = load_json(Config.paths.MODEL_METRICS)

    if df.empty:
        st.warning("Dataset files are missing. Add `dataset/fraudTrain.csv` to populate analytics.")
        return

    target = fraud_column(df)
    amount = amount_column(df)
    total = len(df)
    frauds = int((df[target] == 1).sum()) if target in df else 0
    fraud_rate = (frauds / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{frauds:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.3f}%")
    c4.metric("Model F1", f"{metrics.get('f1', 0):.3f}")

    left, right = st.columns([1, 1])
    with left:
        counts = df[target].map({0: "Legitimate", 1: "Fraud"}).value_counts().reset_index()
        counts.columns = ["class", "count"]
        fig = px.bar(counts, x="class", y="count", color="class", color_discrete_map={"Legitimate": PALETTE["success"], "Fraud": PALETTE["danger"]})
        fig.update_layout(title="Fraud vs Legitimate", showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if amount:
            fig = px.histogram(df, x=amount, color=target, nbins=80, marginal="box", color_discrete_sequence=[PALETTE["blue"], PALETTE["danger"]])
            fig.update_layout(title="Amount Distribution", height=360)
            st.plotly_chart(fig, use_container_width=True)

    if "trans_date_trans_time" in df.columns:
        time_df = df.copy()
        time_df["trans_date_trans_time"] = pd.to_datetime(time_df["trans_date_trans_time"], errors="coerce")
        time_df["hour"] = time_df["trans_date_trans_time"].dt.hour
        trend = time_df.groupby(["hour", target]).size().reset_index(name="transactions")
        fig = px.line(trend, x="hour", y="transactions", color=target, markers=True, color_discrete_sequence=[PALETTE["success"], PALETTE["danger"]])
        fig.update_layout(title="Transaction Trend by Hour", height=380)
        st.plotly_chart(fig, use_container_width=True)

    if "category" in df.columns:
        category = df.groupby("category")[target].agg(["count", "sum"]).reset_index()
        category["fraud_rate"] = category["sum"] / category["count"]
        category = category.sort_values("fraud_rate", ascending=False).head(15)
        fig = px.bar(category, x="fraud_rate", y="category", orientation="h", color="fraud_rate", color_continuous_scale="Reds")
        fig.update_layout(title="Highest-Risk Merchant Categories", height=430, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def single_prediction_page() -> None:
    bundle = require_model()
    header("Real-Time Transaction Scoring", "Score one transaction using the exact trained model feature contract.")

    st.sidebar.subheader("Risk Controls")
    threshold = st.sidebar.slider("Decision threshold", 0.01, 0.99, float(bundle.threshold), 0.01)

    defaults = {feature: 0.0 for feature in bundle.feature_names}
    if Config.paths.PROCESSED_DATASET.exists():
        sample = pd.read_csv(Config.paths.PROCESSED_DATASET, nrows=1)
        for feature in bundle.feature_names:
            if feature in sample.columns:
                defaults[feature] = float(pd.to_numeric(sample[feature], errors="coerce").fillna(0).iloc[0])

    with st.form("single_prediction_form"):
        st.write("Transaction Feature Inputs")
        columns = st.columns(3)
        values = {}
        for index, feature in enumerate(bundle.feature_names):
            values[feature] = columns[index % 3].number_input(feature, value=float(defaults[feature]), format="%.6f")
        submitted = st.form_submit_button("Score Transaction", type="primary")

    if submitted:
        with st.spinner("Scoring transaction..."):
            result = predict(pd.DataFrame([values]), bundle)
        probability = float(result["fraud_probability"].iloc[0])
        label = "Fraud" if probability >= threshold else "Legitimate"
        confidence = max(probability, 1 - probability)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label)
        c2.metric("Fraud Probability", f"{probability:.2%}")
        c3.metric("Confidence", f"{confidence:.2%}")
        st.progress(min(max(probability, 0.0), 1.0), text=f"Risk indicator: {probability:.2%}")

        gauge_color = PALETTE["danger"] if probability >= threshold else PALETTE["success"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": gauge_color}, "threshold": {"line": {"color": PALETTE["danger"], "width": 4}, "value": threshold * 100}},
            title={"text": "Fraud Probability Meter"},
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        append_prediction_history(result)


def batch_prediction_page() -> None:
    bundle = require_model()
    header("Batch Prediction", "Upload model-ready CSV files, validate them safely, and download scored transactions.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("CSV must include trained numeric feature columns. Missing columns are filled with 0 and invalid values are sanitized.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Invalid CSV file: {error}")
        return

    with st.spinner("Running batch scoring..."):
        try:
            scored = predict(batch_df, bundle)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

    append_prediction_history(scored)
    st.success(f"Scored {len(scored):,} transactions.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged Fraud", f"{int((scored['prediction'] == 1).sum()):,}")
    c2.metric("Average Risk", f"{scored['fraud_probability'].mean():.2%}")
    c3.metric("Critical Risk", f"{int((scored['risk_band'] == 'Critical').sum()):,}")

    fig = px.histogram(scored, x="fraud_probability", color="risk_band", nbins=50, color_discrete_sequence=[PALETTE["success"], PALETTE["warning"], PALETTE["blue"], PALETTE["danger"]])
    fig.update_layout(title="Batch Risk Distribution", height=360)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(scored.head(200), use_container_width=True)
    st.download_button("Download scored CSV", scored.to_csv(index=False), file_name="fraud_predictions.csv", mime="text/csv")


def model_performance_page() -> None:
    header("Model Performance", "Validation metrics, model comparison, ROC curve, precision-recall curve, and confusion matrix.")
    metrics = load_json(Config.paths.MODEL_METRICS)
    curves = load_json(Config.paths.MODEL_CURVES if hasattr(Config.paths, "MODEL_CURVES") else Config.paths.OUTPUT_DIR / "model_curves.json")
    comparison = load_comparison()

    if not metrics:
        st.warning("Model metrics are not available yet. Train the model to populate this page.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    c2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    c3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    c4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")

    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True)
        fig = px.bar(comparison, x="Model", y=["Precision", "Recall", "F1", "ROC AUC", "Average Precision"], barmode="group")
        fig.update_layout(title="Model Comparison", height=420)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    if curves.get("roc"):
        roc = curves["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"}))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360)
        left.plotly_chart(fig, use_container_width=True)
    if curves.get("precision_recall"):
        pr = curves["precision_recall"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR"))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=360)
        right.plotly_chart(fig, use_container_width=True)

    cm = metrics.get("confusion_matrix", {})
    if cm:
        matrix = [[cm.get("true_negatives", 0), cm.get("false_positives", 0)], [cm.get("false_negatives", 0), cm.get("true_positives", 0)]]
        fig = px.imshow(matrix, text_auto=True, labels={"x": "Predicted", "y": "Actual"}, x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"], color_continuous_scale="Blues")
        fig.update_layout(title="Confusion Matrix", height=380)
        st.plotly_chart(fig, use_container_width=True)


def explainability_page() -> None:
    header("Explainability", "Feature importance and local contribution approximations for transparent fraud decisions.")
    metrics = load_json(Config.paths.MODEL_METRICS)
    importance = pd.DataFrame(metrics.get("feature_importance", []))
    if importance.empty:
        st.warning("Feature importance is not available. Train the model first.")
        return
    fig = px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Teal")
    fig.update_layout(title="Top Model Features", height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.info("SHAP is optional in this environment. This page uses native model importances when SHAP is not installed.")


def history_page() -> None:
    header("Prediction History", "Audit recent scores, search transactions, and export results.")
    path = Config.paths.PREDICTION_HISTORY
    if not path.exists():
        st.info("No predictions have been recorded yet.")
        return
    history = pd.read_csv(path)
    query = st.text_input("Search history")
    if query:
        mask = history.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        history = history[mask]
    st.dataframe(history.tail(500), use_container_width=True)
    st.download_button("Download history", history.to_csv(index=False), file_name="prediction_history.csv", mime="text/csv")


def health_page() -> None:
    header("System Health", "Deployment readiness checks for datasets, model artifacts, metrics, and configuration.")
    checks = {
        "Training dataset": Config.paths.FRAUD_TRAIN_FILE.exists(),
        "Test dataset": Config.paths.FRAUD_TEST_FILE.exists(),
        "Processed dataset": Config.paths.PROCESSED_DATASET.exists(),
        "Best model": Config.paths.BEST_MODEL.exists(),
        "Scaler": Config.paths.SCALER.exists(),
        "Feature names": Config.paths.FEATURE_NAMES.exists(),
        "Model metadata": Config.paths.MODEL_METADATA.exists(),
        "Metrics": Config.paths.MODEL_METRICS.exists(),
    }
    status_df = pd.DataFrame({"check": checks.keys(), "status": ["Ready" if value else "Missing" for value in checks.values()]})
    st.dataframe(status_df, use_container_width=True)
    ready = sum(checks.values())
    st.progress(ready / len(checks), text=f"{ready}/{len(checks)} deployment checks ready")

    metadata = load_json(Config.paths.MODEL_METADATA)
    if metadata:
        st.json(metadata)


def about_page() -> None:
    header("About", "Enterprise-style fraud detection system for internships, GitHub, and portfolio demonstrations.")
    st.write(
        """
        This application combines fraud-focused feature engineering, imbalance-aware model training,
        calibrated risk scoring, threshold optimization, validation metrics, batch scoring,
        prediction history, and deployment readiness checks.
        """
    )


def main() -> None:
    inject_css()
    pages = {
        "Overview": overview_page,
        "Single Prediction": single_prediction_page,
        "Batch Prediction": batch_prediction_page,
        "Model Performance": model_performance_page,
        "Explainability": explainability_page,
        "History": history_page,
        "System Health": health_page,
        "About": about_page,
    }

    st.sidebar.title("Fraud AI Console")
    st.sidebar.caption("Credit Card Fraud Detection")
    page = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.divider()
    st.sidebar.write("Model threshold and metadata are loaded from saved artifacts after training.")
    pages[page]()


if __name__ == "__main__":
    main()
