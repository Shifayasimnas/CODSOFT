from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FEATURE_IMPORTANCE_PATH, METRICS_PATH, METADATA_PATH
from src.data_utils import find_csv, load_dataset, read_json
from src.inference import predict


st.set_page_config(page_title="Customer Churn Prediction", page_icon="chart_with_upwards_trend", layout="wide")

st.markdown(
    """
    <style>
    .main {background: #f7f9fc;}
    .block-container {padding-top: 1.5rem;}
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: white;
        border: 1px solid #e6ebf2;
        box-shadow: 0 1px 6px rgba(24, 39, 75, 0.05);
    }
    .footer {text-align: center; color: #667085; padding: 2rem 0 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata():
    return read_json(METADATA_PATH)


@st.cache_data
def load_raw_data():
    return load_dataset(find_csv())


metadata = load_metadata()
df = load_raw_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Churn", "Dataset Summary", "Model Information", "Feature Importance"])

st.title("Customer Churn Prediction")
st.caption("Production-style machine learning workflow for identifying customers at risk of leaving.")

if page == "Predict Churn":
    st.subheader("Manual Customer Input")
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(2)
        for index, feature in enumerate(metadata["feature_columns"]):
            target_col = cols[index % 2]
            if feature in metadata["categorical_features"]:
                options = metadata["categorical_values"].get(feature, [])
                values[feature] = target_col.selectbox(feature, options or ["Unknown"])
            else:
                stats = metadata["numeric_ranges"].get(feature, {"min": 0.0, "max": 1.0, "median": 0.0})
                values[feature] = target_col.number_input(
                    feature,
                    value=float(stats["median"]),
                    min_value=float(stats["min"]),
                    max_value=float(stats["max"]),
                )
        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        result = predict(values)
        probability = result["probability"]
        st.divider()
        left, mid, right = st.columns(3)
        left.metric("Prediction", "Will Churn" if result["prediction"] else "Will Stay")
        mid.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        right.metric("Risk Level", result["risk_level"])
        st.progress(min(max(probability, 0.0), 1.0), text=f"Churn probability: {probability * 100:.1f}%")
        if result["risk_level"] == "High":
            st.error("High churn risk. Recommend retention outreach, service review, and targeted offer.")
        elif result["risk_level"] == "Medium":
            st.warning("Medium churn risk. Monitor customer behavior and consider proactive engagement.")
        else:
            st.success("Low churn risk. Customer profile appears stable.")

elif page == "Dataset Summary":
    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{metadata['rows_after_deduplication']:,}")
    c2.metric("Columns", metadata["columns"])
    c3.metric("Target", metadata["target_column"])
    c4.metric("Best Model", metadata["best_model"])
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("EDA Outputs")
    plot_cols = st.columns(2)
    plot_paths = [
        "outputs/class_distribution.png",
        "outputs/missing_values.png",
        "outputs/correlation_heatmap.png",
        "outputs/feature_distributions.png",
    ]
    for idx, plot_path in enumerate(plot_paths):
        if Path(plot_path).exists():
            plot_cols[idx % 2].image(plot_path, use_container_width=True)

elif page == "Model Information":
    st.subheader("Model Comparison")
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True)
    st.subheader("Selected Model")
    st.json(metadata["best_metrics"])
    st.info("XGBoost and SHAP are used automatically when installed. This project currently runs with the installed scikit-learn stack.")

elif page == "Feature Importance":
    st.subheader("Feature Importance")
    if FEATURE_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        st.bar_chart(importance.set_index("feature")["importance"])
        st.dataframe(importance, use_container_width=True)
    else:
        st.warning("Feature importance is not available for the selected estimator.")

st.markdown('<div class="footer">Built for CodSoft Internship Task 3 | Customer Churn Prediction</div>', unsafe_allow_html=True)
