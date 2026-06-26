import contextlib
import io
import importlib.util
import joblib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import plotly.express as px
import streamlit as st


def set_page_config() -> None:
    """Configure the Streamlit app page settings."""
    st.set_page_config(
        page_title="Smart AI SMS Shield",
        page_icon="🛡️",
        layout="centered",
        initial_sidebar_state="expanded",
    )


def load_styles() -> None:
    """Inject custom CSS styles for a modern UI."""
    st.markdown(
        """
        <style>
        .main-container {
            background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%);
            color: #0f172a;
            padding: 0 24px;
        }

        .header-card {
            border-radius: 24px;
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 36px;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
            color: white;
        }

        .header-title {
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 12px;
        }

        .header-subtitle {
            color: rgba(255, 255, 255, 0.88);
            font-size: 1.3rem;
            margin-bottom: 18px;
        }

        .description-text {
            color: #e2e8f0;
            font-size: 1rem;
            max-width: 760px;
        }

        .section-card {
            border-radius: 20px;
            background: white;
            padding: 28px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(226, 232, 240, 0.9);
        }

        .feature-card {
            border-radius: 18px;
            background: #0f172a;
            color: #ffffff;
            padding: 22px;
            min-height: 170px;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .feature-card h4 {
            margin-bottom: 12px;
            color: #93c5fd;
        }

        .feature-card p {
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.96rem;
            line-height: 1.6;
        }

        .footer {
            color: #475569;
            font-size: 0.94rem;
            margin-top: 42px;
            text-align: center;
        }

        @media (max-width: 768px) {
            .header-title {
                font-size: 2.6rem;
            }

            .header-card {
                padding: 28px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Additional styles for prediction card and progress
    st.markdown(
        """
        <style>
        .prediction-card {
            display: flex;
            gap: 20px;
            align-items: center;
            padding: 18px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(249,250,251,0.7));
            box-shadow: 0 10px 30px rgba(2,6,23,0.08);
            border: 1px solid rgba(15,23,42,0.04);
        }
        .prediction-label {
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .prediction-meta {
            color: #475569;
            font-size: 0.95rem;
        }
        .prediction-percent {
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 6px;
        }
        .progress-wrapper {
            width: 100%;
            height: 14px;
            background: rgba(15,23,42,0.06);
            border-radius: 12px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-bar {
            height: 100%;
            border-radius: 12px;
            transition: width 0.9s ease-out;
            box-shadow: inset 0 -4px 10px rgba(0,0,0,0.08);
        }
        .example-box {
            margin-top: 10px;
            color: #0f172a;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


BACKEND_FILE = Path("src") / "spam_detection.py"

@st.cache_resource
def load_backend_module():
    """Import the existing spam_detection backend and reuse its trained model."""
    backend_path = BACKEND_FILE.resolve()
    if not backend_path.exists():
        raise FileNotFoundError(f"Backend file not found: {backend_path}")

    spec = importlib.util.spec_from_file_location("spam_detection_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("Failed to load backend module spec")

    current_dir = Path.cwd()
    try:
        os.chdir(backend_path.parent)
        with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(module)
    finally:
        os.chdir(current_dir)

    return module


# --- Model persistence -------------------------------------------------
MODEL_DIR = Path("models")
NAIVE_BAYES_FILE = MODEL_DIR / "naive_bayes.pkl"
VECTORIZER_FILE = MODEL_DIR / "vectorizer.pkl"
LOGISTIC_FILE = MODEL_DIR / "logistic_regression.pkl"
MODEL_METADATA_FILE = MODEL_DIR / "artifact_version.txt"
MODEL_VERSION = 2


def is_current_model_version() -> bool:
    if not MODEL_METADATA_FILE.exists():
        return False
    try:
        return int(MODEL_METADATA_FILE.read_text().strip()) == MODEL_VERSION
    except ValueError:
        return False


def invalidate_saved_models() -> None:
    for path in [NAIVE_BAYES_FILE, VECTORIZER_FILE, LOGISTIC_FILE, MODEL_METADATA_FILE]:
        if path.exists():
            path.unlink()


@st.cache_resource
def get_model_and_vectorizer():
    """Return (model, vectorizer).

    If persisted artifacts exist under `models/` load them. Otherwise import
    the backend (`src/spam_detection.py`) which trains `model` and
    `vectorizer`, then persist those objects using joblib for future runs.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not is_current_model_version():
        invalidate_saved_models()

    # Load if both artifacts exist
    if NAIVE_BAYES_FILE.exists() and VECTORIZER_FILE.exists():
        model = joblib.load(NAIVE_BAYES_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer

    # Artifacts missing — import backend which performs training when executed
    backend = load_backend_module()
    if not hasattr(backend, "model") or not hasattr(backend, "vectorizer"):
        raise AttributeError("Backend did not expose `model` and `vectorizer` variables")

    model = backend.model
    vectorizer = backend.vectorizer

    # Persist for faster startup next time
    joblib.dump(model, NAIVE_BAYES_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    MODEL_METADATA_FILE.write_text(str(MODEL_VERSION))

    return model, vectorizer


def get_or_train_logistic(vectorizer):
    """Load persisted Logistic Regression or train and persist it using the same dataset.

    Training steps:
    1. Read `dataset/spam.csv` and clean columns (same as backend).
    2. Transform messages using the provided `vectorizer` (which is expected to be fitted).
    3. Split data (test_size=0.2, random_state=42) and train `LogisticRegression`.
    4. Persist the trained model to `models/logistic_regression.pkl`.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not is_current_model_version() and LOGISTIC_FILE.exists():
        LOGISTIC_FILE.unlink()

    if LOGISTIC_FILE.exists():
        return joblib.load(LOGISTIC_FILE)

    # Load dataset and preprocess (same as backend expectations)
    data_file = Path("dataset") / "spam.csv"
    data = pd.read_csv(data_file, encoding="latin-1")
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    data.columns = ["Label", "Message"]
    data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

    X = vectorizer.transform(data["Message"])
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
    clf.fit(X_train, y_train)

    joblib.dump(clf, LOGISTIC_FILE)
    MODEL_METADATA_FILE.write_text(str(MODEL_VERSION))
    return clf


def compute_accuracies(vectorizer, nb_model, logistic_model) -> Dict[str, float]:
    """Compute accuracy for both models on the shared test split and return dict."""
    data_file = Path("dataset") / "spam.csv"
    data = pd.read_csv(data_file, encoding="latin-1")
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    data.columns = ["Label", "Message"]
    data["Label"] = data["Label"].map({"ham": 0, "spam": 1})

    X = vectorizer.transform(data["Message"])
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_acc = accuracy_score(y_test, nb_model.predict(X_test))
    lr_acc = accuracy_score(y_test, logistic_model.predict(X_test))
    return {"Naive Bayes": float(nb_acc * 100), "Logistic Regression": float(lr_acc * 100)}


def predict_sms(message: str, model_choice: str = "Naive Bayes") -> tuple[str, float]:
    """Predict spam or ham and return (label, confidence_percent).

    Confidence is computed as: probability of predicted class * 100. We use
    `model.predict_proba()` to obtain class probabilities.
    """
    # Load base artifacts (Naive Bayes and vectorizer)
    nb_model, vectorizer = get_model_and_vectorizer()

    # Ensure logistic model is available if requested or for comparison
    logistic_model = get_or_train_logistic(vectorizer)

    # Choose model based on user selection
    if model_choice == "Logistic Regression":
        model = logistic_model
    else:
        model = nb_model

    vectorized = vectorizer.transform([message])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(probabilities[prediction] * 100)
    label = "🟢 HAM" if prediction == 0 else "🔴 SPAM"
    return label, confidence


def get_prediction_history() -> list[dict]:
    """Retrieve prediction history from the session state."""
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history


def add_history_entry(message: str, model: str, prediction: str, confidence: float) -> None:
    """Add a prediction entry to the session history."""
    history = get_prediction_history()
    history.append(
        {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "model": model,
        }
    )
    st.session_state.history = history


def delete_history_entry(index: int) -> None:
    """Delete a specific entry from the session history."""
    history = get_prediction_history()
    if 0 <= index < len(history):
        history.pop(index)
        st.session_state.history = history


def render_sidebar() -> str:
    """Build the app sidebar with navigation and project summary."""
    with st.sidebar:
        st.markdown("# Smart AI SMS Shield")
        selection = st.radio("Navigation", ["Home", "Dashboard"], index=0)
        st.markdown("---")
        st.markdown("## 🔍 About")
        st.write(
            "Smart AI SMS Shield is a clean frontend prototype for SMS spam detection. "
            "It is designed for future integration with your existing ML backend."
        )
        st.markdown("---")
        st.markdown("### 🎯 Goal")
        st.write("Showcase a modern interface while keeping the model connection separate.")
        st.markdown("---")
        st.markdown("### 💡 Primary Colors")
        st.write("Blue, White, Dark Gray")
    return selection


def render_header() -> None:
    """Render the page header with title, subtitle, and description."""
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='header-card'>"
        "<div class='header-title'>Smart AI SMS Shield</div>"
        "<div class='header-subtitle'>AI-Powered SMS Spam Detection System</div>"
        "<div class='description-text'>This application uses Machine Learning to classify SMS messages as Spam or Ham.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_feature_cards() -> None:
    """Render a row of feature cards below the header."""
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            "<div class='feature-card'>"
            "<h4>🛡️ Modern Design</h4>"
            "<p>Clean interface with strong visual hierarchy and premium styling.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            "<div class='feature-card'>"
            "<h4>⚡ Fast Prototype</h4>"
            "<p>Ready for integration with your ML backend while keeping the frontend polished.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            "<div class='feature-card'>"
            "<h4>📱 Smart UX</h4>"
            "<p>Responsive layout that scales nicely on desktop and smaller screens.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_dashboard() -> None:
    """Render a dashboard page with model metrics, counts and history."""
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### Dashboard")

    # Load data and get totals
    data_file = Path("dataset") / "spam.csv"
    data = pd.read_csv(data_file, encoding="latin-1")
    data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
    data.columns = ["Label", "Message"]
    spam_count = int((data["Label"] == "spam").sum())
    ham_count = int((data["Label"] == "ham").sum())
    total_messages = int(len(data))

    # Load models and accuracies
    nb_model, vectorizer = get_model_and_vectorizer()
    logistic_model = get_or_train_logistic(vectorizer)
    accuracies = compute_accuracies(vectorizer, nb_model, logistic_model)
    selected_model = st.session_state.get("model_choice", "Naive Bayes")
    selected_accuracy = accuracies[selected_model]

    # Metric cards
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Total Messages", f"{total_messages}")
    col2.metric("Spam Count", f"{spam_count}", delta=f"{spam_count/total_messages:.1%}" if total_messages else "0%")
    col3.metric("Ham Count", f"{ham_count}", delta=f"{ham_count/total_messages:.1%}" if total_messages else "0%")
    col4.metric("Selected Model", selected_model)

    st.markdown("---")
    st.markdown("### Model Accuracy")
    acc_df = pd.DataFrame(
        {
            "Model": ["Naive Bayes", "Logistic Regression"],
            "Accuracy": [accuracies["Naive Bayes"], accuracies["Logistic Regression"]],
        }
    )
    fig = px.bar(
        acc_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        color_discrete_map={"Naive Bayes": "#3b82f6", "Logistic Regression": "#2563eb"},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0, 100]), uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Prediction History")
    history = get_prediction_history()
    if history:
        history_df = pd.DataFrame(history)
        history_df = history_df[["date", "message", "model", "prediction", "confidence"]]
        history_df.columns = ["Date", "Message", "Model", "Prediction", "Confidence"]

        st.markdown("#### Controls")
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Clear All History", key="clear_history"):
                st.session_state.history = []
                st.experimental_rerun()
        with cols[1]:
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="sms_prediction_history.csv",
                mime="text/csv",
                key="download_history",
            )
        with cols[2]:
            st.info(
                "History is stored for this session. Use Clear All History or download a CSV for offline review."
            )

        st.markdown("#### Recent Predictions")
        for idx, row in history_df.iterrows():
            container = st.container()
            row_cols = container.columns([3, 1, 1, 1, 1])
            row_cols[0].markdown(
                f"**{row['Date']}**<br><span style='color:#334155'>{row['Message']}</span>",
                unsafe_allow_html=True,
            )
            row_cols[1].markdown(f"**{row['Model']}**")
            row_cols[2].markdown(f"**{row['Prediction']}**")
            row_cols[3].markdown(f"**{row['Confidence']}**")
            if row_cols[4].button("Delete", key=f"delete_history_{idx}"):
                delete_history_entry(idx)
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("#### Full History Table")
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No prediction history yet. Make predictions on the Home page to populate this dashboard.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_input_section() -> None:
    """Render the main SMS input form and detection button."""
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### Enter Your SMS Message")
    sms_text = st.text_area(
        "SMS Message",
        placeholder="Type or paste your SMS here...",
        height=240,
        key="sms_input",
        label_visibility="hidden",
    )

    # Model selection dropdown (preserve original UI layout)
    nb_model, vect = get_model_and_vectorizer()
    # ensure logistic exists for comparison
    lr_model = get_or_train_logistic(vect)
    accuracies = compute_accuracies(vect, nb_model, lr_model)

    # Render accuracy comparison table with highlighted best model
    best_model = max(accuracies, key=accuracies.get)
    table_html = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:8px;border-bottom:1px solid #ddd'>Model</th>"
        "<th style='text-align:right;padding:8px;border-bottom:1px solid #ddd'>Accuracy</th></tr>"
    )
    for name, acc in accuracies.items():
        if name == best_model:
            row_style = "background:#ecfdf5;font-weight:700;color:#065f46"
        else:
            row_style = ""
        table_html += (
            f"<tr style='{row_style}'><td style='padding:8px'>{name}</td>"
            f"<td style='padding:8px;text-align:right'>{acc:.2f}%</td></tr>"
        )
    table_html += "</table>"
    st.markdown("### Accuracy Comparison")
    st.markdown(table_html, unsafe_allow_html=True)

    model_choice = st.selectbox("Choose model", ["Naive Bayes", "Logistic Regression"], index=0, key="model_choice")

    detect_clicked = st.button("Detect", key="detect_button")
    if detect_clicked:
        if not sms_text or not sms_text.strip():
            st.warning("Please enter an SMS message before detecting.")
        else:
                        prediction_label, confidence = predict_sms(sms_text.strip(), model_choice=model_choice)
                        add_history_entry(sms_text.strip(), model_choice, prediction_label, confidence)

                        # Enhanced prediction card with confidence and progress
                        st.markdown("### Prediction")
                        # derive simple class and styling
                        if "SPAM" in prediction_label:
                                cls = "SPAM"
                                color = "#ef4444"
                                emoji = "🔴"
                                grad = f"linear-gradient(90deg, {color}, #fb7185)"
                        else:
                                cls = "HAM"
                                color = "#10b981"
                                emoji = "🟢"
                                grad = f"linear-gradient(90deg, {color}, #34d399)"

                        percent_text = f"{confidence:.2f}%"
                        bar_width = max(0, min(100, confidence))

                        card_html = f"""
                        <div class='prediction-card'>
                            <div style='min-width:180px'>
                                <div class='prediction-label' style='color:{color}'>{emoji} {cls}</div>
                                <div class='prediction-meta'>Confidence</div>
                                <div class='prediction-percent'>{percent_text}</div>
                            </div>
                            <div style='flex:1'>
                                <div class='progress-wrapper' aria-hidden='true'>
                                    <div class='progress-bar' style='width:{bar_width}%; background:{grad};'></div>
                                </div>
                                <div class='example-box'>Example Prediction: <strong>{emoji} {cls}</strong></div>
                            </div>
                        </div>
                        """

                        st.markdown(card_html, unsafe_allow_html=True)
                        # Also show native progress for accessibility
                        st.progress(int(bar_width) / 100)

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    """Render the page footer message."""
    st.markdown("<div class='footer'>Made by Shifaya Simnas</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Main app entry point."""
    set_page_config()
    load_styles()
    page = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    else:
        render_header()
        render_feature_cards()
        render_input_section()

    render_footer()


if __name__ == "__main__":
    main()
