"""
Fake News Detection - Streamlit Web Application
Run: streamlit run app/app.py
"""

import os
import sys
import json
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import ModelRegistry
from src.sentiment import analyze_sentiment
from src.explainability import (
    explain_baseline_prediction, detect_bias,
    assess_source_credibility, generate_word_importance_html,
    plot_word_importance
)
from src.utils import MODELS_DIR, LOGS_DIR, log_prediction

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",  # noqa
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .fake-badge {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white; padding: 15px 30px; border-radius: 12px;
        font-size: 28px; font-weight: bold; text-align: center;
        box-shadow: 0 4px 15px rgba(255,0,0,0.3);
    }
    .real-badge {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white; padding: 15px 30px; border-radius: 12px;
        font-size: 28px; font-weight: bold; text-align: center;
        box-shadow: 0 4px 15px rgba(0,200,0,0.3);
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 20px; border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Model Performance", "Explainability Dashboard", "Admin Panel"]
)

# Available models
available_models = ModelRegistry.get_available_models()
if not available_models:
    st.sidebar.warning("No trained models found. Train a model first!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Available Models:**")
model_labels = {
    'baseline': 'TF-IDF + Naive Bayes',
    'lstm': 'LSTM (Deep Learning)',
    'bert': 'DistilBERT (Transformer)'
}
for m in available_models:
    st.sidebar.markdown(f"- {model_labels.get(m, m)}")

if not available_models:
    st.sidebar.info(
        "To train the baseline model:\n"
        "1. Place Fake.csv & True.csv in data/raw/\n"
        "2. Run: `python -m src.train_baseline`"
    )


# ============================================================
# PAGE 1: PREDICTION
# ============================================================
def prediction_page():
    st.title("Fake News Detector")
    st.markdown("Analyze news articles to determine if they are **real** or **fake**.")

    if not available_models:
        st.error("No trained models available. Please train a model first.")
        return

    # Model selector
    col1, col2 = st.columns([3, 1])
    with col2:
        model_choice = st.selectbox(
            "Select Model",
            available_models,
            format_func=lambda x: model_labels.get(x, x)
        )

    # Input tabs
    tab_text, tab_file = st.tabs(["Enter Text", "Upload File"])

    input_text = ""

    with tab_text:
        headline = st.text_input("News Headline (optional)", placeholder="Enter headline...")
        body = st.text_area(
            "News Article Text",
            height=200,
            placeholder="Paste the full news article here..."
        )
        if headline and body:
            input_text = f"{headline} {body}"
        elif body:
            input_text = body
        elif headline:
            input_text = headline

    with tab_file:
        uploaded_file = st.file_uploader("Upload a .txt file", type=['txt'])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode('utf-8', errors='ignore')
            st.text_area("File Contents Preview", input_text[:1000], height=150, disabled=True)

    # Analyze button
    if st.button("Analyze Article", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text or upload a file.")
            return

        with st.spinner("Analyzing..."):
            # Get prediction
            try:
                predictor = ModelRegistry.get_predictor(model_choice)
                result = predictor.predict(input_text)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

            # Get sentiment
            sentiment = analyze_sentiment(input_text)

            # Log prediction
            log_prediction(
                input_text[:200], result['label'], result['confidence'],
                model_choice, sentiment['compound']
            )

        # ---- Display Results ----
        st.markdown("---")

        # Main prediction badge
        col_pred, col_conf = st.columns([2, 1])
        with col_pred:
            if result['label'] == 'Fake':
                st.markdown(
                    '<div class="fake-badge">FAKE NEWS</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="real-badge">REAL NEWS</div>',
                    unsafe_allow_html=True
                )

        with col_conf:
            confidence = result['confidence']
            if confidence >= 0.8:
                conf_class = "confidence-high"
            elif confidence >= 0.6:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"

            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <p class="{conf_class}" style="font-size: 32px;">
                    {confidence:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("#### Probability Distribution")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Fake Probability", f"{result['probabilities']['Fake']:.1%}")
            st.progress(result['probabilities']['Fake'])
        with prob_col2:
            st.metric("Real Probability", f"{result['probabilities']['Real']:.1%}")
            st.progress(result['probabilities']['Real'])

        # Explanation section
        st.markdown("---")
        exp_col, sent_col = st.columns(2)

        with exp_col:
            st.markdown("#### Why This Prediction?")
            if model_choice == 'baseline':
                explanation = explain_baseline_prediction(input_text, predictor, top_n=10)
                if explanation['top_words']:
                    html = generate_word_importance_html(explanation['top_words'])
                    st.markdown(html, unsafe_allow_html=True)

                    st.markdown("**Top Contributing Words:**")
                    for item in explanation['top_words'][:7]:
                        icon = "🔴" if 'Fake' in item['direction'] else "🟢"
                        st.markdown(
                            f"- {icon} **{item['word']}** "
                            f"(score: {item['score']:.4f}, {item['direction']})"
                        )
                else:
                    st.info("No significant features found for this text.")
            else:
                st.info(f"Detailed explanations for {model_labels.get(model_choice)} "
                        f"are available after training.")

        with sent_col:
            st.markdown("#### Sentiment Analysis")
            sent_label = sentiment['label']
            if sent_label == 'Positive':
                st.success(f"Sentiment: {sent_label}")
            elif sent_label == 'Negative':
                st.error(f"Sentiment: {sent_label}")
            else:
                st.info(f"Sentiment: {sent_label}")

            sent_cols = st.columns(3)
            sent_cols[0].metric("Positive", f"{sentiment['positive']:.1%}")
            sent_cols[1].metric("Negative", f"{sentiment['negative']:.1%}")
            sent_cols[2].metric("Neutral", f"{sentiment['neutral']:.1%}")

            st.caption(sentiment['interpretation'])

        # Bias & Source Credibility
        st.markdown("---")
        bias_col, source_col = st.columns(2)

        with bias_col:
            st.markdown("#### Bias Detection")
            bias = detect_bias(input_text)
            bias_score = bias['score']

            st.markdown(f"**Political Leaning:** {bias['label']}")
            # Visual bias meter
            meter_val = (bias_score + 1) / 2  # Convert -1..+1 to 0..1
            st.slider(
                "Bias Spectrum (Left ← → Right)",
                0.0, 1.0, meter_val, disabled=True
            )
            st.caption(
                f"Left keywords: {bias['left_count']} | "
                f"Right keywords: {bias['right_count']}"
            )

        with source_col:
            st.markdown("#### Source Credibility")
            credibility = assess_source_credibility(input_text)

            score = credibility['credibility_score']
            if score >= 0.8:
                st.success(f"Credibility Score: {score:.1f}/1.0")
            elif score >= 0.5:
                st.warning(f"Credibility Score: {score:.1f}/1.0")
            else:
                st.error(f"Credibility Score: {score:.1f}/1.0")

            st.caption(credibility['assessment'])
            if credibility['domains_found']:
                for d in credibility['domains_found']:
                    st.markdown(f"- {d}")


# ============================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================
def model_performance_page():
    st.title("Model Performance")
    st.markdown("Compare evaluation metrics across all trained models.")

    if not available_models:
        st.warning("No trained models available.")
        return

    # Load metrics for each model
    all_metrics = {}
    for model_name in available_models:
        metrics_path = os.path.join(MODELS_DIR, model_name, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_metrics[model_name] = json.load(f)

    if not all_metrics:
        st.info("No evaluation metrics found. Train models to generate metrics.")
        return

    # Metrics comparison table
    st.markdown("### Metrics Summary")
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {'Model': model_labels.get(model_name, model_name)}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = f"{v:.4f}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Show evaluation plots
    st.markdown("### Evaluation Plots")
    for model_name in available_models:
        eval_dir = os.path.join(MODELS_DIR, model_name, 'evaluation')
        if not os.path.exists(eval_dir):
            continue

        st.markdown(f"#### {model_labels.get(model_name, model_name)}")
        plot_cols = st.columns(3)

        plots = ['test_confusion_matrix.png', 'test_roc_curve.png', 'test_pr_curve.png']
        titles = ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve']

        for i, (plot_file, title) in enumerate(zip(plots, titles)):
            plot_path = os.path.join(eval_dir, plot_file)
            if os.path.exists(plot_path):
                with plot_cols[i]:
                    st.markdown(f"**{title}**")
                    st.image(plot_path)


# ============================================================
# PAGE 3: EXPLAINABILITY DASHBOARD
# ============================================================
def explainability_page():
    st.title("Explainability Dashboard")
    st.markdown("Understand **why** the model makes its predictions.")

    if 'baseline' not in available_models:
        st.warning("Train the baseline model first to use explainability features.")
        return

    text_input = st.text_area(
        "Enter text to explain",
        height=150,
        placeholder="Paste a news article to see what drives the prediction..."
    )

    if st.button("Explain Prediction", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Generating explanations..."):
            predictor = ModelRegistry.get_predictor('baseline')
            explanation = explain_baseline_prediction(text_input, predictor, top_n=15)

        result = explanation['prediction']
        st.markdown("---")

        # Prediction summary
        label = result['label']
        badge_class = "fake-badge" if label == "Fake" else "real-badge"
        st.markdown(
            f'<div class="{badge_class}">{label} '
            f'({result["confidence"]:.1%} confidence)</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        # Word importance visualization
        col_chart, col_list = st.columns([2, 1])

        with col_chart:
            st.markdown("#### Word Importance Chart")
            if explanation['top_words']:
                fig = plot_word_importance(explanation['top_words'], top_n=12)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

        with col_list:
            st.markdown("#### Key Words")
            for item in explanation['top_words'][:10]:
                icon = "🔴" if 'Fake' in item['direction'] else "🟢"
                st.markdown(
                    f"{icon} **{item['word']}** — "
                    f"score: {item['score']:.4f} ({item['direction']})"
                )

        # Highlighted text
        st.markdown("---")
        st.markdown("#### Highlighted Text")
        html = generate_word_importance_html(explanation['top_words'])
        st.markdown(html, unsafe_allow_html=True)

        # Bias + Sentiment side by side
        st.markdown("---")
        b_col, s_col = st.columns(2)
        with b_col:
            st.markdown("#### Bias Analysis")
            bias = detect_bias(text_input)
            st.json(bias)
        with s_col:
            st.markdown("#### Sentiment Breakdown")
            sentiment = analyze_sentiment(text_input)
            st.json(sentiment)


# ============================================================
# PAGE 4: ADMIN PANEL
# ============================================================
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()


def admin_page():
    st.title("Admin Panel")

    # Simple password auth
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password.")
        st.info("Default password: admin123")
        return

    st.success("Authenticated as admin.")

    admin_tab1, admin_tab2, admin_tab3 = st.tabs([
        "Prediction Logs", "Dataset Management", "Model Management"
    ])

    # ---- Prediction Logs ----
    with admin_tab1:
        st.markdown("### Prediction History")
        log_path = os.path.join(LOGS_DIR, 'predictions.csv')
        if os.path.exists(log_path):
            logs_df = pd.read_csv(log_path)
            st.metric("Total Predictions", len(logs_df))

            if len(logs_df) > 0:
                col1, col2, col3 = st.columns(3)
                fake_count = (logs_df['prediction'] == 'Fake').sum()
                real_count = (logs_df['prediction'] == 'Real').sum()
                col1.metric("Fake Predictions", fake_count)
                col2.metric("Real Predictions", real_count)
                col3.metric("Avg Confidence", f"{logs_df['confidence'].mean():.1%}")

                st.dataframe(logs_df.tail(50).sort_index(ascending=False),
                             use_container_width=True)

                csv = logs_df.to_csv(index=False)
                st.download_button(
                    "Download Logs CSV", csv, "prediction_logs.csv", "text/csv"
                )
        else:
            st.info("No prediction logs yet. Make some predictions first!")

    # ---- Dataset Management ----
    with admin_tab2:
        st.markdown("### Upload New Dataset")
        st.markdown(
            "Upload a CSV file with columns: `title`, `text`, `label` "
            "(0=Fake, 1=Real)"
        )
        uploaded = st.file_uploader("Upload CSV", type=['csv'], key='admin_upload')
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                st.write(f"Shape: {new_df.shape}")
                st.dataframe(new_df.head(10))

                required_cols = ['text', 'label']
                if all(c in new_df.columns for c in required_cols):
                    st.success("Dataset format is valid.")
                    if st.button("Save to data/raw/"):
                        save_path = os.path.join(PROJECT_ROOT, 'data', 'raw', uploaded.name)
                        new_df.to_csv(save_path, index=False)
                        st.success(f"Saved to {save_path}")
                else:
                    st.error(f"Missing required columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ---- Model Management ----
    with admin_tab3:
        st.markdown("### Trained Models")
        for model_name in ['baseline', 'lstm', 'bert']:
            model_dir = os.path.join(MODELS_DIR, model_name)
            st.markdown(f"**{model_labels.get(model_name, model_name)}**")

            if model_name in available_models:
                st.markdown("Status: Trained")
                metrics_path = os.path.join(model_dir, 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        m = json.load(f)
                    st.json(m)
            else:
                st.markdown("Status: Not trained")

            st.markdown("---")

        st.markdown("### Retrain Baseline Model")
        st.warning("This will retrain the TF-IDF + Naive Bayes model from data/raw/.")
        if st.button("Retrain Baseline", type="secondary"):
            with st.spinner("Retraining... this may take a few minutes."):
                try:
                    from src.train_baseline import train_baseline
                    train_baseline()
                    st.success("Baseline model retrained successfully!")
                    ModelRegistry._predictors.pop('baseline', None)
                except Exception as e:
                    st.error(f"Retraining failed: {e}")

    if st.sidebar.button("Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()


# ============================================================
# Route to correct page
# ============================================================
if page == "Prediction":
    prediction_page()
elif page == "Model Performance":
    model_performance_page()
elif page == "Explainability Dashboard":
    explainability_page()
elif page == "Admin Panel":
    admin_page()
