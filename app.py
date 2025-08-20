import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================================
# Page config (sets tab title, emoji, layout)
# =========================================
st.set_page_config(page_title="Iris Oracle", page_icon="🔮", layout="wide")


# =========================================
# Helpers
# =========================================
def inject_css():
    """Inject custom CSS for a playful, dark-themed UI."""
    st.markdown(
        """
        <style>
        body {background-color:#0E1117; color:#FFFFFF;}
        h1 {color:#F72585; text-align:center;}
        h2 {color:#7209B7;}
        .stButton>button {
            background: linear-gradient(90deg, #FF4D6D, #FF8E53);
            color:white; font-size:16px; height:3em;
            border-radius:12px; border:none;
        }
        .stSlider>div>div>div>div>div>div {color:#FF4D6D;}
        .progress-container {margin-bottom: 10px; font-weight: bold;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_artifacts(model_path="models/knn_iris_model.pkl", scaler_path="models/scaler.pkl"):
    """Load trained model and fitted scaler from disk with basic validation."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(
            "Model or scaler not found. Please train the model first (run iris.py) "
            "so that models/knn_iris_model.pkl and models/scaler.pkl exist."
        )
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_species(model, scaler, features_array):
    """Scale inputs and return predicted class index and class probabilities."""
    scaled = scaler.transform(features_array)
    pred_idx = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled) if hasattr(model, "predict_proba") else None
    return pred_idx, proba


def species_palette():
    """Mapping from class index to display info."""
    return {
        0: {"name": "Setosa 🌱", "color": "#4CAF50", "fun_fact": "Setosa is the tiniest and earliest blooming Iris!"},
        1: {"name": "Versicolor 🌷", "color": "#FFB703", "fun_fact": "Versicolor thrives in temperate zones and has delicate petals."},
        2: {"name": "Virginica 🌸", "color": "#9D4EDD", "fun_fact": "Virginica is elegant and often called the queen of Irises."},
    }


# =========================================
# UI: Header
# =========================================
inject_css()
st.title("🔮 Iris Oracle")
st.write("Discover your Iris flower's destiny based on its measurements!")


# =========================================
# Load model + scaler once
# =========================================
model, scaler = load_artifacts()


# =========================================
# Input section
# - Keep the same feature order used in training (sepal_len, sepal_wid, petal_len, petal_wid)
# =========================================
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, step=0.1)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5, step=0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, step=0.1)

# Prepare a 2D array as expected by scikit-learn (n_samples x n_features)
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


# =========================================
# Prediction action
# =========================================
if st.button("Reveal Your Flower 🌸"):
    pred_idx, proba = predict_species(model, scaler, features)
    palette = species_palette()
    info = palette[pred_idx]

    # Pretty header with species name and color
    st.markdown(
        f"<h2 style='color:{info['color']}; text-align:center;'>Prediction: {info['name']}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center; font-size:18px;'>{info['fun_fact']}</p>",
        unsafe_allow_html=True,
    )

    # Probability visualization
    st.subheader("Prediction Probabilities")
    if proba is not None:
        for i, sp in enumerate(["Setosa", "Versicolor", "Virginica"]):
            st.markdown(
                f"<div class='progress-container'>{sp}: {proba[i]*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(proba[i]))
    else:
        st.info("Probability output not available for the current model.")


# =========================================
# Sidebar: Model info + on-demand metrics
# =========================================
with st.sidebar:
    st.header("ℹ️ Model Insights")
    st.write("Algorithm: K-Nearest Neighbors (KNN)")
    st.write("Dataset: Iris (scikit-learn)")
    st.write("Scaler: StandardScaler")

    # Show a quick metrics table using the same scaler on a fresh test split
    with st.expander("Show Classification Report"):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
        )
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # Build a readable DataFrame report
        report = classification_report(
            y_test, y_pred, target_names=iris.target_names, output_dict=True, digits=3
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        # Optional: show overall accuracy prominently
        if "accuracy" in report:
            st.metric(label="Accuracy", value=f"{report['accuracy']*100:.2f}%")
