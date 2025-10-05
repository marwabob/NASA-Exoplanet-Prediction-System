import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ======================================================
# Setup File Paths
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_v8_enriched")

# ======================================================
# Load Model and Components
# ======================================================
@st.cache_resource(show_spinner="📦 Loading model and components...")
def load_model_components():
    """Loads the trained Stacking Ensemble model, Scaler, and Feature list."""
    try:
        model = joblib.load(os.path.join(MODELS_DIR, "model_v8.joblib"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_v8.joblib"))
        features = joblib.load(os.path.join(MODELS_DIR, "features_v8.joblib"))
        return model, scaler, features
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None, None, None

# ======================================================
# SHAP Background Data
# ======================================================
# ✅ FIX: Added underscore before scaler to prevent Streamlit hashing error
@st.cache_data(show_spinner="🔄 Preparing SHAP background data...")
def load_shap_background(features, _scaler):
    """Loads and preprocesses background data for SHAP explanation."""
    data_path = os.path.join(BASE_DIR, "trail", "koi_dataset.csv")
    try:
        df = pd.read_csv(data_path)
        df = df[features].dropna().sample(300, random_state=42)
        X_scaled = _scaler.transform(df)
        return pd.DataFrame(X_scaled, columns=features)
    except Exception as e:
        st.warning(f"⚠️ Could not load SHAP background: {e}")
        return None

# ======================================================
# Streamlit App Layout
# ======================================================
st.set_page_config(page_title="🚀 NASA Exoplanet Predictor", layout="wide")
st.title("🪐 NASA Exoplanet Prediction System")
st.caption("Predict whether an exoplanet candidate is **False Positive**, **Candidate**, or **Confirmed** using an AI-powered ensemble model.")

model, scaler, FEATURES = load_model_components()

if model and scaler and FEATURES:
    st.success(f"✅ Model successfully loaded! Using {len(FEATURES)} features.")
    
    # Input selection mode
    input_mode = st.radio("Select Input Method:", ("Manual Input", "Upload CSV File"))
    X_input = None

    # Manual Input
    if input_mode == "Manual Input":
        st.subheader("🧮 Enter Feature Values:")
        user_input = {}
        cols = st.columns(2)
        for i, feature in enumerate(FEATURES):
            col = cols[i % 2]
            default = 10.0 if "snr" in feature.lower() else 0.0
            user_input[feature] = col.number_input(f"{feature}", value=default)
        X_input = pd.DataFrame([user_input])

    # CSV Upload
    elif input_mode == "Upload CSV File":
        uploaded_file = st.file_uploader("📂 Upload CSV with Feature Columns", type="csv")
        if uploaded_file:
            df_csv = pd.read_csv(uploaded_file)
            missing = [f for f in FEATURES if f not in df_csv.columns]
            if missing:
                st.error(f"Missing required features: {missing}")
                st.stop()
            X_input = df_csv[FEATURES]
        else:
            st.info("Please upload a CSV file to perform batch prediction.")

    # ======================================================
    # Prediction and Visualization
    # ======================================================
    if X_input is not None and st.button("🔍 Predict"):
        try:
            # Scale features
            X_scaled = scaler.transform(X_input)

            # Predict
            proba_all = model.predict_proba(X_scaled)
            preds = np.argmax(proba_all, axis=1)

            CLASS_NAMES = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
            pred_labels = [CLASS_NAMES[i] for i in preds]

            st.subheader("📊 Prediction Results")

            for i in range(len(X_input)):
                st.write(f"### Row {i+1}")
                st.metric("Predicted Class", pred_labels[i])
                st.progress(float(np.max(proba_all[i])))

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(5, 2))
                sns.barplot(x=CLASS_NAMES, y=proba_all[i], palette="viridis", ax=ax)
                ax.set_ylabel("Probability")
                ax.set_title("Class Probability Distribution")
                st.pyplot(fig)
                st.divider()

            # ======================================================
            # If batch mode: show confusion matrix and accuracy
            # ======================================================
            if len(X_input) > 1 and 'koi_disposition' in X_input.columns:
                y_true = X_input['koi_disposition'].map({
                    "FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2
                })
                acc = accuracy_score(y_true, preds)
                st.success(f"✅ Overall Accuracy: {acc*100:.2f}%")

                cm = confusion_matrix(y_true, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                report = classification_report(y_true, preds, target_names=CLASS_NAMES, output_dict=False)
                st.text(report)

            # ======================================================
            # SHAP Explanation (first row)
            # ======================================================
            with st.expander("🔬 SHAP Feature Importance (Row 1 Explanation)"):
                shap_bg = load_shap_background(FEATURES, scaler)
                if shap_bg is not None:
                    explainer = shap.Explainer(model.predict_proba, shap_bg)
                    shap_values = explainer(X_scaled[:1])

                    predicted_idx = preds[0]
                    predicted_label = CLASS_NAMES[predicted_idx]
                    st.markdown(f"**Explaining prediction for class:** `{predicted_label}`")

                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.plots.waterfall(shap_values[0, :, predicted_idx], show=False)
                    st.pyplot(fig)
                    st.info("🔵 Blue: decreases likelihood | 🔴 Red: increases likelihood")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.error("⚠️ Model could not be loaded. Please ensure model files exist in 'models_v8_enriched'.")
