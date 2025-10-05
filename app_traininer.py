import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ======================================================
# Setup File Paths
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Keeping the directory name consistent with the model output
MODELS_DIR = os.path.join(BASE_DIR, "models_v8_enriched")

# ======================================================
# Load Model and Components
# ======================================================
@st.cache_resource(show_spinner="📦 Loading model and components...")
def load_model_components():
    """Loads the Stacking Ensemble model, Scaler, and Feature list."""
    try:
        model_path = os.path.join(MODELS_DIR, "model_v8.joblib")
        scaler_path = os.path.join(MODELS_DIR, "scaler_v8.joblib")
        features_path = os.path.join(MODELS_DIR, "features_v8.joblib")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)

        return model, scaler, features

    except Exception as e:
        # Changed the error message to remove "Generation 8" reference
        st.error(
            f"⚠️ Failed to load Advanced Ensemble model components: {e}\n"
            f"Ensure files (model_v8.joblib, scaler_v8.joblib, features_v8.joblib) "
            f"exist inside '{MODELS_DIR}'"
        )
        return None, None, None

# ======================================================
# SHAP Background Data
# ======================================================
@st.cache_data(show_spinner="🔄 Preparing SHAP background data...")
def load_shap_background(features, _scaler):
    """Loads and preprocesses data for SHAP Explainer."""
    data_path = os.path.join(BASE_DIR, "trail", "koi_dataset.csv")
    try:
        # Load, filter for required features, drop N/A, and sample 500 rows for speed
        df = pd.read_csv(data_path)
        df = df[features].dropna().head(500)
        X_scaled = _scaler.transform(df)
        return pd.DataFrame(X_scaled, columns=features)
    except Exception as e:
        st.error(f"⚠️ Error loading SHAP background data: {e}. Check if 'koi_dataset.csv' is in the 'trail' folder.")
        return None

# ======================================================
# Application Interface
# ======================================================
st.set_page_config(page_title="🚀 NASA Exoplanet Predictor", layout="centered")
# Updated title to remove "Gen 8" reference
st.title("🪐 Kepler Exoplanet Prediction System")
st.caption("This system uses an advanced Stacking Ensemble model trained on 10 enriched features, including `koi_model_snr`.")

# Load model components
model, scaler, FEATURES = load_model_components()

if model is not None and scaler is not None and FEATURES is not None:
    # Updated success message
    st.success(f"✅ Advanced Ensemble model loaded successfully! Uses {len(FEATURES)} features.")

    # Select input mode
    input_mode = st.radio("Select data input method:", ("Manual", "Upload CSV File"))

    X_input = None

    if input_mode == "Manual":
        st.subheader("🧮 Enter feature values for prediction:")
        user_input = {}
        # Set default values for manual input
        col1, col2 = st.columns(2)

        for i, f in enumerate(FEATURES):
            # Alternate columns for better layout
            col = col1 if i < len(FEATURES) / 2 else col2

            # Set a non-zero default for SNR as it's often critical
            default_value = 0.0
            if f == 'koi_model_snr':
                default_value = 10.0

            user_input[f] = col.number_input(f"🔹 {f}", value=default_value, help=f"Required value for feature: {f}")

        X_input = pd.DataFrame([user_input])

    elif input_mode == "Upload CSV File":
        uploaded_file = st.file_uploader("📂 Choose a CSV file containing features", type="csv")
        if uploaded_file is not None:
            try:
                df_csv = pd.read_csv(uploaded_file)
                missing_cols = [f for f in FEATURES if f not in df_csv.columns]

                if missing_cols:
                    st.error(f"⚠️ File is missing the following required columns: {missing_cols}")
                    st.stop()

                X_input = df_csv[FEATURES]

            except Exception as e:
                st.error(f"⚠️ Error reading the file: {e}")
                X_input = None
        else:
             st.info("Please upload a CSV file to proceed with batch prediction.")
             X_input = None


    # Execute prediction if data is ready
    if X_input is not None and st.button("🔍 Execute Prediction"):
        try:
            # Scale input data
            X_scaled = scaler.transform(X_input)

            # Get predictions and probabilities
            proba_all = model.predict_proba(X_scaled)

            # Class names must match the model's output order (0, 1, 2)
            CLASS_NAMES = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']

            # Display results for each row
            for i in range(len(X_input)):
                st.write(f"### Prediction for Row {i+1}:")

                # Format probabilities display
                proba_display = (
                    f"FP: {proba_all[i][0]*100:.2f}% | "
                    f"CAN: {proba_all[i][1]*100:.2f}% | "
                    f"CONF: {proba_all[i][2]*100:.2f}%"
                )
                st.metric("🔢 Probabilities:", proba_display)

                predicted_class_index = np.argmax(proba_all[i])
                predicted_class_name = CLASS_NAMES[predicted_class_index]

                # Style the output for visibility
                color = 'green' if predicted_class_name == 'CONFIRMED' else ('orange' if predicted_class_name == 'CANDIDATE' else 'red')

                st.markdown(f"**🧭 Predicted Result: <span style='color:{color};'>{predicted_class_name}</span>**", unsafe_allow_html=True)

                st.markdown("---")


            # SHAP Analysis for the first row only (for efficiency)
            if X_input is not None and len(X_input) > 0:
                with st.expander("🔬 SHAP Analysis - Decision Explanation (Row 1)"):
                    st.info("⚠️ SHAP value calculation may take a moment, especially on first run.")
                    shap_bg = load_shap_background(FEATURES, scaler)

                    if shap_bg is not None:
                        # Only explain the first prediction
                        X_input_scaled_shap = X_scaled[0:1]

                        # Use model.predict_proba for multiclass explainer (better detail)
                        explainer = shap.Explainer(model.predict_proba, shap_bg)

                        # Calculate SHAP values
                        shap_values_proba = explainer(X_input_scaled_shap)

                        # Get the predicted class index for Row 1
                        predicted_class_idx = np.argmax(proba_all[0])
                        predicted_class_name = CLASS_NAMES[predicted_class_idx]

                        st.write(f"**Explaining Prediction for: {predicted_class_name}**")

                        # Plot the waterfall chart for the predicted class
                        fig, ax = plt.subplots(figsize=(8, 5))
                        shap.plots.waterfall(shap_values_proba[0, :, predicted_class_idx], show=False)
                        st.pyplot(fig)

                        st.markdown(
                            """
                            **Interpretation:**
                            - **Red** features increase the probability of the predicted class.
                            - **Blue** features decrease the probability of the predicted class.
                            - The chart shows how each feature value moves the prediction from the base value (average prediction) to the final output.
                            """
                        )
                    else:
                        st.warning("SHAP analysis skipped due to background data loading error.")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

else:
    # Stop Streamlit execution if model loading failed
    st.stop()
