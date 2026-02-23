import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Leukemia Recognition System",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Deep Learning-Based Leukemia Recognition System")

st.warning(
    "⚠️ Disclaimer: This application is for educational and research purposes only. "
    "It is NOT a medical diagnostic tool."
)

# -----------------------------
# Load Model
# -----------------------------
from pathlib import Path
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "leukemia_cnn_model.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# -----------------------------
# Class Names
# -----------------------------
class_names = ['Benign', 'Early', 'Pre', 'Pro']

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Sidebar Navigation
# -----------------------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Single Image Prediction", "Bulk Image Prediction"]
)

# ======================================================
# SINGLE IMAGE PREDICTION
# ======================================================
if mode == "Single Image Prediction":
    st.subheader("🔍 Single Image Prediction")

    uploaded_file = st.file_uploader(
        "Upload a blood cell image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            processed = preprocess_image(image)
            prediction = model.predict(processed)

            idx = np.argmax(prediction)
            predicted_class = class_names[idx]
            confidence = prediction[0][idx] * 100

            st.success(f"Predicted Class: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

            prob_df = pd.DataFrame({
                "Class": class_names,
                "Probability (%)": prediction[0] * 100
            })

            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(prob_df.set_index("Class"))

# ======================================================
# BULK IMAGE PREDICTION
# ======================================================
else:
    st.subheader("📦 Bulk Image Prediction")

    uploaded_files = st.file_uploader(
        "Upload multiple blood cell images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        with st.spinner("Processing images..."):
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                processed = preprocess_image(image)

                prediction = model.predict(processed, verbose=0)
                idx = np.argmax(prediction)

                results.append({
                    "Image Name": file.name,
                    "Predicted Class": class_names[idx],
                    "Confidence (%)": round(prediction[0][idx] * 100, 2)
                })

        df = pd.DataFrame(results)

        st.success("Bulk prediction completed!")

        # -----------------------------
        # Results Table
        # -----------------------------
        st.subheader("📋 Prediction Results")
        st.dataframe(df)

        # -----------------------------
        # Class Distribution Chart
        # -----------------------------
        st.subheader("📊 Predicted Class Distribution")
        class_dist = df["Predicted Class"].value_counts()
        st.bar_chart(class_dist)

        # -----------------------------
        # Confidence Distribution
        # -----------------------------
        st.subheader("📈 Confidence Distribution")
        st.line_chart(df["Confidence (%)"])

        # -----------------------------
        # Download Results
        # -----------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            "bulk_predictions.csv",
            "text/csv"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Final Year Project | Deep Learning-Based Leukemia Recognition System "
    "| Streamlit Deployment"
)