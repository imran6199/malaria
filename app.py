import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# --- CONFIG: update if your models have different filenames or input sizes ---
MODEL_CONFIG = {
    "Malaria": {"file": "malaria.h5", "input_size": (64, 64)},
    "Pneumonia": {"file": "pneumonia.h5", "input_size": (64, 64)},
}
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(path: str):
    """Load and return a Keras model from the given path."""
    return tf.keras.models.load_model(path)

st.set_page_config(page_title="Malaria & Pneumonia Detector", layout="centered")
st.title("🧫 Malaria & Pneumonia Detection App")
st.write("Select a model, then upload a blood-smear / chest X-ray image to predict.")

# Model selector
choice = st.radio("Choose model", list(MODEL_CONFIG.keys()))

config = MODEL_CONFIG[choice]
model_path = config["file"]
input_size = config["input_size"]

# Try to load model (cached by path)
model = None
if Path(model_path).exists():
    with st.spinner(f"Loading {choice} model..."):
        model = load_model(model_path)
else:
    st.warning(f"Model file `{model_path}` not found in repo. Please add it to the project root.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image (centered)
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize(input_size)
    x = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
    with st.spinner("Predicting..."):
        pred = model.predict(x)[0]
        if pred.shape == ():  # scalar
            score = float(pred)
        else:
            score = float(pred[0]) if np.size(pred) == 1 else float(pred[0])

    # Interpret result (threshold 0.5)
    if score > 0.5:
        label = "🦠 Infected (Positive)"
        color = "red"
        message = "⚠️ Model predicts **positive**. Please consult a medical professional for confirmation."
    else:
        label = "✅ Uninfected (Negative)"
        color = "green"
        message = "🟢 Model predicts **negative**. No infection detected."

    # Display result
    st.subheader(f"{choice} result: {label}")
    st.markdown(
        f"<p style='color:{color}; font-size:18px; font-weight:bold;'>{message}</p>",
        unsafe_allow_html=True,
    )
    st.write(f"Confidence: **{score:.3f}**")