import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria.h5")
    return model

model = load_model()

# Streamlit UI
st.title("🧫 Malaria Detection App")
st.write("Upload a blood smear image to check for malaria infection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", width=300)

    # Resize as per your model input
    img = image.resize((64, 64))  # change to your input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = "🦠 Infected (Malaria Positive)" if prediction[0][0] > 0.5 else "✅ Uninfected (Malaria Negative)"

    st.subheader(f"Result: {result}")
