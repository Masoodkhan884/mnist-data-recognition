import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('bestmodel.h5')

# Page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        font-size: 18px;
        color: #34495E;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 13px;
        text-align: center;
        color: #999;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Instructions
st.markdown('<div class="title">🖌️ MNIST Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload a clear handwritten digit (0–9). The model will classify it using deep learning.</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📁 Upload a digit image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

# Prediction function
def predict_digit(image):
    # Convert to grayscale and resize to 28x28
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

# Process uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼 Uploaded Image',  use_container_width=False, width=200)

    predicted_class, confidence = predict_digit(image)

    st.success(f"✅ **Prediction:** {predicted_class}")
    st.info(f"🔍 **Confidence:** {confidence:.2%}")

    # Optionally show the preprocessed image (28x28)
    if st.checkbox("Show preprocessed image (28x28 grayscale)"):
        processed_img = image.convert('L').resize((28, 28))
        st.image(processed_img, caption="28x28 Preprocessed", width=150)


