import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Accident Detection App", layout="wide")
st.title("Accident Detection App")
st.write("Upload an image to detect if an accident is present.")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'accident_detection_cnn.h5'  # Update with your model path
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Function to preprocess image for prediction
def preprocess_image(image):
    img = image.resize((224, 224))  # Match model input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Confidence threshold slider
confidence_threshold = st.slider(
    "Confidence Threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=1,
    help="Set the minimum confidence required to classify an image as an accident."
) / 100.0

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image (smaller preview)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=300)  # Smaller image preview

    # Preprocess image and predict
    img_array = preprocess_image(image)
    try:
        prediction = model.predict(img_array)[0][0]  # Get probability
        confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100
        is_accident = prediction >= confidence_threshold

        if is_accident:
            st.success(f"Accident detected with {confidence:.2f}% confidence.")
        else:
            st.info(f"No accident found with {confidence:.2f}% confidence.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Instructions
st.markdown("""
### Instructions
1. Upload a JPG, JPEG, or PNG image.
2. Adjust the confidence threshold using the slider.
3. The model will predict if an accident is present based on the threshold.
4. Results will show whether an accident was detected or not, along with the confidence score.
""")