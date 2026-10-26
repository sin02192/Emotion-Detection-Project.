
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("😊 Facial Emotion Detection")
st.write("Upload an image to detect facial emotion.")

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(
        "/content/emotion_classifier.keras"
    )

model = load_emotion_model()

class_names = ["Happy", "Sad", "Neutral"]

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=250)

    image = image.resize((48, 48))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])

    st.success(
        f"Detected Emotion: {class_names[predicted_index]}"
    )

    st.write(
        f"Confidence: {predictions[0][predicted_index] * 100:.2f}%"
    )
