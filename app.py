import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# Page setup
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("Brain Tumor Detection using CNN")
st.write("Upload a Brain MRI image to detect tumor presence.")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    model_path = "brain_tumor_cnn.h5"

    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        file_id = "1q2FExxziI_dA5h1woBjyj-bBHs49fyVL"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path, compile=False)
    return model


model = load_model()

# -------- IMAGE UPLOAD --------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocessing
    image = image.convert("L")
    image = image.resize((128,128))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,128,128,1)

    # Prediction
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    st.write(f"Prediction Score: {score:.4f}")

    if score > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
