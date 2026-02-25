import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection using CNN")
st.write("Upload an MRI image to detect tumor presence.")
@st.cache_resource
def load_model():
    model_file = "brain_tumor_cnn.h5"

    if not os.path.exists(model_file):
        url = "https://drive.google.com/uc?id=1q2FExxziI_dA5h1woBjyj-bBHs49fyVL"
        gdown.download(url, model_file, quiet=False)

    return tf.keras.models.load_model(model_file)

model = load_model()

# -------- IMAGE UPLOAD --------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Prediction Score: {prediction:.4f}")

    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected") 
