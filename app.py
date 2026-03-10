import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("Brain Tumor Detection using CNN")
st.write("Upload an MRI image to detect tumor presence.")

# Load trained model
model = tf.keras.models.load_model("brain_tumor_cnn.h5", compile=False)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    image = image.resize((128,128))
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,128,128,3)

    prediction = model.predict(img_array)[0][0]

    confidence = prediction * 100
    st.write(f"Model Confidence Score: {confidence:.2f}%")

    if prediction > 0.9:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
