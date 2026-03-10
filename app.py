import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from scipy.ndimage import gaussian_laplace

# Load models
svm = joblib.load("svm_model.pkl")
rf = joblib.load("rf_model.pkl")
lr = joblib.load("lr_model.pkl")
ensemble = joblib.load("ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Brain Tumor Detection (ML Models)")
st.write("Using SVM, Random Forest, Logistic Regression and Ensemble")

# Feature extraction (same as training)
def extract_features(image):

    image = cv2.resize(image,(128,128))

    log_img = gaussian_laplace(image.astype(np.float32),sigma=1)

    lbp = local_binary_pattern(image.astype(np.uint8),8,1,"uniform")

    lbp_hist,_ = np.histogram(lbp.ravel(),
                              bins=np.arange(0,11),
                              range=(0,10))

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum()+1e-6)

    hog_features = hog(image,
                       orientations=9,
                       pixels_per_cell=(8,8),
                       cells_per_block=(2,2),
                       feature_vector=True)

    mean = np.mean(log_img)
    std = np.std(log_img)
    var = np.var(log_img)

    features = np.hstack([lbp_hist,hog_features,mean,std,var])

    return features


uploaded_file = st.file_uploader("Upload MRI Image",type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    st.image(image,caption="Uploaded MRI",use_column_width=True)

    image_norm = (image/np.max(image)*255).astype(np.uint8)

    features = extract_features(image_norm)

    features = scaler.transform([features])

    svm_pred = svm.predict(features)[0]
    rf_pred = rf.predict(features)[0]
    lr_pred = lr.predict(features)[0]
    ens_pred = ensemble.predict(features)[0]

    st.subheader("Model Predictions")

    st.write("SVM:", "Tumor" if svm_pred==1 else "No Tumor")
    st.write("Random Forest:", "Tumor" if rf_pred==1 else "No Tumor")
    st.write("Logistic Regression:", "Tumor" if lr_pred==1 else "No Tumor")

    st.subheader("Final Ensemble Result")

    if ens_pred==1:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")
