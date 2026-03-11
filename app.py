import streamlit as st
import numpy as np
import joblib
from PIL import Image

from skimage.feature import local_binary_pattern, hog
from scipy.ndimage import gaussian_laplace
from skimage.transform import resize

# ------------------------------
# Load Models
# ------------------------------

svm = joblib.load("svm_model.pkl")
rf = joblib.load("rf_model.pkl")
lr = joblib.load("lr_model.pkl")
ensemble = joblib.load("ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Brain Tumor Detection")
st.write("Using SVM, Random Forest, Logistic Regression and Ensemble")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg","png","jpeg"]
)

# ------------------------------
# Feature Extraction
# ------------------------------

def extract_features(image):

    image = resize(image,(128,128))

    log_img = gaussian_laplace(image.astype(np.float32),sigma=1)

    lbp = local_binary_pattern(
        image.astype(np.uint8),
        P=8,
        R=1,
        method="uniform"
    )

    lbp_hist,_ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0,11),
        range=(0,10)
    )

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum()+1e-6)

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        feature_vector=True
    )

    mean = np.mean(log_img)
    std = np.std(log_img)
    var = np.var(log_img)

    features = np.hstack([lbp_hist,hog_features,mean,std,var])

    return features

# ------------------------------
# Prediction
# ------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image = np.array(image)

    st.image(image,caption="Uploaded MRI Image",use_column_width=True)

    image_norm = (image/(np.max(image)+1e-6)*255).astype(np.uint8)

    features = extract_features(image_norm)

    features = scaler.transform([features])

    # Model probabilities
    svm_prob = svm.predict_proba(features)[0][1]
    rf_prob = rf.predict_proba(features)[0][1]
    lr_prob = lr.predict_proba(features)[0][1]

    # Ensemble prediction
    ens_pred = ensemble.predict(features)[0]

    st.subheader("Model Predictions")

    st.write("SVM Tumor Probability:", round(svm_prob*100,2),"%")
    st.progress(int(svm_prob*100))

    st.write("RF Tumor Probability:", round(rf_prob*100,2),"%")
    st.progress(int(rf_prob*100))

    st.write("LR Tumor Probability:", round(lr_prob*100,2),"%")
    st.progress(int(lr_prob*100))

    st.subheader("Final Ensemble Prediction")

    if ens_pred == 1:
        st.success("No Tumor Detected")
    else:
        st.error("Tumor Detected")
        
