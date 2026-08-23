import streamlit as st


st.set_page_config(page_title="Plant Disease & Treatment Predictor", layout="centered")

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os
from PIL import Image


@st.cache_resource
def load_resources():
    model = load_model("mobilenetv2_final_model.keras")
    with open("mobilenetv2_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    disease_classes = metadata["disease_classes"]
    severity_classes = metadata["severity_classes"]
    df = pd.read_csv("treatments.csv")
    return model, disease_classes, severity_classes, df

model, disease_classes, severity_classes, treatment_df = load_resources()


def preprocess_image(image: Image.Image, target_size=(256, 256)):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_disease_severity(img_array):
    disease_pred, severity_pred = model.predict(img_array)
    disease = disease_classes[np.argmax(disease_pred[0])]
    severity = severity_classes[np.argmax(severity_pred[0])]
    return disease, severity


def recommend_treatment(disease, severity):
    
    disease_clean = disease.split("___")[-1].replace("_", " ").lower().strip()
    severity_clean = severity.strip().capitalize()

    
    treatment_df["extracted_disease_part_clean"] = treatment_df["extracted_disease_part"].str.lower().str.strip()
    treatment_df["severity_clean"] = treatment_df["severity"].str.strip().str.capitalize()

    
    match = treatment_df[
        (treatment_df["extracted_disease_part_clean"] == disease_clean) &
        (treatment_df["severity_clean"] == severity_clean)
    ]

    if not match.empty:
        return match.iloc[0]["treatment"]
    return " No treatment found for this combination."

st.title(" Plant Disease & Severity Prediction")
st.subheader("Upload a plant leaf image to detect disease and get treatment advice")

uploaded_file = st.file_uploader("Choose an image...", type=None)

if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        st.error(" Please upload a .jpg, .jpeg, or .png image.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        with st.spinner(" Analyzing image..."):
            img_array = preprocess_image(image)
            disease, severity = predict_disease_severity(img_array)
            treatment = recommend_treatment(disease, severity)

        st.success(" Prediction Complete!")
        st.markdown(f"###  Predicted Disease: `{disease}`")
        st.markdown(f"###  Severity Level: `{severity}`")
        st.markdown("### Recommended Treatment:")
        st.info(treatment)
