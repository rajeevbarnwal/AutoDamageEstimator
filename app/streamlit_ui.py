import streamlit as st
from scripts.infer import infer
import cv2

st.title("AutoDamageEstimator")
st.write("Upload an image of a damaged car to estimate repair costs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    detections, cost = infer("temp.jpg")
    
    st.image("temp.jpg", caption="Uploaded Image")
    st.write("Detected Damages:")
    for det in detections:
        st.write(f"- {det['class']} (Severity: {det['severity']}, Confidence: {det['confidence']:.2f})")
    st.write(f"Estimated Repair Cost: ${cost:.2f}")