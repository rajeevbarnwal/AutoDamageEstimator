import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from scripts.infer import infer
import cv2
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stoggle import stoggle
import time

# Page configuration
st.set_page_config(
    page_title="AutoDamageEstimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #104e8b;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
model_options = ["Default (YOLOv8)", "Custom Model"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.button("Admin Dashboard", on_click=lambda: switch_page("admin_dashboard"))

# Main content
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<div class="card"><h2>Upload Your Image</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], key="image_uploader")
if uploaded_file is not None:
    temp_path = os.path.join(os.path.dirname(__file__), "temp.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing image..."):
        time.sleep(1)  # Simulate processing time
        detections, cost = infer(temp_path, conf=confidence_threshold)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>Results</h2>', unsafe_allow_html=True)
    st.image(temp_path, caption="Uploaded Image", use_column_width=True)
    st.markdown('<h3>Detected Damages</h3>', unsafe_allow_html=True)
    if detections:
        for det in detections:
            st.write(f"🚗 **{det['class']}** (Severity: {det['severity']}, Confidence: {det['confidence']:.2f})")
    else:
        st.write("❌ No damages detected with the current threshold.")
    st.markdown(f'<h3>Estimated Repair Cost: ₹{cost:.2f}</h3>', unsafe_allow_html=True)
    os.remove(temp_path)

    # Feedback section
    st.markdown('<h3>Feedback</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Good", key="feedback_good"):
            st.success("Thanks for your feedback!")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("🤔 Neutral", key="feedback_neutral"):
            st.info("Noted, thanks!")
            time.sleep(1)
            st.rerun()
    with col3:
        if st.button("👎 Confusing", key="feedback_confusing"):
            st.error("Sorry about that! Thanks for the feedback.")
            time.sleep(1)
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 10px; color: #6c757d;'>
        © 2025 AutoDamageEstimator | Built with ❤️ by #RajeevBarnwal
    </div>
""", unsafe_allow_html=True)