import sys
import os
import uuid
import csv
import time
import sqlite3

# ── set up feedback database ─────────────────────────────────────────
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "database", "feedback.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_cursor = _conn.cursor()
_cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    rating    TEXT    NOT NULL
);
""")
_conn.commit()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from scripts.infer import _get_model, infer, _get_llm
from langchain.prompts import PromptTemplate
import cv2
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from PIL import Image
import io
from scripts.infer import infer, _get_llm, _get_model   # ← add _get_model

# Page configuration
st.set_page_config(
    page_title="AutoDamageEstimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced theming
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button:hover {
        background-color: #104e8b;
    }
    .dark-mode {
        background-color: #2c3e50 !important;
    }
    .dark-mode .main {
        background-color: #2c3e50;
    }
    .dark-mode .card {
        background-color: #34495e;
        color: #ecf0f1;
    }
    .dark-mode .stButton>button {
        background-color: #3498db;
    }
    .dark-mode .stButton>button:hover {
        background-color: #2980b9;
    }
    .metrics-card {
        background: linear-gradient(135deg, #ffffff, #e6f0fa);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        font-family: 'Roboto', sans-serif;
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    .metrics-card div {
        padding: 5px;
        border-left: 4px solid #1e90ff;
    }
    table {
        font-family: 'Roboto', sans-serif;
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 4px;
        text-align: left;
    }
    th {
        background-color: #1e90ff;
        color: white;
    }
    .summary-card {
        background: linear-gradient(135deg, #ffffff, #f0f8ff);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .result-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .disabled-button {
        background-color: #cccccc !important;
        cursor: not-allowed !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback' not in st.session_state:
    st.session_state.feedback = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = []

# Define storage paths
processed_images_dir = "/Users/rajeevbarnwal/Desktop/Codes/AutoDamageEstimator/database/processed_images"
feedback_file = "/Users/rajeevbarnwal/Desktop/Codes/AutoDamageEstimator/database/feedback/ratings.csv"
os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
os.makedirs(processed_images_dir, exist_ok=True)

def save_feedback(rating: str):
    ts = time.strftime("%F %T")
    _cursor.execute(
        "INSERT INTO feedback (timestamp, rating) VALUES (?, ?)",
        (ts, rating),
    )
    _conn.commit()
    st.session_state.feedback_submitted = True
    st.session_state.feedback = rating

# Sidebar
st.sidebar.title("AutoDamageEstimator")
logo_path = "/Users/rajeevbarnwal/Desktop/Codes/AutoDamageEstimator/app/static/Auto_Damage.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

dark_mode = st.sidebar.toggle("Dark Mode", value=False)
if dark_mode:
    st.markdown('<body class="dark-mode">', unsafe_allow_html=True)
else:
    st.markdown('<body>', unsafe_allow_html=True)

st.sidebar.header("Settings")
model_options = ["Default (YOLOv8)", "Custom Model"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
advanced_options = st.sidebar.expander("Advanced Options", expanded=False)
with advanced_options:
    enable_learning = st.toggle("Enable Automatic Learning", value=False)
    if enable_learning:
        st.warning("Automatic Learning is not implemented yet. Contact admin for setup.")
    show_context = st.toggle("Show Context", value=False)
    if show_context:
        with st.expander("Context (Placeholder)"):
            st.write("Context display is not implemented yet. This will show detailed analysis context in future updates.")

st.sidebar.button("Admin Dashboard", on_click=lambda: switch_page("admin_dashboard"))

# Main content
st.markdown('<div class="main">', unsafe_allow_html=True)

# Input options: File upload or Camera
input_method = st.radio("Select Input Method", ["Upload Images", "Camera Capture"], horizontal=True)
if input_method == "Upload Images":
    st.markdown('<div class="card"><h2>Upload Your Images</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key="image_uploader")
    submit_button = st.button("Submit", key="submit_button")
elif input_method == "Camera Capture":
    st.markdown('<div class="card"><h2>Camera Capture</h2>', unsafe_allow_html=True)
    camera_image = st.camera_input("Take a picture", key="camera_input")
    submit_button = st.button("Submit", key="camera_submit")

results = []
total_cost = 0
saved_image_paths = []

if submit_button:
    if input_method == "Upload Images" and uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(os.path.dirname(__file__), f"temp_{uuid.uuid4()}.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                start_time = time.time()
                detections, cost = infer(temp_path, conf=confidence_threshold)
                
                # ---------- NEW: render YOLO annotations ---------------
                yolo_res      = _get_model().predict(temp_path,
                                                     conf=confidence_threshold,
                                                     save=False)
                plotted_rgb   = cv2.cvtColor(yolo_res[0].plot(), cv2.COLOR_BGR2RGB)
                # -------------------------------------------------------

                total_time = time.time() - start_time
            
            # Save to processed_images directory
            saved_path = os.path.join(processed_images_dir, f"processed_{uuid.uuid4()}.jpg")
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_image_paths.append(saved_path)

            results.append({"file": uploaded_file.name, "detections": detections,
                            "cost": cost, "time": total_time,
                            "plotted": plotted_rgb})
            total_cost += cost
            os.remove(temp_path)
    elif input_method == "Camera Capture" and camera_image:
        temp_path = os.path.join(os.path.dirname(__file__), f"temp_{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as f:
            f.write(camera_image.getbuffer())
        
        with st.spinner("Analyzing camera image..."):
            start_time = time.time()
            detections, cost = infer(temp_path, conf=confidence_threshold)
            total_time = time.time() - start_time
        
        # Save to processed_images directory
        saved_path = os.path.join(processed_images_dir, f"processed_{uuid.uuid4()}.jpg")
        with open(saved_path, "wb") as f:
            f.write(camera_image.getbuffer())
        saved_image_paths.append(saved_path)

        results.append({"file": "Camera Capture", "detections": detections, "cost": cost, "time": total_time, "plotted": plotted_rgb})
        total_cost += cost
        os.remove(temp_path)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>Results</h2>', unsafe_allow_html=True)
    for i, result in enumerate(results, 1):
        st.markdown(f'<div class="result-card"><h3>Result #{i}</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # use YOLO’s own plotted result (already has coloured boxes)
            plotted = results[i-1]["plotted"]
            st.image(plotted, caption="Detected damage", use_container_width=True)
            
        with col2:
            st.markdown('<h4>Detected damage</h4>', unsafe_allow_html=True)
            if result["detections"]:
                df = pd.DataFrame(result["detections"])
                st.dataframe(
                    df,
                    width=320,    # makes it narrower
                    height=min(240, 35 * len(df) + 35),
                    hide_index=True,
                )
            else:
                st.write("❌ No detections with the current threshold.")
            st.markdown(f'**Tentative Repair Cost:** ₹{result["cost"]:.2f}')
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary card
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<h2>Repair Summary</h2>', unsafe_allow_html=True)
    total_parts = sum(len(r["detections"]) for r in results)
    repair_parts = sum(1 for r in results for d in r["detections"] if d["severity"] == "moderate")
    replace_parts = total_parts - repair_parts
    st.markdown(f"**Total Parts Identified:** {total_parts}")
    st.markdown(f"**Parts to Repair:** {repair_parts}")
    st.markdown(f"**Parts to Replace:** {replace_parts}")
    st.markdown(f"**Total Cost of Repair:** ₹{total_cost:.2f} (Sum of individual costs: ₹{sum(r['cost'] for r in results):.2f})")
    st.markdown('</div>', unsafe_allow_html=True)

    # Human-like prompt with LLM
    if total_cost > 0 and _get_llm() is not None:
        llm = _get_llm()
        prompt = PromptTemplate(
            input_variables=["damages", "repair_parts", "replace_parts", "total_cost"],
            template="Greetings! We're so sorry to hear your car has been in an accident—let’s get it back on the road.\nAnswer: The vehicle has the following key damages: {damages}. It requires repair for {repair_parts} parts and replacement of {replace_parts} parts. The tentative cost to repair the vehicle is ₹{total_cost}. Offer a warm, empathetic tone, acknowledge the user's likely frustration, and suggest next steps like contacting a mechanic or scheduling a detailed inspection."
        )
        damages_list = ", ".join(f"{d['class']} ({d['severity']})" for r in results for d in r["detections"])
        with st.spinner("Processing LLM analysis..."):
            response = llm.invoke(prompt.format(damages=damages_list, repair_parts=repair_parts, replace_parts=replace_parts, total_cost=total_cost)).strip()
        st.markdown('<div class="card"><h3>Auto Damage Estimator Analysis</h3>', unsafe_allow_html=True)
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><h3>Feedback</h3>', unsafe_allow_html=True)

# … your "upload + infer" logic that populates results …

if results and not st.session_state.feedback_submitted:
    st.markdown('<div class="card"><h3>Feedback</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("👍 Good", key="fb_good",
                  on_click=save_feedback, args=("Good",))
    with c2:
        st.button("🤔 Neutral", key="fb_neutral",
                  on_click=save_feedback, args=("Neutral",))
    with c3:
        st.button("👎 Confusing", key="fb_confusing",
                  on_click=save_feedback, args=("Confusing",))
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.feedback_submitted:
    st.markdown('<div class="card"><h3>Feedback</h3>', unsafe_allow_html=True)
    st.success("✅ Thanks for your feedback!")
    st.markdown('</div>', unsafe_allow_html=True)

# Metrics card
st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
st.markdown('<h3>Analysis Metrics</h3>', unsafe_allow_html=True)
if results:
    detection_count = sum(len(r["detections"]) for r in results)
    avg_confidence = sum(d["confidence"] for r in results for d in r["detections"]) / detection_count if detection_count > 0 else 0.0
    total_tokens = len(str(results))  # Placeholder for token count
    with st.container():
        st.markdown(f'<div><span style="font-size: 16px;">📊 Tokens:</span> {total_tokens} (Prompt & Response)</div>', unsafe_allow_html=True)
        st.markdown(f'<div><span style="font-size: 16px;">✅ Avg Confidence:</span> {avg_confidence:.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div><span style="font-size: 16px;">📈 YOLO Vectors:</span> 1500</div>', unsafe_allow_html=True)  # Placeholder
        st.markdown(f'<div><span style="font-size: 16px;">⏱️ Total Gen Time:</span> {sum(r["time"] for r in results):.2f} s</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 10px; color: #6c757d;'>
        © 2025 AutoDamageEstimator | Built with ❤️ by #RajeevBarnwal
    </div>
""", unsafe_allow_html=True)