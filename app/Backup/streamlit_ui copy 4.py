import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from scripts.infer import infer, _get_llm
from langchain.prompts import PromptTemplate
import cv2
from streamlit_extras.switch_page_button import switch_page
import time
import pandas as pd
from PIL import Image
import io

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
        font-size: 14px; /* Reduced font size */
    }
    th, td {
        border: 1px solid #ddd;
        padding: 4px; /* Reduced padding */
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
    </style>
""", unsafe_allow_html=True)

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

st.markdown('<div class="card"><h2>Upload Your Images</h2>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key="image_uploader")
submit_button = st.button("Submit", key="submit_button")

results = []
total_cost = 0

if submit_button and uploaded_files:
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(os.path.dirname(__file__), f"temp_{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            start_time = time.time()
            detections, cost = infer(temp_path, conf=confidence_threshold)
            total_time = time.time() - start_time
        
        results.append({"file": uploaded_file.name, "detections": detections, "cost": cost, "time": total_time})
        total_cost += cost
        os.remove(temp_path)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h2>Results</h2>', unsafe_allow_html=True)
    for result in results:
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(io.BytesIO(uploaded_file.getbuffer()))
            img_resized = img.resize((int(img.width * 0.5), int(img.height * 0.5)), Image.Resampling.LANCZOS)
            st.image(img_resized, caption=f"Image: {result['file']}")
        with col2:
            st.markdown('<h3>Detected Damages</h3>', unsafe_allow_html=True)
            if result["detections"]:
                df = pd.DataFrame(result["detections"], columns=["class", "severity", "confidence"])
                st.table(df.style.set_properties(**{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('background-color', '#1e90ff'), ('color', 'white')]}]))
            else:
                st.write("❌ No damages detected with the current threshold.")
            st.markdown(f'<h3>Tentative Repair Cost: ₹{result["cost"]:.2f}</h3>', unsafe_allow_html=True)

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
            template="Provide a human-friendly summary for a vehicle damage assessment. The vehicle has the following key damages: {damages}. It requires repair for {repair_parts} parts and replacement of {replace_parts} parts. The total estimated cost to repair the vehicle is ₹{total_cost}. Use a warm, reassuring tone and suggest next steps."
        )
        damages_list = ", ".join(f"{d['class']} ({d['severity']})" for r in results for d in r["detections"])
        response = llm.invoke(prompt.format(damages=damages_list, repair_parts=repair_parts, replace_parts=replace_parts, total_cost=total_cost))
        st.markdown('<div class="card"><h3>Auto Damage Estimator Analysis</h3>', unsafe_allow_html=True)
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

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

# Metrics card
st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
st.markdown('<h3>Analysis Metrics</h3>', unsafe_allow_html=True)
if results:
    avg_confidence = sum(d["confidence"] for r in results for d in r["detections"]) / len([d for r in results for d in r["detections"]]) if any(results) else 0.0
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