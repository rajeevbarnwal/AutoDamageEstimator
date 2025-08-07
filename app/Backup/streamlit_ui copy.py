# app/streamlit_ui.py
import sys
import os
from pathlib import Path

# Make “…/scripts” importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st
from scripts.infer import infer

st.set_page_config(page_title="AutoDamageEstimator", page_icon="🚗")
st.title("AutoDamageEstimator")
st.markdown("Upload an image of a damaged car and I’ll estimate the repair cost.")

uploaded_file = st.file_uploader("Choose an image …", type=["jpg", "png"])

if uploaded_file is not None:
    # Write the image to a temporary file that YOLO can open
    temp_path = ROOT / "app" / "temp_upload.jpg"
    with temp_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    detections, cost = infer(str(temp_path))

    st.image(str(temp_path), caption="Uploaded image", use_column_width=True)

    if detections:
        st.subheader("Detected damages")
        for d in detections:
            st.write(
                f"- **{d['class']}** &nbsp; "
                f"(severity : {d['severity']}, "
                f"confidence : {d['confidence']:.2f})"
            )
    else:
        st.info("No visible damage detected by the model.")

    st.markdown(f"### Estimated repair cost&nbsp;: &nbsp;₹ {cost:,.0f}")

    temp_path.unlink(missing_ok=True)  # clean-up
