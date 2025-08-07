import streamlit as st
import pandas as pd
import sqlite3
import json
import os
from pathlib import Path

st.title("Admin Dashboard – AutoDamageEstimator")

# ── analytics.json as before ──
ANALYTICS = Path("logs/analytics.json")
# ... your existing analytics loading block ...

st.write("Request Analytics")
st.dataframe(pd.DataFrame(analytics))

# ── now load the same feedback table ──
DB_PATH = Path(__file__).parent.parent / "database" / "feedback.db"
conn = sqlite3.connect(DB_PATH)
try:
    feedback = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)
    st.write("User Feedback")
    st.dataframe(feedback)
except Exception as e:
    st.error(f"Could not load feedback table: {e}")
finally:
    conn.close()