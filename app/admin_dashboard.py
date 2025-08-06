import streamlit as st
import pandas as pd
import sqlite3
import json

st.title("Admin Dashboard - AutoDamageEstimator")

# Load analytics
with open('logs/analytics.json', 'r') as f:
    analytics = json.load(f)

st.write("Request Analytics")
df = pd.DataFrame(analytics)
st.dataframe(df)

# Load feedback
conn = sqlite3.connect('database/feedback.db')
feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
st.write("User Feedback")
st.dataframe(feedback)
conn.close()