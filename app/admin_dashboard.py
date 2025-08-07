import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import json
import sqlite3
import plotly.express as px

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / "database" / "feedback.db"
ANALYTICS_FILE = LOGS_DIR / "analytics.json"
REQUESTS_FILE = LOGS_DIR / "requests.csv"

# Page configuration
st.set_page_config(
    page_title="Admin Dashboard – AutoDamageEstimator",
    page_icon="🚗",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("Admin Dashboard")
pages = ["Overview", "Request Logs", "Request Analytics", "Feedback Analytics", "System Logs"]
page = st.sidebar.radio("Go to", pages)

# Date range filter
today = pd.Timestamp.now()
date_range = st.sidebar.selectbox(
    "Date Range",
    ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
    index=0
)
if date_range != "All Time":
    if date_range == "Last 24 Hours":
        start_date = today - pd.Timedelta(days=1)
    elif date_range == "Last 7 Days":
        start_date = today - pd.Timedelta(days=7)
    else:
        start_date = today - pd.Timedelta(days=30)
else:
    start_date = None

# Data loading functions
def load_analytics():
    if ANALYTICS_FILE.exists():
        try:
            return json.loads(ANALYTICS_FILE.read_text())
        except json.JSONDecodeError:
            st.warning("analytics.json is invalid.")
    return {}

def load_requests():
    if REQUESTS_FILE.exists():
        df = pd.read_csv(REQUESTS_FILE)
        if start_date is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df[df['timestamp'] >= start_date]
        return df
    else:
        st.warning("requests.csv not found.")
        return pd.DataFrame()

def load_feedback():
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)
        conn.close()
        if start_date is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df[df['timestamp'] >= start_date]
        return df
    else:
        st.warning("Feedback database not found.")
        return pd.DataFrame()

analytics = load_analytics()
requests_df = load_requests()
feedback_df = load_feedback()

# Pages
def show_overview():
    st.title("📊 Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    total_requests = analytics.get("total_requests", len(requests_df))
    col1.metric("Total Requests", total_requests)
    success = analytics.get("successful_requests", None)
    if success is not None:
        rate = success / total_requests * 100 if total_requests > 0 else 0
        col2.metric("Success Rate", f"{rate:.1f}%")
    avg_time = analytics.get("average_response_time", requests_df['response_time'].mean() if 'response_time' in requests_df else 0)
    col3.metric("Avg. Response Time", f"{avg_time:.2f}s")

    st.subheader("Requests Over Time")
    if not requests_df.empty and 'timestamp' in requests_df:
        line = requests_df.copy()
        line['hour'] = pd.to_datetime(line['timestamp']).dt.floor('h')
        hourly = line.groupby('hour').size().reset_index(name='count')
        fig = px.line(hourly, x='hour', y='count', title='Requests / Hour')
        st.plotly_chart(fig, use_container_width=True)


def show_request_logs():
    st.title("📋 Request Logs")
    if not requests_df.empty:
        st.dataframe(requests_df, use_container_width=True)
    else:
        st.info("No request data available.")


def show_request_analytics():
    st.title("🔎 Request Analytics")
    st.subheader("Response Time Distribution")
    if 'response_time' in requests_df:
        fig = px.histogram(requests_df, x='response_time', nbins=20, title='Response Time Distribution')
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top Models")
    if 'model' in requests_df:
        top = requests_df['model'].value_counts().reset_index()
        top.columns = ['Model', 'Count']
        fig = px.bar(top, x='Model', y='Count', title='Requests by Model')
        st.plotly_chart(fig, use_container_width=True)


def show_feedback_analytics():
    st.title("⭐ Feedback Analytics")
    if not feedback_df.empty:
        st.subheader("Feedback Overview")
        counts = feedback_df['rating'].value_counts().reset_index()
        counts.columns = ['Rating', 'Count']
        fig = px.pie(counts, names='Rating', values='Count', title='Feedback Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Feedback Over Time")
        df = feedback_df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby(['date','rating']).size().reset_index(name='count')
        fig2 = px.line(daily, x='date', y='count', color='rating', title='Feedback Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No feedback data available.")


def show_system_logs():
    st.title("🛠 System Logs")
    error_file = LOGS_DIR / "error.log"
    if error_file.exists():
        with open(error_file) as f:
            lines = f.readlines()[-100:]
        st.text_area("Recent Errors", "".join(lines), height=300)
    else:
        st.info("error.log not found.")

# Display based on navigation
if page == "Overview":
    show_overview()
elif page == "Request Logs":
    show_request_logs()
elif page == "Request Analytics":
    show_request_analytics()
elif page == "Feedback Analytics":
    show_feedback_analytics()
elif page == "System Logs":
    show_system_logs()
