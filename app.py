import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

st.set_page_config(
    page_title="AI Attendance Dashboard",
    page_icon="🎓",
    layout="wide"
)

def load_data():
    """Load attendance data safely."""
    if not os.path.exists(config.ATTENDANCE_LOG_FILE):
        return pd.DataFrame(columns=["Name", "Date", "Time"])
    
    try:
        # If the file is empty, read_csv will throw empty data error
        if os.path.getsize(config.ATTENDANCE_LOG_FILE) == 0:
            return pd.DataFrame(columns=["Name", "Date", "Time"])
        return pd.read_csv(config.ATTENDANCE_LOG_FILE)
    except Exception as e:
        st.error(f"Error loading attendance records: {e}")
        return pd.DataFrame(columns=["Name", "Date", "Time"])

# Sidebar controls
st.sidebar.header("Controls & Filters")
auto_refresh = st.sidebar.checkbox("Auto-refresh data (every 2s)", value=True)

# Load logs
df = load_data()

st.title("AI Attendance Dashboard")
st.markdown("Monitor real-time student attendance captured by the face recognition system.")

# Create Tabs
tab_dashboard, tab_live = st.tabs(["Attendance Dashboard", "Live Feed"])

# Initialize session state for stream
if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False

with tab_dashboard:
    # Summary Metrics
    col1, col2, col3 = st.columns(3)
    total_records = len(df)
    today_str = datetime.now().strftime('%Y-%m-%d')
    today_records = len(df[df['Date'] == today_str]) if not df.empty and 'Date' in df.columns else 0
    unique_students = df['Name'].nunique() if not df.empty and 'Name' in df.columns else 0

    col1.metric("Total Records", total_records)
    col2.metric("Today's Attendance", today_records)
    col3.metric("Unique Students", unique_students)

    st.divider()

    # Data View
    st.subheader("Attendance Logs")

    if df.empty:
        st.info("No attendance records found yet. Ensure the stream server is running and detecting students.")
    else:
        # Adding a date filter
        if 'Date' in df.columns:
            dates = ["All"] + sorted(list(df['Date'].unique()), reverse=True)
        else:
            dates = ["All"]
            
        selected_date = st.sidebar.selectbox("Filter by Date", dates, index=0)
        
        if selected_date != "All" and 'Date' in df.columns:
            df_display = df[df['Date'] == selected_date]
        else:
            df_display = df

        st.dataframe(df_display, use_container_width=True)
        
        # Download button for excel/csv
        @st.cache_data
        def convert_df(original_df):
            return original_df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df_display)

        st.sidebar.download_button(
            label="Download records as CSV",
            data=csv,
            file_name='attendance_logs.csv',
            mime='text/csv',
        )

with tab_live:
    st.subheader("Live Camera Stream")
    
    # Input for Pi IP
    pi_ip = st.sidebar.text_input("Raspberry Pi IP Address", value=config.PI_IP_ADDRESS)
    stream_url = f"http://{pi_ip}:{config.STREAM_PORT}/video"
    
    st.markdown(f"Streaming from: `{stream_url}`")
    
    col_start, col_stop = st.columns(2)
    if col_start.button("Start Live Stream"):
        st.session_state.stream_active = True
    if col_stop.button("Stop Live Stream"):
        st.session_state.stream_active = False
        st.rerun()

    if st.session_state.stream_active:
        st.image(stream_url, caption="Live Feed from Raspberry Pi", use_container_width=True)
    else:
        st.info("Click 'Start Live Stream' to begin visualization. Note: The stream server must be running on the Pi.")

# Only auto-refresh if the stream is NOT active
if auto_refresh and not st.session_state.stream_active:
    time.sleep(2)
    st.rerun()
