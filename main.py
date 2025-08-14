import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import sys

# --- PATCH: Ensure working directory is set to the script location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

# --- PATCH: Check for data files in repo and adjust paths ---
DATA_FILE = os.path.join(SCRIPT_DIR, "data.jsonl")
SPARSE_ENCODER_FILE = os.path.join(SCRIPT_DIR, "sparse_encoder.pkl")

try:
    from rag_pipeline import run_rag_pipeline
    RAG_AVAILABLE = True
    st.sidebar.success("✅ RAG pipeline imported successfully")
except ImportError as e:
    st.error(f"❌ Could not import RAG pipeline: {e}")
    RAG_AVAILABLE = False

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING (unchanged) ---
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)  # Keep your existing CSS

# --- SESSION STATE INIT ---
for key in ['analysis_complete', 'last_query', 'narrative', 'dashboard_data', 'figures', 'selected_query']:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ['last_query', 'narrative', 'selected_query'] else {}

# --- SYSTEM STATUS ---
def check_system_status():
    status = {
        "rag_pipeline": RAG_AVAILABLE,
        "data_file": os.path.exists(DATA_FILE),
        "sparse_encoder": os.path.exists(SPARSE_ENCODER_FILE),
        "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
        "pinecone_api_key": bool(os.getenv("PINECONE_API_KEY"))
    }
    return status

def display_system_status():
    st.sidebar.markdown("## 🔧 System Status")
    status = check_system_status()
    for comp, available in status.items():
        if available:
            st.sidebar.markdown(f"✅ **{comp.replace('_', ' ').title()}**")
        else:
            st.sidebar.markdown(f"❌ **{comp.replace('_', ' ').title()}**")
    return all(status.values())

# --- SAMPLE QUERIES ---
def display_sample_queries():
    st.sidebar.markdown("## 💡 Sample Queries")
    for i, query in enumerate([
        "Recent developments in political campaigns",
        "Technology trends and AI discussions",
        "Healthcare policy debates",
        "Climate change discussions"
    ], 1):
        if st.sidebar.button(f"📌 {query}", key=f"sample_{i}"):
            st.session_state.selected_query = query
            st.rerun()

# --- RUN ANALYSIS ---
def run_analysis(user_query):
    with st.spinner("🔄 Running analysis..."):
        try:
            narrative, dashboard_data, figures = run_rag_pipeline(user_query)
            return narrative, dashboard_data, figures, None
        except Exception as e:
            return None, None, None, str(e)

# --- DISPLAY NARRATIVE ---
def display_narrative(narrative):
    st.markdown('<h2 class="sub-header">📖 Story Summary</h2>', unsafe_allow_html=True)
    st.markdown(f"""<div class="metric-card"><p class="narrative-text">{narrative}</p></div>""", unsafe_allow_html=True)

# --- DISPLAY VISUALIZATIONS ---
def display_visualizations(figures, dashboard_data):
    if not figures and not dashboard_data:
        st.info("No visualization data available.")
        return
    available_tabs = []
    if figures.get("posts_over_time"):
        available_tabs.append(("📅 Posts Over Time", "posts_over_time"))
    if figures.get("topics"):
        available_tabs.append(("🏷️ Topics Distribution", "topics"))
    if figures.get("sentiment"):
        available_tabs.append(("😊 Sentiment Analysis", "sentiment"))
    if figures.get("community_pie") or dashboard_data.get("subreddits"):
        available_tabs.append(("🥧 Community Distribution", "community_pie"))

    if not available_tabs:
        st.info("No visualizations available.")
        return

    tabs = st.tabs([t[0] for t in available_tabs])
    for tab, (name, key) in zip(tabs, available_tabs):
        with tab:
            if figures.get(key):
                st.plotly_chart(figures[key], use_container_width=True)
            if dashboard_data.get(key):
                df = pd.DataFrame(dashboard_data[key])
                st.dataframe(df)

# --- MAIN FUNCTION ---
def main():
    st.markdown('<h1 class="main-header">📊 Dashboard</h1>', unsafe_allow_html=True)

    system_ready = display_system_status()
    display_sample_queries()

    if not system_ready:
        st.error("⚠️ System not ready.")
        st.stop()

    st.markdown("## 🔍 Enter Your Analysis Query")
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input(
            "What would you like to analyze?",
            value=st.session_state.selected_query or "",
            placeholder="e.g., Recent developments in political campaigns"
        )
    with col2:
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if analyze_button and user_query.strip():
        st.session_state.selected_query = ""
        narrative, dashboard_data, figures, error = run_analysis(user_query)
        if error:
            st.error(f"❌ Analysis failed: {error}")
        else:
            st.session_state.narrative = narrative
            st.session_state.dashboard_data = dashboard_data
            st.session_state.figures = figures
            st.session_state.last_query = user_query
            st.session_state.analysis_complete = True
            st.success("✅ Analysis complete!")

    if st.session_state.analysis_complete:
        if st.session_state.narrative:
            display_narrative(st.session_state.narrative)
        display_visualizations(st.session_state.figures, st.session_state.dashboard_data)

if __name__ == "__main__":
    main()
