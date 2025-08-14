

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os


try:
    from rag_pipeline import run_rag_pipeline
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Could not import RAG pipeline: {e}")
    RAG_AVAILABLE = False


st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
     .metric-card {
        background-color: #000000; /* Changed to black */
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .narrative-text {
        /* You must change the text color to white for readability on a black background */
        color: #ffffff;
        font-size: 1.1rem;
        line-height: 1.7;
        text-align: justify;
        margin: 0;
        font-weight: 400;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 4px;
        color: #1f77b4;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    .query-info {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #333333; /* Add text color */
    }
    
    .metric-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333; /* Add text color */
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'narrative' not in st.session_state:
    st.session_state.narrative = ""
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = {}
if 'figures' not in st.session_state:
    st.session_state.figures = {}
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

import os

def check_system_status():
    """Check if all system components are available."""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Create robust paths for the files
    data_file_path = os.path.join(script_dir, "data.jsonl")
    sparse_encoder_path = os.path.join(script_dir, "sparse_encoder.pkl")
    
    status = {
        "rag_pipeline": RAG_AVAILABLE,
        "data_file": os.path.exists(data_file_path),
        "sparse_encoder": os.path.exists(sparse_encoder_path),
    }
    return status 
def display_system_status():
    """Display system status in the sidebar."""
    st.sidebar.markdown("## 🔧 System Status")
    
    status = check_system_status()
    
    for component, available in status.items():
        if available:
          #  st.sidebar.markdown(f"✅ **{component.replace('_', ' ').title()}**: Ready")
          print("all ready")
        else:
            st.sidebar.markdown(f"❌ **{component.replace('_', ' ').title()}**: Not Available")
    
    all_ready = all(status.values())
    
    if all_ready:
        st.sidebar.markdown('<p class="status-success">🟢 All systems ready!</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">🔴 Some components missing</p>', unsafe_allow_html=True)
        
        st.sidebar.markdown("### 📋 Setup Instructions:")
        
        if not status["data_file"]:
            st.sidebar.markdown("1. Add `data.jsonl` file")
        if not status["sparse_encoder"]:
            st.sidebar.markdown("2. Run data ingestion script")
    
    return all_ready

def display_sample_queries():
    """Display sample queries for user reference."""
    st.sidebar.markdown("## 💡 Sample Queries")
    
    sample_queries = [
        "Recent developments in political campaigns",
        "Technology trends and AI discussions",
        "Healthcare policy debates",
        "Climate change discussions",
        "Economic market trends",
        "Social media platform changes",
        "Educational technology adoption",
        "Remote work culture shifts"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        if st.sidebar.button(f"📌 {query}", key=f"sample_{i}"):
            st.session_state.selected_query = query
            st.rerun()

def run_analysis(user_query):
    """Run the RAG pipeline analysis."""
    with st.spinner("🔄 Running analysis... This may take 1-2 minutes."):
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("🎯 Generating search strategy...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        status_text.text("🔍 Retrieving relevant documents...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        status_text.text("📝 Creating narrative summary...")
        progress_bar.progress(60)
        
        # Run the actual pipeline
        try:
            narrative, dashboard_data, figures = run_rag_pipeline(user_query)
            
            status_text.text("📊 Generating visualizations...")
            progress_bar.progress(80)
            time.sleep(0.5)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return narrative, dashboard_data, figures, None
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            return None, None, None, str(e)

def display_metrics(dashboard_data):
    """Display key metrics from the analysis."""
    if not dashboard_data:
        return
    

def display_narrative(narrative):
    """Display the narrative summary with proper styling."""
    st.markdown('<h2 class="sub-header">📖 Story Summary</h2>', unsafe_allow_html=True)
    
    # Create an attractive container for the narrative with proper text color
    st.markdown(f"""
    <div class="metric-card">
        <p class="narrative-text">{narrative}</p>
    </div>
    """, unsafe_allow_html=True)

def display_visualizations(figures, dashboard_data):
    """Display all visualizations in tabs."""
    if not figures and not dashboard_data:
        st.info("No visualization data available.")
        return
    
    st.markdown('<h2 class="sub-header">📈 Interactive Analytics</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    available_tabs = []
    
    # Check which visualizations are available
    if figures.get("posts_over_time") or dashboard_data.get("posts_over_time"):
        available_tabs.append(("📅 Posts Over Time", "posts_over_time"))
    
    #if figures.get("time_series") or dashboard_data.get("time_series"):
       # available_tabs.append(("📈 Engagement Timeline", "time_series"))
    
    if figures.get("topics") or dashboard_data.get("topics"):
        available_tabs.append(("🏷️ Topics Distribution", "topics"))
    
    if figures.get("sentiment") or dashboard_data.get("sentiment"):
        available_tabs.append(("😊 Sentiment Analysis", "sentiment"))
    
    #if figures.get("top_posts") or dashboard_data.get("top_posts"):
       # available_tabs.append(("🔥 Top Posts", "top_posts"))
    
    #if figures.get("word_cloud") or dashboard_data.get("word_cloud"):
       # available_tabs.append(("💭 Key Words", "word_cloud"))
    
    # if figures.get("subreddits") or dashboard_data.get("subreddits"):
    #     available_tabs.append(("🌐 Communities", "subreddits"))
    
    if figures.get("community_pie") or dashboard_data.get("subreddits"):
        available_tabs.append(("🥧 Community Distribution", "community_pie"))
    
    if figures.get("network_view") or dashboard_data.get("network_view"):
        available_tabs.append(("🌐 Network View", "network_view"))
    
    if not available_tabs:
        st.info("No visualizations available for this query.")
        return
    
    # Create tabs
    tab_names = [tab[0] for tab in available_tabs]
    tab_keys = [tab[1] for tab in available_tabs]
    tabs = st.tabs(tab_names)
    
    for i, (tab, key) in enumerate(zip(tabs, tab_keys)):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display the figure
                figure = figures.get(key)
                if figure:
                    st.plotly_chart(figure, use_container_width=True)
                else:
                    st.info("Visualization not available for this data.")
            
            with col2:
                # Display data summary
                data = dashboard_data.get(key, [])
                if key == "community_pie":  # Special case for community pie
                    data = dashboard_data.get("subreddits", [])
                
                if data:
                    st.markdown("### 📊 Data Summary")
                    if isinstance(data, list) and len(data) > 0:
                        try:
                            df = pd.DataFrame(data)
                            # Handle date columns
                            for col in df.columns:
                                if col == 'date' and not df[col].empty:
                                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                            
                            st.dataframe(df.head(10), use_container_width=True)
                            
                            if len(df) > 10:
                                st.info(f"Showing top 10 of {len(df)} records")
                        except Exception as e:
                            st.error(f"Error displaying data: {str(e)}")
                    else:
                        st.info("No data available")
                else:
                    st.info("No data available for this visualization.")


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">📊  Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666; font-size: 1.1rem;">
        Analyze   trends, sentiment, and engagement patterns using AI-powered insights
    </div>
    """, unsafe_allow_html=True)
    
    # Check system status and display sidebar
    system_ready = display_system_status()
    display_sample_queries()
    
    if not system_ready:
        st.error("⚠️ System not ready. Please complete the setup steps shown in the sidebar.")
        st.stop()
    
    # Main input section
    st.markdown("## 🔍 Enter Your Analysis Query")
    
    # Query input - Use the selected query from session state
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Use the selected query if available, otherwise use empty string
        default_value = st.session_state.selected_query if st.session_state.selected_query else ""
        
        user_query = st.text_input(
            "What would you like to analyze?",
            value=default_value,
            placeholder="e.g., Recent developments in political campaigns and elections",
            key="user_query_input",
            help="Enter a topic, event, or trend you want to analyze from social media data"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Analysis section
    if analyze_button and user_query.strip():
        # Clear the selected query after using it
        st.session_state.selected_query = ""
        
        # Reset previous results
        st.session_state.analysis_complete = False
        
        # Run analysis
        narrative, dashboard_data, figures, error = run_analysis(user_query)
        
        if error:
            st.error(f"❌ Analysis failed: {error}")
            st.info("💡 Try rephrasing your query or check the system status.")
        else:
            # Store results in session state
            st.session_state.narrative = narrative
            st.session_state.dashboard_data = dashboard_data
            st.session_state.figures = figures
            st.session_state.last_query = user_query
            st.session_state.analysis_complete = True
            
            st.success("✅ Analysis completed successfully!")
            st.rerun()

    elif analyze_button and not user_query.strip():
        st.warning("Please enter a query to analyze.")

    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        st.markdown("---")
        
        
        # Display metrics
        display_metrics(st.session_state.dashboard_data)
        
        # Display narrative
        if st.session_state.narrative:
            display_narrative(st.session_state.narrative)
        
        # Display visualizations
        display_visualizations(st.session_state.figures, st.session_state.dashboard_data)
        
        # Export options
        st.markdown("---")
        st.markdown("## 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Summary Report"):
                # Create summary report
                report = f"""
Social Media Analysis Report
===========================
Query: {st.session_state.last_query}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NARRATIVE SUMMARY:
{st.session_state.narrative}

KEY METRICS:
- Total Posts: {len(st.session_state.dashboard_data.get('top_posts', []))}
- Topics Identified: {len(st.session_state.dashboard_data.get('topics', []))}
- Time Points: {len(st.session_state.dashboard_data.get('time_series', []))}
- Communities: {len(st.session_state.dashboard_data.get('subreddits', []))}
"""
                st.download_button(
                    label="📁 Download Report",
                    data=report,
                    file_name=f"social_media_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.session_state.dashboard_data:
                # Convert dashboard data to CSV
                try:
                    # Combine all data into one DataFrame for export
                    export_data = []
                    
                    # Add top posts data
                    for post in st.session_state.dashboard_data.get('top_posts', []):
                        export_data.append({
                            'type': 'post',
                            'title': post.get('title', ''),
                            'score': post.get('score', 0),
                            'comments': post.get('comments', 0),
                            'engagement': post.get('engagement', 0),
                            'subreddit': post.get('subreddit', ''),
                            'url': post.get('url', '')
                        })
                    
                    if export_data:
                        export_df = pd.DataFrame(export_data)
                        csv_data = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📊 Download Data (CSV)",
                            data=csv_data,
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No data available for export")
                except Exception as e:
                    st.error(f"Error preparing export: {str(e)}")
        
        with col3:
            if st.button("🔄 New Analysis"):
                # Clear session state for new analysis
                for key in ['analysis_complete', 'narrative', 'dashboard_data', 'figures', 'selected_query']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Footer with instructions
        st.markdown("---")
        with st.expander("ℹ️ How to Use This Dashboard"):
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Enter a Query**: Type your analysis question in the text box above
            2. **Click Analyze**: The system will search and analyze social media data
            3. **Explore Results**: Navigate through tabs to see different insights
            4. **Export Data**: Download summaries and data for further analysis
            
            ### 💡 Query Tips
            
            - **Be Specific**: "2024 election campaign strategies" vs "politics"
            - **Use Keywords**: Include relevant terms that might appear in posts
            - **Time-Sensitive**: Recent events will have more relevant data
            - **Multi-Aspect**: Ask about trends, sentiment, or specific communities
            
            ### 📊 Visualization Types
            
            - **Posts Over Time**: How posting volume changes over time
            - **Engagement Timeline**: How discussion engagement changes over time
            - **Topics Distribution**: Main themes and their distribution
            - **Sentiment Analysis**: Emotional tone of discussions
            - **Top Posts**: Most engaging content
            - **Key Words**: Most frequently mentioned terms
            - **Communities**: Most active subreddits/platforms
            - **Network View**: Connections between communities and domains
            
            ### 🔧 Technical Notes
            
            - Data is retrieved using hybrid search (semantic + keyword matching)
            - Sentiment analysis uses TextBlob for polarity scoring
            - Topic extraction combines keyword matching and AI analysis
            - All visualizations are interactive - hover for details
            """)

        # About section
        with st.expander("📋 About This Project"):
            st.markdown("""
            ### 🎯 Project Overview
            
            This dashboard is part of a research engineering internship assignment focused on 
            social media analysis and misinformation tracking. It demonstrates:
            
            - **Data Pipeline**: Ingestion, processing, and vectorization of social media data
            - **AI/ML Integration**: Hybrid search, sentiment analysis, and topic modeling 
            - **Interactive Visualization**: Real-time dashboards with multiple chart types
            - **RAG Architecture**: Retrieval-Augmented Generation for intelligent analysis
            
            ### 🛠️ Technology Stack
            
            - **Frontend**: Streamlit for interactive web interface
            - **Vector Database**: Pinecone for hybrid search capabilities
            - **ML Models**: SentenceTransformers for embeddings, BM25 for sparse retrieval
            - **LLM**: Groq API with Llama-3 for text generation and analysis
            - **Visualization**: Plotly for interactive charts and graphs
            - **Data Processing**: Pandas, NumPy for data manipulation
            
            ### 📊 Data Sources
            
            The dashboard analyzes social media posts with the following attributes:
            - Post titles and content
            - Engagement metrics (scores, comments)
            - Community information (subreddits)
            - Temporal data (creation timestamps)
            - External links and references
            """)

if __name__ == "__main__":
    main()
