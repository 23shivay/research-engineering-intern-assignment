

import os
import json
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from urllib.parse import urlparse
from dotenv import load_dotenv
from sklearn.cluster import KMeans

# ML and NLP imports
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from textblob import TextBlob

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "social-media-hybrid-search"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with all necessary components."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0.2
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = None
        
        # Initialize encoders
        self.dense_encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.sparse_encoder = None
        
        print("✅ RAG Pipeline initialized")
    
    def setup_retriever(self) -> bool:
        """
        Setup the retriever components (Pinecone index and encoders).
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Connect to Pinecone index
            if INDEX_NAME not in self.pc.list_indexes().names():
                print(f"❌ Index '{INDEX_NAME}' not found. Please run data ingestion first.")
                return False
            
            self.index = self.pc.Index(INDEX_NAME)
            print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
            
            # Load sparse encoder
            sparse_encoder_path = os.path.join(os.path.dirname(__file__), "sparse_encoder.pkl")
            if os.path.exists(sparse_encoder_path):
                with open(sparse_encoder_path, "rb") as f:
                    self.sparse_encoder = pickle.load(f)
                print("✅ Loaded sparse encoder from file")
            else:
                print("❌ Sparse encoder file not found. Please run data ingestion first.")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up retriever: {str(e)}")
            return False

class PlannerAgent:
    """Agent responsible for generating search strategies."""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the planner agent."""
        instructions = """
You are a Search Strategy Planner for a social media analysis system that retrieves Reddit posts from a database.

Your ONLY job is to generate a valid JSON list of search queries. Do not include any conversational text, explanations, or markdown formatting.

Dataset Schema:
- title: Post title (string)
- selftext: Post body text (string) 
- subreddit: Reddit community (string)
- url: Post/external URL (string)
- created_utc: Unix timestamp (integer)
- score: Upvotes minus downvotes (integer)
- num_comments: Number of comments (integer)
- combined_text: Title + selftext (used for semantic search)

Given a user query about a news story or topic, generate 8-12 targeted search queries that:
1. Cover different aspects: key figures, locations, events, reactions, outcomes
2. Use diverse query types: keyword, semantic, hybrid, temporal
3. Include practical parameters:
   - time_window_days: Recent timeframe (30-180 days)
   - top_k: Results per query (50-200)
4. Consider dashboard visualizations: timelines, topics, sentiment, networks

Output format (JSON only):
[
  {
    "question": "specific search query text",
    "query_type": "keyword/semantic/hybrid/temporal",
    "reasoning": "why this query is important",
    "expected_data_patterns": "what insights this might reveal",
    "time_window_days": 90,
    "top_k": 100
  }
]
"""
        
        template = """
{instructions}

User Query: {user_query}
"""
        
        return PromptTemplate(
            template=template,
            input_variables=["user_query"],
            partial_variables={"instructions": instructions}
        )
    
    def generate_search_plan(self, user_query: str) -> List[Dict]:
        """
        Generate a strategic search plan for the given user query.
        
        Args:
            user_query (str): User's query about a news story or topic
            
        Returns:
            List[Dict]: List of search query specifications
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({"user_query": user_query})
            
            # Parse the JSON response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON array found in planner output")
            
            search_plan = json.loads(json_match.group(0))
            print(f"✅ Generated {len(search_plan)} search queries")
            
            return search_plan
            
        except Exception as e:
            print(f"❌ Error generating search plan: {str(e)}")
            # Return a fallback plan
            return [{
                "question": user_query,
                "query_type": "semantic",
                "reasoning": "Fallback semantic search",
                "expected_data_patterns": "General topic coverage",
                "time_window_days": 90,
                "top_k": 100
            }]

class RetrieverAgent:
    """Agent responsible for retrieving relevant documents from Pinecone with hybrid search."""

    def __init__(self, index, dense_encoder, sparse_encoder):
        """
        Args:
            index: Pinecone index object
            dense_encoder: SentenceTransformer (for dense embeddings)
            sparse_encoder: BM25Encoder (for sparse embeddings)
        """
        self.index = index
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder

    def retrieve_documents(self, search_plan: List[Dict], score_threshold: float = 0.2, alpha: float = 0.7) -> List[Dict]:
        """
        Retrieve documents using hybrid search and prepare them for downstream analysis.

        Args:
            search_plan (List[Dict]): Search queries from planner
            score_threshold (float): Minimum similarity score to include
            alpha (float): Weight given to dense vectors in hybrid search

        Returns:
            List[Dict]: Retrieved documents with metadata and embeddings
        """
        all_results = []
        seen_ids = set()

        print(f"🔍 Executing {len(search_plan)} search queries...")

        for i, query_spec in enumerate(search_plan, 1):
            query = query_spec["question"]
            top_k = query_spec.get("top_k", 100)

            try:
                # Create dense embedding for query
                dense_vector = self.dense_encoder.encode(query, convert_to_tensor=False)
                if hasattr(dense_vector, 'tolist'):
                    dense_vector = dense_vector.tolist()

                # Create sparse embedding for query
                sparse_vector = self.sparse_encoder.encode_queries([query])[0]

                # Hybrid search in Pinecone
                results = self.index.query(
                    vector=dense_vector,
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    include_metadata=True,
                    alpha=alpha  # blending between dense and sparse
                )

                # Process results
                for match in results.matches:
                    if match.score >= score_threshold and match.id not in seen_ids:
                        text = match.metadata.get("combined_text", "")

                        doc_data = {
                            "id": match.id,
                            "score": match.score,
                            "query_type": query_spec.get("query_type", "unknown"),
                            "source_query": query,
                            "metadata": match.metadata,
                            "text": text,
                            # Store dense embedding for topic clustering later
                            "embedding": self.dense_encoder.encode(
                                text, convert_to_numpy=True
                            ).tolist()
                        }

                        created_utc = match.metadata.get("created_utc", 0)
                        if created_utc:
                            doc_data["created_date"] = pd.to_datetime(
                                created_utc, unit='s'
                            ).strftime('%Y-%m-%d')
                        else:
                            doc_data["created_date"] = "unknown"

                        all_results.append(doc_data)
                        seen_ids.add(match.id)

                print(f"  Query {i}: Retrieved {len(results.matches)} documents")

            except Exception as e:
                print(f"  ⚠️ Query {i} failed: {str(e)}")
                continue

        # Sort by relevance score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"✅ Total unique documents retrieved: {len(all_results)}")
        return all_results


class NarrativeAgent:
    """Agent responsible for creating narrative summaries."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_narrative(self, documents: List[Dict], max_docs: int = 10) -> str:
        """
        Create a narrative summary from retrieved documents.
        
        Args:
            documents (List[Dict]): Retrieved documents
            max_docs (int): Maximum documents to use for summary
            
        Returns:
            str: Narrative summary
        """
        if not documents:
            return "No relevant documents found to create a narrative summary."
        
        # Use top documents for context
        top_docs = documents[:max_docs]
        context_texts = []
        
        for doc in top_docs:
            title = doc["metadata"].get("title", "")
            selftext = doc["metadata"].get("selftext", "")
            combined = f"{title}. {selftext}".strip()
            if combined:
                context_texts.append(combined[:500])  # Limit length
        
        context = "\n\n---\n\n".join(context_texts)
        
        prompt = f"""
Based on the following social media posts, create a concise narrative summary that:
1. Identifies the main story or topic
2. Highlights key events, people, and developments
3. Maintains objectivity without adding new information
4. Focuses on factual content from the posts

Social Media Posts:
{context}

Narrative Summary:
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Error creating narrative: {str(e)}")
            return "Unable to generate narrative summary due to processing error."


class AnalysisAgent:
    """Agent responsible for extracting structured data for visualization."""

    def __init__(self, llm):
        self.llm = llm

    def extract_dashboard_data(self, documents: List[Dict]) -> Dict[str, List]:
        if not documents:
            return self._empty_dashboard_data()

        print("📊 Analyzing documents for dashboard data...")

        # Data containers
        time_series_data = []
        posts_over_time_data = []
        topic_data = []
        sentiment_data = []
        engagement_data = []
        word_freq_data = []
        subreddit_data = []
        network_edges = []
        contributor_data = []

        embeddings = []  
        texts_for_topics = []

        for doc in documents:
            metadata = doc["metadata"]
            title = metadata.get("title", "")
            selftext = metadata.get("selftext", "")
            combined_text = f"{title} {selftext}".strip()

            score = metadata.get("score", 0)
            num_comments = metadata.get("num_comments", 0)
            created_utc = metadata.get("created_utc", 0)
            subreddit = metadata.get("subreddit", "unknown")
            url = metadata.get("url", "")
            author = metadata.get("author", "unknown")

            # Store embedding if available
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
                texts_for_topics.append(combined_text)

            # Time series data
            if created_utc:
                try:
                    date = pd.to_datetime(created_utc, unit='s').date()
                    time_series_data.append({
                        "date": date,
                        "score": score,
                        "comments": num_comments,
                        "engagement": score + num_comments
                    })
                    posts_over_time_data.append({"date": date, "posts": 1})
                except:
                    pass

            # Sentiment analysis
            if combined_text:
                try:
                    blob = TextBlob(combined_text)
                    sentiment_score = blob.sentiment.polarity
                except:
                    sentiment_score = 0.0

                sentiment_data.append({
                    "sentiment": sentiment_score,
                    "text_snippet": combined_text[:100] + "..." if len(combined_text) > 100 else combined_text,
                    "score": score
                })

                # Word frequency
                words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
                word_freq_data.extend(words)

            # Engagement data
            engagement_data.append({
                "title": title[:50] + "..." if len(title) > 50 else title,
                "score": score,
                "comments": num_comments,
                "engagement": score + num_comments,
                "subreddit": subreddit,
                "url": url
            })

            # Subreddit data
            subreddit_data.append({
                "subreddit": subreddit,
                "score": score,
                "post_count": 1
            })

            # Network edges: subreddit ↔ domain
            if url:
                domain = urlparse(url).netloc
                if domain:
                    network_edges.append({"source": subreddit, "target": domain})

            # Contributor data
            contributor_data.append({"author": author, "post_count": 1})

        # === Topic clustering ===
        if embeddings:
            try:
                n_clusters = min(5, len(embeddings))
                km = KMeans(n_clusters=n_clusters, random_state=42)
                labels = km.fit_predict(embeddings)

                for cluster_id in range(n_clusters):
                    cluster_texts = [texts_for_topics[i] for i in range(len(labels)) if labels[i] == cluster_id]
                    all_words = re.findall(r'\b[a-zA-Z]{4,}\b', " ".join(cluster_texts).lower())
                    top_words = [w for w, _ in Counter(all_words).most_common(3)]
                    topic_label = " ".join(top_words) if top_words else f"Topic {cluster_id+1}"

                    topic_data.append({
                        "topic": topic_label,
                        "score": sum([engagement_data[i]["score"] for i in range(len(labels)) if labels[i] == cluster_id]),
                        "post_count": sum([1 for i in range(len(labels)) if labels[i] == cluster_id])
                    })
            except Exception as e:
                print(f"⚠️ Topic clustering failed: {str(e)}")

        # Aggregate dashboard data
        dashboard_data = self._process_dashboard_data(
            time_series_data, posts_over_time_data, topic_data, sentiment_data,
            engagement_data, word_freq_data, subreddit_data,
            network_edges, contributor_data
        )

        print("✅ Dashboard data extracted successfully")
        return dashboard_data

    def _process_dashboard_data(self, time_series_data, posts_over_time_data, topic_data,
                                sentiment_data, engagement_data, word_freq_data,
                                subreddit_data, network_edges, contributor_data) -> Dict[str, List]:

        # Engagement over time
        if time_series_data:
            ts_df = pd.DataFrame(time_series_data)
            daily_stats = ts_df.groupby('date').agg({
                'score': 'sum',
                'comments': 'sum',
                'engagement': 'sum'
            }).reset_index()
            time_series_processed = daily_stats.to_dict('records')
        else:
            time_series_processed = []

        # Posts over time
        if posts_over_time_data:
            po_df = pd.DataFrame(posts_over_time_data)
            posts_stats = po_df.groupby('date').agg({'posts': 'sum'}).reset_index()
            posts_over_time_processed = posts_stats.to_dict('records')
        else:
            posts_over_time_processed = []

        # Topics
        topics_processed = pd.DataFrame(topic_data).to_dict('records') if topic_data else []

        # Sentiment
        sentiment_processed = sentiment_data if sentiment_data else []

        # Top posts
        engagement_sorted = sorted(engagement_data, key=lambda x: x['engagement'], reverse=True)
        top_posts = engagement_sorted[:15]

        # Word cloud
        if word_freq_data:
            word_counts = Counter(word_freq_data)
            stop_words = {'that', 'this', 'with', 'from', 'they', 'been', 'have', 'their', 'said', 'each', 'which', 'them'}
            filtered_words = {w: c for w, c in word_counts.items() if w not in stop_words and c > 1}
            word_cloud_data = [{"word": w, "frequency": c} for w, c in Counter(filtered_words).most_common(25)]
        else:
            word_cloud_data = []

        # Subreddits
        subreddits_processed = pd.DataFrame(subreddit_data).groupby('subreddit').agg({
            'score': 'sum',
            'post_count': 'sum'
        }).reset_index().to_dict('records') if subreddit_data else []

        # Network view
        network_processed = network_edges if network_edges else []

        # Community pie
        if contributor_data:
            contrib_df = pd.DataFrame(contributor_data)
            contrib_stats = contrib_df.groupby('author').agg({'post_count': 'sum'}).reset_index()
            community_pie_processed = contrib_stats.to_dict('records')
        else:
            community_pie_processed = []

        return {
            "time_series": time_series_processed,
            "posts_over_time": posts_over_time_processed,
            "topics": topics_processed,
            "sentiment": sentiment_processed,
            "top_posts": top_posts,
            "word_cloud": word_cloud_data,
            "subreddits": subreddits_processed,
            "network_view": network_processed,
            "community_pie": community_pie_processed
        }

    def _empty_dashboard_data(self) -> Dict[str, List]:
        return {
            "time_series": [],
            "posts_over_time": [],
            "topics": [],
            "sentiment": [],
            "top_posts": [],
            "word_cloud": [],
            "subreddits": [],
            "network_view": [],
            "community_pie": []
        }


class VisualizationAgent:
    """Agent responsible for creating visualizations."""

    def create_visualizations(self, dashboard_data: Dict[str, List]) -> Dict[str, Any]:
        figures = {}
        try:
            # Time series chart
            if dashboard_data.get("time_series"):
                figures["time_series"] = self._create_time_series_chart(dashboard_data["time_series"])

            # Posts over time
            if dashboard_data.get("posts_over_time"):
                figures["posts_over_time"] = self._create_posts_over_time_chart(dashboard_data["posts_over_time"])

            # Topics pie chart
            if dashboard_data.get("topics"):
                figures["topics"] = self._create_topics_chart(dashboard_data["topics"])

            # Sentiment histogram
            if dashboard_data.get("sentiment"):
                figures["sentiment"] = self._create_sentiment_chart(dashboard_data["sentiment"])

            # Top posts bar chart
            if dashboard_data.get("top_posts"):
                figures["top_posts"] = self._create_engagement_chart(dashboard_data["top_posts"])

            # Word cloud bar chart
            if dashboard_data.get("word_cloud"):
                figures["word_cloud"] = self._create_word_frequency_chart(dashboard_data["word_cloud"])

            # Subreddits chart
            if dashboard_data.get("subreddits"):
                figures["subreddits"] = self._create_subreddits_chart(dashboard_data["subreddits"])

            # Network view
            if dashboard_data.get("network_view"):
                figures["network_view"] = self._create_network_view_chart(dashboard_data["network_view"])

            # Community pie chart
            if dashboard_data.get("subreddits"):
                figures["community_pie"] = self._create_community_pie_chart(dashboard_data["subreddits"])

        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")

        return figures

    def _create_time_series_chart(self, time_series_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(time_series_data)
        if df.empty:
            return self._create_empty_chart("Time Series", "No time series data available")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['engagement'],
            mode='lines+markers',
            name='Daily Engagement',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Engagement Over Time",
            xaxis_title="Date",
            yaxis_title="Total Engagement (Score + Comments)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        return fig

    def _create_posts_over_time_chart(self, posts_over_time_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(posts_over_time_data)
        if df.empty:
            return self._create_empty_chart("Posts Over Time", "No posts over time data available")
            
        fig = px.bar(
            df,
            x='date',
            y='posts',
            title="Posts Over Time",
            labels={'date': 'Date', 'posts': 'Number of Posts'},
            color='posts',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            template="plotly_white",
            height=400
        )
        return fig

    def _create_topics_chart(self, topics_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(topics_data)
        if df.empty:
            return self._create_empty_chart("Topics Distribution", "No topics data available")
            
        fig = px.pie(
            df,
            values='post_count',
            names='topic',
            title="Distribution of Topics",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            template="plotly_white",
            height=400
        )
        return fig

    def _create_sentiment_chart(self, sentiment_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(sentiment_data)
        if df.empty:
            return self._create_empty_chart("Sentiment Distribution", "No sentiment data available")
            
        fig = px.histogram(
            df,
            x='sentiment',
            title="Sentiment Distribution",
            nbins=20,
            labels={'sentiment': 'Sentiment Score', 'count': 'Number of Posts'},
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Sentiment Score (-1 = Negative, 0 = Neutral, 1 = Positive)",
            height=400
        )
        return fig

    def _create_engagement_chart(self, top_posts_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(top_posts_data[:10])
        if df.empty:
            return self._create_empty_chart("Top Posts", "No engagement data available")
            
        fig = px.bar(
            df,
            x='engagement',
            y='title',
            orientation='h',
            title="Top Posts by Engagement",
            labels={'engagement': 'Engagement Score', 'title': 'Post Title'},
            color='engagement',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        return fig

    def _create_word_frequency_chart(self, word_cloud_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(word_cloud_data[:15])
        if df.empty:
            return self._create_empty_chart("Word Frequency", "No word frequency data available")
            
        fig = px.bar(
            df,
            x='frequency',
            y='word',
            orientation='h',
            title="Most Frequent Words",
            labels={'frequency': 'Frequency', 'word': 'Word'},
            color='frequency',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        return fig

    def _create_subreddits_chart(self, subreddits_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(subreddits_data[:10])
        if df.empty:
            return self._create_empty_chart("Subreddits", "No subreddit data available")
            
        fig = px.bar(
            df,
            x='post_count',
            y='subreddit',
            orientation='h',
            title="Most Active Subreddits",
            labels={'post_count': 'Number of Posts', 'subreddit': 'Subreddit'},
            color='score',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        return fig

    def _create_network_view_chart(self, network_view_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(network_view_data)
        if df.empty:
            return self._create_empty_chart("Network View", "No network data available")
        
        # Create network graph
        G = nx.Graph()
        
        for _, row in df.iterrows():
            G.add_node(row['source'], type='subreddit')
            G.add_node(row['target'], type='domain')
            G.add_edge(row['source'], row['target'])
        
        if len(G.nodes()) == 0:
            return self._create_empty_chart("Network View", "No network connections found")
        
        pos = nx.spring_layout(G, k=0.5, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append('blue' if G.nodes[node]['type'] == 'subreddit' else 'orange')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(size=10, color=node_color),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Network View (Subreddit ↔ Domain)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        
        return fig

    def _create_community_pie_chart(self, subreddits_data: List[Dict]) -> go.Figure:
        df = pd.DataFrame(subreddits_data[:10])  # Top 10 subreddits
        if df.empty:
            return self._create_empty_chart("Community Distribution", "No community data available")
            
        fig = px.pie(
            df,
            values='post_count',
            names='subreddit',
            title="Community Distribution by Subreddit",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            template="plotly_white",
            height=400
        )
        return fig

    def _create_empty_chart(self, title: str, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            template="plotly_white",
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig


# Main RAG Pipeline Orchestrator
def run_rag_pipeline(user_query: str) -> Tuple[str, Dict, Dict]:
    """
    Run the complete RAG pipeline for a user query.
    
    Args:
        user_query (str): User's search query
        
    Returns:
        Tuple[str, Dict, Dict]: (narrative, dashboard_data, figures)
    """
    print(f"🚀 Running RAG pipeline for query: '{user_query}'")
    
    # Initialize pipeline
    rag_pipeline = RAGPipeline()
    
    # Setup retriever
    if not rag_pipeline.setup_retriever():
        return "Setup failed", {}, {}
    
    # Initialize agents
    planner = PlannerAgent(rag_pipeline.llm)
    retriever = RetrieverAgent(rag_pipeline.index, rag_pipeline.dense_encoder, rag_pipeline.sparse_encoder)
    narrative_agent = NarrativeAgent(rag_pipeline.llm)
    analysis_agent = AnalysisAgent(rag_pipeline.llm)
    viz_agent = VisualizationAgent()
    
    try:
        # Step 1: Generate search plan
        search_plan = planner.generate_search_plan(user_query)
        
        # Step 2: Retrieve documents
        documents = retriever.retrieve_documents(search_plan)
        
        # Step 3: Create narrative
        narrative = narrative_agent.create_narrative(documents)
        
        # Step 4: Extract dashboard data
        dashboard_data = analysis_agent.extract_dashboard_data(documents)
        
        # Step 5: Create visualizations
        figures = viz_agent.create_visualizations(dashboard_data)
        
        print("✅ RAG pipeline completed successfully")
        return narrative, dashboard_data, figures
        
    except Exception as e:
        print(f"❌ RAG pipeline error: {str(e)}")
        return f"Error: {str(e)}", {}, {}

if __name__ == "__main__":
    # Test the pipeline
    test_query = "Recent developments in political campaigns and elections"
    narrative, data, figures = run_rag_pipeline(test_query)
    print("\n" + "="*50)
    print("NARRATIVE:")
    print(narrative)
    print("\n" + "="*50)