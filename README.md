# research-engineering-intern-assignment

# 📊  Dashboard

## 🚀 Project Overview

This project is a sophisticated web dashboard designed for real-time social media analysis. It utilizes a **Retrieval-Augmented Generation (RAG)** pipeline to ingest, process, and analyze social media data from sources like Reddit. The goal is to provide a comprehensive, AI-powered platform for tracking trends, understanding sentiment, and identifying key topics related to a given user query.

This dashboard is an ideal showcase for a research engineering internship assignment, demonstrating proficiency in data pipelines, AI/ML integration, and interactive visualization.

### Key Features:

* **RAG Architecture:** Integrates a multi-agent system (Planner, Retriever, Narrative, Analysis, Visualization) to process and generate insights from data.

* **Hybrid Search:** Uses a combination of dense (semantic) and sparse (keyword/BM25) vector embeddings for highly relevant document retrieval from a Pinecone vector database.

* **Interactive Dashboard:** A Streamlit-based interface with dynamic visualizations (charts, graphs) powered by Plotly.

* **AI-Powered Narrative:** Generates a human-readable summary of the key findings from the retrieved data.

* **Sentiment and Topic Analysis:** Identifies the emotional tone and dominant themes within social media discussions.

## 🛠️ Technology Stack

| Category | Technology | Purpose | 
| ----- | ----- | ----- | 
| **Frontend** | `Streamlit` | Interactive web application framework | 
| **Backend/Core** | `Python` | Core logic for the RAG pipeline and data processing | 
| **LLM** | `Groq` with `Llama-3` | Powers the planner and narrative generation agents | 
| **Vector Database** | `Pinecone` | Stores vector embeddings for efficient hybrid search | 
| **ML Models** | `SentenceTransformers`, `BM25` | Generates dense and sparse embeddings for retrieval | 
| **Visualization** | `Plotly`, `pandas`, `numpy` | Creates interactive charts and data manipulation | 
| **Environment** | `.env` | Manages API keys and configuration variables securely | 

## 🖼️ Dashboard Screenshots

## 📋 Setup and Installation

Follow these steps to set up the project and run the dashboard locally.

### Prerequisites

* Python 3.8+

* `pip` package manager

* Groq API Key

* Pinecone API Key

* A Pinecone index named `social-media-hybrid-search`

* A `data.jsonl` file with your social media data.

* A `sparse_encoder.pkl` file generated from your dataset.

### Step 1: Clone the Repository

```
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

```

### Step 2: Install Dependencies

Create a virtual environment (recommended) and install the necessary libraries.

```
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt

```

> **Note:** The `requirements.txt` file should include `streamlit`, `pandas`, `numpy`, `plotly`, `pinecone-client`, `pinecone-text`, `sentence-transformers`, `langchain-groq`, `langchain-core`, `python-dotenv`, `scikit-learn`, `textblob`, and `networkx`.

### Step 3: Configure Environment Variables

Create a `.env` file in the root directory and add your API keys.

```
GROQ_API_KEY="your_groq_api_key"
PINECONE_API_KEY="your_pinecone_api_key"

```

### Step 4: Run the Dashboard

The dashboard can be started using the Streamlit CLI.

```
streamlit run main.py

```

This will open the application in your web browser. The dashboard will automatically check for the required files and dependencies and provide instructions if anything is missing.

## 📁 Project Structure

```
├── main.py                    # Main Streamlit dashboard application
├── rag_pipeline.py            # Contains all RAG agents and core logic
├── data.jsonl                 # Sample social media data file (not in repo)
├── sparse_encoder.pkl         # Serialized sparse encoder model (not in repo)
├── .env                       # Environment variables (gitignore this!)
└── README.md                  # This README file

```
