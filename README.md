# RAG Document Assistant

A full-stack AI application that answers questions from your documents 
using Retrieval-Augmented Generation (RAG). Built with LangChain, 
FastAPI, and Streamlit.

## What it does

Upload a document and ask questions about it. The system retrieves 
the most relevant parts of your document and uses an LLM to generate 
accurate, grounded answers — with source citations. If the answer 
isn't in the document, it says so instead of making something up.

## Architecture

User Question (Streamlit UI)
↓
FastAPI REST API (POST /ask)
↓
FAISS Similarity Search
↓
Relevant chunks passed as context
↓
Groq LLM (LLaMA 3.1) generates answer
↓
Answer + Sources returned to UI

## Project Structure
langchain-projects/
├── api.py              # FastAPI backend — RAG pipeline + /ask endpoint
├── app.py              # Streamlit frontend — user interface
├── rag.py              # Core RAG pipeline (terminal version)
├── rag_chroma.py       # RAG pipeline with ChromaDB persistent storage
├── chunking.py         # Comparison of chunking strategies
├── doc_loader.py       # Document loading and embedding exploration
├── evaluate.py         # RAG evaluation with test Q&A set
├── first_chain.py      # LangChain basics — chains and memory
├── sample.txt          # Sample document for testing
├── requirements.txt    # Project dependencies
└── .env                # API keys (not committed)

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — vector store for similarity search
- **ChromaDB** — persistent vector store alternative
- **HuggingFace Embeddings** — all-MiniLM-L6-v2 (local, free)
- **Groq (LLaMA 3.1-8b-instant)** — LLM for answer generation
- **FastAPI** — REST API backend
- **Streamlit** — web UI frontend
- **Python 3.13**

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/javeria-maryam/langchain-projects
cd langchain-projects
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file in the root folder:
GROQ_API_KEY=your_groq_api_key_here
Get a free Groq API key at: https://console.groq.com

### 5. Add your document
Replace `sample.txt` with your own document.

## Running the Application

You need two terminals running simultaneously.

**Terminal 1 — Start the FastAPI backend:**
```bash
uvicorn api:app --reload
```
API will be available at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

**Terminal 2 — Start the Streamlit frontend:**
```bash
streamlit run app.py
```
UI will open automatically at: `http://localhost:8501`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /ask | Ask a question, returns answer + sources |

**POST /ask request:**
```json
{
    "question": "What is machine learning?"
}
```

**POST /ask response:**
```json
{
    "answer": "Machine learning is a subset of AI...",
    "sources": ["sample.txt"]
}
```

## Key Design Decisions

**Why RecursiveCharacterTextSplitter?**
Tested three chunking strategies (fixed-size, recursive, semantic). 
Recursive respects sentence boundaries without the speed cost of 
semantic chunking — best balance for production use.

**Why HuggingFace embeddings?**
Free, runs locally, no API quota issues. all-MiniLM-L6-v2 produces 
384-dimensional vectors with strong semantic similarity performance 
for most use cases.

**Why FAISS over ChromaDB?**
FAISS for the main pipeline (fast, in-memory, great for demos). 
ChromaDB version (rag_chroma.py) available for persistent storage 
when re-embedding large document sets would be too slow.

**Why Groq?**
Free tier with high rate limits. LLaMA 3.1-8b-instant is fast 
and capable for Q&A tasks.

## Live Demo
👉 [Try it here](https://rag-document-assistant-u2t9ygqgpcwhqt6r6omoju.streamlit.app)

## Author

Jaweria Maryam — [GitHub](https://github.com/javeria-maryam) | 
[LinkedIn](https://www.linkedin.com/in/jaweria-maryam-275096386)