# LangChain RAG Projects

A collection of LangChain-based projects built while learning Generative AI development.

## Projects

### RAG Pipeline (`rag.py`)
A complete Retrieval-Augmented Generation system that answers questions from documents.

**Tech stack:** LangChain, FAISS, HuggingFace Embeddings, Groq (LLaMA 3.1)

**Features:**
- Document loading and intelligent chunking
- Semantic search using FAISS vector store
- Source citations with every answer
- Refuses to answer questions not in the document

### ChromaDB RAG (`rag_chroma.py`)
Same RAG pipeline but with persistent ChromaDB vector store.

**Tech stack:** LangChain, ChromaDB, HuggingFace Embeddings, Groq (LLaMA 3.1)

**Features:**
- Persistent vector store — no re-embedding on restart
- Auto-detects existing database and loads from disk
- Faster startup on subsequent runs

### Chunking Strategies (`chunking.py`)
Comparison of three text splitting strategies for RAG applications.

**Strategies compared:**
- Fixed size — fast but cuts mid-word
- Recursive character — respects sentence boundaries (chosen for production)
- Semantic — groups by meaning, slowest but highest quality

### Conversational Chatbot (`first_chain.py`)
LangChain chatbot with conversation memory using Groq LLaMA 3.1.

**Features:**
- Prompt templates with system instructions
- Full conversation history with HumanMessage/AIMessage
- Topic-restricted responses via system prompt

## Setup

```bash
git clone https://github.com/javeria-maryam/langchain-projects
cd langchain-projects
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your API key to `.env`:

GROQ_API_KEY=your_key_here

## Tech Stack
- LangChain
- FAISS / ChromaDB
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq (LLaMA 3.1-8b-instant)
- Python 3.13

