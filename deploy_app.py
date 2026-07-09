import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_rag_pipeline():
    documents = []
    loaded_files = []

    EXCLUDED_FILES = {
        "requirements.txt", ".env", "README.md",
        "app.py", "api.py", "deploy_app.py", "rag.py",
        "rag_chroma.py", "chunking.py", "doc_loader.py",
        "evaluate.py", "first_chain.py"
    }

    for file in os.listdir("."):
        if file in EXCLUDED_FILES:
            continue
        if file.endswith(".txt"):
            try:
                docs = TextLoader(file).load()
                documents.extend(docs)
                loaded_files.append(file)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
        elif file.endswith(".pdf"):
            try:
                docs = PyPDFLoader(file).load()
                documents.extend(docs)
                loaded_files.append(file)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
Use the following context to answer the question.
Be direct and concise. Answer in 2-3 sentences maximum.
If the answer is not in the context, say 'I don't know based on the provided documents.'

Context: {context}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm
    return vectorstore, chain, loaded_files

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Documents are loaded automatically
    2. Your question is searched semantically
    3. Relevant chunks passed to LLM
    4. Answer grounded in your documents
    """)
    st.markdown("---")
    st.markdown("Built with LangChain, FAISS, Groq & Streamlit")

# Load pipeline
with st.spinner("Loading RAG pipeline..."):
    vectorstore, chain, loaded_files = load_rag_pipeline()

# Main UI
st.title("📄 RAG Document Assistant")
st.markdown("Ask questions about your documents and get grounded answers with source citations.")

if loaded_files:
    st.success(f"Loaded: {', '.join(loaded_files)}")

st.markdown("---")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption(f"Sources: {message['sources']}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_chunks = vectorstore.similarity_search(question, k=2)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

            response = chain.invoke({
                "question": question,
                "context": context
            })

            answer = response.content
            sources = list(set([chunk.metadata["source"] for chunk in relevant_chunks]))
            sources_str = ", ".join(sources)

            st.write(answer)
            st.caption(f"Sources: {sources_str}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_str
            })