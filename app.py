import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Documents are loaded automatically
    2. Your question is searched semantically
    3. Relevant chunks are passed to the LLM
    4. Answer is grounded in your documents
    """)
    st.markdown("---")
    st.markdown("### Loaded Documents")
    try:
        health = requests.get("http://127.0.0.1:8000/")
        if health.status_code == 200:
            st.success("API Connected ✓")
        else:
            st.error("API Error")
    except:
        st.error("API Offline — start uvicorn")
    st.markdown("---")
    st.markdown("Built with LangChain, FAISS, FastAPI & Streamlit")

# Main content
st.title("📄 RAG Document Assistant")
st.markdown("Ask questions about your documents and get grounded answers with source citations.")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            st.caption(f"Sources: {message['sources']}")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    sources = ", ".join(data["sources"])
                    st.caption(f"Sources: {sources}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": sources
                    })
                else:
                    st.error("API returned an error.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")