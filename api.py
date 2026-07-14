from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Load documents
documents = []
loaded_files = []

EXCLUDED_FILES = {
    "requirements.txt", ".env", "README.md", "app.py",
    "api.py", "rag.py", "rag_chroma.py", "chunking.py",
    "doc_loader.py", "evaluate.py", "first_chain.py", "requirements.docker.txt"
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
            print(f"Could not load {file}: {e}")
    elif file.endswith(".pdf"):
        try:
            docs = PyPDFLoader(file).load()
            documents.extend(docs)
            loaded_files.append(file)
        except Exception as e:
            print(f"Could not load {file}: {e}")

print(f"Loaded {len(documents)} document(s) from: {loaded_files}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Use the following context to answer the question.
Be direct and concise. Answer in 2-3 sentences maximum.
If the answer is not in the context, say 'I don't know based on the provided documents.'

Context: {context}"""),
    ("human", "{question}")
])

chain = prompt | llm

# Request and response models
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list[str]

# Endpoints
@app.get("/")
def root():
    return {"status": "RAG API is running", "loaded_files": loaded_files}

@app.post("/ask", response_model=Answer)
def ask_question(request: Question):
    relevant_chunks = vectorstore.similarity_search(request.question, k=2)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    response = chain.invoke({
        "question": request.question,
        "context": context
    })

    sources = list(set([chunk.metadata["source"] for chunk in relevant_chunks]))
    return Answer(answer=response.content, sources=sources)