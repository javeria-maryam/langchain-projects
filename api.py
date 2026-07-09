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

# Initialize FastAPI app
app = FastAPI()


# Load all txt and pdf files in the folder
documents = []
loaded_files = []

EXCLUDED_FILES = {"requirements.txt", ".env", "README.md", "app.py", "api.py", "rag.py", "rag_chroma.py", "chunking.py", "doc_loader.py", "evaluate.py", "first_chain.py"}

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

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
    Answer the question based ONLY on the following context.
    If the context does not contain a clear, direct answer to the question,
    you MUST respond with exactly: 'I don't know based on the provided documents.'
    Do not infer, guess, or use outside knowledge.
    
    Context:
    {context}"""),
])

chain = prompt | llm

# Define request and response models
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: list[str]

# POST endpoint
@app.post("/ask", response_model=Answer)
def ask_question(request: Question):
    relevant_chunks = vectorstore.similarity_search(request.question, k=2)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    response = chain.invoke({
        "question": request.question,
        "context": context
    })
    
    sources = list(set([chunk.metadata['source'] for chunk in relevant_chunks]))
    
    return Answer(answer=response.content, sources=sources)

# GET endpoint to check if API is running
@app.get("/")
def root():
    return {"status": "RAG API is running"}