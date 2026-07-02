from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()


# Step 1: Load document
loader = TextLoader("sample.txt")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings and store in ChromaDB
import os

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("./chroma_db"):
    # Database already exists - just load it
    print("Loading existing ChromaDB from disk...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    # First run - create everything
    print("Creating ChromaDB for the first time...")
    loader = TextLoader("sample.txt")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

print(f"Total chunks stored: {vectorstore._collection.count()}")

# Step 4: Set up LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 5: Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
    Answer the question based only on the following context:
    {context}
    If the answer is not in the context, say 'I don't know based on the provided document.'"""),
    ("human", "{question}")
])

chain = prompt | llm

# Step 6: Interactive loop
while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        break

    relevant_chunks = vectorstore.similarity_search(question, k=2)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])

    response = chain.invoke({
        "question": question,
        "context": context
    })

    print(f"Bot: {response.content}")
    sources = set([chunk.metadata['source'] for chunk in relevant_chunks])
    print(f"Sources: {', '.join(sources)}")