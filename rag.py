from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

# Step 3: Create embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Set up LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 5: Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. 
    Answer the question based only on the following context:
    {context}
    If the answer is not in the context, say 'I don't know based on the provided document.'"""),
    ("human", "{question}")
])

# Step 6: Chain everything together
chain = prompt | llm

# Step 7: Interactive loop
while True:
    question = input("\nYou: ")
    if question.lower() == "quit":
        break
    
    # Get relevant chunks from FAISS
    relevant_chunks = vectorstore.similarity_search(question, k=2)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Get answer from LLM
    response = chain.invoke({
        "question": question,
        "context": context
    })
    
    print(f"Bot: {response.content}")