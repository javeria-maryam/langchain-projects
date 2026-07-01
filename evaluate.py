from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Setup RAG pipeline
loader = TextLoader("sample.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based only on the following context:
    {context}
    If the answer is not in the context, say 'I don't know based on the provided document.'"""),
    ("human", "{question}")
])

chain = prompt | llm

# Test Q&A set - questions and expected answers
test_qa = [
    {
        "question": "What is machine learning?",
        "expected": "learn from data"
    },
    {
        "question": "What industries use AI?",
        "expected": "healthcare, finance, education, transportation"
    },
    {
        "question": "What is deep learning?",
        "expected": "subset of AI"
    },
    {
        "question": "What is the capital of France?",
        "expected": "I don't know based on the provided document"
    },
]

print("RAG System Evaluation")
print("=" * 50)

correct = 0
for i, qa in enumerate(test_qa):
    relevant_chunks = vectorstore.similarity_search(qa["question"], k=2)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    response = chain.invoke({
        "question": qa["question"],
        "context": context
    })
    
    answer = response.content.lower()
    expected = qa["expected"].lower()
    
    keywords = expected.lower().split(",")
    is_correct = all(keyword.strip() in answer.lower() for keyword in keywords)
    if is_correct:
        correct += 1
    
    print(f"\nQ{i+1}: {qa['question']}")
    print(f"Expected: {qa['expected']}")
    print(f"Got: {response.content}")
    print(f"Pass: {'✓' if is_correct else '✗'}")

print(f"\n{'='*50}")
print(f"Score: {correct}/{len(test_qa)} ({round(correct/len(test_qa)*100)}%)")