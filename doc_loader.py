from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Text loader 
loader = TextLoader("sample.txt")
documents = loader.load()
print(f"Text file - Number of documents: {len(documents)}")

# PDF loader 
pdf_loader = PyPDFLoader("sample.pdf")
pdf_documents = pdf_loader.load()
print(f"PDF - Number of documents: {len(pdf_documents)}")
print(f"First page content: {pdf_documents[0].page_content[:200]}")
print(f"Metadata: {pdf_documents[0].metadata}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"After splitting: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content}")
print(f"Second chunk: {chunks[1].page_content}")
print(f"Third chunk: {chunks[1].page_content}")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

sample_text = "Artificial intelligence is transforming the world"
vector = embeddings.embed_query(sample_text)

print(f"Text: {sample_text}")
print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

query = "What is machine learning?"
results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}")
for i, result in enumerate(results):
    print(f"\nResult {i+1}: {result.page_content}")