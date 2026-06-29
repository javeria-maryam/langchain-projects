from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

loader = TextLoader("sample.txt")
documents = loader.load()

print(f"Original text length: {len(documents[0].page_content)} characters")
print(f"Original text: {documents[0].page_content[:100]}...\n")

# Strategy 1 - Fixed Size
fixed_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator=""
)
fixed_chunks = fixed_splitter.split_documents(documents)
print(f"Strategy 1 - Fixed Size: {len(fixed_chunks)} chunks")
for i, chunk in enumerate(fixed_chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:80]}...")

print()

# Strategy 2 - Recursive Character
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
recursive_chunks = recursive_splitter.split_documents(documents)
print(f"Strategy 2 - Recursive: {len(recursive_chunks)} chunks")
for i, chunk in enumerate(recursive_chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:80]}...")

print()

# Strategy 3 - Semantic
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_splitter = SemanticChunker(embeddings)
semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Strategy 3 - Semantic: {len(semantic_chunks)} chunks")
for i, chunk in enumerate(semantic_chunks):
    print(f"  Chunk {i+1}: {chunk.page_content[:80]}...")