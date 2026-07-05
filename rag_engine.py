"""
RAG Engine for Study AI
Handles document ingestion, chunking, embedding, storage, and retrieval.
All processing runs locally via Ollama + ChromaDB.
"""

import os
import ollama
import chromadb
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# Configuration
# -------------------------
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "study_ai_docs"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5


# -------------------------
# Text Extraction
# -------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from an uploaded TXT file."""
    return uploaded_file.read().decode("utf-8")


def extract_text(uploaded_file, filename: str) -> str:
    """Extract text from a file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == ".txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# -------------------------
# Text Chunking
# -------------------------
def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# -------------------------
# Embeddings
# -------------------------
def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a single text using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


# -------------------------
# ChromaDB Operations
# -------------------------
def get_chroma_client():
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_or_create_collection(client=None):
    """Get or create the document collection in ChromaDB."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(chunks: list[str], source_name: str, collection=None) -> int:
    """
    Embed and store document chunks in ChromaDB.
    Returns the number of chunks indexed.
    """
    if collection is None:
        collection = get_or_create_collection()

    if not chunks:
        return 0

    # Generate embeddings in batch
    embeddings = get_embeddings_batch(chunks)

    # Create unique IDs and metadata for each chunk
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def query_documents(question: str, n_results: int = TOP_K_RESULTS, collection=None) -> list[dict]:
    """
    Query ChromaDB for the most relevant chunks.
    Returns a list of dicts with 'text', 'source', and 'distance' keys.
    """
    if collection is None:
        collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    # Generate embedding for the question
    question_embedding = get_embedding(question)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "Unknown"),
            "distance": results["distances"][0][i],
        })

    return retrieved


def clear_collection(collection=None):
    """Delete all documents from the collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def get_indexed_sources(collection=None) -> dict:
    """
    Get a summary of all indexed document sources.
    Returns a dict: {source_name: chunk_count}
    """
    if collection is None:
        collection = get_or_create_collection()

    if collection.count() == 0:
        return {}

    # Get all metadata
    all_data = collection.get(include=["metadatas"])
    source_counts = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    return source_counts


# -------------------------
# RAG Prompt Builder
# -------------------------
def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a prompt that includes retrieved document context
    for grounded answer generation.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a study assistant. Answer the following question using ONLY the context provided below. 
If the context doesn't contain enough information to answer fully, say so clearly.

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based strictly on the provided context
2. Structure your answer with clear headings and bullet points
3. Cite which source document the information comes from
4. If the context is insufficient, state what's missing
5. Keep the answer concise and study-friendly

ANSWER:"""

    return prompt
