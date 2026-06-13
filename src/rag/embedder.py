"""
src/rag/embedder.py
Loads processed chunks, generates embeddings via sentence-transformers (free, local),
and stores vectors in ChromaDB (free, local).
"""

import json
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed"))
COLLECTION_NAME = "Zeladija_knowledge_base"

_chroma_client: Optional[chromadb.PersistentClient] = None
_embedding_model: Optional[SentenceTransformer] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return (or initialize) the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        #Create the database directory if it does not already exist
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info(f"ChromaDB client initialized at {CHROMA_DB_PATH}")
    return _chroma_client


def get_embedding_model() -> SentenceTransformer:
    """Return (or initialize) the sentence-transformers embedding model."""
    global _embedding_model
    if _embedding_model is None:
        #Load the embedding model only once and reuse it
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_or_create_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if it does not exist."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_index(chunks: list[dict], batch_size: int = 64) -> chromadb.Collection:
    """
    Embed a list of chunk dicts and upsert them into ChromaDB.
    Skips chunks that are already indexed (by chunk_id).
    """
    collection = get_or_create_collection()
    model = get_embedding_model()

    #Collect IDs that are already stored in the database
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All chunks already indexed. Nothing to do.")
        return collection

    logger.info(f"Embedding {len(new_chunks)} new chunks (batch_size={batch_size})...")

    #Process chunks in batches to avoid loading everything at once
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]

        #Keep only metadata fields that should be stored separately
        metadatas = [
            {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
            for c in batch
        ]

        #Convert text into vector embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"  Upserted batch {i // batch_size + 1}: {len(batch)} chunks")

    logger.success(f"Indexing complete. Collection now has {collection.count()} documents.")
    return collection


def run_embedder() -> chromadb.Collection:
    """Load all processed chunks and embed them into ChromaDB."""
    combined_path = PROCESSED_PATH / "all_chunks.json"

    #Ensure the chunk file exists before attempting to load it
    if not combined_path.exists():
        raise FileNotFoundError(
            f"No chunks found at {combined_path}. Run chunker.py first."
        )

    with open(combined_path) as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {combined_path}")
    return embed_and_index(chunks)


def reset_collection() -> None:
    """Delete and recreate the ChromaDB collection (use for re-indexing)."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.warning(f"Deleted collection: {COLLECTION_NAME}")
    except Exception:
        #Ignore the error if the collection does not already exist
        pass

    #Create a fresh empty collection after deletion
    client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    logger.info(f"Created fresh collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    #Uncomment this to clear the existing database before indexing again
    #reset_collection()
    run_embedder()