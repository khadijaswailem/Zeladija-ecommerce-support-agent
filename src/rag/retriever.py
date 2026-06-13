"""
src/rag/retriever.py
Provides two retrieval strategies:
  - Naive:    simple cosine similarity top-k from ChromaDB (baseline for RAGAS).
  - Advanced: hybrid BM25 keyword + dense vector via rank_bm25, then reciprocal
              rank fusion. Optional doc_type metadata filter.

Use naive_retrieve() for baseline RAGAS scoring.
Use advanced_retrieve() (default) in all agent nodes.
"""

import os
from typing import Optional

from rank_bm25 import BM25Okapi
from loguru import logger
from dotenv import load_dotenv

from src.rag.embedder import get_or_create_collection, get_embedding_model

load_dotenv()

TOP_K = int(os.getenv("RAG_TOP_K", "7"))  #5 was too tight post-fusion
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "15"))  #wider BM25 net before fusion


#Naive retriever (baseline)

def naive_retrieve(
    query: str,
    top_k: int = TOP_K,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Baseline cosine-similarity retrieval from ChromaDB.
    Used ONLY for RAGAS baseline measurement in rag_eval.ipynb.
    Returns a list of dicts with 'text', 'metadata', and 'score' keys.
    """
    collection = get_or_create_collection()
    model = get_embedding_model()

    #Generate an embedding for the user query
    query_embedding = model.encode([query]).tolist()

    #Apply an optional metadata filter when requested
    where_filter = {"doc_type": doc_type} if doc_type else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    #Convert query results into a consistent output format
    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"text": text, "metadata": meta, "score": 1 - dist})

    logger.debug(f"[naive_retrieve] query='{query[:60]}...' → {len(docs)} results")
    return docs


#Advanced retriever (hybrid BM25 + dense vector with RRF fusion)

def _reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists of doc IDs using Reciprocal Rank Fusion.
    Returns list of (doc_id, rrf_score) sorted descending.
    """
    #Accumulate scores from multiple ranking methods
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    #Return documents ordered by their fused score
    return sorted(scores.items(), key=lambda x: -x[1])


import re


#Normalize text before BM25 tokenization
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]+\b", text.lower())

    #Apply simple suffix stripping to reduce word variations
    normalized = []
    for t in tokens:
        if t.endswith("ing"):
            t = t[:-3]
        elif t.endswith("ed"):
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3:
            t = t[:-1]
        normalized.append(t)

    return normalized


def advanced_retrieve(
    query: str,
    top_k: int = TOP_K,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 keyword search + dense vector search, fused with RRF.
    Optional doc_type metadata filter ('policy', 'faq', 'catalog').
    Returns a list of dicts with 'text', 'metadata', and 'score' keys.
    """
    collection = get_or_create_collection()
    model = get_embedding_model()

    #Apply an optional metadata filter before retrieval
    where_filter = {"doc_type": doc_type} if doc_type else None

    #Expand queries with related keywords for common e-commerce concepts
    QUERY_EXPANSIONS = {
        "return": "return refund policy window",
        "refund": "refund return money back policy",
        "cancel": "cancel cancellation order",
        "ship": "shipping delivery dispatch",
        "damage": "damaged broken defective item",
    }

    expanded_query = query
    for kw, expansion in QUERY_EXPANSIONS.items():
        if kw in query.lower():
            expanded_query = f"{query} {expansion}"
            break

    #Load candidate documents for BM25 processing
    all_docs = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )

    if not all_docs["ids"]:
        logger.warning("No documents found in collection with given filter.")
        return []

    ids = all_docs["ids"]
    texts = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    #Build a BM25 index and rank documents by keyword relevance
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(expanded_query))
    bm25_ranking = [
        ids[i] for i in sorted(range(len(ids)), key=lambda x: -bm25_scores[x])
    ][:BM25_TOP_K]

    #Retrieve documents using dense vector similarity
    query_embedding = model.encode([expanded_query]).tolist()
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(BM25_TOP_K, len(ids)),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    dense_ranking = dense_results["ids"][0]

    #Combine BM25 and dense rankings into a single ranking
    fused = _reciprocal_rank_fusion([bm25_ranking, dense_ranking])

    #Map fused document IDs back to their text and metadata
    id_to_data = {
        id_: (text, meta)
        for id_, text, meta in zip(ids, texts, metadatas)
    }

    output = []
    for doc_id, rrf_score in fused[:top_k]:
        if doc_id in id_to_data:
            text, meta = id_to_data[doc_id]
            output.append(
                {"text": text, "metadata": meta, "score": rrf_score}
            )

    logger.debug(
        f"[advanced_retrieve] query='{query[:60]}...' | filter={doc_type} → {len(output)} results"
    )
    return output


#Default export used by agent nodes

def retrieve(
    query: str,
    top_k: int = TOP_K,
    doc_type: Optional[str] = None,
    mode: str = "advanced",
) -> list[dict]:
    """
    Unified retrieval entry point.
    mode='advanced' (default) uses hybrid BM25 + dense.
    mode='naive' uses simple cosine similarity (for eval baselines only).
    """
    #Select the retrieval strategy based on the requested mode
    if mode == "naive":
        return naive_retrieve(query, top_k=top_k, doc_type=doc_type)

    return advanced_retrieve(query, top_k=top_k, doc_type=doc_type)