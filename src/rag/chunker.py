"""
src/rag/chunker.py
Splits raw documents into overlapping chunks with metadata tagging.
Chunk size: 512 tokens (~400 words), overlap: 50 tokens (~40 words).
"""

import json
import os
import re
from pathlib import Path
from typing import Generator
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed"))

CHUNK_SIZE = 200
OVERLAP = 30

#maps source files to their document type for retrieval metadata
DOC_TYPE_MAP = {
    "shipping_policy.txt": "policy",
    "returns_policy.txt": "policy",
    "warranty_policy.txt": "policy",
    "faq_policy.txt": "faq",
    "bitext_faq.json": "faq",
    "product_catalog.json": "catalog",
}


def word_chunks(text: str, size: int, overlap: int) -> Generator[str, None, None]:
    """Yield overlapping word-window chunks from a block of text."""
    words = text.split()
    start = 0

    #slide a window through the document while preserving overlap
    while start < len(words):
        end = min(start + size, len(words))
        yield " ".join(words[start:end])

        if end == len(words):
            break

        start += size - overlap


def chunk_text_file(filepath: Path, doc_type: str) -> list[dict]:
    """Chunk a plain-text policy document."""
    text = filepath.read_text(encoding="utf-8")

    #normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []

    for idx, chunk_text in enumerate(word_chunks(text, CHUNK_SIZE, OVERLAP)):
        chunks.append(
            {
                "chunk_id": f"{filepath.stem}_chunk_{idx:04d}",
                "source_file": filepath.name,
                "doc_type": doc_type,
                "text": chunk_text.strip(),
                "chunk_index": idx,
            }
        )

    logger.info(f"Chunked {filepath.name} → {len(chunks)} chunks")
    return chunks


def chunk_faq(filepath: Path) -> list[dict]:
    """Convert the Bitext FAQ json into RAG chunks."""
    with open(filepath) as f:
        faqs = json.load(f)

    chunks = []

    #convert each FAQ entry into a retrieval-friendly text block
    for idx, item in enumerate(faqs):
        intent = item.get("intent", "general")
        examples = "\n- ".join(item.get("examples", []))
        response = item.get("response", "No response provided.")

        text = (
            f"Topic: {intent}\n"
            f"Customer Questions:\n- {examples}\n"
            f"Official Response:\n{response}"
        )

        chunks.append(
            {
                "chunk_id": f"faq_{intent}_{idx}",
                "source_file": filepath.name,
                "doc_type": "faq",
                "text": text.strip(),
                "chunk_index": idx,
            }
        )

    logger.info(f"Chunked {filepath.name} → {len(chunks)} FAQ entries")
    return chunks


def chunk_catalog(filepath: Path) -> list[dict]:
    """Convert each Amazon product entry into a single descriptive chunk."""
    with open(filepath) as f:
        products = json.load(f)

    chunks = []

    #convert each product record into a searchable text chunk
    for idx, product in enumerate(products):
        raw_features = product.get("features", [])

        #protect against malformed feature values
        if not isinstance(raw_features, list):
            raw_features = []

        features_str = ", ".join(str(f) for f in raw_features)

        #provide safe defaults for missing fields
        title = product.get("title", "Unknown Product")
        sku = product.get("sku", f"AMZ-{idx:04d}")
        category = product.get("category", "Electronics")
        price = product.get("price", "N/A")
        description = product.get("description", "No description available.")

        text = (
            f"Product: {title}\n"
            f"SKU: {sku}\n"
            f"Category: {category}\n"
            f"Price: {price}\n"
            f"Description: {description}\n"
            f"Key Features: {features_str}"
        )

        chunks.append(
            {
                "chunk_id": f"catalog_{sku}",
                "source_file": filepath.name,
                "doc_type": "catalog",
                "sku": sku,
                "text": text.strip(),
                "chunk_index": idx,
            }
        )

    logger.info(f"Chunked {filepath.name} → {len(chunks)} product entries")
    return chunks


def chunk_bitext(filepath: Path) -> list[dict]:
    """Convert each Bitext intent group into a chunk."""
    with open(filepath) as f:
        intents = json.load(f)

    chunks = []

    #store each intent and its examples as a single retrieval unit
    for idx, intent in enumerate(intents):
        examples = "\n".join(intent.get("examples", []))
        response = intent.get("response", "")

        text = (
            f"Intent: {intent['intent']}\n"
            f"Example customer messages:\n{examples}\n"
            f"Typical response: {response}"
        )

        chunks.append({
            "chunk_id": f"bitext_{intent['intent']}",
            "source_file": filepath.name,
            "doc_type": "faq",
            "intent": intent["intent"],
            "text": text.strip(),
            "chunk_index": idx,
        })

    logger.info(f"Chunked {filepath.name} → {len(chunks)} intent groups")
    return chunks


def run_chunker() -> list[dict]:
    """Process all raw documents and write chunks to data/processed/."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []

    #process every configured source document
    for filename, doc_type in DOC_TYPE_MAP.items():
        filepath = RAW_PATH / filename

        if not filepath.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        #select the appropriate chunking strategy for each file type
        if filename == "bitext_faq.json":
            chunks = chunk_bitext(filepath)
        elif filename.endswith(".json"):
            chunks = chunk_catalog(filepath)
        else:
            chunks = chunk_text_file(filepath, doc_type)

        all_chunks.extend(chunks)

        #save chunks generated from this document
        out_path = PROCESSED_PATH / f"{filepath.stem}_chunks.json"

        with open(out_path, "w") as f:
            json.dump(chunks, f, indent=2)

    #save all chunks into a single file for embedding
    combined_path = PROCESSED_PATH / "all_chunks.json"

    with open(combined_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Total chunks produced: {len(all_chunks)}")
    logger.info(f"Combined chunks saved to {combined_path}")

    return all_chunks


if __name__ == "__main__":
    run_chunker()