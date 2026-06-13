"""
scripts/fetch_amazon_catalog.py
Downloads a subset of the Amazon Reviews 2023 dataset (Electronics)
and formats 50 products into a clean JSON catalog for RAG indexing.
"""
import os
import json
from loguru import logger
from datasets import load_dataset

def fetch_catalog():
    logger.info("Downloading Amazon Electronics dataset...")
    # Load metadata only (descriptions and specs)
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Electronics",
        split="full",
        trust_remote_code=True
    )
    
    products = []
    for r in ds:
        desc = r.get('description', [])
        # Only keep products that actually have descriptions
        if desc and len(desc) > 0 and isinstance(desc, list):
            products.append({
                "title": r.get("title", "Unknown Product"),
                "description": " ".join(desc),
                "price": r.get("price", "N/A"),
                "features": r.get("features", [])
            })
        if len(products) >= 50:
            break
            
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/product_catalog.json'
    
    with open(output_path, 'w') as f:
        json.dump(products, f, indent=2)
        
    logger.success(f"Saved {len(products)} Amazon products to {output_path}")

if __name__ == "__main__":
    fetch_catalog()