"""
scripts/fetch_bitext_faq.py
Downloads the Bitext Customer Support LLM dataset, filters for e-commerce,
and saves a formatted FAQ knowledge base to data/raw/faq.json.
"""
import os
import json
from loguru import logger
import pandas as pd
from datasets import load_dataset

def fetch_and_format_faq():
    logger.info("Downloading Bitext dataset from Hugging Face...")
    ds = load_dataset('bitext/Bitext-customer-support-llm-chatbot-training-dataset')
    df = ds['train'].to_pandas()
    
    keep_intents = [
        'track_order', 'return_item', 'cancel_order', 'delivery_period',
        'refund_not_received', 'check_refund_policy', 'get_invoice',
        'payment_issue', 'change_shipping_address', 'contact_human_agent'
    ]
    
    faq_df = df[df['intent'].isin(keep_intents)]
    logger.info(f"Filtered to {len(faq_df)} relevant e-commerce support tickets.")
    
    faq_docs = []
    for intent, group in faq_df.groupby('intent'):
        doc = {
            'intent': intent,
            'examples': group['instruction'].head(5).tolist(),
            'response': group['response'].iloc[0]
        }
        faq_docs.append(doc)
    
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/faq.json'
    
    with open(output_path, 'w') as f:
        json.dump(faq_docs, f, indent=2)
        
    logger.success(f"Saved {len(faq_docs)} FAQ documents to {output_path}")

if __name__ == "__main__":
    fetch_and_format_faq()