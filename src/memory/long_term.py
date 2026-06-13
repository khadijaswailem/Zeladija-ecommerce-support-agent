"""
src/memory/long_term.py
Long-term memory: persists customer facts across sessions using SQLite.
Schema: customer_id, fact_type, fact_value, created_at, session_id.
"""

import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/Zeladija_memory.db")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return (or initialize) the OpenAI client."""
    global _openai_client

    #create the client once and reuse it across calls
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    return _openai_client


def get_db_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the database and table if needed."""
    db_path = Path(SQLITE_DB_PATH)

    #ensure the database directory exists before connecting
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    #create required tables and indexes if missing
    _ensure_schema(conn)

    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the customer_facts table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT    NOT NULL,
            fact_type   TEXT    NOT NULL,
            fact_value  TEXT    NOT NULL,
            session_id  TEXT,
            created_at  TEXT    NOT NULL
        )
    """)

    #index customer_id to speed up history lookups
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_facts(customer_id)
    """)

    conn.commit()


#supported categories of customer information that can be stored
FACT_TYPES = {
    "preferred_name": "Customer's preferred name or alias.",
    "past_order_id": "An order ID mentioned in a prior session.",
    "reported_issue": "A problem or complaint the customer reported.",
    "resolution_status": "Whether the customer's last issue was resolved.",
    "refund_requested": "A refund amount the customer has requested.",
    "product_interest": "A product category or item the customer asked about.",
}

#prompt used to instruct the LLM to extract persistent customer facts
EXTRACTION_PROMPT = """You are a fact extractor for a customer support system.
Analyze the conversation and extract persistent facts worth remembering for future sessions.

Fact types to look for:
{fact_types}

Conversation:
{conversation}

Return a JSON array of facts. Each fact has:
  {{"fact_type": "<one of the fact types above>", "fact_value": "<concise value, max 100 chars>"}}

Return [] if no meaningful facts found. Output ONLY valid JSON array, no other text."""


def extract_facts_from_session(
    customer_id: str,
    session_id: str,
    messages: list[dict],
) -> list[dict]:
    """
    Use LLM to extract persistent facts from a completed session.
    Saves extracted facts to SQLite.
    """
    #convert the conversation into a format suitable for the LLM
    conversation = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in messages]
    )

    #format the supported fact types for inclusion in the prompt
    fact_types_str = "\n".join(
        [f"  - {k}: {v}" for k, v in FACT_TYPES.items()]
    )

    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(
                    fact_types=fact_types_str,
                    conversation=conversation,
                ),
            }],
            max_tokens=400,
            temperature=0.0,
        )

        #clean and parse the model output as JSON
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        facts = json.loads(raw)

    except Exception as e:
        logger.warning(f"[long_term] Fact extraction failed: {e}")
        facts = []

    #persist extracted facts for future sessions
    if facts:
        save_facts(customer_id, session_id, facts)
        logger.info(f"[long_term] Saved {len(facts)} facts for {customer_id}")

    return facts


def save_facts(
    customer_id: str,
    session_id: str,
    facts: list[dict],
) -> None:
    """Write extracted facts to the SQLite customer_facts table."""
    conn = get_db_connection()
    now = datetime.utcnow().isoformat()

    #filter out malformed facts before insertion
    rows = [
        (customer_id, f["fact_type"], f["fact_value"], session_id, now)
        for f in facts
        if "fact_type" in f and "fact_value" in f
    ]

    conn.executemany(
        "INSERT INTO customer_facts (customer_id, fact_type, fact_value, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()


def get_customer_history(customer_id: str, limit: int = 10) -> Optional[str]:
    """
    Retrieve recent facts for a customer and format as a system prompt injection.
    Returns None if no history exists.

    Example return:
      "Customer history: preferred_name=Alex, past_order_id=ORD-ABC123,
       reported_issue=Lost package on order ORD-ABC123"
    """
    conn = get_db_connection()

    #retrieve the most recent stored facts for the customer
    rows = conn.execute(
        """
        SELECT fact_type, fact_value, created_at
        FROM customer_facts
        WHERE customer_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (customer_id, limit),
    ).fetchall()

    conn.close()

    if not rows:
        return None

    #format facts into a compact string for prompt injection
    facts_str = ", ".join([f"{r['fact_type']}={r['fact_value']}" for r in rows])

    return f"Customer history from past sessions: {facts_str}"


def clear_customer_history(customer_id: str) -> None:
    """Delete all stored facts for a customer (for testing or GDPR requests)."""
    conn = get_db_connection()

    #remove all stored records associated with the customer
    conn.execute("DELETE FROM customer_facts WHERE customer_id = ?", (customer_id,))

    conn.commit()
    conn.close()

    logger.info(f"[long_term] Cleared all history for {customer_id}")