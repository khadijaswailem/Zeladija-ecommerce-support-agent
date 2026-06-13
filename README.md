# Zeladija AI Customer Support Agent

Multi-agent AI system for e-commerce customer support. Built on LangGraph, this system handles order tracking, return and refund eligibility, policy questions, and escalations through a fully automated pipeline with guardrails, long-term memory, and an evaluation suite.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the System](#running-the-system)
- [Running the Evaluation Suite](#running-the-evaluation-suite)
- [Environment Variables](#environment-variables)
- [Data and Knowledge Base](#data-and-knowledge-base)

---

## Architecture Overview

```
User Message
    |
    v
Input Guardrail         <-- detects and sanitizes prompt injection
    |
    v
Toxicity Guardrail      <-- flags hostile messages, routes to escalation
    |
    v
Supervisor Agent        <-- classifies intent, loads long-term memory
    |
    +--------+-----------+-----------+
    |        |           |           |
    v        v           v           v
Order    Policy &    General     Escalation
Lookup   Returns     FAQ         Agent
Agent    Agent       Agent
    |        |           |           |
    +--------+-----------+-----------+
                    |
                    v
             Policy Guardrail    <-- scans responses for unauthorized commitments
                    |
                    v
             Respond Node        <-- single terminal for logging and post-processing
                    |
                    v
             Final Response
```

**RAG pipeline** feeds into the Policy and FAQ agents:
```
Raw Documents 
    |
    v
Chunker 
    |
    v
Embedder 
    |
    v
Retriever (Hybrid BM25 + Dense Vector + RRF fusion)
```

**Memory** flows through the entire graph:
```
Short-term: AgentState.messages (50-message cap, within session)
Long-term:  SQLite customer_facts table (persists across sessions)
```

---

## Project Structure

```
.
├── data/
│   ├── raw/                      # Source documents and generated orders
│   │   ├── product_catalog.json
│   │   ├── shipping_policy.txt
│   │   ├── returns_policy.txt
│   │   ├── faq_policy.txt
│   │   ├── warranty_policy.txt
│   │   └── orders.json
│   ├── processed/                # Chunked outputs from chunker.py
│   └── chroma_db/                # ChromaDB vector index (auto-generated)
│
├── scripts/
│   └── generate_orders.py        # Generates 200 synthetic orders (Faker, seed=42)
│
├── src/
│   ├── agents/
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── supervisor.py         # Intent classification + routing
│   │   ├── order_lookup.py       # Order tracking agent
│   │   ├── policy_agent.py       # Returns, refunds, FAQ agent
│   │   ├── escalation_agent.py   # Human handoff agent
│   │   └── graph.py              # LangGraph StateGraph assembly
│   ├── rag/
│   │   ├── chunker.py            # Document chunking
│   │   ├── embedder.py           # Embedding + ChromaDB indexing
│   │   └── retriever.py          # Naive and hybrid retrieval
│   ├── memory/
│   │   ├── short_term.py         # Session message management
│   │   └── long_term.py          # SQLite fact persistence
│   ├── guardrails/
│   │   ├── input_guardrail.py    # Prompt injection detection
│   │   ├── policy_guardrail.py   # Unauthorized commitment detection
│   │   └── toxicity_guardrail.py # Toxic message classification
│   └── tools/
│       └── mock_api.py           # LangChain @tool order API (called via .invoke())
│
├── eval/
│   ├── metrics.py                # Resolution rate, latency, guardrail accuracy
│   ├── llm_judge.py              # LLM-as-judge policy compliance scoring
│   ├── run_eval.py               # 30-case eval runner
│   ├── dashboard.py              # Streamlit metrics dashboard
│   ├── test_cases/               # JSON test cases by category
│   │   ├── happy_paths/
│   │   ├── policy_edge/
│   │   ├── adversarial/
│   │   └── toxic/
│   └── results/                  # Eval output JSONs
│
├── notebooks/
│   └── rag_eval.ipynb            # Naive vs advanced RAG comparison
│
├── chainlit_app/                 # Frontend (optional)
│   ├── app.py
│   ├── config.py
│   ├── services/
│   └── ui/
│
├── .env.example
└── requirements.txt
```

---

## Setup and Installation

### 1. Clone the repository and install dependencies

```bash
git clone <repo-url>
cd zeladija-support-agent
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. Generate synthetic order data

```bash
python scripts/generate_orders.py
```

This creates `data/raw/orders.json` with 200 orders across 40 customers using a fixed seed for reproducibility.

### 4. Build the RAG knowledge base

Run these in order. Each step depends on the previous.

```bash
# Step 1: Chunk all policy documents and the product catalog
python -m src.rag.chunker

# Step 2: Embed chunks and index them in ChromaDB
python -m src.rag.embedder
```

ChromaDB will persist automatically to `data/chroma_db/`. The embedder is safe to run multiple times since it skips already-indexed chunks.

---

## Running the System

### Command line (direct graph invocation)

```bash
python -m src.agents.graph
```

This runs a demo message through the full graph and prints the node structure, edges, and response.

### Chainlit frontend

```bash
pip install -r requirements_chainlit.txt
chainlit run chainlit_app/app.py
```

Open `http://localhost:8000` in your browser.

---

## Running the Evaluation Suite

### Full 30-case eval

```bash
python eval/run_eval.py
```

Results are saved to `eval/results/latest.json` and a timestamped file. The runner prints live progress for each test case.

### Evaluation dashboard

```bash
streamlit run eval/dashboard.py
```

Shows resolution rate, policy compliance, P95 latency, intent accuracy, and guardrail precision/recall.

### RAG comparison notebook

```bash
jupyter notebook notebooks/rag_eval.ipynb
```

Compares naive cosine similarity retrieval against the hybrid BM25 + RRF pipeline across 20 generated question/answer pairs.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key for LLM calls |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `STORE_NAME` | `Zeladija` | Used in prompts and responses |
| `REFUND_THRESHOLD` | `150.00` | Maximum refund the agent can approve |
| `TOXICITY_THRESHOLD` | `0.75` | Confidence cutoff for toxic-bert |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB persistence directory |
| `SQLITE_DB_PATH` | `./data/zeladija_memory.db` | Long-term memory database |
| `ORDERS_JSON_PATH` | `./data/raw/orders.json` | Path to generated orders |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

---

## Data and Knowledge Base

All data files are synthetic and were generated specifically for this project.

| File | Contents |
|---|---|
| `shipping_policy.txt` | Standard / Express / Overnight rates, 2PM ET cutoff, carrier partners |
| `returns_policy.txt` | 30-day return window, $150 escalation threshold, refund timeline |
| `faq_policy.txt` | 40+ Q&A pairs covering accounts, payments, loyalty, and accessibility |
| `warranty_policy.txt` | 1-year electronics / 90-day accessories coverage and claim process |
| `product_catalog.json` | 25 electronics products with SKUs, prices, and feature descriptions |
| `orders.json` | 200 synthetic orders across 40 customers (generated by Faker) |

The $150 refund threshold is enforced in three places: the `returns_policy.txt` document, the `policy_agent.py` pre-generation check, and the `policy_guardrail.py` post-generation check. All three must agree for the threshold to hold under all conditions.
