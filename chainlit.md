# Zeladija E-Commerce AI Support Agent 🛒🤖

Welcome to the **Zeladija E-Commerce Support Agent** —  multi-agent AI system built for intelligent customer support.

## What This Agent Can Do

Ask me about anything related to your order or our store:

- 📦 **Order Tracking** — Look up order status, estimated delivery, and carrier info
- 🔄 **Returns & Refunds** — Get guidance on return eligibility, timelines, and refund limits
- ❓ **Product & FAQ** — Answers pulled directly from our knowledge base using RAG
- 🚨 **Escalation** — Complex or high-value issues are automatically routed for human review

## How It Works

Your message flows through a **LangGraph multi-agent pipeline**:

1. A **Supervisor** classifies your intent and routes your request
2. A **Specialist Agent** handles your query (Order Lookup, Policy, or General FAQ)
3. **Guardrails** run on every message to ensure safe, compliant responses
4. An **Escalation Agent** steps in when human oversight is needed

## Tips for Getting Started

- Try: *"Where is my order #1042?"*
- Try: *"What is your return policy for damaged items?"*
- Try: *"Do you offer international shipping?"*

---

