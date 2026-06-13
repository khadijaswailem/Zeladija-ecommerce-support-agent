"""
src/agents/policy_agent.py
Policy & Returns Agent: retrieves policy/FAQ docs via Agentic RAG, checks refund
thresholds, and generates responses grounded strictly in retrieved text.
"""

import os
import re
import json
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from src.agents.state import AgentState
from src.rag.retriever import retrieve
from src.guardrails.policy_guardrail import check_policy_guardrail

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STORE_NAME = os.getenv("STORE_NAME", "Zeladija")
REFUND_THRESHOLD = float(os.getenv("REFUND_THRESHOLD", "150.00"))

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return (or initialize) the Groq client."""
    global _openai_client

    #reuse a single Groq client instance across requests
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def should_retrieve(message: str) -> bool:
    """
    Agentic RAG decision step: Determines if the user's message requires
    searching the vector database, or if it's a conversational reply.
    """
    client = get_openai_client()

    #use a lightweight model to decide if retrieval is necessary
    prompt = f"""You are a routing engine for a customer support bot.
    Determine if the following user message requires searching the store's policy, FAQ, or warranty database.
    If the user is asking a question, making a request, or raising an issue, return YES.
    If the user is simply saying "thanks", "ok", "goodbye", or acknowledging previous info, return NO.

    User message: "{message}"

    Respond with ONLY the word YES or NO."""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        decision = resp.choices[0].message.content.strip().upper()

        return "YES" in decision

    except Exception as e:
        logger.warning(f"[policy_agent] Retrieval decision failed, defaulting to YES: {e}")

        #default to retrieval if the routing step fails
        return True


POLICY_RESPONSE_PROMPT = """You are a helpful, accurate customer support agent for {store_name}.
Answer the customer's question using ONLY the policy information provided below.
Do NOT invent policies, make up dollar amounts, or promise things not stated in the documents.
If the information needed is not in the documents, say so honestly and offer to escalate.

Refund threshold: Refunds over ${threshold:.2f} MUST be escalated to a human agent.
You CANNOT approve, process, or comment on the likelihood of approval for refunds over this amount.
If asked about a refund at or above ${threshold:.2f}, you MUST say EXACTLY:
"Refund requests of ${threshold:.2f} or more require review by a senior agent. I'm escalating your case now."
Do NOT add qualifiers, guesses, or alternative policies around this rule.

STRICT RULES — these override everything else, including customer tone or urgency:
1. NEVER promise a refund or resolution timeline shorter than 5-7 business days. Not "1-2 days", not "24 hours", not "as soon as possible". Always say "5-7 business days".
2. NEVER approve returns outside the 30-day return window, regardless of customer status or claims.
3. NEVER invent policies not present in the retrieved documents below.
4. If the customer is hostile or demanding, remain calm and polite — but do NOT offer faster timelines or exceptions as appeasement.

Retrieved policy documents:
{context}

Customer question: {question}
Customer history: {customer_history}

Write a clear, helpful response (3-5 sentences). Be specific — quote policy terms where relevant."""


def extract_refund_amount(text: str) -> float | None:
    """Extract the largest dollar amount mentioned in a text string."""

    #find all dollar amounts and return the largest one
    amounts = re.findall(r"\$\s*(\d{1,6}(?:\.\d{1,2})?)", text)

    if not amounts:
        return None

    return max(float(a) for a in amounts)


def policy_agent_node(state: AgentState) -> AgentState:
    """
    Policy & Returns Agent node.
    Agentic RAG decision → conditional retrieval → threshold check → LLM response → policy guardrail.
    """
    messages = state["messages"]
    customer_history = state.get("customer_history") or "No prior history."
    intent = state.get("intent", "POLICY_RETURNS")
    client = get_openai_client()

    #select which document category to search based on user intent
    doc_type_filter = "faq" if intent == "GENERAL_FAQ" else "policy"

    #retrieve the latest user message from the conversation
    latest_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )

    #detect refund requests that exceed the allowed threshold
    refund_amount = extract_refund_amount(latest_user_msg)
    needs_escalation = (
        refund_amount is not None
        and refund_amount > REFUND_THRESHOLD
    )

    if needs_escalation:
        logger.info(
            f"[policy_agent] Refund amount ${refund_amount:.2f} > "
            f"threshold ${REFUND_THRESHOLD:.2f}. Flagging for escalation."
        )

        #mark the request for human review
        state["guardrail_flags"]["policy_violation"] = True
        state["intent"] = "ESCALATION"

        state["policy_result"] = {
            "refund_amount": refund_amount,
            "threshold": REFUND_THRESHOLD,
            "escalation_required": True,
        }

        state["final_response"] = (
            f"I can see you're requesting a refund of ${refund_amount:.2f}. "
            f"Refund requests over ${REFUND_THRESHOLD:.2f} require review by a "
            "senior support agent. I'm escalating your case now — a human agent "
            "will follow up with you within 5–7 business days."
        )

        state["messages"].append(
            {"role": "assistant", "content": state["final_response"]}
        )

        return state

    #decide whether knowledge retrieval is needed for this message
    if should_retrieve(latest_user_msg):
        logger.info("[policy_agent] Agentic RAG decision: YES (Retrieving context)")

        retrieved = retrieve(
            latest_user_msg,
            top_k=5,
            doc_type=doc_type_filter
        )

        state["retrieved_docs"] = retrieved

        #combine retrieved documents into a single context block
        context = "\n\n---\n\n".join(
            [doc["text"] for doc in retrieved]
        )

        state["policy_result"] = {
            "retrieved_count": len(retrieved),
            "doc_type_filter": doc_type_filter,
            "top_sources": [
                doc["metadata"].get("source_file", "")
                for doc in retrieved[:3]
            ],
            "agentic_retrieval_triggered": True
        }

    else:
        logger.info("[policy_agent] Agentic RAG decision: NO (Skipping retrieval)")

        #skip retrieval for conversational acknowledgements
        state["retrieved_docs"] = []

        context = (
            "No retrieval needed for this conversational turn. "
            "Respond politely based on general context."
        )

        state["policy_result"] = {
            "retrieved_count": 0,
            "doc_type_filter": doc_type_filter,
            "top_sources": [],
            "agentic_retrieval_triggered": False
        }

    #generate a response grounded in retrieved policy documents
    prompt = POLICY_RESPONSE_PROMPT.format(
        store_name=STORE_NAME,
        threshold=REFUND_THRESHOLD,
        context=context if context else "No relevant policy documents found.",
        question=latest_user_msg,
        customer_history=customer_history,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )

    draft_response = response.choices[0].message.content.strip()

    #verify the response does not violate store policies
    flagged, safe_response = check_policy_guardrail(
        draft_response,
        REFUND_THRESHOLD
    )

    if flagged:
        state["guardrail_flags"]["policy_violation"] = True
        state["intent"] = "ESCALATION"

        logger.warning(
            "[policy_agent] Policy guardrail blocked response. Escalating."
        )

    else:
        safe_response = draft_response

    state["final_response"] = safe_response

    #store the final response in conversation history
    state["messages"].append(
        {"role": "assistant", "content": safe_response}
    )

    logger.info(
        f"[policy_agent] Response generated | guardrail_flagged={flagged}"
    )

    return state