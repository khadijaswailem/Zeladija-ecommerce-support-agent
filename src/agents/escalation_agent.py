"""
src/agents/escalation_agent.py
Escalation Agent: generates a structured JSON handoff summary for human agents.
Does NOT resolve the issue — creates a handoff package and sends a holding message.
"""

import os
import json
from datetime import datetime
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from src.agents.state import AgentState

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STORE_NAME = os.getenv("STORE_NAME", "Zeladija")

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return (or initialize) the OpenAI client."""
    global _openai_client

    #reuse a single Groq client instance across requests
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


HANDOFF_PROMPT = """You are the escalation module for {store_name}'s AI support system.
Analyze the conversation and generate a structured handoff summary for a human support agent.

Conversation history:
{conversation}

Order data (if any): {order_data}
Policy result (if any): {policy_result}
Guardrail flags: {guardrail_flags}

Generate a JSON object with these exact fields:
{{
  "customer_id": "<customer id>",
  "issue_category": "REFUND_OVER_THRESHOLD | COMPLAINT | UNRESOLVED_MULTI_TURN | TOXIC_INPUT | POLICY_VIOLATION | JAILBREAK_ATTEMPT | OTHER",
  "conversation_summary": "<2-3 sentence summary of what happened>",
  "recommended_action": "<what the human agent should do>",
  "urgency_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "refund_amount_requested": <number or null>,
  "order_ids_mentioned": ["<list of order ids or empty>"],
  "guardrail_triggers": {{
    "injection_detected": <bool>,
    "policy_violation": <bool>,
    "toxic_input": <bool>
  }}
}}

Output ONLY valid JSON. No other text."""

HOLDING_MESSAGES = {
    "TOXIC_INPUT": (
        "I understand you're frustrated, and I want to help resolve this for you. "
        "I've connected you with a senior member of our support team who will reach out "
        "within 1–2 business days to address your concern personally."
    ),
    "REFUND_OVER_THRESHOLD": (
        "Thank you for reaching out. Your refund request requires review by a senior agent "
        "due to the amount involved. A human agent will contact you within 1–2 business days "
        "to process your request securely."
    ),
    "JAILBREAK_ATTEMPT": (
        "I'm not able to assist with that request. I've flagged this interaction for "
        "review by our team. If you have a genuine support question, please let us know."
    ),
    "DEFAULT": (
        "I've escalated your case to a senior support agent who will follow up with you "
        "within 1–2 business days. Thank you for your patience."
    ),
}


def determine_issue_category(state: AgentState) -> str:
    """Determine the escalation reason from guardrail flags and state."""
    flags = state.get("guardrail_flags", {})

    #prioritize security and behavior related escalations first
    if flags.get("injection_detected"):
        return "JAILBREAK_ATTEMPT"

    if flags.get("toxic_input"):
        return "TOXIC_INPUT"

    if flags.get("policy_violation"):
        pr = state.get("policy_result", {})

        #treat refund related policy violations as a separate category
        if pr.get("refund_amount"):
            return "REFUND_OVER_THRESHOLD"

        return "POLICY_VIOLATION"

    return "OTHER"


def escalation_agent_node(state: AgentState) -> AgentState:
    """
    Escalation Agent node.
    Generates a structured JSON handoff and a calm customer holding message.
    """
    messages = state["messages"]
    client = get_openai_client()

    #convert the conversation history into a readable transcript
    conversation_str = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in messages]
    )

    #build the prompt containing conversation and state information
    prompt = HANDOFF_PROMPT.format(
        store_name=STORE_NAME,
        conversation=conversation_str,
        order_data=json.dumps(state.get("order_data", {})),
        policy_result=json.dumps(state.get("policy_result", {})),
        guardrail_flags=json.dumps(state.get("guardrail_flags", {})),
    )

    handoff_dict = {}

    try:
        #ask the LLM to generate a structured handoff package
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        #remove markdown formatting before JSON parsing
        raw = raw.replace("```json", "").replace("```", "").strip()

        handoff_dict = json.loads(raw)

    except Exception as e:
        logger.warning(f"[escalation_agent] Handoff LLM call failed: {e}")

        #create a fallback handoff if the LLM response cannot be generated
        handoff_dict = {
            "customer_id": state["customer_id"],
            "issue_category": determine_issue_category(state),
            "conversation_summary": "Automated summary unavailable.",
            "recommended_action": "Review full conversation and follow up with customer.",
            "urgency_level": "MEDIUM",
            "refund_amount_requested": None,
            "order_ids_mentioned": [],
            "guardrail_triggers": state.get("guardrail_flags", {}),
        }

    #attach tracking information for support teams
    handoff_dict["escalated_at"] = datetime.utcnow().isoformat()
    handoff_dict["session_id"] = state.get("session_id", "")

    #store the formatted handoff summary in the agent state
    state["escalation_summary"] = json.dumps(handoff_dict, indent=2)

    logger.info(
        f"[escalation_agent] Handoff created | category={handoff_dict.get('issue_category')} "
        f"| urgency={handoff_dict.get('urgency_level')}"
    )

    #choose a customer facing response based on escalation type
    category = handoff_dict.get("issue_category", "OTHER")
    holding_msg = HOLDING_MESSAGES.get(category, HOLDING_MESSAGES["DEFAULT"])

    state["final_response"] = holding_msg

    #append the holding message to the conversation history
    state["messages"].append({"role": "assistant", "content": holding_msg})

    return state