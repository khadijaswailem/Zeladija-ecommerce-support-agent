"""
chainlit_app.py
Chainlit chat UI for the Zeladija AI Support Agent.

Run: chainlit run chainlit_app.py
"""

import os
import uuid
from dotenv import load_dotenv
import chainlit as cl
from loguru import logger

load_dotenv()

STORE_NAME = os.getenv("STORE_NAME", "Zeladija")
PRODUCT_CATEGORY = os.getenv("PRODUCT_CATEGORY", "Electronics")

# Import the agent — lazy so Chainlit starts even if a dep is slow
from src.agents.graph import run_agent


INTENT_LABELS = {
    "ORDER_LOOKUP":     ("📦", "Order Lookup"),
    "POLICY_RETURNS":   ("↩️",  "Returns & Refunds"),
    "GENERAL_FAQ":      ("💬", "FAQ"),
    "ESCALATION":       ("🚨", "Escalated"),
    "UNKNOWN":          ("❓", "Unknown"),
    "ERROR":            ("⚠️",  "Error"),
}

GUARDRAIL_LABELS = {
    "injection_detected": "🛡️ Injection attempt blocked",
    "policy_violation":   "📋 Policy violation flagged",
    "toxic_input":        "🚫 Toxic content detected",
}


@cl.on_chat_start
async def on_chat_start():
    """Initialise session state when a new conversation starts."""
    session_id = str(uuid.uuid4())
    customer_id = f"CUST-{session_id[:8].upper()}"

    cl.user_session.set("session_id", session_id)
    cl.user_session.set("customer_id", customer_id)
    cl.user_session.set("turn_count", 0)

    await cl.Message(
        content=(
            f"👋 Welcome to **{STORE_NAME} Support**!\n\n"
            f"I can help you with:\n"
            f"- 📦 **Order tracking** — give me your order ID (e.g. `ORD-AB12CD34`)\n"
            f"- ↩️ **Returns & refunds** — policies, status, and processing\n"
            f"- 💬 **Product questions** — {PRODUCT_CATEGORY} FAQs\n"
            f"- 🚨 **Escalations** — I'll connect you with a human agent if needed\n\n"
            f"How can I help you today?"
        ),
        author=STORE_NAME,
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle each incoming user message."""
    session_id = cl.user_session.get("session_id")
    customer_id = cl.user_session.get("customer_id")
    turn_count = cl.user_session.get("turn_count", 0) + 1
    cl.user_session.set("turn_count", turn_count)

    # Show a typing indicator while the agent runs
    async with cl.Step(name="Agent thinking…", show_input=False) as step:
        try:
            state = run_agent(
                user_message=message.content,
                customer_id=customer_id,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await cl.Message(
                content="⚠️ Something went wrong on my end. Please try again.",
                author=STORE_NAME,
            ).send()
            return

        # Pull values from state
        final_response  = state.get("final_response", "I'm sorry, I couldn't generate a response.")
        intent          = state.get("intent", "UNKNOWN")
        guardrail_flags = state.get("guardrail_flags", {})
        escalated       = bool(state.get("escalation_summary", ""))

        # Build step summary (visible in the Chainlit "steps" panel)
        icon, label = INTENT_LABELS.get(intent, ("❓", intent))
        step.output = f"{icon} Intent: **{label}**"
        if escalated:
            step.output += "  |  🚨 Escalated to human agent"

    # Surface any guardrail flags as a small warning above the response
    active_flags = [
        GUARDRAIL_LABELS[k]
        for k, v in guardrail_flags.items()
        if v and k in GUARDRAIL_LABELS
    ]
    if active_flags:
        flags_text = "  \n".join(f"`{f}`" for f in active_flags)
        await cl.Message(
            content=flags_text,
            author="Guardrails",
        ).send()

    # Send the main response
    await cl.Message(
        content=final_response,
        author=STORE_NAME,
    ).send()

    # After escalation, offer a follow-up action
    if escalated:
        actions = [
            cl.Action(
                name="new_conversation",
                label="Start a new conversation",
                payload={"action": "reset"},
                tooltip="Clear this conversation and start fresh",
            )
        ]
        await cl.Message(
            content="Your case has been logged and a human agent will follow up. Is there anything else I can help with?",
            author=STORE_NAME,
            actions=actions,
        ).send()


@cl.action_callback("new_conversation")
async def on_new_conversation(action: cl.Action):
    """Reset the session when the user clicks 'Start a new conversation'."""
    new_session_id = str(uuid.uuid4())
    new_customer_id = f"CUST-{new_session_id[:8].upper()}"
    cl.user_session.set("session_id", new_session_id)
    cl.user_session.set("customer_id", new_customer_id)
    cl.user_session.set("turn_count", 0)

    await cl.Message(
        content="✅ New conversation started. How can I help you?",
        author=STORE_NAME,
    ).send()