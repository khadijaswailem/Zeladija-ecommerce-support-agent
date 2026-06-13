"""
src/agents/order_lookup.py
Order Lookup Agent: extracts order ID from conversation, calls mock API,
formats a customer-facing response with live order data.
"""

import os
import re
import json
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from src.agents.state import AgentState
from src.tools.mock_api import get_order_status, get_customer_orders

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
STORE_NAME = os.getenv("STORE_NAME", "Zeladija")

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Return (or initialize) the OpenAI client."""
    global _openai_client

    #reuse a single client instance across requests
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


#match order IDs such as ORD-ABC123
ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Z0-9]{6,10}\b", re.IGNORECASE)

RESPONSE_PROMPT = """You are a helpful customer support agent for {store_name}, an electronics store.
A customer has asked about an order. Here is the real order data:

{order_data}

Write a friendly, concise response (2–4 sentences) that:
- Confirms the order ID and current status
- Mentions the estimated delivery date if the order is in transit
- Provides the tracking URL if available
- Is warm and professional in tone

Do not make up information not in the order data above.
Customer history context: {customer_history}"""

CLARIFY_PROMPT = """You are a helpful customer support agent for {store_name}.
The customer is asking about an order but has not provided an order ID.
Customer's recent orders: {recent_orders}

Write a short, friendly response (1–2 sentences) asking the customer to
provide their order ID, or offer to look up by customer ID if they have it.
If recent orders are available, list them briefly."""


def extract_order_id(messages: list[dict]) -> str | None:
    """Extract the most recent order ID from conversation history using regex."""

    #search from newest message to oldest to find the latest order reference
    for msg in reversed(messages):
        match = ORDER_ID_PATTERN.search(msg.get("content", ""))

        if match:
            return match.group(0).upper()

    return None


def order_lookup_node(state: AgentState) -> AgentState:
    """
    Order Lookup Agent node.
    Extracts order ID → calls mock API → formats response → writes to state.
    """
    messages = state["messages"]
    customer_id = state["customer_id"]
    customer_history = state.get("customer_history") or "No prior history."
    client = get_openai_client()

    #extract an order ID from the conversation history
    order_id = extract_order_id(messages)

    if not order_id:

        #retrieve recent customer orders to help identify the correct order
        recent_order_ids = get_customer_orders.invoke({"customer_id": customer_id})

        prompt = CLARIFY_PROMPT.format(
            store_name=STORE_NAME,
            recent_orders=", ".join(recent_order_ids) if recent_order_ids else "none found",
        )

        #generate a clarification request when no order ID is available
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        reply = response.choices[0].message.content.strip()

        state["final_response"] = reply
        state["order_data"] = {
            "status": "order_id_missing",
            "customer_id": customer_id
        }

        logger.info(f"[order_lookup] No order ID found for customer {customer_id}")

    else:

        #query the order system using the extracted order ID
        order = get_order_status.invoke({"order_id": order_id})
        state["order_data"] = order

        if "error" in order:

            #return a helpful message if the order does not exist
            state["final_response"] = (
                f"I couldn't find order {order_id} in our system. "
                "Please double-check the order ID from your confirmation email."
            )

            logger.warning(f"[order_lookup] Order not found: {order_id}")

        else:

            #convert order details into a prompt for response generation
            order_str = json.dumps(order, indent=2)

            prompt = RESPONSE_PROMPT.format(
                store_name=STORE_NAME,
                order_data=order_str,
                customer_history=customer_history,
            )

            #generate a customer friendly summary of the order status
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )

            reply = response.choices[0].message.content.strip()

            state["final_response"] = reply

            logger.info(
                f"[order_lookup] Resolved order {order_id} status={order.get('status')}"
            )

    #store the assistant response in the conversation history
    state["messages"].append(
        {"role": "assistant", "content": state["final_response"]}
    )

    return state