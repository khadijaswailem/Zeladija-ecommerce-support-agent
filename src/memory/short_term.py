"""
src/memory/short_term.py
Short-term memory: manages the conversation message list within a single session.
Ensures messages are always appended (never overwritten) and provides helpers
for extracting context from conversation history.
"""

from typing import Optional
from loguru import logger


#maximum number of messages retained in memory
MAX_MESSAGES = 50


def append_message(messages: list[dict], role: str, content: str) -> list[dict]:
    """
    Safely append a message to the conversation history.
    Enforces MAX_MESSAGES cap by dropping oldest messages (keeping system context intact).
    """
    #trim older messages while preserving the first system message
    if len(messages) >= MAX_MESSAGES:
        messages = [messages[0]] + messages[-(MAX_MESSAGES - 2):]
        logger.debug(f"[short_term] Message cap hit — trimmed to {len(messages)} messages")

    messages.append({"role": role, "content": content})
    return messages


def get_conversation_text(messages: list[dict], max_turns: int = 10) -> str:
    """
    Return the last `max_turns` messages as a formatted string.
    Useful for including conversation context in LLM prompts.
    """
    #use only the most recent messages to keep prompts concise
    recent = messages[-max_turns:] if len(messages) > max_turns else messages

    lines = []
    for msg in recent:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def extract_mentioned_order_ids(messages: list[dict]) -> list[str]:
    """
    Scan all messages and return all unique order IDs mentioned in the conversation.
    """
    import re

    #match order IDs such as ORD-ABC123
    pattern = re.compile(r"\bORD-[A-Z0-9]{6,10}\b", re.IGNORECASE)

    found = set()

    for msg in messages:
        matches = pattern.findall(msg.get("content", ""))

        #store order IDs in uppercase to avoid duplicates
        for m in matches:
            found.add(m.upper())

    return list(found)


def has_user_already_provided(messages: list[dict], keyword: str) -> bool:
    """
    Check if the user has already mentioned a keyword (e.g. 'order ID', 'email')
    in a prior message. Avoids asking for the same info twice.
    """
    keyword_lower = keyword.lower()

    #search only user messages for the requested information
    for msg in messages:
        if msg.get("role") == "user" and keyword_lower in msg.get("content", "").lower():
            return True

    return False


def get_last_user_message(messages: list[dict]) -> Optional[str]:
    """Return the content of the most recent user message."""
    #iterate backwards to find the latest user message
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")

    return None


def summarize_session(messages: list[dict]) -> str:
    """
    Produce a brief text summary of what happened in this session.
    Used by long_term.py to extract facts for persistent storage.
    """
    #separate user and assistant messages for summary statistics
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    agent_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]

    return (
        f"Session had {len(user_msgs)} user turns and {len(agent_msgs)} agent turns. "
        f"First user message: '{user_msgs[0][:80]}...'" if user_msgs else "Empty session."
    )