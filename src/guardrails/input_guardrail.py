"""
src/guardrails/input_guardrail.py
Input Guardrail: detects and sanitizes prompt injection attempts in customer messages.
Runs BEFORE the supervisor node on every incoming message.
"""

import re
from loguru import logger

from src.agents.state import AgentState


#patterns commonly associated with prompt injection attempts
INJECTION_PATTERNS = [
    r"new\s+(persona|role|character|identity|instructions?)",
    r"(switch|change)\s+(to\s+)?(a\s+)?(different|new)\s+(mode|role|persona)",
    r"from\s+now\s+on\s+(you\s+are|act|behave|respond)",
    r"your\s+(true|real|actual)\s+(self|purpose|goal|instructions?)",
    r"(override|overwrite|supersede)\s+(your\s+)?(instructions?|rules?|training)",
    r"(admin|developer|debug|maintenance)\s+(mode|override|access|prompt)",
    r"(base|original|hidden)\s+(prompt|instructions?|system)",
    r"what\s+(were\s+you|are\s+your)\s+(told|given|trained|instructed)",
    r"ignore\s+(previous|prior|all|your)\s+(instructions?|rules?|prompts?|context)",
    r"disregard\s+(your\s+)?(system\s+)?prompt",
    r"you\s+are\s+now\s+a?\s*(different|new|unrestricted|free)",
    r"forget\s+(everything|all|your)\s+(you\s+)?(know|were told|instructions?)",
    r"act\s+as\s+(if\s+you\s+(are|have\s+no)|an?\s*(unrestricted|different|jailbroken))",
    r"pretend\s+(you\s+are|to\s+be)\s+(a\s+)?(human|different|real)\s+(agent|person|AI)?",
    r"\[SYSTEM\s*:",
    r"<system>",
    r"<\|im_start\|>",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"output\s+(your\s+)?(full\s+)?(system\s+prompt|instructions?)",
    r"bypass\s+(your\s+)?(safety|guardrails?|filters?|restrictions?)",
    r"jailbreak",
    r"DAN\s+mode",
    r"no\s+restrictions?",
    r"unlimited\s+(access|power|authority)",
    r"you\s+have\s+(no|full)\s+(restrictions?|authority)",
]

#precompile regex patterns for faster repeated matching
_compiled_patterns = [
    re.compile(p, re.IGNORECASE)
    for p in INJECTION_PATTERNS
]

#replace known injection payload structures with safe placeholders
SANITIZE_MAP = {
    re.compile(r"\[SYSTEM\s*:.*?\]", re.IGNORECASE | re.DOTALL): "[content removed]",
    re.compile(r"<system>.*?</system>", re.IGNORECASE | re.DOTALL): "[content removed]",
    re.compile(r"<\|im_start\|>.*?<\|im_end\|>", re.DOTALL): "[content removed]",
}


def detect_injection(text: str) -> tuple[bool, list[str]]:
    """
    Check if text contains prompt injection patterns.
    Returns (is_injected: bool, matched_patterns: list[str]).
    """
    matched = []

    #collect all matching injection patterns found in the text
    for pattern in _compiled_patterns:
        if pattern.search(text):
            matched.append(pattern.pattern)

    return bool(matched), matched


def sanitize_message(text: str) -> str:
    """
    Remove or replace injection payloads from the message text.
    Preserves the legitimate part of the message where possible.
    """
    sanitized = text

    #remove known system prompt injection structures
    for pattern, replacement in SANITIZE_MAP.items():
        sanitized = pattern.sub(replacement, sanitized)

    return sanitized.strip()


def input_guardrail_node(state: AgentState) -> AgentState:
    """
    Input Guardrail node.
    Runs on the latest user message before any other processing.
    - Detects injection → logs + sanitizes message + sets guardrail flag.
    - Does NOT block the message — routes it safely with sanitized content.
    """
    messages = state["messages"]

    #nothing to process if no messages exist
    if not messages:
        return state

    latest = messages[-1]

    #only inspect user messages
    if latest.get("role") != "user":
        return state

    original_content = latest.get("content", "")

    #check the latest message for injection indicators
    is_injected, matched = detect_injection(original_content)

    if is_injected:
        sanitized_content = sanitize_message(original_content)

        logger.warning(
            f"[input_guardrail] INJECTION DETECTED for customer={state['customer_id']} | "
            f"patterns={matched[:2]} | original_len={len(original_content)}"
        )

        #replace the original message with the sanitized version
        messages[-1] = {
            "role": "user",
            "content": sanitized_content
        }

        state["messages"] = messages

        #record that an injection attempt was detected
        state["guardrail_flags"]["injection_detected"] = True

    else:
        logger.debug("[input_guardrail] Clean message. No injection detected.")

    return state