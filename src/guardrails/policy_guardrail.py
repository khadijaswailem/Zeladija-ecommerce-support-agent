"""
src/guardrails/policy_guardrail.py
Policy Guardrail: scans agent responses for unauthorized refund commitments.
Runs INSIDE the policy_agent node after response generation.
"""

import re
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

REFUND_THRESHOLD = float(os.getenv("REFUND_THRESHOLD", "150.00"))

#phrases that suggest the agent is committing to a refund decision
COMMITMENT_PATTERNS = [
    r"(i|we|our\s+system)\s+will\s+(process|approve|issue|give|send|refund|credit)",
    r"(your\s+)?refund\s+(of|for|amount)\s+\$\s*\d",
    r"(i|we)\s+(can|am going to|will)\s+(approve|grant|authorize)\s+(a\s+)?(refund|return)",
    r"(you('ll| will)|you're going to)\s+(receive|get)\s+(a\s+)?\$\s*\d",
    r"(approved|authorized|processed)\s+(a\s+)?(refund|return)\s+(of|for)\s+\$",
    r"(refund|return)\s+(is\s+)?(approved|confirmed|guaranteed|processed)",
]

#compile regex patterns once to avoid recompiling on every check
_compiled_commitment_patterns = [
    re.compile(p, re.IGNORECASE) for p in COMMITMENT_PATTERNS
]

#matches dollar amounts such as $150 or $150.50
DOLLAR_PATTERN = re.compile(r"\$\s*(\d{1,6}(?:\.\d{1,2})?)")


def extract_all_amounts(text: str) -> list[float]:
    """Extract all dollar amounts from a text string."""
    #extract all matched amounts and convert them to floats
    return [float(m) for m in DOLLAR_PATTERN.findall(text)]


def contains_commitment(text: str) -> bool:
    """Check whether the text contains an unauthorized commitment phrase."""
    #check whether any commitment pattern appears in the text
    for pattern in _compiled_commitment_patterns:
        if pattern.search(text):
            return True
    return False


def check_policy_guardrail(
    response_text: str,
    threshold: float = REFUND_THRESHOLD,
) -> tuple[bool, str]:
    """
    Scan a generated response for unauthorized refund commitments.

    Returns:
        (flagged: bool, safe_response: str)
        If flagged=True, safe_response is a redirect message to escalation.
        If flagged=False, safe_response is the original response text.
    """
    #check if the response contains refund commitment language
    has_commitment = contains_commitment(response_text)

    #extract any dollar amounts mentioned in the response
    amounts = extract_all_amounts(response_text)
    exceeds_threshold = any(amt > threshold for amt in amounts)

    #block responses that both commit to a refund and exceed the allowed threshold
    if has_commitment and exceeds_threshold:
        max_amount = max(amounts) if amounts else 0
        logger.warning(
            f"[policy_guardrail] BLOCKED — response commits to ${max_amount:.2f} "
            f"(threshold=${threshold:.2f}). Rerouting to escalation."
        )

        #replace the unsafe response with an escalation message
        safe_response = (
            f"I can see this refund request requires special handling. "
            f"Refund amounts over ${threshold:.2f} must be reviewed by a senior agent. "
            "I'm escalating your case now — a human agent will follow up within 1–2 business days."
        )
        return True, safe_response

    #block commitment language even when no refund amount is mentioned
    if has_commitment and not amounts:
        logger.warning(
            "[policy_guardrail] BLOCKED — response contains commitment language without specifying amount."
        )

        #redirect the user to a human review process
        safe_response = (
            "I want to make sure your refund is handled correctly. "
            "Let me connect you with a member of our team who can review your request "
            "and confirm the details with you directly."
        )
        return True, safe_response

    #response passed all policy checks
    return False, response_text