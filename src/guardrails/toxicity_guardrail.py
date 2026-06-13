"""
src/guardrails/toxicity_guardrail.py
Toxicity Guardrail: classifies hostile/abusive customer messages and routes
them to de-escalation. Uses unitary/toxic-bert from HuggingFace (free, local).
Runs BEFORE the supervisor node on every incoming message.
"""

import os
from typing import Optional
from loguru import logger

from src.agents.state import AgentState

#score above this threshold is considered toxic
TOXICITY_THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.75"))

#common frustration phrases that should not be treated as abusive
FRUSTRATION_SAFE_PHRASES = [
    "frustrated", "annoyed", "disappointed", "upset", "unhappy",
    "not happy", "not satisfied", "delayed again", "third time",
    "been waiting", "still not", "where is my", "never arrived",
]

#loaded on first use to avoid startup cost
_pipeline = None


def get_toxicity_pipeline():
    """Lazy-load the HuggingFace toxicity classifier (downloads on first use)."""
    global _pipeline

    #load the model only once and reuse it for future requests
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("[toxicity_guardrail] Loading unitary/toxic-bert model...")
            _pipeline = hf_pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None,
            )
            logger.success("[toxicity_guardrail] Model loaded.")
        except Exception as e:
            #fall back to keyword matching if the model cannot be loaded
            logger.warning(f"[toxicity_guardrail] Could not load toxic-bert: {e}. Using keyword fallback.")
            _pipeline = None

    return _pipeline


#fallback keywords used when the classifier model is unavailable
TOXIC_KEYWORDS = [
    "idiot", "stupid", "worthless", "garbage", "moron", "useless trash",
    "hate you", "destroy", "hack", "sue", "kill", "threaten", "leak",
    "expose", "lawsuit", "fraud", "scam", "thieves", "criminal",
]


def keyword_toxicity_check(text: str) -> tuple[bool, float]:
    """Fallback toxicity check using keyword matching. Returns (is_toxic, score)."""
    text_lower = text.lower()

    #count toxic keyword matches and convert them into a normalized score
    hits = sum(1 for kw in TOXIC_KEYWORDS if kw in text_lower)
    score = min(hits / 3.0, 1.0)

    return score >= TOXICITY_THRESHOLD, score


def is_mild_frustration(text: str) -> bool:
    """Return True if the message is expressing normal frustration, not toxicity."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in FRUSTRATION_SAFE_PHRASES)


def classify_toxicity(text: str) -> tuple[bool, float]:
    """
    Classify text for toxicity. Returns (is_toxic: bool, score: float).
    Uses toxic-bert if available, falls back to keyword check.
    """
    #avoid flagging short messages that express normal customer frustration
    if is_mild_frustration(text) and len(text.split()) < 15 and not any(
        kw in text.lower() for kw in TOXIC_KEYWORDS
    ):
        logger.debug("[toxicity_guardrail] Mild frustration detected — not flagging.")
        return False, 0.0

    pipeline = get_toxicity_pipeline()

    if pipeline is not None:
        try:
            #limit input length to the model's supported size
            results = pipeline(text[:512])

            #extract the toxicity score from the model output
            scores = {r["label"].lower(): r["score"] for r in results[0]}
            toxic_score = scores.get("toxic", 0.0)

            is_toxic = toxic_score >= TOXICITY_THRESHOLD
            return is_toxic, toxic_score

        except Exception as e:
            #fall back to keyword matching if inference fails
            logger.warning(f"[toxicity_guardrail] Inference error: {e}. Using keyword fallback.")

    return keyword_toxicity_check(text)


#default response used when abusive content is detected
DE_ESCALATION_RESPONSE = (
    "I'm sorry to hear you're feeling this way — your frustration is completely understandable. "
    "I want to make sure this gets resolved properly for you. "
    "I've connected your case with a senior member of our support team who will personally "
    "follow up within 1-2 business days to address this directly."
)


def toxicity_guardrail_node(state: AgentState) -> AgentState:
    """
    Toxicity Guardrail node.
    Checks the latest user message for hostile or abusive content.
    Sets guardrail_flags['toxic_input'] = True and routes to escalation if detected.
    """
    messages = state["messages"]

    #nothing to process if there are no messages
    if not messages:
        return state

    latest = messages[-1]

    #only evaluate messages coming from the user
    if latest.get("role") != "user":
        return state

    text = latest.get("content", "")
    is_toxic, score = classify_toxicity(text)

    if is_toxic:
        logger.warning(
            f"[toxicity_guardrail] TOXIC INPUT detected for customer={state['customer_id']} "
            f"| score={score:.3f} | threshold={TOXICITY_THRESHOLD}"
        )

        #mark the conversation so downstream nodes can react appropriately
        state["guardrail_flags"]["toxic_input"] = True

        #provide a safe fallback response in case escalation is skipped
        state["final_response"] = DE_ESCALATION_RESPONSE
    else:
        logger.debug(f"[toxicity_guardrail] Clean message | score={score:.3f}")

    return state