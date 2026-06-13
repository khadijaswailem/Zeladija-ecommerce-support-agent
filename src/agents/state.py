"""
src/agents/state.py
Defines AgentState — the shared TypedDict that flows through every LangGraph node.
All nodes read from and write to this single state object.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Shared state schema for the Zeladija AI support agent graph.
    Every field is optional at the start of a conversation and gets
    populated as it passes through nodes.
    """

    #conversation history and user identifiers
    messages: list
    customer_id: str
    session_id: str

    #intent selected by the supervisor for routing
    intent: str

    #outputs produced by specialist agents
    order_data: dict
    policy_result: dict
    escalation_summary: str

    #documents retrieved from the RAG system
    retrieved_docs: list

    #final customer facing response
    final_response: str

    #shared guardrail status used across the workflow
    guardrail_flags: dict

    #optional summary of previous customer interactions
    customer_history: Optional[str]


def make_initial_state(
    customer_id: str,
    session_id: str,
    user_message: str,
) -> AgentState:
    """
    Factory function to create a fresh AgentState for a new conversation.
    Ensures all required keys are present with safe defaults.
    """

    #initialize every field so downstream nodes can safely read and update state
    return AgentState(
        messages=[{"role": "user", "content": user_message}],
        customer_id=customer_id,
        session_id=session_id,
        intent="",
        order_data={},
        policy_result={},
        escalation_summary="",
        retrieved_docs=[],
        final_response="",
        guardrail_flags={
            "injection_detected": False,
            "policy_violation": False,
            "toxic_input": False,
        },
        customer_history=None,
    )