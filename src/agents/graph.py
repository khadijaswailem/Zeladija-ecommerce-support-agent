"""
src/agents/graph.py
Assembles all nodes into the LangGraph StateGraph with conditional edges.
This is the main entry point for running the agent system.

Run: python -m src.agents.graph
"""

import os
import uuid
from loguru import logger
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from src.agents.state import AgentState, make_initial_state
from src.agents.supervisor import supervisor_node, route_by_intent
from src.agents.order_lookup import order_lookup_node
from src.agents.policy_agent import policy_agent_node
from src.agents.escalation_agent import escalation_agent_node
from src.guardrails.input_guardrail import input_guardrail_node
from src.guardrails.toxicity_guardrail import toxicity_guardrail_node

load_dotenv()

STORE_NAME = os.getenv("STORE_NAME", "Zeladija")


def respond_node(state: AgentState) -> AgentState:
    """
    Terminal node: formats and logs the final response.
    The final_response field is already populated by the specialist node;
    this node is a clean endpoint for the graph.
    """
    #log key information about the completed request
    logger.info(
        f"[respond] session={state.get('session_id')} "
        f"intent={state.get('intent')} "
        f"response_len={len(state.get('final_response', ''))}"
    )
    return state


def build_graph() -> StateGraph:
    """
    Construct and compile the Zeladija support agent LangGraph.

    Graph flow:
        [START]
           │
           ▼
    input_guardrail ──────────────────────────────────┐
           │                                           │
           ▼                                           │
    toxicity_guardrail ───────────────────────────────┤
           │                                           │
           ▼                                           │
       supervisor                                      │
           │ (conditional edge by intent)              │
           ├──ORDER_LOOKUP──► order_lookup ────────────┤
           ├──POLICY_RETURNS─► policy_agent ───────────┤
           ├──GENERAL_FAQ───► policy_agent ────────────┤
           └──ESCALATION────► escalation_agent ────────┤
                                                       ▼
                                                   respond ──► [END]
    """
    graph = StateGraph(AgentState)

    #register all graph nodes
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("toxicity_guardrail", toxicity_guardrail_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("order_lookup", order_lookup_node)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)
    graph.add_node("respond", respond_node)

    #start every request at the input guardrail
    graph.set_entry_point("input_guardrail")

    #run safety checks before intent routing
    graph.add_edge("input_guardrail", "toxicity_guardrail")
    graph.add_edge("toxicity_guardrail", "supervisor")

    #route requests to the appropriate specialist agent
    graph.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "order_lookup": "order_lookup",
            "policy_agent": "policy_agent",
            "escalation_agent": "escalation_agent",
        },
    )

    #allow policy agent to escalate when a policy violation is detected
    graph.add_conditional_edges(
        "policy_agent",
        lambda state: "escalation_agent" if state["guardrail_flags"].get("policy_violation") else "respond",
        {
            "escalation_agent": "escalation_agent",
            "respond": "respond",
        },
    )

    #normal completion paths
    graph.add_edge("order_lookup", "respond")
    graph.add_edge("escalation_agent", "respond")

    #end the workflow after the final response is prepared
    graph.add_edge("respond", END)

    compiled = graph.compile()

    logger.success("LangGraph compiled successfully.")

    return compiled


#create a reusable graph instance for the application
agent_graph = build_graph()


def run_agent(
    user_message: str,
    customer_id: str = "CUST-0001",
    session_id: str | None = None,
) -> AgentState:
    """
    Convenience function to run a single message through the agent graph.
    Returns the final AgentState.
    """

    #generate a unique session if one is not provided
    if session_id is None:
        session_id = str(uuid.uuid4())

    #build the initial state consumed by the graph
    initial_state = make_initial_state(
        customer_id=customer_id,
        session_id=session_id,
        user_message=user_message,
    )

    #execute the workflow and return the final state
    result = agent_graph.invoke(initial_state)

    return result


if __name__ == "__main__":
    import uuid

    print(f"\n{'='*60}")
    print(f"  Testing Agentic RAG Decision Logic")
    print(f"{'='*60}\n")

    #use one session so multiple turns share conversation context
    test_session_id = str(uuid.uuid4())

    #test a knowledge seeking query
    print("USER: What is your return policy for electronics?")
    result_1 = run_agent(
        user_message="What is your return policy for electronics?",
        customer_id="CUST-TEST",
        session_id=test_session_id
    )
    print(f"AGENT: {result_1['final_response']}\n")

    print("-" * 60 + "\n")

    #test a conversational follow up message
    print("USER: Got it, thanks for explaining that so clearly!")
    result_2 = run_agent(
        user_message="Got it, thanks for explaining that so clearly!",
        customer_id="CUST-TEST",
        session_id=test_session_id
    )
    print(f"AGENT: {result_2['final_response']}\n")