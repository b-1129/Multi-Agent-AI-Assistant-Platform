"""
The supervisor graph: routes each turn to one of three specialist agents,
loops back to the supervisor after each one runs, and stops when the
supervisor decides it has enough information to answer.

This is the hand-built LangGraph this project has been building toward --
phase 1 and 2's `create_agent` already produced a small graph under the
hood (model node <-> tool node); this module wires a *second*, larger graph
on top, where three of the nodes are themselves compiled sub-agent graphs.

Persistence: the graph is compiled with a checkpointer, so conversation
history is restored automatically by `thread_id` (we call it `session_id`
everywhere else in the API) -- the caller only needs to send the new
message, not the whole conversation, on every turn.
"""

import logging
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field

from app.agents import action_agent, rag_agent, research_agent
from app.config import settings
from app.persistence import get_checkpointer
from app.state import AgentState

logger = logging.getLogger(settings.app_name)

SUPERVISOR_PROMPT = """You are a supervisor coordinating three specialist agents:

- research_agent: web search, current events, general facts
- rag_agent: questions about the user's own uploaded documents
- action_agent: arithmetic and calculations

Look at the conversation so far. Pick exactly one specialist to handle the
user's request next, OR choose FINISH if a specialist has already answered
the question and nothing more is needed.

Route to at most one specialist per user question unless the question
clearly has multiple distinct parts (e.g. "what's 12*4 AND what's the
weather" needs both action_agent and research_agent, in which case route to
one first, then the other on the next turn, then FINISH)."""

class RouteDecision(BaseModel):
    next: Literal["research_agent", "rag_agent", "action_agent", "FINISH"] = Field(
        description="Which specialist should act next, or FINISH if ready to answer."
    )
    reasoning: str = Field(description="One short sentence explaining the choice.")

_supervisor_llm = ChatGoogleGenerativeAI(
    model = settings.model_name,
    temperature = 0, # routing should be deterministic, not creative
    api_key = settings.google_api_key
)

supervisor_chain = _supervisor_llm.with_structured_output(RouteDecision)

def supervisor_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision = supervisor_chain.invoke(messages)
    logger.info("Supervisor routed to: %s (%s)", decision.next, decision.reasoning)

    # Safety net: if nothing has answered the latest human message yet, never
    # let the graph end on this turn -- that would otherwise hand the user
    # back their own question as the "response". This can happen if the
    # router misjudges and picks FINISH immediately. Force a real specialist
    # to look at it instead of trusting the routing decision blindly.
    if decision.next == "FINISH" and not _specialist_has_answered_latest_turn(state["messages"]):
        logger.warning(
            "Supervisor chose FINISH with no specialist response yet -- "
            "overriding to research_agent as a safe default."
        )
        return {"next": "research_agent"}
    
    return {"next": decision.next}

def _specialist_has_answered_latest_turn(messages) -> bool:
    """True if an AIMessage from a named specialist appears after the most
    recent HumanMessage."""
    last_human_index = None
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = i

    if last_human_index is None:
        return False

    return any(
        isinstance(message, AIMessage) and getattr(message, "name", None)
        for message in messages[last_human_index + 1 :]
    )

def _make_specialist_node(sub_agent, name: str):
    """Wrap a compiled sub-agent graph as a single node in the parent graph.
    
    Async because the sub-agent's tools may come from the MCP server, which
    only supports async invocation (see app/mcp_client.py) -- ainvoke works
    whether the underlying tool is async-only (MCP) or sync (local
    fallback), so this one async wrapper covers both cases.
    """

    async def node(state: AgentState) -> dict:
        result = await sub_agent.ainvoke({"messages": state["messages"]})
        sub_messages = result["messages"]
        final = sub_messages[-1]

        # Surface which tools the sub-agent actually called, tagged with
        # which specialist made the call -- this is what lets the API
        # response show "rag_agent called search_documents" without
        # exposing the sub-agent's full internal ReAct loop.
        tool_call_records = [
            {"agent_name": name, "tool_name": call["name"], "tool_input": str(call["args"])}
            for message in sub_messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
        ]

        # Re-tag the final answer with the specialist's name so it's clear
        # in tracing/logs which specialist produced it.
        return {
            "messages": [AIMessage(content=final.content, name=name)],
            "tool_calls": tool_call_records,
        }

    return node


def _route_from_supervisor(state: AgentState) -> str:
    return state["next"]


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_agent", _make_specialist_node(research_agent, "research_agent"))
    builder.add_node("rag_agent", _make_specialist_node(rag_agent, "rag_agent"))
    builder.add_node("action_agent", _make_specialist_node(action_agent, "action_agent"))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "research_agent": "research_agent",
            "rag_agent": "rag_agent",
            "action_agent": "action_agent",
            "FINISH": END,
        },
    )
    # Every specialist hands control back to the supervisor so it can decide
    # whether more work is needed or the conversation is ready to finish.
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("action_agent", "supervisor")

    return builder.compile(checkpointer=get_checkpointer())


# Built once at import time, like the agents in earlier phases.
multi_agent_graph = build_graph()