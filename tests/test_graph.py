"""
Tests for the supervisor + specialist multi-agent graph.

The supervisor's routing decision and each specialist's sub-agent are
swapped for deterministic fakes via monkeypatch -- the same
dependency-injection seam used for embeddings in phase 2
(`_build_embeddings` there, `supervisor_chain` / the imported specialist
names here). That keeps these tests free, fast, and runnable with no
internet access or API key, while still exercising the real graph
topology, routing logic, and persistence behavior.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

import app.graph as graph_module


class FakeDecision:
    """Stand-in for the supervisor's structured RouteDecision output."""

    def __init__(self, next: str, reasoning: str = "test"):
        self.next = next
        self.reasoning = reasoning


class ScriptedSupervisorChain:
    """Returns a pre-scripted sequence of routing decisions, one per call.
    Falls back to FINISH once the script runs out."""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def invoke(self, _messages):
        if self.decisions:
            return self.decisions.pop(0)
        return FakeDecision("FINISH")


class FakeSpecialist:
    """Stand-in for a compiled sub-agent: returns a fixed answer and records
    a fake tool call, without making any real model call."""

    def __init__(self, name: str, answer: str):
        self.name = name
        self.answer = answer

    def invoke(self, state):
        message = AIMessage(content=self.answer, name=self.name)
        message.tool_calls = [
            {"name": f"{self.name}_tool", "args": {}, "id": "fake-call-1"}
        ]
        return {"messages": state["messages"] + [message]}


@pytest.fixture
def stub_specialists(monkeypatch):
    """Replace all three specialists with fakes by default. Individual
    tests can still override one further if they need a specific answer."""
    monkeypatch.setattr(graph_module, "research_agent", FakeSpecialist("research_agent", "research answer"))
    monkeypatch.setattr(graph_module, "rag_agent", FakeSpecialist("rag_agent", "rag answer"))
    monkeypatch.setattr(graph_module, "action_agent", FakeSpecialist("action_agent", "42"))


def test_routes_to_one_specialist_then_finishes(monkeypatch, stub_specialists):
    monkeypatch.setattr(
        graph_module,
        "supervisor_chain",
        ScriptedSupervisorChain([FakeDecision("action_agent"), FakeDecision("FINISH")]),
    )

    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-single-route"}}
    result = graph.invoke({"messages": [HumanMessage(content="what is 6 times 7?")]}, config=config)

    assert result["messages"][-1].content == "42"
    assert result["messages"][-1].name == "action_agent"
    assert result["tool_calls"] == [
        {"agent_name": "action_agent", "tool_name": "action_agent_tool", "tool_input": "{}"}
    ]


def test_can_route_to_multiple_specialists_in_one_turn(monkeypatch, stub_specialists):
    monkeypatch.setattr(
        graph_module,
        "supervisor_chain",
        ScriptedSupervisorChain(
            [FakeDecision("action_agent"), FakeDecision("research_agent"), FakeDecision("FINISH")]
        ),
    )

    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-multi-route"}}
    result = graph.invoke({"messages": [HumanMessage(content="calc this and look that up")]}, config=config)

    specialist_names = [m.name for m in result["messages"] if isinstance(m, AIMessage)]
    assert specialist_names == ["action_agent", "research_agent"]
    assert len(result["tool_calls"]) == 2


def test_safety_net_prevents_echoing_user_message_back(monkeypatch, stub_specialists):
    # Supervisor misjudges and tries to FINISH before any specialist has
    # answered the fresh human message -- the safety net in supervisor_node
    # must override this rather than let the graph end with the user's own
    # message as the last one in state.
    monkeypatch.setattr(
        graph_module, "supervisor_chain", ScriptedSupervisorChain([FakeDecision("FINISH")])
    )

    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-safety-net"}}
    result = graph.invoke({"messages": [HumanMessage(content="hello?")]}, config=config)

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content != "hello?"


def test_conversation_persists_across_turns_with_same_thread_id(monkeypatch, stub_specialists):
    monkeypatch.setattr(
        graph_module,
        "supervisor_chain",
        ScriptedSupervisorChain(
            [FakeDecision("action_agent"), FakeDecision("FINISH"), FakeDecision("rag_agent"), FakeDecision("FINISH")]
        ),
    )

    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-persistence"}}

    graph.invoke({"messages": [HumanMessage(content="first question")]}, config=config)
    result = graph.invoke({"messages": [HumanMessage(content="second question")]}, config=config)

    # All four messages (2 human + 2 ai) should be present -- proving the
    # checkpointer restored turn 1's history before turn 2 ran.
    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "first question"
    assert result["messages"][2].content == "second question"


def test_different_thread_ids_do_not_share_history(monkeypatch, stub_specialists):
    monkeypatch.setattr(
        graph_module,
        "supervisor_chain",
        ScriptedSupervisorChain([FakeDecision("action_agent"), FakeDecision("FINISH")] * 2),
    )

    graph = graph_module.build_graph()

    graph.invoke(
        {"messages": [HumanMessage(content="session A message")]},
        config={"configurable": {"thread_id": "session-a"}},
    )
    result_b = graph.invoke(
        {"messages": [HumanMessage(content="session B message")]},
        config={"configurable": {"thread_id": "session-b"}},
    )

    # Session B should only see its own message, not session A's.
    assert len(result_b["messages"]) == 2
    assert result_b["messages"][0].content == "session B message"


def test_runaway_supervisor_loop_hits_recursion_limit(monkeypatch, stub_specialists):
    # A supervisor that never says FINISH should be caught by the
    # recursion limit rather than looping forever.
    class InfiniteSupervisorChain:
        def invoke(self, _messages):
            return FakeDecision("action_agent")

    monkeypatch.setattr(graph_module, "supervisor_chain", InfiniteSupervisorChain())

    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "test-recursion"}, "recursion_limit": 8}

    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="loop forever")]}, config=config)
