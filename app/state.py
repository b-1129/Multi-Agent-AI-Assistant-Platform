"""
The state that flows through the supervisor + specialist graph.

A TypedDict + `add_messages` reducer is the standard LangGraph pattern (not
a Pydantic model, even though Pydantic is used everywhere else in this
project) -- `add_messages` knows how to append new messages to existing
history, merge duplicate IDs, etc. Pydantic models don't get that reducer
behavior for free, so plain LangGraph state stays a TypedDict.

`tool_calls` uses a simple list-concatenation reducer so that every
specialist node's tool calls accumulate across a single turn -- useful for
showing the caller exactly what happened, without exposing the sub-agents'
full internal message history.
"""

import operator
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: str
    tool_calls: Annotated[List[dict], operator.add]