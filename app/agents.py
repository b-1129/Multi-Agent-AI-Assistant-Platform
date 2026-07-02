"""
The specialist sub-agents.

Each one is its own small `create_agent` graph with exactly one tool -- same
as phase 3. What changed in phase 4: the tools themselves now come from the
MCP server (mcp_server/server.py) instead of being imported directly as
local Python objects. `app.mcp_client.get_tools_with_fallback()` is the one
place that decides "MCP server, or local fallback" -- this module just asks
for tools by name and doesn't care which transport actually served them.

One real consequence of using MCP tools: they're async-only (the protocol
is async I/O under the hood), so these sub-agents must be invoked with
`.ainvoke()`, not `.invoke()` -- see app/graph.py, where the specialist node
wrappers are async functions for exactly this reason. The local fallback
tools (app/tools.py) support sync invocation too, so the system still works
if the MCP server is down, just over a slightly different code path under
the hood.
"""

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.mcp_client import get_tools_with_fallback

_llm = ChatGoogleGenerativeAI(
    model = settings.model_name,
    temperature = settings.model_temperature,
    api_key = settings.google_api_key,
)

_tools = get_tools_with_fallback()

research_agent = create_agent(
    model=_llm,
    tools=[_tools["web_search"]],
    system_prompt=(
        "You are a research specialist. Use web_search to answer questions "
        "about current events, facts, or anything you're not certain about. "
        "Be concise and cite what you found."
    ),
)

rag_agent = create_agent(
    model=_llm,
    tools=[_tools["search_documents"]],
    system_prompt=(
        "You are a document specialist. Use search_documents to answer "
        "questions about the user's uploaded files. If nothing relevant "
        "comes back, say so plainly rather than guessing. Mention which "
        "source document an answer came from."
    ),
)

action_agent = create_agent(
    model=_llm,
    tools=[_tools["calculator"]],
    system_prompt=(
        "You are an action specialist. Use the calculator tool for any "
        "arithmetic. Be concise -- state the result, not your work."
    ),
)

