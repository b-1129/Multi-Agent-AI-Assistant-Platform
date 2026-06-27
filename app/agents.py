"""
The specialist sub-agents.

Each one is its own small `create_agent` graph with exactly one tool -- the
same pattern phase 1 and 2 used for the single agent, just narrower in scope.
The supervisor graph (app/graph.py) treats each of these as a single node:
it doesn't know or care that a sub-agent is itself a multi-step ReAct loop
internally. That's the point of hierarchical multi-agent design -- nesting
graphs inside graphs, each one a clean black box to whatever calls it.
"""

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.tools import get_research_tools, get_rag_tools, get_action_tools

_llm = ChatGoogleGenerativeAI(
    model = settings.model_name,
    temperature = settings.model_temperature,
    api_key = settings.google_api_key,
)

research_agent = create_agent(
    model=_llm,
    tools=get_research_tools,
    system_prompt=(
        "You are a research specialist. Use web_search to answer questions "
        "about current events, facts, or anything you're not certain about. "
        "Be concise and cite what you found."
    ),
)

rag_agent = create_agent(
    model=_llm,
    tools=get_rag_tools,
    system_prompt=(
        "You are a document specialist. Use search_documents to answer "
        "questions about the user's uploaded files. If nothing relevant "
        "comes back, say so plainly rather than guessing. Mention which "
        "source document an answer came from."
    ),
)

action_agent = create_agent(
    model=_llm,
    tools=get_action_tools(),
    system_prompt=(
        "You are an action specialist. Use the calculator tool for any "
        "arithmetic. Be concise -- state the result, not your work."
    ),
)

