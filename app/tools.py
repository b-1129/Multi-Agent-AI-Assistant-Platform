"""
Local, in-process tool wrappers -- these exist as the fallback path for
when the MCP server (mcp_server/server.py) is unreachable. See
app.mcp_client.get_tools_with_fallback(), which is what app.agents actually
calls.

The tool *logic* lives in app/core_tools.py and is shared with the MCP
server, so these two transports (direct Python call vs MCP) never drift
apart into two different implementations of "what calculator actually
does."

web_search is the one exception: DuckDuckGoSearchRun is already a
ready-made LangChain tool, so there's no separate "logic" module for it --
app.core_tools.web_search() just calls it directly, and this module wraps
the same underlying call for local use.
"""

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core_tools import calculate, search_documents


class CalculatorInput(BaseModel):
    expression: str = Field(
        ..., description="A basic arithmetic expression, e.g. '12 * (3 + 4)'."
    )


calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).",
    args_schema=CalculatorInput,
)

web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current information. Use for anything "
    "you don't already know or that may have changed recently.",
)


class DocumentSearchInput(BaseModel):
    query: str = Field(
        ..., description="What to search for in the uploaded documents."
    )


document_search_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_documents",
    description="Search the user's uploaded documents for relevant passages. "
    "Use this whenever the question could be answered from documents the "
    "user has uploaded, before falling back to web_search or general knowledge.",
    args_schema=DocumentSearchInput,
)