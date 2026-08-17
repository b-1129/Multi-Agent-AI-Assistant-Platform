"""
The MCP server: exposes this project's tools over the Model Context
Protocol instead of importing them directly into an agent process.

This is a genuinely separate, standalone server -- run it as its own
process (or container, see docker-compose.yml) and any MCP-compatible
client (this project's agents, Claude Desktop, another agent framework
entirely) can discover and call these same tools over HTTP. The tool
*logic* is unchanged from phase 1-3 -- see app/core_tools.py -- only the
transport is new.

Run directly:  python -m mcp_server.server
Or via Docker:  see the `mcp-server` service in docker-compose.yml
"""

import logging

from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.core_tools import calculate, search_documents, web_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

mcp = FastMCP(
    "agent-paltform-tools",
    instructions=(
        "Tools for an agentic platform: arithmetic, web search, and"
        "search over a user's uploaded documents."
    ),
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
)

@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).
    Example: '12 * (3 + 4)' """
    return calculate(expression)

@mcp.tool()
def web_search_tool(query: str) -> str:
    """Search the web for current information -- news, facts, anything that
    may have changed since training data was collected."""
    return web_search(query)

mcp.tool()
def search_documents_tool(query: str, k: int = 4) -> str:
    """Search the user's uploaded documents (ingested via the main API's
    /documents/upload endpoint) for passages relevant to the query."""
    return search_documents(query, k=k)

if __name__ == "__main__":
    logger.info(
        "Starting MCP server on %s:%s (streamable-http, patch=/mcp)",
        settings.mcp_server_host,
        settings.mcp_server_port,
    )
    mcp.run(transport= "streamable-http")
