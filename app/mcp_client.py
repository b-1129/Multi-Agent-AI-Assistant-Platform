"""
Fetches this project's tools from the MCP server (app.agents uses this
instead of importing tool objects directly, which is the whole point of
phase 4).

If the MCP server can't be reached at startup -- not running yet, wrong
URL, whatever -- this falls back to the local, in-process tools from
app.tools instead of crashing the whole app. That's a real production
concern: an agent platform shouldn't go down just because one downstream
tool server is unavailable, and it's worth being able to point at this
fallback path and explain why it's there.
"""

import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings

logger = logging.getLogger(settings.app_name)

MCP_TOOL_NAMES = {
    "calculator": "calculator",
    "web_search": "web_search_tool",
    "search_documents": "search_documents_tool",
}

async def _fetch_mcp_tools() -> dict[str, BaseTool]:
    client = MultiServerMCPClient(
        {
            "agent_platform": {
                "transport": "streamable_http",
                "url": settings.mcp_server_url,
            }
        }
    )

    tools = await client.get_tools()
    return {tool.name: tool for tool in tools}

def get_tools_with_fallback() -> dict[str, BaseTool]:
    """Returns {'calculator': tool, 'web_search': tool, 'search_documents': tool},
    sourced from the MCP server if reachable, or local fallback tools if not."""
    try:
        by_name = asyncio.run(_fetch_mcp_tools())
        resolved = {
            local_name: by_name[mcp_name] for local_name, mcp_name in MCP_TOOL_NAMES.items()
        }
        logger.info("Loaded tools from MCP server at %s", settings.mcp_server_url)
        return resolved
    except Exception as exc:
        logger.warning(
            "Could not reach MCP server at %s (%s) -- falling back to local tools.",
            settings.mcp_server_url,
            exc,
        )
        from app.tools import calculator_tool, document_search_tool, web_search_tool

        return {
            "calculator": calculator_tool,
            "web_search": web_search_tool,
            "search_documents": document_search_tool,
        }