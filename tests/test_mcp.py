"""
Tests for the MCP server (mcp_server/server.py) and the fallback logic in
app/mcp_client.py.

The server test spins up the *real* server as a subprocess on a throwaway
port and connects to it with a real MCP client over real HTTP -- there's no
useful way to fake a network protocol server, so this is a genuine
integration test rather than a stubbed unit test. Only `calculator` is
exercised this way: it's pure and deterministic. `search_documents` and
`web_search` have their own logic already covered elsewhere (core_tools is
shared code; document search is tested via test_documents.py) without
needing a real embedding-model download or a real internet search inside
this test.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.mcp_client import get_tools_with_fallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.1)
    raise TimeoutError(f"Nothing listening on port {port} after {timeout}s")


@pytest.fixture
def mcp_server_url(tmp_path):
    """Starts the real MCP server as a subprocess on a free port, isolated
    from the real chroma_data/data dirs, and tears it down afterward."""
    port = _find_free_port()
    env = {
        **os.environ,
        "MCP_SERVER_HOST": "127.0.0.1",
        "MCP_SERVER_PORT": str(port),
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "DATA_DIR": str(tmp_path / "data"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_mcp_server_exposes_all_three_tools(mcp_server_url):
    import asyncio

    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def fetch():
        client = MultiServerMCPClient(
            {"agent_platform": {"transport": "streamable_http", "url": mcp_server_url}}
        )
        return await client.get_tools()

    tools = asyncio.run(fetch())
    tool_names = {t.name for t in tools}
    assert tool_names == {"calculator", "web_search_tool", "search_documents_tool"}


def test_mcp_server_calculator_tool_real_round_trip(mcp_server_url):
    import asyncio

    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def call():
        client = MultiServerMCPClient(
            {"agent_platform": {"transport": "streamable_http", "url": mcp_server_url}}
        )
        tools = await client.get_tools()
        calculator = next(t for t in tools if t.name == "calculator")
        return await calculator.ainvoke({"expression": "12 * 4"})

    result = asyncio.run(call())
    assert "48" in str(result)


def test_get_tools_with_fallback_uses_mcp_when_reachable(mcp_server_url, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mcp_server_url", mcp_server_url)
    tools = get_tools_with_fallback()

    assert set(tools.keys()) == {"calculator", "web_search", "search_documents"}

    # MCP-sourced tools are async-only -- confirm async invoke works and
    # returns the right answer.
    import asyncio

    async_result = asyncio.run(tools["calculator"].ainvoke({"expression": "7 * 6"}))
    assert "42" in str(async_result)


def test_get_tools_with_fallback_falls_back_when_unreachable(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mcp_server_url", "http://127.0.0.1:1/mcp")  # nothing listens here
    tools = get_tools_with_fallback()

    assert set(tools.keys()) == {"calculator", "web_search", "search_documents"}
    # Local fallback tools support sync invocation -- this is the whole
    # point of falling back rather than just failing.
    result = tools["calculator"].invoke({"expression": "7 * 6"})
    assert result == "42"