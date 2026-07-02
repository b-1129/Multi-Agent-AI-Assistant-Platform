# Agent platform — phases 1-4 (MCP)

A supervisor agent coordinates three specialist agents (research, RAG,
action) built with LangGraph, with persistent per-session memory — and as
of phase 4, the specialists' tools are served over **MCP** (Model Context
Protocol) by a separate, standalone server process, instead of being
imported directly into the agent code.

This is the full project so far: FastAPI + Pydantic (phase 1), RAG over
uploaded documents (phase 2), multi-agent LangGraph + Postgres memory
(phase 3), and now tools-over-MCP with automatic fallback (phase 4).

## What's new in phase 4

- `app/core_tools.py` — the actual tool logic (calculator, document search,
  web search), with no framework wrapping. Shared by both the MCP server and
  the local fallback tools, so there's exactly one implementation of each tool.
- `mcp_server/server.py` — a standalone MCP server (FastMCP, streamable-HTTP
  transport) exposing `calculator`, `web_search_tool`, and
  `search_documents_tool`. Runs as its own process / container.
- `app/mcp_client.py` — fetches tools from the MCP server at startup; if the
  server is unreachable, falls back to local in-process tools automatically
  instead of crashing the app.
- `app/agents.py` — now sources tools via `get_tools_with_fallback()` instead
  of importing local tool objects directly.
- `app/graph.py` — specialist nodes are now **async** (`ainvoke`), because
  MCP-sourced tools are async-only.
- `app/main.py` — `/chat` is now an async endpoint.
- `tests/test_mcp.py` — spins up the real MCP server as a subprocess and
  tests it over real HTTP, plus tests of the fallback logic.
- `docker-compose.yml` — adds an `mcp-server` service; the `api` service
  depends on its health and reaches it by service name.

## Why MCP, conceptually

In phases 1-3, a tool was just a Python function imported directly into the
agent process. That's simple, but it means every agent that wants to use
`calculator` has to live in the same codebase, same language, same process.

MCP turns each tool into something served over a standard protocol instead:
any MCP-compatible client — this project's agents, Claude Desktop, a
completely different agent framework — can discover and call the same
tools without knowing anything about how they're implemented. The tool
*logic* in `core_tools.py` hasn't changed at all from phase 3; only how
it's reached has.

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│  api process (FastAPI)      │   MCP   │  mcp-server process       │
│  supervisor + 3 specialists │◀───────▶│  calculator                │
│  (app/graph.py, agents.py)  │  HTTP   │  web_search_tool           │
│                              │         │  search_documents_tool     │
└─────────────────────────────┘         └──────────────────────────┘
        │ on startup: can't reach MCP server?
        ▼
   falls back to local in-process tools (app/tools.py)
   instead of crashing
```

## Run it locally (no Docker)

Two processes now: the MCP server, and the API.

**Terminal 1 — start the MCP server:**

```bash
cd agent-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your real ANTHROPIC_API_KEY

python -m mcp_server.server
# -> Starting MCP server on 0.0.0.0:8001 (streamable-http, path=/mcp)
```

**Terminal 2 — start the API:**

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
# logs should show: "Loaded tools from MCP server at http://localhost:8001/mcp"
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what is 17 * 23?"}'
```

**Try the fallback path:** stop the MCP server (Ctrl+C in terminal 1), then
restart the API. The startup log should now say "Could not reach MCP server
... falling back to local tools" — and `/chat` still works correctly,
just using local in-process tools instead of MCP.

## Run it with Docker (all services together)

```bash
cp .env.example .env   # add your real ANTHROPIC_API_KEY
docker compose up --build
```

This starts `mcp-server`, `postgres`, and `api` together, in the right
order (`api` waits for both to be healthy). `MCP_SERVER_URL` is set
automatically to `http://mcp-server:8001/mcp` — the one setting that
differs between local and Docker, since the API container reaches the MCP
server by its Docker service name, not `localhost`.

## Run the tests

```bash
pytest
```

- `test_mcp.py` — starts the **real** MCP server as a subprocess on a free
  port and connects to it with a real client over real HTTP. Also tests
  that `get_tools_with_fallback()` correctly falls back to local tools when
  nothing is listening. No mocking of the protocol itself — there's no
  useful way to fake a network server, so this is a genuine integration test.
- `test_graph.py` — updated for async (`ainvoke`) graph execution; routing,
  persistence, and the FINISH safety net, all with stubbed LLM/specialist
  calls, no real model or MCP server needed.
- `test_documents.py`, `test_main.py` — unchanged in spirit from earlier
  phases (fake embeddings, isolated tmp dirs); `test_main.py`'s real-model
  test is skipped unless `ANTHROPIC_API_KEY` is set.

Note: simply importing `app.main` (which `test_documents.py` and
`test_main.py` both do) triggers `get_tools_with_fallback()` at module load
time. In CI / local test runs with no MCP server running, this correctly
and quickly falls back to local tools — you'll see the fallback warning in
test output, which is expected, not a failure.

## Why it's built this way (for revision later)

- **One implementation of each tool, two transports**: `core_tools.py`
  holds the actual logic; `app/tools.py` wraps it as local LangChain tools,
  `mcp_server/server.py` wraps the *same functions* as MCP tools. Without
  this split, the local fallback and the MCP-served version would drift
  into two different implementations of "what calculator actually does" —
  worth being able to explain why that's a real risk in larger systems with
  more tools and more contributors.
- **Fallback instead of a hard dependency**: an agent platform shouldn't go
  down just because one downstream tool server is slow or unavailable.
  `get_tools_with_fallback()` is the one seam that decides MCP vs local —
  tests monkeypatch `settings.mcp_server_url` to exercise both paths
  directly, the same dependency-injection pattern used throughout this
  project (`_build_embeddings` in phase 2, `supervisor_chain` in phase 3).
- **Async all the way through the specialist nodes**: MCP tools are
  async-only because the protocol itself is async I/O under the hood. Once
  one tool in the graph is async-only, the whole call path has to be — this
  is why `/chat` became an `async def` and specialist nodes use `ainvoke`,
  not just a local style preference.
- **A real subprocess in the MCP tests, not a mock**: protocols are exactly
  the kind of thing that's easy to mock incorrectly and end up testing your
  mock instead of your integration. `test_mcp.py` starts the actual server
  binary on a free port and talks to it over real HTTP — slower than a unit
  test, but it actually proves the wire protocol works.
- **Same FastMCP tool, different system prompt per specialist**: each
  specialist agent still only gets *one* tool (`research_agent` only sees
  `web_search`, etc.) even though all three tools live on the same MCP
  server. MCP doesn't force "one server = one tool exposed to everyone" —
  `app/agents.py` still picks which tools each specialist is allowed to use.

## Project roadmap (where this fits)

1. DONE — Phase 1: FastAPI + Pydantic + single LangChain agent + Docker
2. DONE — Phase 2: RAG — document upload, chunking, local embeddings, Chroma vector DB, agentic retrieval
3. DONE — Phase 3: Multi-agent with LangGraph — supervisor + research/RAG/action specialists, Postgres-backed session memory
4. DONE — Phase 4 (this one): MCP — tools served over a standalone MCP server, with automatic local fallback
5. Phase 5 — Security: guardrails (input/output checks) + gateway with model fallback
6. Phase 6 — Observability: LangSmith tracing + an automated agent eval set
7. Phase 7 — Ship it: Docker Compose locally, then AWS (ECS + RDS)
