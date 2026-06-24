# Agent platform — phase 1

A single LangChain agent (calculator + web search tools), validated end to end
with Pydantic, served over a FastAPI REST API, and containerized with Docker.
This is the foundation phase of a larger multi-agent platform (see the project
roadmap at the bottom).

## What's in this phase

- `app/config.py` — Pydantic settings, loaded from environment variables
- `app/schemas.py` — Pydantic request/response models for the API
- `app/tools.py` — one custom tool (calculator) and one community tool (web search)
- `app/agent.py` — the LangChain agent + `AgentExecutor`
- `app/main.py` — FastAPI app exposing `/health` and `/chat`
- `tests/test_main.py` — smoke tests
- `Dockerfile` / `docker-compose.yml` — containerized local run

## Run it locally (no Docker)

```bash
cd agent-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your real GOOGLE_API_KEY

uvicorn app.main:app --reload
```

Then visit http://localhost:8000/docs for the interactive Swagger UI, or:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 23 * 17, and who won the last World Cup?"}'
```

You should see the agent call `calculator` for the arithmetic and `web_search`
for the World Cup question - check the `tool_calls` field in the response.

## Run it with Docker

```bash
cp .env.example .env   # add your real key
docker compose up --build
```

API is then available at http://localhost:8000.

## Run the tests

```bash
pytest
```

`test_health` always runs. `test_chat_calculator` only runs if
`GOOGLE_API_KEY` is set, since it calls the real model.

## Why it's built this way (for revision later)

- **Pydantic everywhere**: settings, request/response schemas, and the tool's
  input schema are all Pydantic models. This is the same discipline you'll
  carry into every later phase — agent state, gateway payloads, eval records.
- **`create_agent`, not manual prompt-parsing**: LangChain's current agent
  API builds a small LangGraph graph for you under the hood (a model node and
  a tool node, looping until the model stops calling tools). That means
  phase 1 is already standing on LangGraph — phase 3 is where you take the
  wheel and build a custom multi-node graph by hand instead of using this helper.
- **Extracting tool calls from the message list**: the agent returns a list
  of messages (`HumanMessage`, `AIMessage`, `ToolMessage`); the API pulls out
  each `AIMessage.tool_calls` so you can see exactly which tools ran - useful
  for debugging now, and a preview of what you'll formalize into full
  LangSmith traces in a later phase.
- **One safe custom tool**: the calculator uses a restricted AST evaluator
  instead of `eval()` - a small, real example of the "don't trust model or
  user input blindly" mindset that becomes the guardrails phase.

## Project roadmap (where this fits)

1. DONE — Phase 1 (this one): FastAPI + Pydantic + single LangChain agent + Docker
2. Phase 2 - RAG: document upload, chunking, embeddings, Chroma vector DB
3. Phase 3 - Multi-agent with LangGraph: supervisor + research/RAG/action agents
4. Phase 4 - MCP: expose tools via an MCP server instead of inline functions
5. Phase 5 - Security: guardrails (input/output checks) + gateway with model fallback
6. Phase 6 - Observability: LangSmith tracing + an automated agent eval set
7. Phase 7 - Ship it: Docker Compose locally, then AWS (ECS + RDS)
