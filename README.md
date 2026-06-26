# Agent platform — phase 1 + phase 2 (RAG)

A LangChain agent (calculator, web search, and now document search) behind a
FastAPI REST API, validated end to end with Pydantic, and containerized with
Docker. Phase 2 adds Retrieval-Augmented Generation: upload your own
documents, and the agent decides for itself when to search them.

## What's in this phase

- `app/config.py` — Pydantic settings, loaded from environment variables
- `app/schemas.py` — Pydantic request/response models for the API
- `app/vectorstore.py` — local embeddings (FastEmbed) + the Chroma vector store instance
- `app/ingestion.py` — load → chunk → embed → store pipeline, plus a small JSON registry of uploaded files
- `app/tools.py` — calculator, web search, and `search_documents` (the RAG tool)
- `app/agent.py` — the LangChain agent (`create_agent`, built on LangGraph under the hood)
- `app/main.py` — FastAPI app: `/health`, `/documents/upload`, `/documents`, `/chat`
- `tests/test_main.py` — health + chat smoke tests
- `tests/test_documents.py` — upload/list tests with fake embeddings and isolated tmp dirs (no network needed)
- `Dockerfile` / `docker-compose.yml` — containerized local run with persistent volumes for the vector store and registry

## Run it locally (no Docker)

```bash
cd agent-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your real ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

The first time you upload a document, FastEmbed downloads its embedding
model from Hugging Face (a few seconds to a minute, one-time, needs
internet). After that, embedding runs fully offline on CPU.

Visit http://localhost:8000/docs for the interactive Swagger UI, or use curl:

```bash
curl http://localhost:8000/health

# upload a document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/your_notes.txt"

# see what's been ingested
curl http://localhost:8000/documents

# ask the agent something about it
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does my uploaded document say about X?"}'

# the agent still has its other tools too
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 23 * 17, and who won the last World Cup?"}'
```

Check the `tool_calls` field in the `/chat` response to see which tool the
agent actually picked — `search_documents`, `web_search`, or `calculator` —
based on the question, not because you told it which one to use.

## Run it with Docker

```bash
cp .env.example .env   # add your real key
docker compose up --build
```

The vector store and document registry persist in named Docker volumes
(`chroma_data`, `document_data`), so re-uploading after a restart isn't
necessary.

## Run the tests

```bash
pytest
```

`test_health` always runs. `test_documents.py` runs with a fake, deterministic
embeddings class and isolated tmp directories — no internet or API key
needed, and it won't touch your real `chroma_data`/`data` folders.
`test_chat_calculator` is skipped unless `ANTHROPIC_API_KEY` is set, since it
calls the real model.

## Supported document types

`.txt`, `.md`, `.pdf` — anything else is rejected with a 400. Empty or
unextractable documents (e.g. a scanned/image-only PDF with no text layer)
are also rejected with a clear error rather than silently storing nothing.

## Why it's built this way (for revision later)

- **Agentic RAG, not a fixed pipeline**: `search_documents` is just another
  tool. The agent decides per-question whether to use it, fall back to
  `web_search`, or answer directly — that's the real difference between
  "RAG" as a static retrieve-then-answer chain and RAG as one capability of
  an agent.
- **Local embeddings (FastEmbed)**: no second paid API key, and it makes the
  embedding step a clean swap-in point later (Voyage AI, OpenAI, Cohere) —
  the `Embeddings` interface stays identical either way. `_build_embeddings()`
  in `vectorstore.py` is the seam: tests monkeypatch that one function to
  inject a fake, instead of needing real network access.
- **Chunking with overlap**: `RecursiveCharacterTextSplitter` with a
  `chunk_overlap` means a fact that lands near a chunk boundary still gets
  surfaced via the neighboring chunk — worth being able to explain why
  overlap exists, not just that it does.
- **A small JSON registry, on purpose**: tracking which files were ingested
  doesn't need a database yet — Chroma stores the vectors, a flat JSON file
  tracks filenames and chunk counts. Phase 3 replaces this with Postgres once
  there's real multi-user/session state to justify it.
- **Pydantic everywhere**: settings, request/response schemas, and tool input
  schemas are all Pydantic models — the same discipline carried over from
  phase 1.

## Project roadmap (where this fits)

1. DONE — Phase 1: FastAPI + Pydantic + single LangChain agent + Docker
2. DONE — Phase 2 (this one): RAG — document upload, chunking, local embeddings, Chroma vector DB, agentic retrieval
3. Phase 3 — Multi-agent with LangGraph: supervisor + research/RAG/action agents, Postgres for chat history
4. Phase 4 — MCP: expose tools via an MCP server instead of inline functions
5. Phase 5 — Security: guardrails (input/output checks) + gateway with model fallback
6. Phase 6 — Observability: LangSmith tracing + an automated agent eval set
7. Phase 7 — Ship it: Docker Compose locally, then AWS (ECS + RDS)
