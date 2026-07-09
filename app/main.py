"""
FastAPI app: the REST interface in front of the AI gateway.

Phase 5 inserts app.gateway between the HTTP endpoint and the multi-agent
graph. Phase 6 adds LangSmith tracing via configure_tracing() called at
the top of this module, before any LangChain imports, so auto-tracing is
active for every graph invocation, LLM call, and tool call.

Every /chat request now flows:
  HTTP request
    -> rate limiter
    -> input guardrails (injection, PII, blocked topics)
    -> multi-agent graph (primary model, with fallback to secondary)
       [every LLM call + tool call traced in LangSmith when enabled]
    -> output guardrails
    -> HTTP response
"""

# configure_tracing() must run before any LangChain/LangGraph imports so
# the LANGCHAIN_* env vars are in place when those packages are first loaded.
from app.tracing import configure_tracing  # noqa: E402 (intentional early import)
configure_tracing()

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.gateway import process_request
from app.graph import multi_agent_graph
from app.ingestion import ingest_file, list_documents
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationMessage,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    SessionHistoryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(settings.app_name)

app = FastAPI(
    title="Agent Platform -- Phase 6",
    description=(
        "Multi-agent platform with AI security, MCP tools, and LangSmith "
        "observability + automated agent evaluation."
    ),
    version="0.6.0",
)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        guardrails_enabled=settings.guardrails_enabled,
        fallback_model_configured=bool(settings.groq_api_key),
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return DocumentUploadResponse(**result)


@app.get("/documents", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    docs = [DocumentInfo(**doc) for doc in list_documents()]
    return DocumentListResponse(documents=docs, total_documents=len(docs))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    # Get client IP for rate limiting. X-Forwarded-For is checked first so
    # this works correctly behind a proxy/load balancer (e.g. AWS ALB).
    forwarded_for = http_request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        http_request.client.host if http_request.client else "unknown"
    )

    return await process_request(
        message=request.message,
        session_id=session_id,
        client_ip=client_ip,
    )


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    config = {"configurable": {"thread_id": session_id}}
    snapshot = multi_agent_graph.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No session found with id '{session_id}'")

    messages = [
        ConversationMessage(
            role="ai" if isinstance(message, AIMessage) else "human",
            content=message.content,
            agent_name=getattr(message, "name", None),
        )
        for message in snapshot.values.get("messages", [])
    ]

    return SessionHistoryResponse(session_id=session_id, messages=messages)
