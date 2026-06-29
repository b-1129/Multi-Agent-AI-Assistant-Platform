"""
FastAPI app: the REST interface infront of the agent.

phase 5 puts a gateway (guardrails + fallbacks) in front of this; phase 6
adds langsmith tracing around it. For now it's deliberately plain so the FastAPI +
Pydantic + Langchain wiring is easy to see on its own.
"""

import logging
import shutil
import tempfile
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app.config import settings
from app.graph import multi_agent_graph
from app.ingestion import ingest_file, list_documents
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCallRecord, DocumentInfo, DocumentListResponse, DocumentUploadResponse, ConversationMessage, SessionHistoryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(settings.app_name)

app = FastAPI(
    title="Agent platform -- phase 3",
    description="A supervisor + specialist multi-agent system built with LangGraph.",
    version="0.3.0",
)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_RECURSION_LIMIT = 25

@app.get("/health", response_model=HealthResponse)
def health()-> HealthResponse:
    return HealthResponse(
        status="ok", app_name=settings.app_name, environment=settings.environment
    )

@app.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile)-> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}'"
        )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion Failed")
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return DocumentUploadResponse(**result)


@app.get("/documents", response_model=DocumentListResponse)
def get_documents()-> DocumentListResponse:
    docs = [DocumentInfo(**doc) for doc in list_documents()]
    return DocumentListResponse(documents=docs, total_documents=len(docs))


@app.post("/chat", response_model=ChatResponse)
def chat(request:ChatRequest)-> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_RECURSION_LIMIT,
    }

    try:
        result = multi_agent_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

    except GraphRecursionError:
        logger.warning("Session %s hit the recursion limit", session_id)
        return ChatResponse(
            response=(
                "I went back and forth between specialists longer than expected "
                "without reaching an answer. Could you rephrase or split your "
                "question into smaller parts?"
            ),
            session_id=session_id,
            tool_calls=[],
            model=settings.model_name,
        )

    except Exception as exc:
        logger.exception("Agent Execution Failed")
        raise HTTPException(status_code=502, detail=f"Agent Error: {exc}") from exc
    
    messages = result["messages"][-1]

    tool_calls = [ToolCallRecord(**record) for record in result.get("tool_calls", [])]
    
    final_message = messages[-1]

    return ChatResponse(
        response=final_message.content,
        session_id=session_id,
        tool_calls=tool_calls,
        model=settings.model_name,
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
