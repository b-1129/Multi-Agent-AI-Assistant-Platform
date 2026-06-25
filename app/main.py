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

from fastapi import FastAPI, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import agent
from app.config import settings
from app.ingestion import ingest_file, list_documents
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCallRecord, DocumentInfo, DocumentListResponse, DocumentUploadResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(settings.app_name)

app = FastAPI(
    title="Agent platform -- phase 2",
    description="A LangChain agent with calculator, web search, and RAG over uploaded documents.",
    version="0.2.0",
)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


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
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.messages)]}
        )
    except Exception as exc:
        logger.exception("Agent Execution Failed")
        raise HTTPException(status_code=502, detail=f"Agent Error: {exc}") from exc
    
    messages = result["messages"]

    tool_calls = [ToolCallRecord(tool_name=call["name"], tool_input=str(call["args"])) 
                  for message in messages 
                  if isinstance(message, AIMessage)
                  for call in (message.tool_calls or [])]
    
    final_message = messages[-1]

    return ChatResponse(
        response=final_message.content,
        tool_calls=tool_calls,
        model=settings.model_name,
    )
