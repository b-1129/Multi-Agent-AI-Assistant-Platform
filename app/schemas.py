"""
Pydantic schemas define the contract for every request and response in the
API. This is the pattern you'll extend in every later phase: tool inputs,
agent state, and gateway payloads will all be Pydantic models too.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message to the agent.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session id to continue an existing conversation. Omit "
        "on the first message of a new conversation -- the response will "
        "include a session_id to reuse on follow-up messages.",
    )


class ToolCallRecord(BaseModel):
    agent_name: str
    tool_name: str
    tool_input: str


class ChatResponse(BaseModel):
    response: str
    session_id: str = Field(
        description="Echo this back on the next request to continue the same conversation."
    )
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    model: str
    # Phase 5: gateway fields -- tell callers whether (and why) a request
    # was blocked so they can show a helpful message rather than a raw error.
    blocked: bool = False
    block_reason: Optional[str] = Field(
        default=None,
        description="Why the request was blocked: 'rate_limited', "
        "'input_guardrail', or 'output_guardrail'. None if not blocked.",
    )
    pii_entities: List[str] = Field(
        default_factory=list,
        description="PII entity types found in the input (e.g. EMAIL_ADDRESS, US_SSN). "
        "Only populated when block_reason is 'input_guardrail' and the cause is PII.",
    )


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    guardrails_enabled: bool = True
    fallback_model_configured: bool = False


class ConversationMessage(BaseModel):
    role: str  # "human" or "ai"
    content: str
    agent_name: Optional[str] = None  # which specialist produced this, if any


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[ConversationMessage]


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    ingested_at: float


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int

