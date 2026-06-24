"""
Pydantic schemas define the contract for every request and response in the
API. This is the pattern you will extend in every later phase: tool inputs,
agent state, and gateway payloads will all be pydantic models too.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    messages: str= Field(..., min_length=1, max_length=4000, description="The user's message to the agent.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id to group a conversation." \
        "Not persisted yet in phase 1 - wired up when the database lands in a later phase."
    )

class ToolCallRecord(BaseModel):
    tool_name:str
    tool_input:str

class ChatResponse(BaseModel):
    response:str
    tool_calls:List[ToolCallRecord] = Field(default_factory=list)
    model:str

class HealthResponse(BaseModel):
    status:str
    app_name: str
    environment: str