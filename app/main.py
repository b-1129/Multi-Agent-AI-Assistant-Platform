import logging

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import agent
from app.config import settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse, ToolCallRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(settings.app_name)

app = FastAPI(
    title="Agent platform -- phase 1",
    description="A single Langchain Agent (Calculator + Web Search Tools) behind a REST API",
    version="0.1.0",
)

@app.get("/health", response_model=HealthResponse)
def health()-> HealthResponse:
    return HealthResponse(
        status="ok", app_name=settings.app_name, environment=settings.environment
    )

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
