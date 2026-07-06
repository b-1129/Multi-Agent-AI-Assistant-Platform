"""
The AI gateway: the single entry point that every /chat request flows
through before touching the multi-agent graph.

Responsibilities (in request order):
  1. Rate limiting  — reject requests from IPs that are sending too fast
  2. Input guardrails — block injection, PII, blocked topics
  3. Primary model invoke — ainvoke the multi-agent graph
  4. Model fallback — if the primary call fails, rebuild the graph with the
     fallback LLM and retry once
  5. Output guardrails — check the response before returning it

This separation is intentional: the graph (app/graph.py) stays focused on
*routing and orchestration*; the gateway owns *safety and reliability*.
Keeping them apart also means the gateway can be unit-tested independently
by injecting stub graph callables, the same pattern used throughout this
project.

Rate limiting is in-memory here (a defaultdict of timestamps per IP).
For production you would use Redis + a sliding window -- but the shape is
the same and the replacement is a one-function swap.
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.guardrails import check_input, check_output
from app.schemas import ChatResponse, ToolCallRecord
from app.tracing import traceable, get_run_metadata

logger = logging.getLogger(settings.app_name)

SAFE_FALLBACK_MESSAGE = (
    "I'm sorry, I'm unable to help with that request."
)

# In-memory rate limiter (per IP, sliding window)

_request_timestamps: dict[str, list[float]] = defaultdict(list)

def _is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    window = 60.0
    limit = settings.rate_limit_per_minute

    timestamps = _request_timestamps[client_ip]
    # Drop timestamps outside the rolling window
    _request_timestamps[client_ip] = [t for t in timestamps if now - t < window]
    
    if  len(_request_timestamps[client_ip]) >= limit:
        return True
    
    _request_timestamps[client_ip].append(now)
    return False

# Fallback Graph Builder

def _build_fallback_graph():
    """Rebuild the multi-agent graph using the fallback LLM (GROQ).

    This is intentionally lazy: we only import and instantiate the GROQ
    client when we actually need it, so the app starts fine even if no
    GROQ_API_KEY is set (the fallback path simply won't be available).
    """

    if not settings.groq_api_key:
        raise RuntimeError(
            "Primary model failed and no GROQ_API_KEY is configured for fallback."
        )
    
    from langchain_groq import ChatGroq
    from langchain.agents import create_agent
    from app.mcp_client import get_tools_with_fallback
    from app.graph import build_graph
    import app.agents as agents_module
    import app.graph as graph_module

    fallback_llm = ChatGroq(
        model = settings.fallback_model_name,
        api_key = settings.groq_api_key,
        temperature = settings.model_temperature,
    )

    tools = get_tools_with_fallback()

    agents_module.research_agent = create_agent(
        model=fallback_llm,
        tools=[tools["web_search"]],
        system_prompt="You are a research specialist. Use web search to answer questions about current facts.",
    )
    agents_module.rag_agent = create_agent(
        model=fallback_llm,
        tools=[tools["search_documents"]],
        system_prompt="You are a document specialist. Search uploaded documents to answer questions.",
    )
    agents_module.action_agent = create_agent(
        model=fallback_llm,
        tools=[tools["calculator"]],
        system_prompt="You are a calculation specialist. Use the calculator for arithmetic.",
    )

    from typing import Literal
    from pydantic import BaseModel, Field

    fallback_supervisor_llm = ChatGroq(
        model = settings.fallback_model_name,
        api_key = settings.groq_api_key,
        temperature = 0,
    )
    
    class RouteDecision(BaseModel):
        next: Literal["research_agent", "rag_agent", "action_agent", "FINISH"] = Field(
            ..., description="Which specialist should act next, or FINISH if ready to answer."
        )
        reasoning: str = Field(..., description="One short sentence explaining the choice.")

    graph_module.supervisor_chain = fallback_supervisor_llm.with_structured_output(RouteDecision)

    return build_graph()

# Main gateway entry point

@traceable(name="gateway-process-request", run_type="chain", tags=["gateway"])
async def process_request(
    message: str,
    session_id: str,
    client_ip: str,
    graph_invoke: Optional[Callable] = None,
) -> ChatResponse:
    """
    Full gateway pipeline for a single chat request.

    `graph_invoke` is injected in tests so the gateway can be tested
    without running a real multi-agent graph.
    """
    # 1. Rate limiting
    if _is_rate_limited(client_ip):
        logger.warning("Rate limit exceeded for IP %s", client_ip)
        return ChatResponse(
            response="Too many requests. Please wait a moment and try again.",
            session_id=session_id,
            tool_calls=[],
            model=settings.model_name,
            blocked=True,
            block_reason="rate_limited",
        )

    # 2. Input guardrails
    input_check = check_input(message)
    if input_check.blocked:
        logger.info("Input blocked for session %s: %s", session_id, input_check.reason)
        return ChatResponse(
            response=input_check.reason or SAFE_FALLBACK_MESSAGE,
            session_id=session_id,
            tool_calls=[],
            model=settings.model_name,
            blocked=True,
            block_reason="input_guardrail",
            pii_entities=input_check.pii_entities,
        )

    # 3. Primary invoke (with fallback)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,
    }
    model_used = settings.model_name

    if graph_invoke is None:
        from app.graph import multi_agent_graph
        graph_invoke = multi_agent_graph.ainvoke

    result = None
    for attempt in range(settings.primary_model_max_retries + 1):
        try:
            result = await graph_invoke(
                {"messages": [HumanMessage(content=message)]},
                config,
            )
            break
        except Exception as exc:  # noqa: BLE001
            is_last = attempt == settings.primary_model_max_retries
            if is_last:
                logger.error("Primary model failed after %d attempts: %s", attempt + 1, exc)
                # 4. Model fallback
                try:
                    logger.info("Switching to fallback model: %s", settings.fallback_model_name)
                    fallback_graph = _build_fallback_graph()
                    result = await fallback_graph.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config,
                    )
                    model_used = settings.fallback_model_name
                    logger.info("Fallback model succeeded.")
                except Exception as fb_exc:
                    logger.error("Fallback model also failed: %s", fb_exc)
                    return ChatResponse(
                        response="I'm experiencing technical difficulties. Please try again shortly.",
                        session_id=session_id,
                        tool_calls=[],
                        model="none",
                        blocked=False,
                    )
            else:
                logger.warning(
                    "Primary model attempt %d failed (%s), retrying...", attempt + 1, exc
                )
                import asyncio
                await asyncio.sleep(0.5 * (attempt + 1))

    if result is None:
        return ChatResponse(
            response="Unexpected error. Please try again.",
            session_id=session_id,
            tool_calls=[],
            model="none",
            blocked=False,
        )

    # 5. Output guardrails
    final_message = result["messages"][-1]
    response_text = final_message.content

    output_check = check_output(response_text)
    if output_check.blocked:
        logger.warning("Output blocked for session %s: %s", session_id, output_check.reason)
        return ChatResponse(
            response=SAFE_FALLBACK_MESSAGE,
            session_id=session_id,
            tool_calls=[],
            model=model_used,
            blocked=True,
            block_reason="output_guardrail",
        )

    tool_calls = [
        ToolCallRecord(**record) for record in result.get("tool_calls", [])
    ]

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        tool_calls=tool_calls,
        model=model_used,
        blocked=False,
    )