"""
LangSmith tracing setup (phase 6).

How tracing works in this project:

1. AUTO-TRACING (zero code change):
   LangChain and LangGraph automatically emit traces to LangSmith when
   LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set. Every graph
   invocation, every LLM call, every tool call is captured with input,
   output, latency, and token usage -- no code in graph.py or agents.py
   needs to change.

2. CUSTOM SPANS with @traceable:
   The gateway and eval runner wrap their top-level logic with @traceable
   so they appear as named root spans in the LangSmith UI, making it easy
   to see "this chat request" as one trace rather than disconnected LLM
   calls.

3. METADATA TAGGING:
   Every trace gets tagged with session_id, model_used, and whether a
   guardrail fired -- so you can filter in LangSmith by "show me all
   requests that used the fallback model" or "show me all blocked requests."

4. OFFLINE MODE:
   If LANGCHAIN_TRACING_V2 is false (the default), @traceable still calls
   the wrapped function normally -- it just doesn't upload anything. Tests
   and local dev work without any API key.
"""

import logging
import os

from langsmith import traceable as _ls_traceable

from app.config import settings

logger = logging.getLogger(settings.app_name)


def configure_tracing() -> bool:
    """
    Push LangSmith config into environment variables.

    LangChain reads LANGCHAIN_* env vars at import time, so this must be
    called before any LangChain/LangGraph import if you want tracing enabled.
    In practice, call it at the top of app/main.py before the other imports.

    Returns True if tracing is enabled, False if running in offline mode.
    """
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(
            "LangSmith tracing ENABLED -- project: %s", settings.langchain_project
        )
        return True
    else:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
        logger.info("LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable)")
        return False


def traceable(name: str, run_type: str = "chain", tags: list | None = None):
    """
    Wraps a function/coroutine with a named LangSmith span.

    Usage:
        @traceable("gateway-process-request")
        async def process_request(...):
            ...

    When tracing is disabled this is a no-op decorator -- the function runs
    exactly as normal, with no overhead.
    """
    return _ls_traceable(name=name, run_type=run_type, tags=tags or [])


def get_run_metadata(session_id: str, model_used: str, blocked: bool, block_reason: str | None = None) -> dict:
    """
    Returns a metadata dict to attach to a LangSmith run via the `metadata`
    kwarg on traceable functions or via `langsmith_extra`.

    LangSmith stores this as searchable key-value pairs on the run, so you
    can filter the trace list by session_id, model, or block_reason.
    """
    return {
        "session_id": session_id,
        "model_used": model_used,
        "blocked": blocked,
        "block_reason": block_reason or "none",
        "environment": settings.environment,
    }
