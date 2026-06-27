"""
Conversation persistence for the multi-agent graph.

LangGraph's checkpointer is what gives the graph memory across requests: it
stores the full state (message history, routing state) keyed by a thread
id, so a second request with the same `session_id` resumes where the last
one left off -- the API only needs the new message each turn, not the
whole conversation.

DATABASE_URL set -> Postgres-backed, durable across restarts (the real
deployment story). Not set -> in-memory, good enough for trying things out
locally without standing up Postgres first.
"""

import logging

from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings

logger = logging.getLogger(settings.app_name)

_postgres_saver = None

def get_checkpointer():
    if not settings.database_url:
        logger.info(
            "DATABASE_URL not set -- using in-memory checkpointer "
            "(no persistence across restarts)."
        )

    global _postgres_saver
    if _postgres_saver is None:
        from langgraph.checkpoint.postgres import PostgresSaver

        saver_cm = PostgresSaver.from_conn_string(settings.database_url)
        _postgres_saver = saver_cm.__enter__()
        _postgres_saver.setup()  # creates checkpoint tables if they don't exist yet
        logger.info("Using postgres-backed checkpointer.")

    return _postgres_saver
