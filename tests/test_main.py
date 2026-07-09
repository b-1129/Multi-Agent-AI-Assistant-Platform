"""
Minimal smoke tests for the core chat path. Document upload/list/search and
the multi-agent graph's routing logic have their own dedicated test files
(test_documents.py, test_graph.py) with isolated fakes -- this file is just
an end-to-end sanity check with the real Anthropic model.

/health needs no API key. /chat does -- this now exercises the *real*
supervisor + specialist multi-agent graph end to end (real routing decision,
real action_agent, real calculator tool call), so it's skipped automatically
if GOOGLE_API_KEY isn't set.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="requires a real Google API key"
)
def test_chat_calculator_end_to_end():
    """Full real run: supervisor must route to action_agent, which must call
    the calculator tool and return the right answer."""
    response = client.post("/chat", json={"message": "what is 12 times 4? Use your calculator tool."})
    assert response.status_code == 200
    body = response.json()
    assert "48" in body["response"]
    assert body["session_id"]
    assert any(call["agent_name"] == "action_agent" for call in body["tool_calls"])
