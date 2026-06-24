"""
Minimal smoke tests. /health needs no API key; /chat does (it calls the real
agent), so it's skipped automatically if GOOGLE_API_KEY isn't set - that
way 'pytest' still works for anyone just checking the API wiring.
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
    not os.getenv("GOOGLE_API_KEY"), reason="Requires a real Google API Key"
)
def test_chat_calculator():
    response = client.post("/chat", json={"message": "what is 12 times 4?"})
    assert response.status_code == 200
    body = response.json()
    assert "48" in body["response"]