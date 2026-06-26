"""
Tests for the document upload/list/search pipeline.

These use a fake, deterministic Embeddings class instead of the real
FastEmbed model - that keeps the tests free, fast, and runnable with no
internet access. This is a same dependency-injection seam you would use
to swap in a different embeddings provided later (Voyage AI, OpenAI, etc.)
"""

import hashlib        # generate unique, repeatable math fingerprints from text strings
import random         # andom number generation tool,
import tempfile       # module to handle temporary files and folders

import pytest         # engine used to run, organize, and evaluate these automated tests
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings: same text always maps to the same vector."""

    def _vec(self, text: str):

        # Converts the text to bytes, runs it through an MD5 hashing algorithm, turns that hash into a giant integer number, and constrains it to fit a standard 32-bit computer system limit. This turns any phrase (e.g., "hello") into a permanent, unique starting number (seed).
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2*32)

        # Creates an isolated random number generator (rng) locked to that specific seed number. This ensures the "randomness" is perfectly reproducible every single time that specific text is evaluated.
        rng = random.Random(seed)

        # generate a list of 16 random decimals between 0.0 and 1.0. This acts as a fake 16-dimensional vector embedding.
        return [rng.random() for _ in range(16)]
    
    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]
    
    def embed_query(self, text):
        return self._vec(text)
    

# Marks the function below as a Pytest fixture. Fixtures set up state or tools needed before tests run. It asks Pytest for monkeypatch (to rewrite code on the fly) and tmp_path (a unique temporary directory).
@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.vectorstore as vectorstore_module
    from app.config import settings

    #  Uses monkeypatching to hijack the real _build_embeddings function inside the vector store module, replacing it with a fake function (lambda) that returns your FakeEmbeddings object instead.
    monkeypatch.setattr(vectorstore_module, "_build_embeddings", lambda: FakeEmbeddings())

    # Resets the app's vector database variable to None, forcing the application to completely rebuild a clean, empty database instance for this specific test.
    monkeypatch.setattr(vectorstore_module, "_vectorstore", None)

    # Diverts the vector database's storage directory away from your hard drive and routes it into the disposable Pytest temporary directory.
    settings.chroma_persist_dir = str(tmp_path/"data")

    # Diverts the application's file upload data storage directory into the disposable temporary folder as well.
    settings.data_dir = str(tmp_path/"data")

    from app.main import app
    return TestClient(app)

def test_upload_and_list_document(client):
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"FastAPI is a Python web framework.", "text/plain")},
    )
    assert upload_resp.status_code == 200
    body = upload_resp.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_added"] >= 1

    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    documents = list_resp.json()["documents"]
    assert any(doc["filename"] == "notes.txt" for doc in documents)

def test_upload_rejects_unsupported_file_type(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_document(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"   ", "text/plain")},
    )
    assert resp.status_code == 400