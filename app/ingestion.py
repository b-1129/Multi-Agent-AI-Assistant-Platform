"""
Ingestion pipeline: file -> text -> chunks -> embeddings -> vector store.

Kept deliberately small and synchronous for phase 2. A later phase could move
this to a background task/queue, but understanding the linear version first
is the point.
"""

import json
import time
import uuid
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.vectorstore import get_vectorstore

def _registry_path():
    return Path(settings.data_dir) / "document_registry.json"

def _read_text_from_file(path:Path)-> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")

def _load_registry() -> dict:
    registry_path = _registry_path()
    if registry_path.exists():
        return json.loads(registry_path.read_text())
    return {}

def _save_registry(registry:dict)-> None:
    registry_path = _registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))

def chunk_text(text:str, filename:str)-> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata = {"source": filename, "chunk_index":i})
        for i, chunk in enumerate(chunks)
    ]

def ingest_file(path:Path, filename:str)-> dict:
    """Read, Chunk, Embedd and Store one file. Returns a summary dict."""

    text = _read_text_from_file(path)
    if not text.strip():
        raise ValueError(f"No extractable text found in '{filename}'.")
    
    documents = chunk_text(text, filename)
    ids = [str(uuid.uuid4()) for _ in documents]

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=documents, ids=ids)

    registry = _load_registry()
    registry[filename] = {
        "chunk_count": len(documents),
        "ingested_at": time.time(),
    }
    _save_registry()

    return {"filename": filename, "chunks_added": len(documents)}

def list_documents()->List[dict]:
    registry = _load_registry()
    return [
        {"filename":name, "chunk_count":info["chunk_count"], "ingested_at":info["ingested_at"]}
        for name, info in registry.items()
    ]