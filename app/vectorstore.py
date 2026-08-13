"""
Embeddings + the Chroma vector store instance.

Embeddings use FastEmbed -- a local, ONNX-based embedding model that runs on
CPU with no API key and no per-token cost. It downloads its model from
Hugging Face the first time it runs (needs internet once), then runs fully
offline after that. This keeps the project self-contained: the agent's LLM
calls Google(Gemini), but retrieval doesn't depend on a second paid API.

Swapping this for a hosted embedding API (e.g. Voyage AI, OpenAI) later is a
one-line change here, since LangChain's 'Embeddings' interface is the same
either way -- a good interview answer for "how would you scale this."

Chunking and loading live in 'app.ingestion', not here -- this module's only
job is producing the shared embedding function and the shared store instance
that both ingestion and the 'search_documents' tool read from.
"""

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from app.config import settings

_embeddings = None
_vectorstore = None

def _build_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=settings.embedding_model_name)

def get_embeddings()-> FastEmbedEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = _build_embeddings()
    return _embeddings

def get_vectorstore()-> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
        )
    return _vectorstore