"""
App configuration, loaded from environment variables (or a .env file).
Using pydantic-settings: this is the same pattern you will reuse for every
later phase (gateway config, eval config, etc.)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "agent_platform"
    environment: str = "local"

    # LLM provider config
    google_api_key: str = "GOOGLE_API_KEY"
    model_name: str = "MODEL_NAME"
    model_temperature: float = 0.2

    # Agent behaviour
    agent_max_iteration: int = 6
    agent_verbose: bool = True

    # RAG: Embeddings(local FastEmbed, No API Key) + Vector Store (Chroma)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_k: int = 4

    # Where the document registry (filename -> chunk count) is stored.
    # A real database takes over this job in phase 3.
    data_dir: str="./data"

settings = Settings()