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

    # RAG: Embeddings(local FastEmbed, No API Key) + Vector Store (Chroma)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_k: int = 4

    # Document registry (filename -> chunk count, ingestion timestamp).
    # Still a flat JSON file, not a database -- there's no multi-user
    # state here yet that would justify moving it to Postgres.
    data_dir: str="./data"

    # Multi-agent conversation persistence (phase 3). Empty -> in-memory
    # checkpointer, good for local trial use. Set -> Postgres-backed,
    # durable across restarts.
    database_url: str = ""

    # MCP server (phase 4): host/port are what the server itself binds to;
    # mcp_server_url is what the agent process (the client) connects to --
    # they differ in Docker, where the server binds 0.0.0.0 but the client
    # reaches it by service name.
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8001
    mcp_server_url: str = "http://localhost:8001/mcp"

settings = Settings()