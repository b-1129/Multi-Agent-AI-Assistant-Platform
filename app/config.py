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
    google_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
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

    # AI Security / Gateway (phase 5)
    # Fallback model: used when the primary Gemini model fails or is
    # unavailable. Set to an Groq model (Groq is cheap and fast
    # as a fallback) if you have an GROQ_API_KEY; leave blank to disable.
    groq_api_key: str = ""
    fallback_model_name: str = "llama-3.1-8b-instant"
    primary_model_max_retries: int = 2

    # Rate limiting: max requests per IP per minute.
    rate_limit_per_minute: int = 30

    # Guardrails: set to False to disable individual checks (useful for testing).
    guardrails_enabled: bool = True
    pii_detection_enabled: bool = True
    injection_detection_enabled: bool = True
    blocked_topics_enabled: bool = True
    output_safety_enabled: bool = True

    # LangSmith tracing (phase 6)
    # Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable.
    # LangChain/LangGraph auto-detect these env vars -- no code change needed
    # to activate tracing, the settings below add project metadata so runs
    # are grouped correctly in the LangSmith UI.
    langchain_api_key: str = ""
    langchain_project: str = "agent-platform"
    langchain_tracing_v2: bool = False

    # Agent evaluation (phase 6)
    eval_dataset_path: str = "./evals/dataset.json"
    eval_results_dir: str = "./evals/results"

settings = Settings()