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

settings = Settings()