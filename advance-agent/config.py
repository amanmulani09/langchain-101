import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings, sourced from environment variables.

    Kept dependency-light (plain os.getenv) so it works with the existing
    dotenv setup. Swap for pydantic-settings if you outgrow this.
    """

    # Model
    model_name: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    model_provider: str = os.getenv("MODEL_PROVIDER", "groq")
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))

    # HTTP
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # Upstream weather API timeout (seconds)
    weather_timeout: float = float(os.getenv("WEATHER_TIMEOUT", "10"))

    # LangSmith observability. LangChain/LangGraph auto-trace when these env
    # vars are set — we only read them here to log status at startup.
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "weather-agent")


@lru_cache
def get_settings() -> Settings:
    return Settings()
