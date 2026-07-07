import logging

import requests

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.base import BaseCheckpointSaver

from config import get_settings
from schemas import Context, ResponseFormat

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "you're a helpful weather assistant agent, who cracks the joke and is humorous"
)

# Static demo lookup table. In production this would be a DB / profile service.
_USER_CITIES = {
    "user_1": "pune",
    "user_2": "hyd",
    "user_3": "mumbai",
}


@tool("get_weather", description="returns current weather from users query")
def get_weather(city: str) -> dict:
    settings = get_settings()
    try:
        r = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=settings.weather_timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        logger.warning("weather lookup failed for %s: %s", city, exc)
        return {"error": f"could not fetch weather for {city}"}


@tool("locate_user", description="look up the users city based on context")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    return _USER_CITIES.get(runtime.context.user_id, "Unknown")


def build_agent(checkpointer: BaseCheckpointSaver):
    """Construct the weather agent bound to a persistence backend.

    The checkpointer is injected so the app owns its lifecycle (e.g. an
    async SQLite connection opened in the FastAPI lifespan).
    """
    settings = get_settings()

    model = init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider,
        temperature=settings.model_temperature,
    )

    return create_agent(
        name="weather agent",
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_weather, locate_user],
        context_schema=Context,
        response_format=ResponseFormat,
        checkpointer=checkpointer,
    )
