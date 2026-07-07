from pydantic import BaseModel, Field


class Context(BaseModel):
    """Runtime context passed to the agent/tools on every call.

    This identifies *who* is asking (used by tools like locate_user).
    It is NOT the memory key — see thread_id in the request below.
    """

    user_id: str


class ResponseFormat(BaseModel):
    """Structured output the agent is forced to return."""

    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float


class InitRequest(BaseModel):
    """Start a new conversation for a user."""

    user_id: str = Field(..., min_length=1, examples=["user_1"])


class InitResponse(BaseModel):
    """Handed back to the frontend; thread_id is used on every later call."""

    thread_id: str
    user_id: str


class ChatRequest(BaseModel):
    """A turn in an existing conversation.

    The frontend sends only the opaque thread_id (from /chat/init) plus the
    message. The server resolves the owning user_id from the thread_id.
    """

    thread_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, examples=["what is the weather like"])


class ChatResponse(BaseModel):
    thread_id: str
    response: ResponseFormat
