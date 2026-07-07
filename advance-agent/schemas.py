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


class SessionInitRequest(BaseModel):
    """Log in: start a session for a user. Spans many chats until logout."""

    user_id: str = Field(..., min_length=1, examples=["user_1"])


class SessionInitResponse(BaseModel):
    session_id: str
    user_id: str


class ThreadInitRequest(BaseModel):
    """Open a new chat inside an existing session."""

    session_id: str = Field(..., min_length=1)


class ThreadInitResponse(BaseModel):
    thread_id: str
    session_id: str


class ChatRequest(BaseModel):
    """A turn in an existing chat.

    The frontend sends only the opaque thread_id (from /chat/init) plus the
    message. The server resolves the owning session/user from the thread_id.
    """

    thread_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, examples=["what is the weather like"])


class ChatResponse(BaseModel):
    thread_id: str
    response: ResponseFormat
