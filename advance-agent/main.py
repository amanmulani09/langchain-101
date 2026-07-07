import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver

from agent import build_agent
from config import get_settings
from schemas import (
    ChatRequest,
    ChatResponse,
    Context,
    SessionInitRequest,
    SessionInitResponse,
    ThreadInitRequest,
    ThreadInitResponse,
)
from session import SessionStore, Thread

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advance-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once at startup with short-term (in-memory) memory.

    InMemorySaver keeps conversation state in RAM, keyed by thread_id, for the
    lifetime of this process only. Memory is lost on restart and is NOT shared
    across workers — swap for a durable checkpointer (SQLite/Postgres) if you
    need persistence or multi-worker deployments.
    """
    app.state.agent = build_agent(InMemorySaver())
    app.state.sessions = SessionStore()
    logger.info("agent ready (short-term in-memory checkpointer)")
    yield


app = FastAPI(title="Weather Agent", version="1.0.0", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_thread(http_request: Request, thread_id: str) -> Thread:
    """Look up the thread for a thread_id, or 404 if it was never opened."""
    thread = http_request.app.state.sessions.get_thread(thread_id)
    if thread is None:
        raise HTTPException(
            status_code=404,
            detail="unknown thread_id — call POST /chat/init first",
        )
    return thread


def _config(thread_id: str) -> dict:
    """LangGraph config. thread_id is the memory key for this conversation."""
    return {"configurable": {"thread_id": thread_id}}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/session", response_model=SessionInitResponse)
async def session_init(request: SessionInitRequest, http_request: Request):
    """Log in: start a session for a user. One session spans many chats."""
    session = http_request.app.state.sessions.create_session(request.user_id)
    return SessionInitResponse(session_id=session.session_id, user_id=session.user_id)


@app.delete("/session/{session_id}", status_code=204)
async def session_end(session_id: str, http_request: Request):
    """Log out: end the session and drop all its chats."""
    if not http_request.app.state.sessions.end_session(session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")


@app.post("/chat/init", response_model=ThreadInitResponse)
async def chat_init(request: ThreadInitRequest, http_request: Request):
    """Open a new chat inside a session and return an opaque thread_id.

    The frontend calls this per conversation, stores the thread_id, and sends
    it on every subsequent /chat call.
    """
    thread = http_request.app.state.sessions.create_thread(request.session_id)
    if thread is None:
        raise HTTPException(
            status_code=404,
            detail="unknown session_id — call POST /session first",
        )
    return ThreadInitResponse(thread_id=thread.thread_id, session_id=thread.session_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    thread = _resolve_thread(http_request, request.thread_id)
    agent = http_request.app.state.agent
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": request.message}]},
            config=_config(thread.thread_id),
            context=Context(user_id=thread.user_id),
        )
    except Exception as exc:  # upstream model/tool failure
        logger.exception("chat failed")
        raise HTTPException(status_code=502, detail="agent invocation failed") from exc

    return ChatResponse(
        thread_id=thread.thread_id,
        response=result["structured_response"],
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    thread = _resolve_thread(http_request, request.thread_id)
    agent = http_request.app.state.agent

    async def generate():
        async for chunk, _meta in agent.astream(
            {"messages": [{"role": "user", "content": request.message}]},
            config=_config(thread.thread_id),
            context=Context(user_id=thread.user_id),
            stream_mode="messages",
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")
