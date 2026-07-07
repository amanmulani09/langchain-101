import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    """A user's login session (login → logout). Owns many chat threads."""

    session_id: str
    user_id: str
    thread_ids: list[str] = field(default_factory=list)


@dataclass
class Thread:
    """A single chat/conversation. Its thread_id is the agent's memory key."""

    thread_id: str
    session_id: str
    user_id: str


class SessionStore:
    """Two-level registry: sessions (login span) → threads (individual chats).

    Short-term / in-memory, matching the InMemorySaver checkpointer: entries
    live for the process lifetime only and are not shared across workers.
    Swap the dicts for Redis/a DB when you need persistence or multi-worker.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._threads: dict[str, Thread] = {}

    # --- sessions (login / logout) ---------------------------------------
    def create_session(self, user_id: str) -> Session:
        session_id = uuid.uuid4().hex
        session = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """Log out: drop the session and all its threads. False if unknown."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        for thread_id in session.thread_ids:
            self._threads.pop(thread_id, None)
        return True

    # --- threads (individual chats) --------------------------------------
    def create_thread(self, session_id: str) -> Thread | None:
        """Open a new chat inside a session. None if the session is unknown."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        thread_id = uuid.uuid4().hex
        thread = Thread(
            thread_id=thread_id,
            session_id=session_id,
            user_id=session.user_id,
        )
        self._threads[thread_id] = thread
        session.thread_ids.append(thread_id)
        return thread

    def get_thread(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)
