# Weather Agent — System Design

A conversational AI agent served over HTTP. A user chats in natural language;
the agent figures out their city, fetches live weather, remembers the
conversation, and replies in a structured JSON shape. Built on FastAPI +
LangChain agents + LangGraph, backed by a Groq-hosted LLM.

This document is written from a **system-design-interview perspective**: what
the requirements are, how the pieces fit, *why* each decision was made, and what
changes when it needs to scale.

---

## 1. Requirements

**Functional**
- A user sends a message; the agent answers about the weather.
- The agent resolves the user's city from their identity (no need to type it).
- The agent calls a live weather API and grounds its answer in that data.
- Multi-turn: follow-ups ("is this usual?") understand prior context.
- Output is a fixed JSON schema, safe for a frontend to render.
- Streaming responses for a typing-indicator UX.

**Non-functional**
- Concurrency: many users at once → async, non-blocking I/O.
- Isolation: one user's conversation must never leak into another's.
- Predictable output: no free-form prose the UI can't parse.
- Extensible: adding a tool or swapping the model shouldn't ripple.
- Operable: health checks, structured errors, config via env.

**Explicitly out of scope (stated up front in an interview)**
- Authentication (identity is trusted from the request for now).
- Long-term persistence (memory is short-term / in-process by design).
- Horizontal scale-out (single-worker today; see §7 for the path).

---

## 2. High-level architecture

```
                    ┌──────────────────────────────────────────────┐
   Frontend  ──────▶│                  FastAPI app                  │
   (browser)        │  /session  /chat/init  /chat  /chat/stream    │
                    │                                                │
                    │   ┌───────────────┐      ┌──────────────────┐ │
                    │   │ SessionStore  │      │   LangChain agent │ │
                    │   │ session→thread│      │  (LangGraph loop) │ │
                    │   │   →user (RAM) │      │                   │ │
                    │   └───────────────┘      │  ┌─────────────┐  │ │
                    │                          │  │ InMemorySaver│ │ │
                    │                          │  │ (checkpointer)│ │ │
                    │                          │  └─────────────┘  │ │
                    │                          └────────┬──────────┘ │
                    └───────────────────────────────────┼───────────┘
                                                         │
                            ┌────────────────────────────┼───────────────┐
                            ▼                            ▼                 ▼
                     Groq LLM API              locate_user tool     get_weather tool
                  (llama-3.3-70b)              (user_id → city)     (wttr.in HTTP)
```

**Layered responsibilities**

| Layer | File | Responsibility |
|-------|------|----------------|
| HTTP / API | `main.py` | Routing, validation, session resolution, error mapping, streaming |
| Identity / state | `session.py` | Session → thread → user registry |
| Agent | `agent.py` | Model + tools + agent assembly (a factory) |
| Contracts | `schemas.py` | Pydantic request/response + agent I/O schemas |
| Config | `config.py` | Env-driven settings |

The separation is deliberate: the **API layer never talks to the model
directly**, and the **agent layer knows nothing about HTTP**. You can unit-test
the agent without a server, and swap the transport without touching the agent.

---

## 3. The identity model (the core design decision)

Three IDs with three different lifetimes. Conflating them is the classic
mistake; keeping them separate is what makes the system correct.

```
user_id            who they are            permanent      (from auth)
   └─ session_id   one login → logout      minutes–hours  (POST /session)
        ├─ thread_id  chat A               one conversation (POST /chat/init)
        ├─ thread_id  chat B
        └─ thread_id  chat C
```

| ID | Scope | Created by | Consumed as |
|----|-------|-----------|-------------|
| `user_id` | The person | Auth (trusted from body today) | **Tool context** — who's asking |
| `session_id` | Login span | `POST /session` | Groups a user's chats; logout cascade |
| `thread_id` | One chat | `POST /chat/init` | **Agent memory key** (LangGraph config) |

**Why two layers of ID (session vs thread)?**
A user's login is one thing; a single conversation is another. One login can
own many parallel chats, each needing its *own isolated memory*. So `session_id`
is the coarse grouping (and the logout unit), while `thread_id` is the fine
grouping the agent actually keys memory on. Logout ends the session and
cascades to drop every thread under it.

**Why does the frontend only hold opaque IDs?**
The client never constructs a memory key or asserts a user on each turn. It
holds a random `thread_id`; the server maps it back to `session → user`. This
(a) prevents the client from forging another user's memory namespace and
(b) lets the key format evolve server-side without a frontend change.

---

## 4. Request lifecycle

### 4a. The API flow (frontend ↔ server)

```
1. Login        POST /session      { user_id }            → { session_id }
2. Open a chat  POST /chat/init    { session_id }         → { thread_id }
3. Each turn    POST /chat         { thread_id, message } → { response }
4. Logout       DELETE /session/{session_id}              → 204
```

At step 3 the server resolves `thread_id → Thread`, then feeds the agent two
different things from it:
- `thread.thread_id` → LangGraph `config` → **which conversation's memory** to load
- `thread.user_id`   → `Context`          → **which city** the tools resolve

Unknown `thread_id` → `404` (client re-inits). Model/tool failure → `502`.
Malformed body → `422` (Pydantic, before the handler runs).

### 4b. The agent loop (inside a single `/chat` call)

An "agent" is not one LLM call — it's a **reason–act loop** (ReAct):

```
  load prior messages for thread_id  ──(checkpointer read)
              │
              ▼
  ┌───────────────────────────────────────────┐
  │ LLM sees: system prompt + history +        │◀──┐
  │ new message + tool definitions             │   │
  └───────────────────┬───────────────────────┘   │
                      ▼                            │
             wants a tool? ── yes ── run tool ─────┘  (feed result back, loop)
                      │
                      no
                      ▼
  emit final answer as ResponseFormat (validated JSON)
                      │
                      ▼
  save messages under thread_id  ──(checkpointer write)
```

Concrete trace for *"what is the weather like"*:
1. LLM has no city → calls `locate_user`, which reads `context.user_id` → `"pune"`.
2. LLM calls `get_weather("pune")` → live JSON from wttr.in.
3. LLM stops calling tools and emits `ResponseFormat`
   (`summary`, `temperature_celsius`, `temperature_fahrenheit`, `humidity`).

---

## 5. Component deep-dives

### Memory / checkpointer
- **What:** LangGraph's `InMemorySaver`. After each step it snapshots the full
  message list, keyed by `thread_id`, in a RAM dict.
- **Why it enables multi-turn:** the next turn on the same `thread_id` reloads
  that history, so "is this usual?" has referents.
- **Short-term by design:** state lives for the process lifetime only. Chosen
  deliberately (see §1 scope) to keep the demo simple and stateless-ish.
- **Tradeoff:** lost on restart, and **not shared across workers** — so today
  the service runs single-worker. §7 covers the durable path.

### Tools
- `locate_user(runtime)` — reads `runtime.context.user_id`, returns a city.
  Notably takes **no LLM-supplied arguments**; it's grounded in server-side
  identity, so the model can't spoof another user's location.
- `get_weather(city)` — HTTP GET to wttr.in with a timeout + error handling; on
  failure returns an `{"error": ...}` dict rather than throwing, so the agent
  can recover gracefully.
- Extensibility: adding a capability = writing one `@tool` function and adding
  it to the `tools=[...]` list. No routing code, no prompt surgery — the LLM
  decides when to call it from the description.

### Structured output
- `response_format=ResponseFormat` forces the model's final answer through a
  Pydantic schema. The framework validates (and makes the model retry on
  mismatch), so the frontend always gets parseable JSON — no regex-scraping
  prose. This is a reliability decision, not a cosmetic one.

### Model
- `init_chat_model("llama-3.3-70b-versatile", provider="groq")`. Provider-
  agnostic call site: swapping to another model/provider is a config change,
  not a code change. `temperature=0.3` keeps answers fairly deterministic.

### Config
- All knobs (model, temperature, CORS, timeout) come from env via a cached
  `Settings` object — 12-factor style, no redeploy to retune.

---

## 5b. Guardrails & evaluation

Two different quality mechanisms, easily confused in interviews:

**Guardrails — runtime, in the request path (`guardrails.py`).** Cheap,
deterministic, rule-based (no extra LLM call, ~zero latency):
- *Input* (`check_input`, before the model): message length cap + prompt-
  injection pattern match. Trips → `400` (client's fault).
- *Output* (`check_output`, after the model): plausible temperature range,
  humidity ∈ [0,100], and °C/°F internal consistency. Trips → `502` (our fault
  — the model produced something incoherent, so we don't ship it).
- Wired via a FastAPI exception handler that maps `stage → status`.
- Upgrade path: add an LLM-as-judge guardrail for semantic checks
  (toxicity, off-topic, PII) where rules can't reach.

**Evaluation — offline, out of the request path (`evaluation.py`).** A fixed
dataset run through the real agent, scored on *properties* (you can't string-
match a non-deterministic LLM):
- `structured_ok` — returned a valid `ResponseFormat`.
- `output_valid` — passes the output guardrails.
- `city_grounded` — the correct city (from `user_id`) appears in the tool trace,
  i.e. `locate_user` → `get_weather` wired up correctly.
- Emits per-metric pass rates. Run it when you change the model, prompt, or
  tools to catch regressions before they ship.

The relationship: **guardrails protect each live request; evaluation measures
aggregate quality across releases.** Guardrail functions are reused inside the
eval (`check_output`), so the two stay consistent.

## 6. Failure modes & handling

| Failure | Where | Response |
|---------|-------|----------|
| Malformed request body | Pydantic | `422` before handler runs |
| Oversized / injection input | `check_input` | `400` before the model runs |
| Incoherent model output | `check_output` | `502`, answer withheld |
| Unknown / expired `thread_id` | `_resolve_thread` | `404` → client re-inits |
| Unknown `session_id` on chat init | `create_thread` | `404` → client re-logs-in |
| Weather API down / slow | `get_weather` | tool returns `{error}`, agent copes; request still `200` |
| LLM / provider error | `/chat` try-except | `502`, logged with stack trace |
| Server restart | in-memory stores | all sessions/threads gone → clients get `404`, re-init |

The design **fails safe**: a flaky upstream weather call degrades one answer
rather than 500-ing the request, and lost state surfaces as a recoverable 404
rather than silent cross-user leakage.

---

## 7. Scaling & production hardening (the "what would you change" question)

Ordered by what an interviewer usually probes:

1. **Multi-worker / horizontal scale.** The two in-memory stores
   (`InMemorySaver` + `SessionStore`) are process-local, so today it's
   single-worker. To scale out: move both behind a shared backend —
   a durable LangGraph checkpointer (`langgraph-checkpoint-postgres`/`sqlite`)
   for memory, and Redis/Postgres for the session registry. The factory
   (`build_agent(checkpointer)`) already injects the checkpointer, so this is a
   wiring change, not a rewrite.

2. **Persistence.** Same swap gives durability across restarts — conversations
   survive deploys.

3. **Authentication.** `user_id` is trusted from the `/session` body. In
   production, derive it from a verified JWT/OAuth session at `/session`;
   everything downstream inherits the real identity automatically. Treat
   `thread_id` as a capability token and check ownership on each call.

4. **Session lifecycle.** Add TTL / idle expiry and eviction so the stores
   don't grow unbounded (an in-memory dict is an unbounded leak otherwise).

5. **Resilience & cost.** Timeouts + retries/circuit-breaker around the LLM and
   weather calls; a semaphore or queue to cap concurrent model calls; response
   caching for identical city lookups.

6. **Observability.** Structured logs (already started) plus **LangSmith**:
   LangChain/LangGraph auto-trace every agent run, tool call, and LLM call when
   `LANGSMITH_TRACING`/`LANGSMITH_API_KEY` are set — giving per-step traces,
   token usage, and latency in a UI with zero code change. LangSmith also hosts
   the eval datasets + experiments (`evaluation_langsmith.py`), so runtime
   traces and offline scores live in one place.

7. **Safety.** Rate limiting per user, input size caps, and prompt-injection
   hardening on tool outputs.

---

## 8. Key tradeoffs, summarized

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Memory backend | In-memory (short-term) | SQLite/Postgres | Simplicity now; injectable later |
| ID model | 3-tier user/session/thread | single conversation id | Correct isolation + logout semantics |
| Output | Forced Pydantic schema | free-form text | Frontend reliability |
| Tool location | server-side `context` | LLM-supplied args | Can't spoof identity |
| Agent build | once at startup (factory) | per request | Avoid rebuild cost |
| Concurrency | fully async (`ainvoke`/`astream`) | sync | Non-blocking under load |
| Config | env-driven settings | hardcoded | Retune without redeploy |

---

## 9. Talking-points cheat sheet (30-second version)

> "It's a FastAPI service wrapping a LangChain ReAct agent on a Groq LLM.
> Identity is modeled in three tiers — a permanent `user_id`, a `session_id`
> per login, and a `thread_id` per chat, which is what the LangGraph
> checkpointer keys conversation memory on. The frontend logs in to get a
> session, opens a thread per chat, and posts turns with just the opaque
> thread_id; the server maps that back to the user. The agent loops between the
> LLM and two tools — one that resolves the user's city from server-side
> context, one that hits a live weather API — and is forced to answer in a
> validated JSON schema. Memory is in-process/short-term by design, but the
> checkpointer is injected via a factory, so swapping to a durable, shared
> backend for multi-worker scale-out is a wiring change, not a rewrite."
