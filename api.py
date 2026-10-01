"""FastAPI backend: a thin HTTP layer over the LangGraph orchestrator.

    React / curl --HTTP--> FastAPI (validate, login, owner checks, errors) --> graph --> agents

Every endpoint except /health and /auth/* needs "Authorization: Bearer <token>" (from POST /auth/login) and runs
AS that user: their threads, profile, workspace, mailbox, calendar and approvals only.

Run:   .venv/Scripts/python -m uvicorn api:app --port 8000        then open http://127.0.0.1:8000/docs
Test:  .venv/Scripts/python api_test.py
"""
import json
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from agents.document import _safe_path, free_path
from agents.rag import RAG_TYPES, index_file
from auth import AuthError, login, logout, register, user_for_token
from memory import get_profile, list_threads, save_profile, thread_owner, touch_thread
from orchestrator import (AGENTS, graph, invoke_traced, new_request, thread_config, traced_stream,
                          waiting_for_approval)
from previews import changed_since, is_user_file, preview, snapshot
from userdata import as_user

# Agents print() LLM text for logging. On Windows the server's console is cp1252, so one "↑" in an answer
# raised UnicodeEncodeError and killed the whole task (seen in Phase 7). Logging must never crash a request.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

app = FastAPI(title="Multi-Agent Task Orchestrator", version="0.11")

# CORS: browsers block a page on one origin (the React dev server) from calling another (this API) unless allowed.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type", "Authorization"],
                   expose_headers=["Content-Disposition"])


# --- AUTH: "Authorization: Bearer <token>" -> the user this request runs as ---
def current_user(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    found = user_for_token(token) if token else None
    if not found:
        raise HTTPException(status_code=401, detail="Please log in.", headers={"WWW-Authenticate": "Bearer"})
    return found[0]


def _own_thread(thread_id: str, user: str) -> None:
    """Another user's conversation answers 404, not 403: nobody can even confirm that an id exists."""
    if thread_owner(thread_id) != user:
        raise HTTPException(status_code=404, detail="No conversation with that id.")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=200)


# --- REQUEST / RESPONSE MODELS: validated automatically, and they document the API at /docs ---
class TaskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="The user's request in plain language.")
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{1,64}$",
                                  description="Continue this conversation. Omit to start a new one.")


class Step(BaseModel):
    turn: int
    agent: str
    inputs: list[str]


class TaskResponse(BaseModel):
    final_answer: str
    agents_used: list[str]
    history: list[Step]
    agent_results: dict[str, str]
    files: list[str]  # workspace files this task created or changed
    thread_id: str    # pass it back to ask a follow-up in the same conversation
    pending_approval: list[dict] = []  # non-empty = the run is PAUSED until POST /tasks/{thread_id}/resume


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    # emails only: the human's final wording. The recipient is NOT editable (it's what the agent proposed/you saw).
    edits: dict[Literal["subject", "body"], str] = Field(default_factory=dict)

    @field_validator("edits")
    @classmethod
    def _limit(cls, edits):
        if any(len(v) > 20_000 for v in edits.values()):
            raise ValueError("edit too long")
        return edits


class ResumeRequest(BaseModel):
    # one decision per pending action id: "approve" | "reject" | {"decision": ..., "edits": {...}}.
    # Anything missing counts as "reject" (safe default).
    decisions: dict[str, Literal["approve", "reject"] | Decision]


class AgentInfo(BaseModel):
    name: str
    description: str


class Profile(BaseModel):  # long-term memory about the user; "" clears a field
    name: str = Field(default="", max_length=100)
    email: str = Field(default="", max_length=200)
    role: str = Field(default="", max_length=100)
    sign_off: str = Field(default="", max_length=200, description='e.g. "Best regards,\nVatsal"')
    preferences: str = Field(default="", max_length=1000, description='e.g. "Keep answers short."')


def _start(req: TaskRequest, user: str) -> tuple[dict, dict, str]:
    """(graph input, config, thread_id) for one request; records the thread in the task history."""
    thread_id = req.thread_id or str(uuid.uuid4())
    if req.thread_id and thread_owner(thread_id) not in (None, user):  # someone else's conversation
        raise HTTPException(status_code=404, detail="No conversation with that id.")
    if req.thread_id and waiting_for_approval(thread_id):  # can't sneak past a pending approval
        raise HTTPException(status_code=409, detail="This mission is waiting for your approval. Approve or reject first.")
    touch_thread(thread_id, req.message)
    return new_request(req.message), thread_config(thread_id), thread_id


PROVIDER_MODULES = {"groq", "google", "langchain_google_genai", "httpx", "tavily", "llm"}  # llm: AllModelsFailed  # where LLM/API errors come from


def _task_error(e: Exception) -> HTTPException:
    """503 = an external provider failed (retrying later may help); 500 = our bug (retrying won't help).
    Lumping everything into 503 once hid a real bug (a UnicodeEncodeError) behind "providers overloaded"."""
    print(f"[api] task failed: {type(e).__name__}: {str(e)[:200]}")  # details stay in the server log
    if type(e).__module__.split(".")[0] in PROVIDER_MODULES:
        return HTTPException(status_code=503, detail="The AI providers are unavailable or overloaded. Try again shortly.")
    return HTTPException(status_code=500, detail="Internal error while running the task. It has been logged.")


# --- ENDPOINTS ---
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def auth_register(creds: Credentials):
    try:
        return {"user_id": register(creds.username, creds.password), "username": creds.username}
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/auth/login")
def auth_login(creds: Credentials):
    try:
        token, _ = login(creds.username, creds.password)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    return {"token": token, "username": creds.username}


@app.post("/auth/logout", status_code=204)
def auth_logout(authorization: str = Header(default=""), user: str = Depends(current_user)):
    logout(authorization.removeprefix("Bearer ").strip())
    return Response(status_code=204)


@app.get("/auth/me")
def auth_me(authorization: str = Header(default="")):
    found = user_for_token(authorization.removeprefix("Bearer ").strip())
    if not found:
        raise HTTPException(status_code=401, detail="Please log in.")
    return {"user_id": found[0], "username": found[1]}


@app.get("/agents", response_model=list[AgentInfo], dependencies=[Depends(current_user)])
def list_agents():
    return [AgentInfo(name=n, description=d) for n, (_, d) in AGENTS.items()]


# Plain `def` (not async): the graph makes BLOCKING LLM calls, so FastAPI runs this in a worker thread
# and the server stays responsive. Blocking inside `async def` would freeze every other request.
@app.post("/tasks", response_model=TaskResponse)
def run_task(req: TaskRequest, user: str = Depends(current_user)):
    with as_user(user):  # everything below (agents, tools, mailbox, profile...) works on THIS user's data
        before = snapshot()
        state, config, thread_id = _start(req, user)
        try:
            result = invoke_traced(state, thread_id)
        except Exception as e:
            raise _task_error(e) from e
        history = result.get("history", [])
        return TaskResponse(final_answer=result.get("final_answer", ""),
                            agents_used=list(dict.fromkeys(h["agent"] for h in history)),
                            history=history, agent_results=result.get("agent_results", {}),
                            files=changed_since(before), thread_id=thread_id,
                            pending_approval=waiting_for_approval(thread_id))


def _sse(event: str, data: dict) -> str:
    """Server-Sent Events wire format: 'event: <name>\\ndata: <json>\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_as(user: str, fn, *args):
    with as_user(user):
        return fn(*args)


def _stream(graph_input, config: dict, thread_id: str, user: str) -> StreamingResponse:
    """SSE for a new request OR a resume: thread -> plan/agent... -> approval (paused) | final -> files.

    The generator runs LATER, step by step, possibly on different threads - so the user can't simply be "set"
    around it. Each piece of work runs explicitly as the user (and traced_stream carries it into the graph)."""

    def events():  # a normal generator: Starlette iterates it in a worker thread, like the endpoints
        before = _run_as(user, snapshot)  # compare after the run: exact list of files created/changed
        yield _sse("thread", {"thread_id": thread_id})  # so the UI can send follow-ups to the same conversation
        try:
            # stream_mode="updates": yields {node_name: what_that_node_returned} after every node finishes
            for update in traced_stream(graph_input, config, user):  # graph.stream + run id + user
                for node, out in update.items():
                    if node == "__interrupt__":  # the approval node paused the run: ask the human
                        yield _sse("approval", {"actions": [a for i in out for a in i.value["actions"]]})
                    elif node == "approval":
                        yield _sse("approved", {"outcomes": out.get("approvals", [])})
                    elif node == "orchestrator":
                        yield _sse("plan", {"turn": out.get("turns"), "agents": [p["agent"] for p in out.get("plan", [])]})
                    elif node == "finalize":
                        yield _sse("final", {"final_answer": out["final_answer"]})
                    elif node == "compact":  # housekeeping after the answer (older turns -> summary): nothing to show
                        continue
                    else:  # an agent finished
                        yield _sse("agent", {"agent": node, "result": out["agent_results"][node]})
        except Exception as e:  # headers are already sent, so errors travel as an event, not a status code
            yield _sse("error", {"message": _task_error(e).detail})
        files = _run_as(user, changed_since, before)  # also after an error: a partial run may have made files
        if files:
            yield _sse("files", {"files": files})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/tasks/stream")
def stream_task(req: TaskRequest, user: str = Depends(current_user)):
    """Live progress as Server-Sent Events. May end with an `approval` event: the run is paused."""
    with as_user(user):
        state, config, thread_id = _start(req, user)
    return _stream(state, config, thread_id, user)


@app.post("/tasks/{thread_id}/resume")
def resume_task(thread_id: str, req: ResumeRequest, user: str = Depends(current_user)):
    """The human's decisions for a paused run. Only the OWNER can decide - never another user, never an agent."""
    _own_thread(thread_id, user)
    waiting = {a["id"] for a in waiting_for_approval(thread_id)}
    if not waiting:
        raise HTTPException(status_code=409, detail="This mission isn't waiting for approval.")
    unknown = set(req.decisions) - waiting
    if unknown:  # decisions must refer to exactly the actions shown to the user
        raise HTTPException(status_code=422, detail=f"Unknown action ids: {sorted(unknown)}")
    decisions = {}
    for action_id, d in req.decisions.items():
        if isinstance(d, Decision):
            if d.edits and not action_id.startswith("email:"):
                raise HTTPException(status_code=422, detail=f"Only emails can be edited ({action_id}).")
            decisions[action_id] = d.model_dump()
        else:
            decisions[action_id] = d
    return _stream(Command(resume=decisions), thread_config(thread_id), thread_id, user)


# --- OUTPUT FILES: preview (JSON the UI draws) + download. Same security check as the agents' tools. ---
def _workspace_file(name: str):
    try:
        path = _safe_path(name)  # only plain names inside workspace/ ("../api.py" is refused)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No file '{name}' in the workspace.")
    return path


@app.get("/files/{name}/preview")
def file_preview(name: str, user: str = Depends(current_user)):
    with as_user(user):  # only this user's workspace
        return _preview(name)


def _preview(name: str):
    _workspace_file(name)
    try:
        return preview(name)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except Exception as e:  # corrupt/locked file. An unhandled 500 skips the CORS middleware, and the browser
        # then only reports "Failed to fetch" - so turn it into a normal, readable HTTP error.
        raise HTTPException(status_code=422, detail=f"Can't read this file ({type(e).__name__}).") from e


# --- UPLOADS: the user provides a document -> workspace (every agent can use it) + their own RAG index ---
UPLOAD_TYPES = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"}
MAX_UPLOAD = 20 * 1024 * 1024  # 20 MB
MAGIC = {".pdf": b"%PDF", ".docx": b"PK", ".xlsx": b"PK", ".pptx": b"PK"}  # Office files are zip archives


@app.put("/files/{name}", status_code=201)
async def upload_file(name: str, request: Request, user: str = Depends(current_user)):
    """Raw file bytes as the body (no multipart, so no extra dependency). Never overwrites: a taken name gets _2."""
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_TYPES or not is_user_file(name):
        raise HTTPException(status_code=400, detail=f"Allowed types: {', '.join(sorted(UPLOAD_TYPES))}.")
    data = bytearray()
    async for part in request.stream():  # count while reading: a lying or missing Content-Length can't exhaust memory
        data += part
        if len(data) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="File is larger than 20 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if suffix in MAGIC and not data.startswith(MAGIC[suffix]):
        raise HTTPException(status_code=415, detail=f"This doesn't look like a real {suffix} file.")
    if suffix in (".txt", ".md"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=415, detail="Text files must be UTF-8.") from e
    return await run_in_threadpool(_store_upload, user, name, bytes(data))  # file + embedding calls block


def _store_upload(user: str, name: str, data: bytes) -> dict:
    with as_user(user):  # this user's workspace and RAG index
        try:
            path = free_path(_safe_path(name))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        path.write_bytes(data)
        result = {"name": path.name, "size": len(data), "rag_chunks": None, "rag_error": ""}
        if path.suffix.lower() in RAG_TYPES:
            try:
                result["rag_chunks"] = index_file(path.name)
            except Exception as e:  # noqa: BLE001 - the file is saved either way; only the index failed
                print(f"[upload] indexing {path.name} failed: {type(e).__name__}: {e}")
                result["rag_error"] = "Saved, but couldn't add it to the knowledge base right now (try again later)."
        return result


@app.get("/files/{name}")
def download_file(name: str, user: str = Depends(current_user)):
    with as_user(user):
        return FileResponse(_workspace_file(name), filename=name)  # sets Content-Disposition: attachment


# --- MEMORY: task history (threads) + long-term profile ---
@app.get("/threads")
def threads(user: str = Depends(current_user)):
    with as_user(user):
        return list_threads()  # only this user's


@app.get("/threads/{thread_id}")
def thread_messages(thread_id: str, user: str = Depends(current_user)):
    """The saved conversation of one thread, read back from the checkpointer."""
    _own_thread(thread_id, user)
    snap = graph.get_state(thread_config(thread_id))
    if not snap.values:
        raise HTTPException(status_code=404, detail="No conversation with that id.")
    return {"thread_id": thread_id,
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.text}
                         for m in snap.values.get("messages", [])],
            "summary": snap.values.get("summary", ""),  # older turns, compacted
            "pending_approval": waiting_for_approval(thread_id)}


@app.get("/profile")
def read_profile(user: str = Depends(current_user)):
    with as_user(user):
        return get_profile()


@app.put("/profile")
def update_profile(profile: Profile, user: str = Depends(current_user)):
    with as_user(user):
        return save_profile(profile.model_dump())
