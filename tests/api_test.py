"""Phase 6 tests: the HTTP layer (validation, auth, errors, streaming) via FastAPI's TestClient (no server needed).

Only tests 5 and 6 call real LLMs.   Run:  python -m tests.api_test
"""
import os
import tempfile

# Use a throwaway memory database: tests must never write into the user's real history/profile.
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")  # ...and never into the real trace log
import json  # noqa: E402
import sys  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from userdata import use_temp_data, workspace_dir  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
use_temp_data()  # all per-user files (workspace, mailbox, calendar) go to a temp folder, never the real ones
WORKSPACE = workspace_dir()
client = TestClient(api.app)

# Log in (the first account in this fresh test DB becomes "local"); accounts themselves are tested in auth_test.py
client.post("/auth/register", json={"username": "tester", "password": "test-pass-1"})
client.headers["Authorization"] = "Bearer " + client.post(
    "/auth/login", json={"username": "tester", "password": "test-pass-1"}).json()["token"]

# 1. Health + agent list
assert client.get("/health").json() == {"status": "ok"}
agents = client.get("/agents").json()
assert len(agents) == 8 and {"name", "description"} <= set(agents[0]), agents
print("1. /health and /agents OK")

# 2. Validation: empty or oversized messages are rejected BEFORE any LLM is called
assert client.post("/tasks", json={"message": ""}).status_code == 422
assert client.post("/tasks", json={"message": "x" * 4001}).status_code == 422
assert client.post("/tasks", json={}).status_code == 422
print("2. request validation (422) OK")

# 3. Auth: no/invalid token -> 401 (full account + isolation tests live in auth_test.py)
assert client.get("/agents", headers={"Authorization": ""}).status_code == 401
assert client.get("/agents", headers={"Authorization": "Bearer wrong"}).status_code == 401
assert client.get("/health", headers={"Authorization": ""}).status_code == 200  # public for uptime checks
print("3. login required (401) OK")

# 3b. Memory endpoints (no LLM): profile round-trip, unknown thread -> 404, bad thread id -> 422
saved = client.get("/profile").json()
r = client.put("/profile", json={"name": "Test User", "sign_off": "Cheers,\nTest"})
assert r.json()["name"] == "Test User" and client.get("/profile").json()["sign_off"] == "Cheers,\nTest"
client.put("/profile", json={k: saved.get(k, "") for k in ("name", "email", "role", "sign_off", "preferences")})
assert client.get("/threads/does-not-exist").status_code == 404
assert client.post("/tasks", json={"message": "hi", "thread_id": "../../etc"}).status_code == 422
assert isinstance(client.get("/threads").json(), list)
print("3b. profile / threads endpoints OK")

# 4. Failures (simulated, no LLM calls): provider down -> 503 "try again"; our own bug -> 500, NOT "providers down"
real_invoke, real_stream = api.graph.invoke, api.graph.stream


class RateLimitError(Exception):
    pass


RateLimitError.__module__ = "groq"  # looks like the Groq SDK's error


def raising(exc):
    def f(*a, **k):
        raise exc
    return f


api.graph.invoke = api.graph.stream = raising(RateLimitError("429 on every model"))
r = client.post("/tasks", json={"message": "hi"})
assert r.status_code == 503 and "unavailable" in r.json()["detail"], r.text
assert "event: error" in client.post("/tasks/stream", json={"message": "hi"}).text

# the real Phase 7 bug: a print() of "↑" on a cp1252 console crashed the task
real_invoke_traced = api.invoke_traced  # /tasks runs through the traced path since Phase 10
api.invoke_traced = raising(UnicodeEncodeError("charmap", "↑", 0, 1, "character maps to <undefined>"))
r = client.post("/tasks", json={"message": "hi"})
assert r.status_code == 500 and "Internal error" in r.json()["detail"], r.text
api.invoke_traced = real_invoke_traced
api.graph.invoke, api.graph.stream = real_invoke, real_stream
print("4. provider failure -> 503, internal bug -> 500 OK")

# 4b. Output files: preview JSON per type, download, path safety, and the "files" event (no LLM calls)
from agents.document import create_document  # noqa: E402
from agents.ppt import create_presentation  # noqa: E402

for f in ("t_prev.pptx", "t_prev.docx", "t_new.docx"):
    (WORKSPACE / f).unlink(missing_ok=True)
create_presentation.invoke({"filename": "t_prev.pptx", "title": "Deck", "subtitle": "Sub",
                            "slides": [{"title": "Findings", "bullets": ["82% satisfied"], "notes": "say hi"}]})
create_document.invoke({"filename": "t_prev.docx", "content": "# Report\n## Part\n- point\nplain text"})

deck = client.get("/files/t_prev.pptx/preview").json()
assert deck["type"] == "pptx" and [s["title"] for s in deck["slides"]] == ["Deck", "Findings"], deck
assert deck["slides"][1]["bullets"] == ["82% satisfied"] and deck["slides"][1]["notes"] == "say hi"
doc = client.get("/files/t_prev.docx/preview").json()
assert [b["kind"] for b in doc["blocks"]] == ["h1", "h2", "bullet", "p"], doc
sheet = client.get("/files/sales.xlsx/preview").json()
assert sheet["sheets"][0]["rows"][0] == ["Month", "Region", "Product", "Units", "Revenue"]
assert "82 percent" in client.get("/files/survey_report.pdf/preview").json()["text"]

r = client.get("/files/t_prev.pptx")
assert r.status_code == 200 and r.content[:2] == b"PK" and "attachment" in r.headers["content-disposition"]
# Path traversal is refused at two layers: an encoded "/" can't match the {name} route at all (404),
# and a backslash reaches our code, where _safe_path rejects it (400). Either way: never served.
for attack in ["..%2Fapi.py", "..%2F..%2Fapi.py", "C:%2FWindows%2Fwin.ini", "..%5Capi.py"]:
    assert client.get(f"/files/{attack}").status_code in (400, 404), attack
assert client.get("/files/..%5Capi.py").status_code == 400
assert client.get("/files/nope.pptx/preview").status_code == 404


(WORKSPACE / "t_bad.pptx").write_bytes(b"not a real pptx")  # corrupt/locked file -> readable 422, not a 500
assert client.get("/files/t_bad.pptx/preview").status_code == 422
(WORKSPACE / "t_bad.pptx").unlink()


def fake_run(*a, **k):  # a "run" that creates a file, to check the files event
    create_document.invoke({"filename": "t_new.docx", "content": "hello"})
    (WORKSPACE / "~$t_new.docx").write_bytes(b"lock")  # what Word/PowerPoint create while a file is open
    yield {"finalize": {"final_answer": "done"}}


api.graph.stream = fake_run
text = client.post("/tasks/stream", json={"message": "hi"}).text
api.graph.stream = real_stream
assert 'event: files\ndata: {"files": ["t_new.docx"]}' in text, text  # ONLY the new file: no old ones, no lock file
for f in ("t_prev.pptx", "t_prev.docx", "t_new.docx", "~$t_new.docx"):
    (WORKSPACE / f).unlink()
print("4b. file preview / download / safety / files event OK")

# 5. Real task, wait-for-answer style
r = client.post("/tasks", json={"message": "How many vacation days do I get per year?"})
assert r.status_code == 200, r.text
body = r.json()
print("   ", body["agents_used"], "|", body["final_answer"][:100])
assert body["agents_used"] == ["rag_agent"] and "24" in body["final_answer"]
print("5. POST /tasks OK")

# 6. Real task, streaming: parse the SSE events in order
events = []
with client.stream("POST", "/tasks/stream", json={"message": "What's the capital of Australia?"}) as r:
    assert r.headers["content-type"].startswith("text/event-stream")
    for block in r.iter_text():
        for chunk in block.strip().split("\n\n"):
            if chunk.startswith("event:"):
                name, data = chunk.split("\n", 1)
                events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
print("    events:", [(n, d.get("agents") or d.get("agent") or "…") for n, d in events])
names = [n for n, _ in events]
assert names[:2] == ["thread", "plan"] and "agent" in names and names[-1] == "final", names
assert "canberra" in events[-1][1]["final_answer"].lower()
print("6. POST /tasks/stream OK")

print("\nAPI OK")
