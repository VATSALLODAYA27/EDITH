"""User accounts: registration, login, sessions, and ISOLATION between users. No LLM calls.

Run:  python -m tests.auth_test
"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")  # fresh DB: no users yet
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")
import sys  # noqa: E402

from userdata import as_user, use_temp_data  # noqa: E402

use_temp_data()  # never touch the real mailbox/calendar/workspace
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import approvals  # noqa: E402
import auth  # noqa: E402
from agents import mail  # noqa: E402
from agents.document import create_document  # noqa: E402
from memory import conn  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
client = TestClient(api.app)

# 1. Registration rules; the FIRST account becomes "local" (it owns the data from before accounts existed)
assert client.post("/auth/register", json={"username": "alice", "password": "short"}).status_code == 422
assert client.post("/auth/register", json={"username": "bad name!", "password": "longenough1"}).status_code == 400
r = client.post("/auth/register", json={"username": "alice", "password": "alice-pass-1"})
assert r.status_code == 201 and r.json()["user_id"] == "local", r.text
r = client.post("/auth/register", json={"username": "bob", "password": "bob-pass-12"})
assert r.status_code == 201 and r.json()["user_id"] != "local"
assert client.post("/auth/register", json={"username": "ALICE", "password": "whatever12"}).status_code == 400
print("1. registration OK")

# 2. Login: same message for unknown user and wrong password; the DB never stores raw passwords or tokens
unknown = client.post("/auth/login", json={"username": "nobody", "password": "whatever12"})
wrong = client.post("/auth/login", json={"username": "alice", "password": "wrong-pass-1"})
assert unknown.status_code == wrong.status_code == 401 and unknown.json() == wrong.json()
auth._failures.clear()
token_a = client.post("/auth/login", json={"username": "alice", "password": "alice-pass-1"}).json()["token"]
token_b = client.post("/auth/login", json={"username": "bob", "password": "bob-pass-12"}).json()["token"]
A, B = {"Authorization": f"Bearer {token_a}"}, {"Authorization": f"Bearer {token_b}"}
stored = str(conn.execute("SELECT * FROM sessions").fetchall()) + str(conn.execute("SELECT * FROM users").fetchall())
assert token_a not in stored and "alice-pass-1" not in stored
assert client.get("/auth/me", headers=A).json()["username"] == "alice"
print("2. login + hashed storage OK")

# 3. Everything except /health and /auth needs a valid token
for method, path in [("get", "/threads"), ("get", "/profile"), ("get", "/agents"), ("get", "/files/sales.xlsx")]:
    assert getattr(client, method)(path).status_code == 401, path
    assert getattr(client, method)(path, headers={"Authorization": "Bearer made-up"}).status_code == 401, path
assert client.post("/tasks", json={"message": "hi"}).status_code == 401
assert client.get("/health").status_code == 200
print("3. endpoints require login OK")

# 4. ISOLATION between alice and bob
client.put("/profile", headers=A, json={"name": "Alice", "sign_off": "Best, Alice"})
assert client.get("/profile", headers=B).json() == {}
client.put("/profile", headers=B, json={"name": "Bob"})
assert client.get("/profile", headers=A).json()["name"] == "Alice"

real_invoke = api.invoke_traced
api.invoke_traced = lambda state, thread_id: {"final_answer": "ok", "history": [], "agent_results": {}}
thread_a = client.post("/tasks", headers=A, json={"message": "alice's secret plan"}).json()["thread_id"]
api.invoke_traced = real_invoke
assert [t["id"] for t in client.get("/threads", headers=A).json()] == [thread_a]
assert client.get("/threads", headers=B).json() == []
assert client.get(f"/threads/{thread_a}", headers=B).status_code == 404            # can't read it
assert client.post("/tasks", headers=B, json={"message": "hi", "thread_id": thread_a}).status_code == 404  # continue
assert client.post(f"/tasks/{thread_a}/resume", headers=B, json={"decisions": {}}).status_code == 404       # approve

with as_user("local"):  # alice
    create_document.invoke({"filename": "alice_note.docx", "content": "private"})
    mail.draft_email.invoke({"to": "x@example.com", "subject": "alice only", "body": "hi"})
assert client.get("/files/alice_note.docx/preview", headers=A).status_code == 200
assert client.get("/files/alice_note.docx/preview", headers=B).status_code == 404
assert client.get("/files/alice_note.docx", headers=B).status_code == 404
bob_id = client.get("/auth/me", headers=B).json()["user_id"]
with as_user(bob_id):
    assert approvals.pending_ids() == [], "bob must not see alice's pending email"
    assert "sales.xlsx" in [f for f in os.listdir(__import__("userdata").workspace_dir())]  # own seeded workspace
print("4. isolation: profile, threads, approvals, files, mailbox OK")

# 5. Logout ends the session
assert client.post("/auth/logout", headers=B).status_code == 204
assert client.get("/threads", headers=B).status_code == 401
print("5. logout OK")

# 6. Brute force: 5 wrong passwords lock the account for a while, even for the right password
for _ in range(auth.MAX_FAILURES):
    client.post("/auth/login", json={"username": "bob", "password": "wrong-pass-1"})
r = client.post("/auth/login", json={"username": "bob", "password": "bob-pass-12"})
assert r.status_code == 401 and "Too many" in r.json()["detail"], r.text
auth._failures.clear()
print("6. lockout OK")

# 7. Expired sessions are refused
conn.execute("UPDATE sessions SET expires = '2000-01-01T00:00:00'")
conn.commit()
assert client.get("/threads", headers=A).status_code == 401
print("7. expiry OK")

print("\nAuth OK")
