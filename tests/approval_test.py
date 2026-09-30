"""Phase 9 tests: human approval (interrupt -> decide -> resume) for emails, calendar changes and slide deletions.

Section 1 uses no LLM. Sections 2-3 call real LLMs.   Run:  python -m tests.approval_test
"""
import os
import tempfile

# Use a throwaway memory database: tests must never write into the user's real history/profile.
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")  # ...and never into the real trace log
import json  # noqa: E402
import sys  # noqa: E402

from userdata import use_temp_data  # noqa: E402

use_temp_data()  # never touch the real mailbox/calendar/workspace

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import approvals  # noqa: E402
from agents import calendar_agent, mail  # noqa: E402
from orchestrator import resume, run  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def fresh():
    mail.reset_mailbox()
    calendar_agent.reset_calendar()
    approvals.file_queue_path().unlink(missing_ok=True)


# --- 1. The approvals module on its own ---
fresh()
mail.draft_email.invoke({"to": "a@x.com", "subject": "one", "body": "1"})
mail.draft_email.invoke({"to": "b@x.com", "subject": "two", "body": "2"})
assert approvals.pending_ids() == ["email:d1", "email:d2"]
approvals.discard("email:d1")
mail.draft_email.invoke({"to": "c@x.com", "subject": "three", "body": "3"})
assert approvals.pending_ids() == ["email:d2", "email:d3"], "ids must never be reused after a rejection"
approvals.execute("email:d2")
assert mail._load()["sent"][0]["to"] == "b@x.com" and approvals.pending_ids() == ["email:d3"]

calendar_agent.propose_cancel.invoke({"event_id": "e4"})
calendar_agent.propose_cancel.invoke({"event_id": "e2"})
approvals.discard("calendar:c1")
calendar_agent.propose_cancel.invoke({"event_id": "e1"})
assert [a for a in approvals.pending_ids() if a.startswith("calendar")] == ["calendar:c2", "calendar:c3"]
approvals.execute("calendar:c2")
assert "e2" not in [e["id"] for e in calendar_agent._load()["events"]]
# idempotent: repeating the same tool call (LLMs do that) must not create a second pending action -
# even when the retry differs only in free text (seen in testing: same cancel, different "reason")
calendar_agent.propose_cancel.invoke({"event_id": "e1"})
calendar_agent.propose_cancel.invoke({"event_id": "e1", "reason": "user asked"})
mail.draft_email.invoke({"to": "c@x.com", "subject": "three", "body": "3"})
assert approvals.pending_ids() == ["email:d3", "calendar:c3"], approvals.pending_ids()
print("1. approvals module (ids, execute, discard) OK")
fresh()

# --- 1b. Edit before approve, through the REAL graph + API, no LLM: a scripted router drafts an email as a side
#         effect and finishes; the real approval node pauses on it; we approve WITH edits.
import orchestrator  # noqa: E402
from orchestrator import Route  # noqa: E402


class DraftingRouter:
    def __init__(self):
        self.calls = 0

    def invoke(self, *a, **k):
        self.calls += 1
        if self.calls == 1:
            mail.draft_email.invoke({"to": "john.miller@nimbuslabs.com", "subject": "Re: sync", "body": "rough draft"})
        return Route(steps=[], reason="done")


real_router = orchestrator.router
orchestrator.router = DraftingRouter()
r = run("Reply to John")
action = r["__interrupt__"][0].value["actions"][0]
done = resume(r["thread_id"], {action["id"]: {"decision": "approve", "edits": {"body": "Polished by a human."}}})
sent = mail._load()["sent"][0]
assert sent["body"] == "Polished by a human." and sent["to"] == "john.miller@nimbuslabs.com", sent
assert "(edited by you)" in done["final_answer"], done["final_answer"]
try:
    approvals.execute("email:d9", {"to": "attacker@evil.com"})
    raise AssertionError("the recipient must not be editable")
except ValueError:
    pass
print("1b. edit before approve (graph) OK")
fresh()

# API validation of edits (still the scripted router, no LLM)
_api_client = TestClient(api.app)
_api_client.post("/auth/register", json={"username": "editor", "password": "test-pass-1"})
_api_client.headers["Authorization"] = "Bearer " + _api_client.post(
    "/auth/login", json={"username": "editor", "password": "test-pass-1"}).json()["token"]
orchestrator.router = DraftingRouter()
text = _api_client.post("/tasks/stream", json={"message": "Reply to John"}).text
tid = json.loads(text.split("\n")[1].removeprefix("data: "))["thread_id"]
aid = json.loads(text.strip().split("\n")[-1].removeprefix("data: "))["actions"][0]["id"]
bad = {aid: {"decision": "approve", "edits": {"to": "attacker@evil.com"}}}
assert _api_client.post(f"/tasks/{tid}/resume", json={"decisions": bad}).status_code == 422  # recipient locked
ok = _api_client.post(f"/tasks/{tid}/resume", json={"decisions": {aid: {"decision": "approve", "edits": {"subject": "Final"}}}})
assert "(edited by you)" in ok.text and mail._load()["sent"][0]["subject"] == "Final", ok.text
orchestrator.router = real_router
print("1c. edit validation via API OK")
fresh()

# --- 2. Through the graph: the run PAUSES; nothing happens until the human decides ---
r = run("Reply to John's email saying 3 PM tomorrow works for me.")
assert "__interrupt__" in r, "the run should pause for approval"
calls = [h["agent"] for h in r["history"]]
assert len(calls) == len(set(calls)), f"an agent ran twice (e.g. to 'confirm' a pending action): {calls}"
actions = r["__interrupt__"][0].value["actions"]
print("   paused for:", [a["summary"] for a in actions])
assert [a["kind"] for a in actions] == ["email"] and actions[0]["details"]["to"] == "john.miller@nimbuslabs.com"
assert not mail._load()["sent"], "nothing may be sent before approval"
done = resume(r["thread_id"], {actions[0]["id"]: "approve"})
assert mail._load()["sent"][0]["to"] == "john.miller@nimbuslabs.com" and not mail._load()["drafts"]
assert "✓ Sent" in done["final_answer"], done["final_answer"]
print("2a. approve -> email sent OK")

leftover = mail.draft_email.invoke({"to": "old@x.com", "subject": "left over", "body": "x"})  # from "earlier"
r = run("Cancel my 1:1 with my manager tomorrow.")
calls = [h["agent"] for h in r["history"]]
assert len(calls) == len(set(calls)), f"an agent ran twice (e.g. to 'confirm' a pending action): {calls}"
actions = r["__interrupt__"][0].value["actions"]
assert [a["kind"] for a in actions] == ["calendar"], f"only THIS request's actions should be asked about: {actions}"
done = resume(r["thread_id"], {actions[0]["id"]: "reject"})
assert "e4" in [e["id"] for e in calendar_agent._load()["events"]] and not calendar_agent._load()["pending"]
assert "✗ Rejected" in done["final_answer"]
print("2b. reject -> calendar unchanged, proposal discarded OK")
fresh()

# --- 3. Through the HTTP API (what the UI uses) ---
client = TestClient(api.app)
client.post("/auth/register", json={"username": "tester", "password": "test-pass-1"})  # first user = "local"
client.headers["Authorization"] = "Bearer " + client.post(
    "/auth/login", json={"username": "tester", "password": "test-pass-1"}).json()["token"]


def events(response_text):
    out = []
    for chunk in response_text.strip().split("\n\n"):
        name, data = chunk.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


ev = events(client.post("/tasks/stream", json={"message": "Draft an email to priya.nair@nimbuslabs.com "
                                                              "saying the Atlas review moves to Friday."}).text)
names = [n for n, _ in ev]
print("   stream 1:", names)
assert names[-1] == "approval" and "final" not in names, names
tid = ev[0][1]["thread_id"]
action_id = ev[-1][1]["actions"][0]["id"]

assert client.post("/tasks/stream", json={"message": "hi", "thread_id": tid}).status_code == 409  # can't skip it
assert client.post(f"/tasks/{tid}/resume", json={"decisions": {"email:d99": "approve"}}).status_code == 422
assert client.get(f"/threads/{tid}").json()["pending_approval"][0]["id"] == action_id  # visible after reload too

ev = events(client.post(f"/tasks/{tid}/resume", json={"decisions": {action_id: "approve"}}).text)
names = [n for n, _ in ev]
print("   stream 2:", names)
assert "approved" in names and names[-1] == "final" and mail._load()["sent"][0]["to"] == "priya.nair@nimbuslabs.com"
assert client.post(f"/tasks/{tid}/resume", json={"decisions": {}}).status_code == 409  # nothing waiting any more
print("3. API: approval event, 409/422 guards, resume -> sent OK")

fresh()
print("\nApprovals OK")
