"""Phase 10 tests: cooldowns/backoff, agent isolation, output verification, limits, recovery, tracing.

Fully deterministic: fake models raise scripted errors, and a fake router/agent drive the real graph.
No API calls, no quota.   Run:  python reliability_test.py
"""
import os
import tempfile

# throwaway memory DB + trace log: tests never touch the user's real data
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")
import sys  # noqa: E402

from userdata import use_temp_data  # noqa: E402

use_temp_data()  # never touch the real mailbox/calendar/workspace
import time  # noqa: E402

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.errors import GraphInterrupt  # noqa: E402

import llm  # noqa: E402
import orchestrator  # noqa: E402
from agents import research  # noqa: E402
from llm import AllModelsFailed, ResilientLLM, classify  # noqa: E402
from orchestrator import MAX_TURNS, Route, Step, guarded, verify_outputs  # noqa: E402
from tracing import read_run  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RATE = Exception("Error code: 429 - Rate limit reached. Please try again in 1.0s")
DAILY = Exception("Error code: 429 - Rate limit reached on tokens per day (TPD): Limit 200000")
BAD = Exception("Error code: 400 - Tool choice is required, but model did not call a tool")
DOWN = Exception("503 UNAVAILABLE. The model is overloaded")


class Fake:
    """A 'model' that plays back scripted outcomes (an Exception is raised, anything else is returned)."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def invoke(self, messages, config=None, **kw):
        self.calls += 1
        out = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(out, Exception):
            raise out
        return out


def fresh():
    llm._cooldown_until.clear()


# 1. Error classification -> cooldown length
assert classify(RATE) == ("rate_limit", 1.0) and classify(DAILY)[0] == "daily_limit" and classify(DAILY)[1] >= 1800
assert classify(BAD) == ("bad_request", 0) and classify(DOWN)[0] == "unavailable"
print("1. error classification OK")

# 2. Circuit breaker: after a 429 the model is SKIPPED (no wasted round trip) until its cooldown ends
fresh()
a, b = Fake(RATE), Fake("B1", "B2")
chain = ResilientLLM([("a", a), ("b", b)])
assert chain.invoke([]) == "B1" and chain.invoke([]) == "B2"
assert a.calls == 1, "a cooling model must not be called again"
print("2. cooldown skips a rate-limited model OK")

# 3. A bad request (400) is about THIS call, not the model: no cooldown, the model is tried next time
fresh()
a, b = Fake(BAD, "A2"), Fake("B1")
chain = ResilientLLM([("a", a), ("b", b)])
assert chain.invoke([]) == "B1" and chain.invoke([]) == "A2" and a.calls == 2
print("3. bad request -> no cooldown OK")

# 4. Everyone briefly rate-limited -> wait for the soonest cooldown, then retry (backoff) instead of failing
fresh()
a = Fake(RATE, "A")
start = time.monotonic()
assert ResilientLLM([("a", a)]).invoke([]) == "A" and time.monotonic() - start >= 0.9
print(f"4. backoff + retry after {time.monotonic() - start:.1f}s OK")

# 5. Nothing temporary about it (400s, a bad key) -> fail fast, no retry of the same bad request,
#    and the error lists EVERY model (not just the first)
fresh()
a, b, c = Fake(BAD, "A"), Fake(Exception("Error code: 401 - Invalid API Key")), Fake(BAD, "C")
start = time.monotonic()
try:
    ResilientLLM([("a", a), ("b", b), ("c", c)]).invoke([])
    raise AssertionError("should have failed")
except AllModelsFailed as e:
    assert [n for n, _ in e.errors] == ["a", "b", "c"] and "a:" in str(e) and "c:" in str(e), e
assert a.calls == c.calls == 1 and time.monotonic() - start < 1, "a 400 must not be retried within the same call"
print("5. AllModelsFailed lists every model, no pointless retries OK")

# 6. A daily limit parks the model for a long time; a call while EVERY model is parked fails immediately
fresh()
d = Fake(DAILY)
chain = ResilientLLM([("d", d)])
start = time.monotonic()
for _ in range(2):
    try:
        chain.invoke([])
    except AllModelsFailed:
        pass
assert d.calls == 1 and time.monotonic() - start < 1, "must not wait 30 minutes or retry a daily-limited model"
print("6. daily limit -> long cooldown, fail fast OK")
fresh()


# 7. Agent isolation: a crashing agent becomes an ERROR result; LangGraph's own signals still pass through
def boom(state):
    raise RuntimeError("disk on fire")


out = guarded("excel_agent", boom)({"task": "x"})
assert out["agent_results"]["excel_agent"].startswith("ERROR: excel_agent failed (RuntimeError: disk on fire)")


def pause(state):
    raise GraphInterrupt(())


try:
    guarded("x", pause)({"task": "x"})
    raise AssertionError("GraphInterrupt must not be swallowed")
except GraphInterrupt:
    pass
print("7. agent isolation OK")

# 8. Output verification: claims about files are checked against the workspace
assert verify_outputs({"ppt_agent": "Created **ghost_deck.pptx** with 3 slides."}) == \
    ["ppt_agent said it created or changed ghost_deck.pptx, but that file doesn't exist."]
assert verify_outputs({"excel_agent": "Updated sales.xlsx with a total row."}) == []  # exists -> fine
assert verify_outputs({"rag_agent": "The policy says 24 days."}) == []                 # no claim -> nothing to check
print("8. output verification OK")

# 9. Limits and recovery inside the orchestrator node (no LLM needed)
base = {"messages": [], "agent_results": {}, "turns": 0, "started": time.time()}
assert orchestrator.orchestrator({**base, "turns": MAX_TURNS})["plan"] == []
assert orchestrator.orchestrator({**base, "started": time.time() - 10_000})["plan"] == []
real_router, real_llm = orchestrator.router, orchestrator.llm
orchestrator.router = Fake(AllModelsFailed([("m", DOWN)]), AllModelsFailed([("m", DOWN)]))
assert orchestrator.orchestrator({**base, "agent_results": {"rag_agent": "24 days"}})["plan"] == []  # keep progress
try:
    orchestrator.orchestrator(base)  # nothing done yet -> the request fails (API: 503)
    raise AssertionError("should raise")
except AllModelsFailed:
    pass
orchestrator.llm = Fake(AllModelsFailed([("m", DOWN)]))
answer = orchestrator.finalize({"messages": [], "agent_results": {"rag_agent": "24 days", "research_agent": "Canberra"}})
assert "24 days" in answer["final_answer"] and "Canberra" in answer["final_answer"]  # shown as-is, not lost
print("9. turn/time limits, router + finalize recovery OK")

# 10. End to end through the REAL graph with a scripted router: a crashing agent doesn't kill the run, and the
#     trace records the whole run under one run id.
orchestrator.router = Fake(Route(steps=[Step(agent="research_agent", task="Capital of Australia?")], reason="r"),
                           Route(steps=[], reason="done"))
orchestrator.llm = real_llm
real_research_llm = research.llm
research.llm = Fake(RuntimeError("model exploded"))
result = orchestrator.run("What's the capital of Australia?")
assert result["agent_results"]["research_agent"].startswith("ERROR: research_agent failed")
assert "ERROR" in result["final_answer"], result["final_answer"]
events = read_run()
kinds = [e["event"] for e in events]
assert kinds[0] == "run_start" and kinds[-1] == "run_end" and events[-1]["status"] == "done", kinds
assert {"route", "agent"} <= set(kinds) and len({e["run"] for e in events}) == 1
assert next(e for e in events if e["event"] == "agent") | {} and not next(e for e in events if e["event"] == "agent")["ok"]
orchestrator.router, research.llm = real_router, real_research_llm
print("10. graph survives a crashing agent + full trace OK")

# 11. Concurrency: many writers at once must neither lose updates nor corrupt the file.
#     Before jsonstore.py, 20 drafts written at once CORRUPTED the mailbox (interleaved JSON).
import threading  # noqa: E402

from agents import calendar_agent, mail  # noqa: E402

mail.reset_mailbox()
calendar_agent.reset_calendar()
workers = [threading.Thread(target=mail.draft_email.invoke, args=({"to": f"p{i}@x.com", "subject": f"s{i}", "body": "b"},))
           for i in range(20)]
workers += [threading.Thread(target=calendar_agent.propose_cancel.invoke, args=({"event_id": f"e{1 + i % 6}", "reason": str(i)},))
            for i in range(12)]
for w in workers:
    w.start()
for w in workers:
    w.join()
drafts = mail._load()["drafts"]
assert len(drafts) == 20 and len({d["id"] for d in drafts}) == 20, f"lost or duplicate drafts: {len(drafts)}"
pending = calendar_agent._load()["pending"]
assert len(pending) == 6 and len({c["id"] for c in pending}) == 6, pending  # 6 distinct events, duplicates merged
print("11. 32 concurrent writers: no lost updates, no corruption OK")

print("\nReliability OK")
