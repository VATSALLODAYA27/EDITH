"""Long documents (map-reduce) and long conversations (rolling summary). No LLM calls: scripted fakes.

Run:  python long_context_test.py
"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")
import sys  # noqa: E402

from userdata import use_temp_data, workspace_dir  # noqa: E402

use_temp_data()
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import orchestrator  # noqa: E402
from agents import document  # noqa: E402
from llm import AllModelsFailed  # noqa: E402
from orchestrator import COMPACT_AFTER, KEEP_RECENT, Route  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


class FakeLLM:
    """Records every prompt; 'summarizes' by keeping lines that contain a CODE-like token."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages, *a, **k):
        self.prompts.append(messages)
        text = messages[-1].content
        keep = [line for line in text.split("\n") if "CODE-" in line] or ["(nothing important)"]
        return AIMessage("\n".join(keep))


# --- 1. Long document: 60k characters, the key fact is at the very END ---
paragraphs = [f"Paragraph {i}: routine operational detail number {i} about office logistics." for i in range(900)]
paragraphs[-3] = "Final decision: the launch code is CODE-ORCHID-7 and ships in March."
(workspace_dir() / "big_report.txt").write_text("\n".join(paragraphs), encoding="utf-8")

shown = document.read_document.invoke({"filename": "big_report.txt"})
assert "CODE-ORCHID-7" not in shown and "summarize_long_document" in shown, "read_document should truncate + hint"

fake = FakeLLM()
document._summarizer = fake
summary = document.summarize_long_document.invoke({"filename": "big_report.txt", "focus": "decisions"})
parts = len(document._chunks("\n".join(paragraphs)))
print(f"   {parts} parts -> {len(fake.prompts)} LLM calls (map {parts} + reduce 1)")
assert "CODE-ORCHID-7" in summary, "map-reduce must cover the WHOLE file, including the end"
assert len(fake.prompts) == parts + 1 and parts > 1
assert all(len(c) <= document.CHUNK_CHARS for c in document._chunks("\n".join(paragraphs)))
assert "decisions" in fake.prompts[0][0].content  # the focus reaches every step
try:
    (workspace_dir() / "huge.txt").write_text("x" * (document.CHUNK_CHARS * (document.MAX_CHUNKS + 1)), encoding="utf-8")
    document.summarize_long_document.invoke({"filename": "huge.txt"})
    raise AssertionError("should refuse files that are too long")
except ValueError:
    pass
document._summarizer = None
print("1. long document: whole file covered, bounded cost OK")

# --- 2. Long conversation: older turns fold into a summary, recent ones stay verbatim ---
msgs = []
for i in range(10):
    msgs += [HumanMessage(f"request {i}" + (" (keep CODE-DECK-1: we made survey_deck.pptx)" if i == 0 else ""), id=f"h{i}"),
             AIMessage(f"answer {i}", id=f"a{i}")]
state = {"messages": msgs, "summary": ""}
assert orchestrator.compact({"messages": msgs[:COMPACT_AFTER], "summary": ""}) == {}  # short: nothing to do

real_llm = orchestrator.llm
orchestrator.llm = FakeLLM()
out = orchestrator.compact(state)
removed = [m.id for m in out["messages"]]
assert removed == [m.id for m in msgs[:-KEEP_RECENT]], removed          # everything but the last few
assert "CODE-DECK-1" in out["summary"], out["summary"]                   # the old fact survives in the summary

orchestrator.llm = type("Down", (), {"invoke": lambda *a, **k: (_ for _ in ()).throw(AllModelsFailed([]))})()
assert orchestrator.compact(state) == {}, "if no LLM is available, keep the messages (nothing lost)"
orchestrator.llm = real_llm
print("2. compaction: old turns -> summary, recent kept, safe on failure OK")

# --- 3. The router actually sees the summary ---
seen = []
real_router = orchestrator.router
orchestrator.router = type("R", (), {"invoke": lambda self, m, *a, **k: seen.append(m) or Route(steps=[], reason="x")})()
orchestrator.orchestrator({"messages": msgs[-KEEP_RECENT:], "summary": "User made survey_deck.pptx (CODE-DECK-1).",
                           "agent_results": {}, "turns": 0})
orchestrator.router = real_router
assert "CODE-DECK-1" in seen[0][0].content, "the summary must be in the router's prompt"
print("3. router sees the conversation summary OK")

print("\nLong context OK")
