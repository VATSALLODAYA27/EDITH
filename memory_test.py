"""Phase 8 tests: per-run state reset, short-term memory (threads), persistence, long-term memory (profile).

Tests 1-2 use no LLM. Tests 3-5 call real LLMs.   Run:  python memory_test.py
"""
import os
import tempfile

# Use a throwaway memory database: tests must never write into the user's real history/profile.
_tmp = tempfile.mkdtemp()
os.environ["MEMORY_DB"] = os.path.join(_tmp, "test_memory.sqlite")
os.environ["TRACE_LOG"] = os.path.join(_tmp, "trace.jsonl")  # ...and never into the real trace log
import sqlite3
import sys

from userdata import use_temp_data  # noqa: E402

use_temp_data()  # never touch the real mailbox/calendar/workspace

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import orchestrator
from agents.mail import _load as load_mailbox, email_agent, reset_mailbox
from agents.tool_agent import run_tool_agent
from memory import DB_PATH, get_profile, save_profile
from orchestrator import RESET, append_list, merge_dict, run, thread_config

sys.stdout.reconfigure(encoding="utf-8")
original_profile = get_profile()
FIELDS = ("name", "email", "role", "sign_off", "preferences")

# 1. Per-run fields reset; merging still works within a run
assert merge_dict({"rag_agent": "old"}, RESET) == {} and merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
assert append_list([{"turn": 1}], RESET) == [] and append_list([1], [2]) == [1, 2]
print("1. RESET reducers OK")


# 2. Long-term memory reaches every tool agent's prompt (scripted LLM, no API call)
class ScriptedLLM:
    def __init__(self):
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage("ok")


save_profile({"name": "Asha Rao", "sign_off": "Best,\nAsha", "email": "", "role": "", "preferences": ""})
fake = ScriptedLLM()
run_tool_agent("t", "sys", "x", [], llm=fake)
assert '"name": "Asha Rao"' in fake.seen[0][0].content, fake.seen[0][0].content
save_profile({k: "" for k in FIELDS})
fake = ScriptedLLM()
run_tool_agent("t", "sys", "x", [], llm=fake)
assert "About the user" not in fake.seen[0][0].content  # no profile -> nothing invented
print("2. profile -> agent prompt OK")

# 3. Short-term memory: a follow-up that only makes sense with the previous turn ("those")
r1 = run("How many vacation days do I get per year?")
tid = r1["thread_id"]
r2 = run("And how many of those can I carry over to next year?", thread_id=tid)
print(f"   turn 1: {r1['final_answer'][:90]}\n   turn 2: {r2['final_answer'][:90]}")
assert "5" in r2["final_answer"], r2["final_answer"]
msgs = r2["messages"]
assert [m.type for m in msgs] == ["human", "ai", "human", "ai"], [m.type for m in msgs]
# per-run fields started fresh: only THIS run's work, turn counter restarted
assert list(r2["agent_results"]) == ["rag_agent"] and all(h["turn"] <= 2 for h in r2["history"]), r2["history"]
print("3. follow-up in the same thread OK")

# 4. Persistence: a brand-new connection (= server restart) reads the same conversation from disk
fresh_graph = orchestrator.builder.compile(checkpointer=SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False)))
restored = fresh_graph.get_state(thread_config(tid)).values["messages"]
assert len(restored) == 4 and "carry over" in restored[2].text
assert not fresh_graph.get_state(thread_config("some-other-thread")).values  # threads are isolated
print("4. persisted across a restart OK")

# 5. Long-term memory in action: the email draft is signed with the profile, not "[Your Name]"
reset_mailbox()
save_profile({"name": "Asha Rao", "sign_off": "Best,\nAsha", "email": "", "role": "", "preferences": ""})
email_agent({"task": "Reply to John's email (m1) saying 3 PM tomorrow works for me."})
draft = next(d for d in load_mailbox()["drafts"] if d["reply_to"] == "m1")
print(f"   draft ends: {draft['body'][-40:]!r}")
assert draft["body"].rstrip().endswith("Best,\nAsha") and "[Your Name]" not in draft["body"], draft["body"]
print("5. profile used in email signature OK")

reset_mailbox()
save_profile({k: original_profile.get(k, "") for k in FIELDS})
print("\nMemory OK")
