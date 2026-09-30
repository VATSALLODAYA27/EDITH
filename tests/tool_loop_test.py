"""Phase 5 test: Agent -> Tool -> Result, checked WITHOUT any LLM/API calls.

A ScriptedLLM replays pre-written AI replies (incl. tool calls) and records what the loop sends back,
so we can test the mechanics exactly: free, instant and deterministic.   Run:  python -m tests.tool_loop_test
"""
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agents.tool_agent import MAX_TOOL_OUTPUT, run_tool_agent

sys.stdout.reconfigure(encoding="utf-8")


class ScriptedLLM:
    def __init__(self, *replies: AIMessage):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(list(messages))  # what the agent loop sent to the "LLM" this turn
        return self.replies.pop(0)


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


def tool_replies(llm) -> list[str]:
    return [m.content for m in llm.seen[-1] if isinstance(m, ToolMessage)]


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def explode(reason: str) -> str:
    """Always fails."""
    raise FileNotFoundError(reason)


@tool
def huge() -> str:
    """Returns a very long text."""
    return "x" * (MAX_TOOL_OUTPUT + 5_000)


# 1. Happy path: tool call -> our code runs it -> result goes back -> final answer
llm = ScriptedLLM(AIMessage("", tool_calls=[call("add", {"a": 2, "b": 3})]), AIMessage("2 + 3 = 5"))
assert run_tool_agent("t", "sys", "add 2 and 3", [add], llm=llm) == "2 + 3 = 5"
assert tool_replies(llm) == ["5"]
assert llm.seen[-1][-1].tool_call_id == "c1"  # the result is linked to the exact call that asked for it
print("1. tool call -> result -> answer OK")

# 2. Tool raises -> error text (with type) goes back to the LLM instead of crashing
llm = ScriptedLLM(AIMessage("", tool_calls=[call("explode", {"reason": "no such file"})]), AIMessage("sorry"))
run_tool_agent("t", "sys", "x", [explode], llm=llm)
assert tool_replies(llm) == ["Error: FileNotFoundError: no such file"], tool_replies(llm)
print("2. tool error fed back OK")

# 3. Invented tool -> helpful error listing the real tools
llm = ScriptedLLM(AIMessage("", tool_calls=[call("web_search", {"q": "x"})]), AIMessage("ok"))
run_tool_agent("t", "sys", "x", [add], llm=llm)
assert tool_replies(llm) == ["Error: unknown tool 'web_search'. Available tools: add"], tool_replies(llm)
print("3. unknown tool OK")

# 4. Bad arguments -> schema validation error (before the function even runs)
llm = ScriptedLLM(AIMessage("", tool_calls=[call("add", {"a": "two"})]), AIMessage("ok"))
run_tool_agent("t", "sys", "x", [add], llm=llm)
err = tool_replies(llm)[0]
assert err.startswith("Error: ValidationError") and "b" in err, err
print("4. argument validation OK")

# 5. Several tool calls in ONE reply -> each runs and gets its own ToolMessage, matched by id
llm = ScriptedLLM(AIMessage("", tool_calls=[call("add", {"a": 1, "b": 1}, "c1"), call("add", {"a": 5, "b": 5}, "c2")]),
                  AIMessage("done"))
run_tool_agent("t", "sys", "x", [add], llm=llm)
msgs = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
assert [(m.tool_call_id, m.content) for m in msgs] == [("c1", "2"), ("c2", "10")]
print("5. parallel tool calls OK")

# 6. Huge output -> truncated before it reaches the LLM
llm = ScriptedLLM(AIMessage("", tool_calls=[call("huge", {})]), AIMessage("ok"))
run_tool_agent("t", "sys", "x", [huge], llm=llm)
out = tool_replies(llm)[0]
assert len(out) < MAX_TOOL_OUTPUT + 100 and out.endswith("[...truncated 5000 characters]")
print("6. output cap OK")

# 7. An LLM that never stops calling tools -> the loop stops at max_turns
llm = ScriptedLLM(*[AIMessage("", tool_calls=[call("add", {"a": 1, "b": 1}, f"c{i}")]) for i in range(3)])
assert run_tool_agent("t", "sys", "x", [add], max_turns=3, llm=llm) == "Stopped after 3 turns without finishing."
print("7. max_turns stop OK")

# 8. Invented / empty argument names are removed before the call is stored in the history
#    (seen in testing: list_files({'': ''}) made Gemini reject the whole conversation with 400 INVALID_ARGUMENT)
llm = ScriptedLLM(AIMessage("", tool_calls=[call("add", {"a": 1, "b": 2, "": "", "top_k": 5})]), AIMessage("ok"))
run_tool_agent("t", "sys", "x", [add], llm=llm)
stored = llm.seen[-1][2].tool_calls[0]["args"]  # [system, human, AI(tool call), ToolMessage]
assert stored == {"a": 1, "b": 2} and tool_replies(llm) == ["3"], stored
print("8. argument cleanup OK")

# 8b. Cross-provider history: a Groq-made tool call in the history must get Gemini's placeholder thought signature,
#     otherwise Gemini 3 rejects the whole conversation (Phase 9 bug: our "-latest" aliases disabled the fix).
import langchain_google_genai.chat_models as gm  # noqa: E402
from llm import models  # noqa: E402

for m in models:
    if "gemini" in m.model:
        _, contents = gm._parse_chat_history(
            [HumanMessage("x"), AIMessage("", tool_calls=[call("add", {"a": 1, "b": 2})]), ToolMessage("3", tool_call_id="c1")],
            model=m.model)
        sig = [p.thought_signature for c in contents if c.role == "model" for p in c.parts if p.function_call]
        assert sig == ["skip_thought_signature_validator"], f"{m.model}: foreign tool calls would be rejected ({sig})"
print("8b. Gemini fallback accepts other providers' tool calls OK")

# 9. The shared rules are always in the system prompt
assert "Never say you created" in llm.seen[0][0].content and "include the actual information" in llm.seen[0][0].content
print("9. shared rules present OK")

print("\nTool loop OK")
