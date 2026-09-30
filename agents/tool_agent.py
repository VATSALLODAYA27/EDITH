"""The agent loop, written by hand so you can see exactly what an "agent" is.

    LLM thinks -> asks for tool call(s) -> OUR code runs them -> results go back to the LLM -> repeat
    ... until the LLM answers without asking for a tool.

Test without any API calls:  python tool_loop_test.py
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from llm import get_llm
from memory import profile_prompt
from tracing import Timer, trace

MAX_TOOL_OUTPUT = 20_000  # chars; protects the context window and rate limits (Groq counts tokens per minute)

# Shared rule for every tool agent: LLMs sometimes DESCRIBE an action instead of calling the tool
# (seen in testing: the Calendar Agent said "proposed cancellation of e4" without calling propose_cancel).
HONESTY_RULE = ("\n\nActions only happen through tool calls. Never say you created, changed, drafted, proposed or "
                "deleted anything unless a tool call in this conversation actually did it and returned success.")
# Phase 4: in a chain, an agent's final reply IS the data the next agent receives. "I created a file" isn't enough.
HANDOFF_RULE = ("\nYour final reply may be passed to other agents: include the actual information you found or "
                "produced (facts, numbers, names, file names), not just a description of what you did.")


def run_tool(agent: str, tools_by_name: dict, call: dict) -> str:
    """Execute ONE tool call safely. Whatever happens, the LLM gets a string back - never a crash."""
    print(f"[{agent}] tool: {call['name']}({call['args']})")
    tool = tools_by_name.get(call["name"])
    if tool is None:  # LLMs sometimes invent tools (gpt-oss "remembers" tools from its training)
        trace("tool", agent=agent, tool=call["name"], ok=False, error="unknown tool")
        return f"Error: unknown tool '{call['name']}'. Available tools: {', '.join(tools_by_name)}"
    with Timer() as t:
        try:
            output = str(tool.invoke(call["args"]))  # .invoke validates args against the tool's schema first
            error = ""
        except Exception as e:  # errors go BACK to the LLM so it can fix its call
            output = error = f"Error: {type(e).__name__}: {e}"
    trace("tool", agent=agent, tool=call["name"], ok=not error, ms=t.ms, error=error[:200])
    if error:
        return error
    if len(output) > MAX_TOOL_OUTPUT:
        output = output[:MAX_TOOL_OUTPUT] + f"\n[...truncated {len(output) - MAX_TOOL_OUTPUT} characters]"
    return output


def run_tool_agent(name: str, system: str, task: str, tools: list, max_turns: int = 8, llm=None) -> str:
    """`llm` can be injected (e.g. a scripted fake in tests); by default it's our fallback chain with tools bound."""
    llm = llm or get_llm(tools=tools)
    by_name = {t.name: t for t in tools}
    # profile_prompt(): long-term memory about the user (name, email, sign-off...), read fresh on every run
    messages = [SystemMessage(system + HONESTY_RULE + HANDOFF_RULE + profile_prompt()), HumanMessage(task)]

    for _ in range(max_turns):  # hard cap: a confused LLM can't call tools forever
        ai = llm.invoke(messages)
        for call in ai.tool_calls:  # drop arguments the tool doesn't have: invented ones (gpt-oss adds "top_k")
            if call["name"] in by_name:  # or '' keys, which poison the history for the NEXT fallback provider
                call["args"] = {k: v for k, v in call["args"].items() if k in by_name[call["name"]].args}
        messages.append(ai)
        if not ai.tool_calls:  # no tool requested -> this is the final answer
            return ai.text
        for call in ai.tool_calls:  # the LLM may ask for several tools at once; each gets its own reply
            messages.append(ToolMessage(run_tool(name, by_name, call), tool_call_id=call["id"]))

    return f"Stopped after {max_turns} turns without finishing."
