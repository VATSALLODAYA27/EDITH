"""Phase 1: the smallest useful LangGraph.

    START -> agent -> END
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm import get_llm

llm = get_llm()


# --- 1. STATE: the data that flows through the graph ---
class State(TypedDict):
    # add_messages is a "reducer": new messages are APPENDED instead of overwriting the list.
    messages: Annotated[list[AnyMessage], add_messages]


# --- 2. NODE: a function that takes the State and returns an update ---
def agent(state: State) -> dict:
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}  # only the key we changed; LangGraph merges it in


# --- 3. GRAPH: register nodes, connect them with edges, compile ---
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252; LLMs emit Unicode (–, ‑, emoji)

    # --- 4. RUN: give it a starting State, get the final State back ---
    result = graph.invoke({"messages": [HumanMessage("Say hello to LangGraph in one sentence.")]})

    for m in result["messages"]:
        print(f"{m.type:>5}: {m.content}")

    # Self-check: reducer appended (not replaced) -> user msg + AI reply
    assert len(result["messages"]) == 2
    assert result["messages"][-1].type == "ai" and result["messages"][-1].content
    print("\nPhase 1 OK")
