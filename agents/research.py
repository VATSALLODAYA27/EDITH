"""Research Agent: answers general-knowledge questions from the LLM's own training data (no tools yet)."""
from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm

llm = get_llm()


def research_agent(state: dict) -> dict:
    # The agent only sees its TASK, not the whole conversation. The orchestrator decides what it needs.
    answer = llm.invoke([
        SystemMessage("You are a research assistant. Answer from general knowledge in at most 3 sentences."),
        HumanMessage(state["task"]),
    ]).text
    print(f"[research_agent] {answer[:80]}...")
    return {"agent_results": {"research_agent": answer}}
