"""The Orchestrator: plans which agents run, in what order, and what data flows between them.

    START -> orchestrator --(Send: 1+ steps in PARALLEL)--> agent(s) --+
                 ^        |-> no steps left -> approval -> finalize -> END   (approval may PAUSE for a human)
                 +------------------ agents report back ----------------+

Each turn the orchestrator returns a batch of steps that don't depend on each other (they run in parallel).
A step that needs another agent's output lists it in `inputs`; Python attaches that output word for word.

Phase 8: compiled with a checkpointer, so each conversation (thread_id) is saved and continues on the next request.
Phase 9: the approval node PAUSES the graph (interrupt) until a human approves/rejects risky actions.
Phase 10: failing agents are isolated, outputs are verified, runs have a time limit, everything is traced.
"""
import contextvars
import re
import time
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field

from agents.browser import browser_agent
from agents.calendar_agent import calendar_agent
from agents.document import document_agent
from agents.editor import document_editor
from agents.excel import excel_agent
from agents.mail import email_agent
from agents.ppt import ppt_agent
from agents.rag import rag_agent
from agents.research import research_agent
from llm import AllModelsFailed, get_llm
from approvals import discard, execute, list_pending, pending_ids
from memory import checkpointer
from tracing import RUN_ID, trace
from userdata import CURRENT_USER, workspace_dir

MAX_TURNS = 6          # hard stop: max routing decisions, so orchestrator <-> agents can never loop forever
MAX_RUN_SECONDS = 300  # hard stop on wall-clock time for one request (on top of the turn/tool-call limits)

# --- AGENT REGISTRY: adding an agent = one line here. Route, prompt and graph are all built from it. ---
# The description is what the orchestrator LLM reads to decide who gets the task.
AGENTS = {
    "research_agent": (research_agent, "stable general knowledge that doesn't change (science, history, "
                                       "how things work, definitions). No internet access."),
    "browser_agent": (browser_agent, "the live public web: current/recent info (latest versions, news, prices, "
                                     "anything that may have changed recently) or reading a given URL. Cites sources."),
    "rag_agent": (rag_agent, "ANSWER QUESTIONS from the knowledge base: company docs (HR/leave, travel & expenses, "
                             "internal projects, offices) and documents the user uploaded, with citations. "
                             "Never creates, edits or rewrites files."),
    "document_agent": (document_agent, "read, summarize or CREATE NEW document files in the user's workspace "
                                       "(Word .docx, PDF, .txt, .md; NOT spreadsheets), or add a section to a .docx. "
                                       "Use when the user names such a file or wants a new document made. NOT for "
                                       "rewriting/editing an existing document or using the user's template or "
                                       "letterhead: that's document_editor."),
    "document_editor": (document_editor, "CHANGE the content of the user's existing documents while keeping their "
                                         "look: rewrite a PDF/document into the user's template or letterhead .docx, "
                                         "or make precise text edits (names, dates, sentences) in a .docx. Use it "
                                         "alone for any request to edit, rewrite or re-template a document."),
    "excel_agent": (excel_agent, "spreadsheets (.xlsx): read data, calculate totals/averages/per-group numbers, "
                                 "create or edit sheets, formulas and charts."),
    "ppt_agent": (ppt_agent, "PowerPoint presentations (.pptx): create decks, add/edit/delete/reorder slides."),
    "email_agent": (email_agent, "the user's mailbox: read, search and summarize emails, and DRAFT emails or replies "
                                 "(it cannot send; drafts wait for the user's approval)."),
    "calendar_agent": (calendar_agent, "the user's calendar: list meetings, find free time, and PROPOSE new, "
                                       "moved or cancelled events (changes wait for the user's approval)."),
}


# --- STATE ---
# With a checkpointer the WHOLE state carries over to the next request in a thread. `messages` should (that's the
# conversation), but the per-run fields must start fresh. Merging reducers can't be cleared by passing {} or [],
# so they understand a RESET value.
RESET = "__reset__"


def merge_dict(old: dict, new):
    return {} if new == RESET else {**(old or {}), **new}


def append_list(old: list, new):
    return [] if new == RESET else (old or []) + new


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # the CONVERSATION: kept across requests in a thread
    summary: str                                          # older conversation, compacted (kept across requests)
    plan: list[dict]                                      # --- per-run fields (reset on every new request) ---
    agent_results: Annotated[dict, merge_dict]            # {"rag_agent": "..."}; MERGES parallel writes
    history: Annotated[list[dict], append_list]           # every step this run dispatched: {turn, agent, inputs}
    turns: int
    final_answer: str
    approval_baseline: list[str]  # pending actions that already existed BEFORE this request (not ours to ask about)
    approvals: list[str]          # what happened to this request's actions after the human decided
    started: float                # time.time() when this request began (time limit)


# --- ORCHESTRATOR ---
# The LLM must answer in this exact shape, so it can't invent an agent name.
class Step(BaseModel):
    agent: Literal[*AGENTS]  # only real agent names are allowed
    task: str = Field(description="Clear, self-contained instruction for this agent.")
    inputs: list[Literal[*AGENTS]] = Field(
        default_factory=list,
        description="Agents whose results this step needs. They must ALREADY appear in 'Agent results so far'.")


class Route(BaseModel):
    steps: list[Step] = Field(description="Steps to run NOW, in parallel. Empty list = the request is complete.")
    reason: str = Field(description="One short sentence: why this choice.")


ORCHESTRATOR_PROMPT = """You are the orchestrator of a team of agents. Plan the next step(s) for the user's request.

Agents:
{agents}

Rules:
- Use only the agents needed, and never redo work that is already in the results.
- Steps you return together run IN PARALLEL, so they must not depend on each other.
- If a step needs another agent's output, run that agent first; in a LATER turn, run the step with
  inputs=[that agent]. The system then attaches that agent's result to the step word for word.
  Always use inputs for this - don't copy facts from the results into the task yourself.
- When an agent's output will feed a later step, ask it to REPLY with the information (not to save it to a file,
  unless the user asked for that file).
- Give each agent its WHOLE job in one step (e.g. "find John's email and draft a reply"), not split across turns.
- Drafted emails, proposed calendar changes and requested deletions are shown to the user for approval
  automatically AFTER you finish. Never call an agent again to "confirm", "send" or "apply" them - just finish.
- Return an empty steps list when the results already fully answer or complete the request.
- The conversation may contain earlier requests and answers. Plan ONLY for the LATEST user request; use earlier
  turns as context (e.g. "that deck" = the file created earlier) and write it explicitly into the task,
  because agents don't see the conversation.""".format(
    agents="\n".join(f"- {name}: {desc}" for name, (_, desc) in AGENTS.items()))

router = get_llm(schema=Route)


MAX_HISTORY = 12      # recent messages the router sees (older ones live on in `summary`)
COMPACT_AFTER = 16    # once a thread has more messages than this...
KEEP_RECENT = 6       # ...fold all but the last few into the running summary


def summary_note(state) -> str:
    """The compacted older conversation, appended to a system prompt (one system message works for every provider)."""
    return f"\n\nSummary of the earlier conversation in this thread:\n{state['summary']}" if state.get("summary") else ""


def format_results(results: dict) -> str:
    return "\n\n".join(f"[{agent}]\n{answer}" for agent, answer in results.items()) or "(none yet)"


def orchestrator(state: State) -> dict:
    turns, results = state.get("turns", 0), state.get("agent_results", {})
    if turns >= MAX_TURNS:
        print("[orchestrator] turn limit reached -> finalize")
        trace("limit", kind="turns", turns=turns)
        return {"plan": []}
    if time.time() - state.get("started", time.time()) > MAX_RUN_SECONDS:
        print("[orchestrator] time limit reached -> finalize")
        trace("limit", kind="time", seconds=MAX_RUN_SECONDS)
        return {"plan": []}

    # Results go in as an explicit, labelled message, so the LLM can't mistake them for its own words.
    shown = HumanMessage(f"Agent results so far:\n{format_results(results)}")
    try:
        route = router.invoke([SystemMessage(ORCHESTRATOR_PROMPT + summary_note(state))]
                          + state["messages"][-MAX_HISTORY:] + [shown])
    except AllModelsFailed:
        if not results:
            raise  # nothing done yet: fail the request (the API answers 503 "try again")
        print("[orchestrator] router unavailable -> finishing with the results we already have")
        trace("recover", kind="router_failed_after_progress", agents=list(results))
        return {"plan": []}

    plan = []
    for step in route.steps:
        # Each agent runs at most once per request. Prompt rules didn't stop the router re-calling an agent to
        # "confirm"/"retrieve" its own pending draft, so it's enforced here (don't rely on the LLM for plumbing).
        # ponytail: blocks a deliberate second call (read now, write later); allow it per agent if a workflow needs it
        if step.agent in results:
            print(f"[orchestrator] skipping {step.agent}: it already ran in this request")
            continue
        missing = [a for a in step.inputs if a not in results]
        if missing:  # the LLM batched a step with the step it depends on -> it gets planned again next turn
            print(f"[orchestrator] deferring {step.agent}: waits for {missing}")
            continue
        if any(p["agent"] == step.agent for p in plan):  # same agent twice in parallel would overwrite its result
            continue
        # DATA FLOW: Python copies the needed results exactly - the LLM never has to re-type them.
        # The router often forgets `inputs` (seen in testing), so default to everything produced so far.
        inputs = step.inputs or list(results)
        context = "\n\n".join(f"[{a}]\n{results[a]}" for a in inputs)
        task = step.task + (f"\n\nInputs from other agents (use these facts):\n{context}" if context else "")
        plan.append({"agent": step.agent, "task": task, "inputs": inputs})

    names = [p["agent"] + (f"(inputs={p['inputs']})" if p["inputs"] else "") for p in plan] or ["FINISH"]
    print(f"[orchestrator] turn {turns + 1} -> {' + '.join(names)}: {route.reason}")
    trace("route", turn=turns + 1, agents=[p["agent"] for p in plan], reason=route.reason)
    history = [{"turn": turns + 1, "agent": p["agent"], "inputs": p["inputs"]} for p in plan]
    return {"plan": plan, "turns": turns + 1, "history": history}


# --- CONDITIONAL EDGE: one Send per planned step -> LangGraph runs them in parallel ---
def route_next(state: State):
    if not state.get("plan"):
        return "approval"
    # Send(node, payload): the agent receives ONLY this payload (its task), not the whole State.
    return [Send(p["agent"], {"task": p["task"]}) for p in state["plan"]]


# --- APPROVAL: pause for a human before anything irreversible happens ---
# On resume LangGraph re-runs this node FROM THE START, so everything before interrupt() must be safe to repeat.
# That's why the pause lives here (no LLM, just "list -> ask -> do") and not inside an agent's tool loop.
def approval(state: State) -> dict:
    ours = [a for a in list_pending() if a["id"] not in state.get("approval_baseline", [])]
    if not ours:
        return {"approvals": []}
    trace("approval_requested", actions=[a["id"] for a in ours])
    decisions = interrupt({"actions": ours})  # PAUSE: state saved; resumes with {action_id: "approve" | "reject"}
    trace("approval_decided", decisions=decisions)
    outcomes = []
    for action in ours:
        try:  # anything not explicitly approved is rejected (safe default)
            d = decisions.get(action["id"], "reject")
            d = d if isinstance(d, dict) else {"decision": d}  # "approve" | {"decision": "approve", "edits": {...}}
            if d.get("decision") == "approve":
                outcomes.append("✓ " + execute(action["id"], d.get("edits") or None))
            else:
                outcomes.append("✗ " + discard(action["id"]) + f" ({action['summary']})")
        except Exception as e:  # e.g. the slide was already gone: report it, don't crash the whole run
            outcomes.append(f"⚠ {action['summary']}: failed ({e})")
    print(f"[approval] {outcomes}")
    return {"approvals": outcomes}


llm = get_llm()  # used by finalize


# --- FINALIZE: turn the agent results into one answer for the user ---
def with_approvals(answer: str, state: State) -> str:
    done = state.get("approvals") or []
    answer += ("\n\nAfter your review:\n" + "\n".join(f"- {o}" for o in done) if done else "")
    problems = verify_outputs(state.get("agent_results", {}))
    return answer + ("\n\n⚠ Verification:\n" + "\n".join(f"- {p}" for p in problems) if problems else "")


# --- VERIFY: don't trust claims. An agent saying "Created x.pptx" must have really created it. ---
CLAIMED_FILE = re.compile(r"(?:created|saved|wrote|added to|updated)[^\n.]{0,80}?\b([\w\-]+\.(?:docx|pptx|xlsx))",
                          re.IGNORECASE)


def verify_outputs(results: dict) -> list[str]:
    problems = []
    for agent, text in results.items():
        for name in set(CLAIMED_FILE.findall(text)):
            if not (workspace_dir() / name).exists():
                problems.append(f"{agent} said it created or changed {name}, but that file doesn't exist.")
    if problems:
        trace("verify_failed", problems=problems)
    return problems


def finalize(state: State) -> dict:
    results = state.get("agent_results", {})
    if len(results) <= 1:  # one agent answered -> use it as-is, no extra LLM call
        answer = with_approvals(next(iter(results.values()), "Sorry, I couldn't answer that."), state)
        return {"final_answer": answer, "messages": [AIMessage(answer)]}
    try:
        answer = combine(state, results)
    except AllModelsFailed:  # no LLM to merge the answers: show them as they are rather than losing them
        trace("recover", kind="finalize_without_llm")
        answer = "\n\n".join(f"**{a}:** {r}" for a, r in results.items())
    answer = with_approvals(answer, state)
    # The answer joins the conversation, so the next request in this thread can refer to it ("that deck").
    return {"final_answer": answer, "messages": [AIMessage(answer)]}


def combine(state: State, results: dict) -> str:
    return llm.invoke([
        SystemMessage("Combine these agent results into one clear answer to the user's LATEST request. "
                      "Keep every fact and citation; don't add new information. Format: markdown (short paragraphs, "
                      "bullet lists, **bold**, tables where they help); never HTML (the UI drops it)."),
        *state["messages"][-MAX_HISTORY:],
        HumanMessage(format_results(results)),
    ]).text


# --- COMPACT: long conversations keep a rolling summary instead of silently forgetting old turns ---
def compact(state: State) -> dict:
    msgs = state.get("messages", [])
    if len(msgs) <= COMPACT_AFTER:
        return {}
    old = msgs[:-KEEP_RECENT]
    transcript = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {m.text[:1500]}" for m in old)
    try:
        summary = llm.invoke([
            SystemMessage("Update the running summary of this conversation. Keep what the user may refer to later: "
                          "their requests, decisions, file names, numbers, names, and what was sent or approved. "
                          "At most 200 words."),
            HumanMessage(f"Current summary:\n{state.get('summary') or '(none)'}\n\nOlder messages to fold in:\n{transcript}"),
        ]).text
    except AllModelsFailed:  # nothing is lost: the messages stay and we try again after the next request
        trace("recover", kind="compact_skipped")
        return {}
    trace("compact", folded=len(old), kept=KEEP_RECENT)
    # RemoveMessage(id) is understood by the add_messages reducer: those messages leave the saved state
    return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}


# --- ISOLATION: one agent crashing must not kill the whole request ---
def guarded(name: str, node):
    def run_agent(state: dict) -> dict:
        start = time.perf_counter()
        try:
            out, ok, error = node(state), True, ""
        except GraphBubbleUp:  # LangGraph's own control flow (e.g. an interrupt): must pass through
            raise
        except Exception as e:  # noqa: BLE001
            ok, error = False, f"{type(e).__name__}: {e}"
            print(f"[{name}] FAILED: {error[:200]}")
            out = {"agent_results": {name: f"ERROR: {name} failed ({error[:300]}). This part was not done."}}
        trace("agent", agent=name, ok=ok, ms=round((time.perf_counter() - start) * 1000), error=error[:200])
        return out
    return run_agent


# --- GRAPH ---
builder = StateGraph(State)
builder.add_node("orchestrator", orchestrator)
builder.add_node("approval", approval)
builder.add_node("finalize", finalize)
builder.add_node("compact", compact)
for name, (node, _) in AGENTS.items():
    builder.add_node(name, guarded(name, node))
    builder.add_edge(name, "orchestrator")  # agents always report back

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", route_next, [*AGENTS, "approval"])  # possible destinations
builder.add_edge("approval", "finalize")
builder.add_edge("finalize", "compact")
builder.add_edge("compact", END)
# The checkpointer saves the full state after every step, keyed by thread_id (short-term memory + resume).
graph = builder.compile(checkpointer=checkpointer)


def new_request(message: str) -> dict:
    """Input for one request: append the message to the conversation, reset every per-run field."""
    return {"messages": [HumanMessage(message)], "plan": [], "agent_results": RESET, "history": RESET,
            "turns": 0, "final_answer": "", "approval_baseline": pending_ids(), "approvals": [],
            "started": time.time()}


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def traced_stream(graph_input, config: dict, user: str | None = None):
    """graph.stream(), with every step run inside ONE context that carries this run's id (for tracing).

    Why ctx.run(next, ...): a web server may resume a generator on different threads, and a context variable set
    in one step would be gone in the next. Running each step inside the same copied context keeps the id.
    """
    thread_id = config["configurable"]["thread_id"]
    ctx = contextvars.copy_context()
    ctx.run(RUN_ID.set, f"{thread_id[:8]}-{uuid.uuid4().hex[:4]}")
    if user:  # every agent, tool and data file in this run belongs to this user
        ctx.run(CURRENT_USER.set, user)
    ctx.run(trace, "run_start", thread=thread_id, resume=isinstance(graph_input, Command))
    steps = ctx.run(graph.stream, graph_input, config, stream_mode="updates")
    start, status = time.perf_counter(), "done"
    try:
        while True:
            try:
                update = ctx.run(next, steps)
            except StopIteration:
                break
            status = "paused" if "__interrupt__" in update else status
            yield update
    except GeneratorExit:  # the client went away (e.g. pressed Stop)
        status = "stopped"
        raise
    except Exception as e:
        status = "failed"
        ctx.run(trace, "error", error=f"{type(e).__name__}: {str(e)[:300]}")
        raise
    finally:
        ctx.run(trace, "run_end", status=status, ms=round((time.perf_counter() - start) * 1000))


def invoke_traced(graph_input, thread_id: str) -> dict:
    """Like graph.invoke (final state), but through traced_stream so the run is traced."""
    for _ in traced_stream(graph_input, thread_config(thread_id)):
        pass
    snap = graph.get_state(thread_config(thread_id))
    result = {**snap.values, "thread_id": thread_id}
    if snap.interrupts:
        result["__interrupt__"] = list(snap.interrupts)
    return result


def run(question: str, thread_id: str | None = None) -> dict:
    """Run one request. Pass the same thread_id to continue a conversation; None starts a new one."""
    return invoke_traced(new_request(question), thread_id or str(uuid.uuid4()))


def resume(thread_id: str, decisions: dict) -> dict:
    """Continue a paused run with the human's decisions, e.g. {"email:d1": "approve", "calendar:c2": "reject"}."""
    return invoke_traced(Command(resume=decisions), thread_id)


def waiting_for_approval(thread_id: str) -> list[dict]:
    """The actions a paused thread is waiting on ([] if it isn't paused)."""
    snap = graph.get_state(thread_config(thread_id))
    return [a for i in snap.interrupts for a in i.value.get("actions", [])]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    tests = [
        ("What causes the seasons on Earth?", {"research_agent"}),
        ("How many vacation days do I get per year?", {"rag_agent"}),
        ("When does Project Atlas launch, and what is a data platform in general?", {"rag_agent", "research_agent"}),
        ("What's the daily meal allowance when I travel for work?", {"rag_agent"}),  # only in the new docs
        ("Summarize survey_report.pdf in two sentences.", {"document_agent"}),
        ("Create a Word document called lunch_invite.docx inviting the team to lunch on Friday at 1pm.",
         {"document_agent"}),
        ("Rewrite survey_report.pdf into my company template acme_template.docx.", {"document_editor"}),
        ("In offer_letter.docx change the joining date from 1 May to 3 June.", {"document_editor"}),
        ("Which product sold the most units in total in sales.xlsx?", {"excel_agent"}),
        ("Create offsite.pptx with a title slide 'Team Offsite' and one slide with: Date 12 Dec, "
         "Venue Pune office, Agenda planning + lunch.", {"ppt_agent"}),
        ("What is the latest stable version of Python right now?", {"browser_agent"}),
        ("Read https://example.com and tell me in one sentence what it says.", {"browser_agent"}),
        ("Draft an email to priya.nair@nimbuslabs.com saying I'll review the Atlas dashboards by Friday.",
         {"email_agent"}),
        ("What meetings do I have tomorrow, and when am I free for an hour?", {"calendar_agent"}),
    ]
    from userdata import use_temp_data, workspace_dir

    use_temp_data()
    WORKSPACE = workspace_dir()
    from agents.calendar_agent import reset_calendar
    from agents.mail import reset_mailbox
    reset_mailbox()
    reset_calendar()
    for f in ["lunch_invite.docx", "offsite.pptx"]:  # make the create-tests repeatable
        (WORKSPACE / f).unlink(missing_ok=True)
    for question, expected in tests:
        print(f"\n=== {question}")
        result = run(question)
        if "__interrupt__" in result:  # paused for approval (e.g. the email draft): reject, so nothing is sent
            result = resume(result["thread_id"], {})
        print(f"FINAL: {result['final_answer']}")
        used = set(result["agent_results"])
        assert used == expected, f"expected {expected}, orchestrator used {used}"
        calls = [h["agent"] for h in result["history"]]
        assert len(calls) == len(set(calls)), f"an agent ran twice (wasted work): {calls}"

    print("\nOrchestrator OK")
