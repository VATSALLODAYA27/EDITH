"""Calendar Agent: reads the calendar, finds free time, and PROPOSES new/changed/cancelled events.

    task -> tool-calling loop (list / find_free_slots / propose_*) -> answer or "change c1 awaits approval" -> result

KEY IDEAS:
- Time arithmetic is done in Python (find_free_slots), never by the LLM - same lesson as the Excel Agent.
- Changes affect other people (invites), so the agent can only PROPOSE them. apply_change() is human-only
  (Phase 9 adds the approval step) - same lesson as the Email Agent.

Calendar: a simulated JSON file (data/calendar.json) whose sample events are generated relative to today.
Run the standalone test from the project root:  python -m agents.calendar_agent
"""
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool

from agents.mail import _valid_address  # same email-address check as the Email Agent
from agents.tool_agent import run_tool_agent
from jsonstore import locked, read_json, write_json
from userdata import data_file

FMT, DAY = "%Y-%m-%d %H:%M", "%Y-%m-%d"
# ponytail: naive local times, no time zones; add zoneinfo when attendees span time zones.


# --- calendar storage ---
def reset_calendar() -> None:
    """Sample week, relative to today, so 'tomorrow' always has meetings."""
    t = date.today() + timedelta(days=1)
    d2 = t + timedelta(days=1)

    def ev(i, title, day, start, end, attendees=(), location="Pune office"):
        return {"id": f"e{i}", "title": title, "start": f"{day:%Y-%m-%d} {start}", "end": f"{day:%Y-%m-%d} {end}",
                "attendees": list(attendees), "location": location}

    events = [
        ev(1, "Daily standup", t, "09:30", "10:00", ["team@nimbuslabs.com"], "Online"),
        ev(2, "Atlas dashboards review", t, "11:00", "12:00", ["priya.nair@nimbuslabs.com"]),
        ev(3, "Customer call: Acme Retail", t, "14:00", "15:00", ["support@nimbuslabs.com"], "Online"),
        ev(4, "1:1 with manager", t, "16:30", "17:30", ["manager@nimbuslabs.com"]),
        ev(5, "Daily standup", d2, "09:30", "10:00", ["team@nimbuslabs.com"], "Online"),
        ev(6, "Q4 planning workshop", d2, "13:00", "16:00", ["all@nimbuslabs.com"]),
    ]
    _save({"events": events, "pending": []})


def _path():
    return data_file("calendar.json")  # the CURRENT user's calendar


def _load() -> dict:
    with locked(_path()):
        if not _path().exists():
            reset_calendar()
        return read_json(_path())


def _save(cal: dict) -> None:
    with locked(_path()):
        write_json(_path(), cal)  # atomic (see jsonstore.py)


@contextmanager
def transaction():
    """with transaction() as cal: ...  = read, change, save atomically, one writer at a time."""
    with locked(_path()):
        cal = _load()
        yield cal
        _save(cal)


def _parse(value: str, fmt: str = FMT) -> datetime:
    try:
        return datetime.strptime(value.strip(), fmt)
    except ValueError:
        raise ValueError(f"'{value}' must look like {datetime(2026, 10, 1, 15, 0).strftime(fmt)}") from None


def _event(cal: dict, event_id: str) -> dict:
    ev = next((e for e in cal["events"] if e["id"] == event_id), None)
    if ev is None:
        raise ValueError(f"No event '{event_id}'. Use list_events to find ids.")
    return ev


def _fmt_event(e: dict) -> str:
    who = f" | with {', '.join(e['attendees'])}" if e["attendees"] else ""
    return f"[{e['id']}] {e['start']} - {e['end'][-5:]} | {e['title']} | {e['location']}{who}"


def _conflicts(cal: dict, start: datetime, end: datetime, ignore_id: str = "") -> list[dict]:
    return [e for e in cal["events"] if e["id"] != ignore_id
            and _parse(e["start"]) < end and start < _parse(e["end"])]  # intervals overlap


def _propose(cal: dict, change: dict) -> str:
    # Idempotent: LLMs sometimes repeat a tool call; the user must not see (or approve) the same change twice.
    # Compare WHAT the change does (action + event), not free text: a retry once differed only in "reason".
    def effect(c: dict) -> tuple:
        ev = c.get("event", {})
        return c["action"], c.get("event_id"), ev.get("title"), ev.get("start"), ev.get("end")

    for existing in cal["pending"]:
        if effect(existing) == effect(change):
            return f"Change {existing['id']} ({change['action']}) is already proposed and awaiting the user's approval."
    # max+1 over ids ever used: len+1 could reuse an id after a rejected change was discarded
    cal["last_change_id"] = cal.get("last_change_id", 0) + 1
    change["id"] = f"c{cal['last_change_id']}"
    cal["pending"].append(change)
    return f"Proposed change {change['id']} ({change['action']}). NOT applied - it needs the user's approval."


# --- TOOLS ---
@tool
def list_events(start_date: str, end_date: str = "") -> str:
    """List events from start_date to end_date inclusive (dates as YYYY-MM-DD; end_date empty = same day)."""
    first, last = _parse(start_date, DAY), _parse(end_date or start_date, DAY) + timedelta(days=1)
    evs = sorted((e for e in _load()["events"] if first <= _parse(e["start"]) < last), key=lambda e: e["start"])
    return "\n".join(_fmt_event(e) for e in evs) or "No events in that period."


@tool
def find_free_slots(day: str, duration_minutes: int = 30, day_start: str = "09:00", day_end: str = "18:00") -> str:
    """Compute free time on a day (YYYY-MM-DD) within working hours, keeping gaps of at least duration_minutes.
    ALWAYS use this instead of working out free time yourself."""
    lo, hi = _parse(f"{day} {day_start}"), _parse(f"{day} {day_end}")
    busy = sorted((max(_parse(e["start"]), lo), min(_parse(e["end"]), hi)) for e in _load()["events"]
                  if _parse(e["start"]) < hi and lo < _parse(e["end"]))
    free, cursor = [], lo
    for s, e in busy:  # walk the sorted busy blocks; the space before each one is free
        if s > cursor:
            free.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        free.append((cursor, hi))
    need = timedelta(minutes=duration_minutes)
    slots = [f"{s:%H:%M}-{e:%H:%M}" for s, e in free if e - s >= need]
    return f"Free on {day} (>= {duration_minutes} min, {day_start}-{day_end}): " + (", ".join(slots) or "none")


@tool
def propose_event(title: str, start: str, end: str, attendees: list[str] = [], location: str = "") -> str:
    """Propose a NEW event (start/end as 'YYYY-MM-DD HH:MM'). Not created until the user approves."""
    s, e = _parse(start), _parse(end)
    if e <= s:
        raise ValueError("end must be after start")
    with transaction() as cal:  # no other writer between read and save
        clash = _conflicts(cal, s, e)
        msg = _propose(cal, {"action": "create", "event": {
            "title": title, "start": f"{s:{FMT}}", "end": f"{e:{FMT}}",
            "attendees": [_valid_address(a) for a in attendees], "location": location or "Online"}})
        return msg + (f" WARNING: overlaps {', '.join(c['title'] for c in clash)}." if clash else "")


@tool
def propose_update(event_id: str, title: str = "", start: str = "", end: str = "", location: str = "") -> str:
    """Propose changing an existing event. Only the fields you pass are changed. Needs the user's approval."""
    with transaction() as cal:  # no other writer between read and save
        ev = _event(cal, event_id)
        new = {**ev, **{k: v for k, v in {"title": title, "location": location}.items() if v}}
        if start or end:
            s, e = _parse(start or ev["start"]), _parse(end or ev["end"])
            if e <= s:
                raise ValueError("end must be after start")
            new["start"], new["end"] = f"{s:{FMT}}", f"{e:{FMT}}"
        clash = _conflicts(cal, _parse(new["start"]), _parse(new["end"]), ignore_id=event_id)
        msg = _propose(cal, {"action": "update", "event_id": event_id, "event": new})
        return msg + (f" WARNING: overlaps {', '.join(c['title'] for c in clash)}." if clash else "")


@tool
def propose_cancel(event_id: str, reason: str = "") -> str:
    """Propose cancelling an existing event. Needs the user's approval."""
    with transaction() as cal:  # no other writer between read and save
        ev = _event(cal, event_id)
        return _propose(cal, {"action": "cancel", "event_id": event_id, "title": ev["title"], "reason": reason})


def apply_change(change_id: str) -> str:
    """HUMAN-ONLY: apply a pending change. Deliberately NOT a tool - the agent can never call this."""
    with transaction() as cal:  # no other writer between read and save
        ch = next((c for c in cal["pending"] if c["id"] == change_id), None)
        if ch is None:
            raise ValueError(f"No pending change '{change_id}'")
        if ch["action"] == "create":
            next_id = max((int(e["id"][1:]) for e in cal["events"]), default=0) + 1  # default: empty calendar
            cal["events"].append({"id": f"e{next_id}", **ch["event"]})
        elif ch["action"] == "update":
            cal["events"] = [ch["event"] if e["id"] == ch["event_id"] else e for e in cal["events"]]
        else:
            cal["events"] = [e for e in cal["events"] if e["id"] != ch["event_id"]]
        cal["pending"].remove(ch)
        return f"Applied {change_id} ({ch['action']})"


TOOLS = [list_events, find_free_slots, propose_event, propose_update, propose_cancel]  # note: no apply

SYSTEM = """You are the Calendar Agent. Today is {today}. Times are local, format 'YYYY-MM-DD HH:MM'.
- Resolve relative dates ("tomorrow", "next Monday") from today's date above.
- Use list_events to see what's scheduled. NEVER work out free time yourself: use find_free_slots.
- You cannot change the calendar directly: propose_event / propose_update / propose_cancel create proposals
  that the user must approve. To propose a change you MUST call one of these tools - describing a change in
  your reply does nothing. Only report a proposal if the tool returned "Proposed change cN", and quote that id.
- Before proposing, check for conflicts; mention any WARNING you get.
- To change or cancel an event, first find its id with list_events.
- Don't invent attendees or email addresses; only use ones given in the task or found in events.
- End with a short answer, listing any proposed changes (id, what, when) as awaiting approval."""


def calendar_agent(state: dict) -> dict:
    system = SYSTEM.format(today=f"{date.today():%A %Y-%m-%d}")  # computed per call, never stale
    answer = run_tool_agent("calendar_agent", system, state["task"], TOOLS)
    print(f"[calendar_agent] {answer[:80]}...")
    return {"agent_results": {"calendar_agent": answer}}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tomorrow = f"{date.today() + timedelta(days=1):%Y-%m-%d}"

    # 1. Tools alone (no LLM)
    reset_calendar()
    assert not any(t.name.startswith("apply") for t in TOOLS), "the agent must not be able to apply changes"
    free = find_free_slots.invoke({"day": tomorrow, "duration_minutes": 30})
    print(free)
    assert "15:00-16:30" in free and "12:00-14:00" in free and "09:00-09:30" in free, free
    assert "10:00-11:00" in free and "17:30-18:00" in free
    assert "none" in find_free_slots.invoke({"day": tomorrow, "duration_minutes": 180})
    try:
        propose_event.invoke({"title": "x", "start": f"{tomorrow} 15:00", "end": f"{tomorrow} 14:00"})
        raise AssertionError("end before start accepted")
    except ValueError:
        pass
    assert "WARNING" in propose_event.invoke({"title": "clash", "start": f"{tomorrow} 14:30", "end": f"{tomorrow} 15:30"})
    print(apply_change("c1"))  # the human-only path
    assert len(_load()["events"]) == 7 and not _load()["pending"]
    reset_calendar()
    print("tool checks OK\n")

    # 2. Read the calendar
    ans = calendar_agent({"task": "What's on my calendar tomorrow?"})["agent_results"]["calendar_agent"]
    assert "atlas" in ans.lower() and "acme" in ans.lower(), ans

    # 3. Free-time question (answer depends on real maths: 15:00-16:30 is free)
    ans = calendar_agent({"task": "Am I free tomorrow at 3 PM for 30 minutes?"})["agent_results"]["calendar_agent"]
    print(f"\n{ans}\n")
    assert "free" in ans.lower() or "yes" in ans.lower(), ans

    # 4. Propose a new event (must NOT be created yet)
    calendar_agent({"task": "Schedule a 30-minute meeting 'Q4 roadmap sync' with john.miller@nimbuslabs.com "
                            "tomorrow at 3 PM."})
    cal = _load()
    create = next(c for c in cal["pending"] if c["action"] == "create")
    print("pending:", create)
    assert create["event"]["start"] == f"{tomorrow} 15:00" and create["event"]["end"] == f"{tomorrow} 15:30"
    assert "john.miller@nimbuslabs.com" in create["event"]["attendees"] and len(cal["events"]) == 6

    # 5. Cancel by meaning (agent must look up the right id)
    calendar_agent({"task": "Cancel my 1:1 with my manager tomorrow."})
    cancel = next(c for c in _load()["pending"] if c["action"] == "cancel")
    assert cancel["event_id"] == "e4", cancel

    reset_calendar()
    print("\nCalendar agent OK")
