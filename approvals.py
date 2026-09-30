"""Phase 9: every action that needs a human "yes" before it happens, in one place.

    email:d1     -> send a drafted email        (agents/mail.py drafts)
    calendar:c1  -> create/move/cancel an event (agents/calendar_agent.py pending changes)
    file:f1      -> delete a slide              (queued here by the PPT agent's delete_slide tool)

Agents can only CREATE these pending actions. execute() / discard() are called by the approval node, with the
decision the human made in the UI - never by an LLM.
"""
from agents import calendar_agent, mail
from jsonstore import locked, read_json, write_json
from userdata import data_file


def file_queue_path():
    return data_file("pending_file_actions.json")  # the CURRENT user's queue


# --- queue for risky file operations (the PPT agent's delete_slide adds to it) ---
def _file_queue() -> list[dict]:
    path = file_queue_path()
    with locked(path):
        return read_json(path) if path.exists() else []


def _save_file_queue(items: list[dict]) -> None:
    with locked(file_queue_path()):
        write_json(file_queue_path(), items)  # atomic


def queue_file_action(action: dict) -> str:
    with locked(file_queue_path()):  # read -> add -> save with no other writer in between
        items = _file_queue()
        action_id = f"f{max((int(a['id'][1:]) for a in items), default=0) + 1}"
        items.append({"id": action_id, **action})
        _save_file_queue(items)
    return action_id


# --- one list of everything waiting for approval ---
def list_pending() -> list[dict]:
    actions = []
    for d in mail._load()["drafts"]:
        actions.append({"id": f"email:{d['id']}", "kind": "email", "summary": f"Send email to {d['to']}: {d['subject']}",
                        "details": {"to": d["to"], "subject": d["subject"], "body": d["body"]}})
    for c in calendar_agent._load()["pending"]:
        ev = c.get("event", {})
        what = {"create": f"Create '{ev.get('title')}'", "update": f"Change '{ev.get('title')}'",
                "cancel": f"Cancel '{c.get('title')}'"}[c["action"]]
        when = f" on {ev['start']}-{ev['end'][-5:]}" if ev else ""
        actions.append({"id": f"calendar:{c['id']}", "kind": "calendar", "summary": what + when,
                        "details": {"action": c["action"], **({k: ev[k] for k in ("title", "start", "end", "attendees") if k in ev}),
                                    **({"reason": c["reason"]} if c.get("reason") else {})}})
    for f in _file_queue():
        actions.append({"id": f"file:{f['id']}", "kind": "file",
                        "summary": f"Delete slide {f['slide_number']} ('{f['title']}') from {f['filename']}", "details": f})
    return actions


def pending_ids() -> list[str]:
    return [a["id"] for a in list_pending()]


EDITABLE = {"email": ("subject", "body")}  # what a human may change before approving (never the recipient)


def execute(action_id: str, edits: dict | None = None) -> str:
    """Carry out ONE approved action, optionally with the human's edits applied first."""
    kind, local_id = action_id.split(":", 1)
    if edits and set(edits) - set(EDITABLE.get(kind, ())):
        raise ValueError(f"Can't edit {sorted(set(edits) - set(EDITABLE.get(kind, ())))} of a {kind} action")
    if kind == "email":
        if edits:
            with mail.transaction() as box:  # apply the edits to the draft, then send exactly that
                draft = next(d for d in box["drafts"] if d["id"] == local_id)
                draft.update({k: v.strip() for k, v in edits.items() if v and v.strip()})
        return mail.send_draft(local_id) + (" (edited by you)" if edits else "")
    if kind == "calendar":
        return calendar_agent.apply_change(local_id)
    if kind == "file":
        from agents.ppt import delete_slide_now  # imported here: ppt imports this module for queue_file_action
        item = next(a for a in _file_queue() if a["id"] == local_id)
        result = delete_slide_now(item["filename"], item["slide_number"], item["title"])
        _save_file_queue([a for a in _file_queue() if a["id"] != local_id])
        return result
    raise ValueError(f"Unknown action '{action_id}'")


def discard(action_id: str) -> str:
    """Throw away ONE rejected action, so it can never run later by accident."""
    kind, local_id = action_id.split(":", 1)
    if kind == "email":
        with mail.transaction() as box:
            box["drafts"] = [d for d in box["drafts"] if d["id"] != local_id]
    elif kind == "calendar":
        with calendar_agent.transaction() as cal:
            cal["pending"] = [c for c in cal["pending"] if c["id"] != local_id]
    elif kind == "file":
        with locked(file_queue_path()):
            _save_file_queue([a for a in _file_queue() if a["id"] != local_id])
    else:
        raise ValueError(f"Unknown action '{action_id}'")
    return f"Rejected {action_id}"
