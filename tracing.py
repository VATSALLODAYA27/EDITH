"""Phase 10: structured tracing. One JSON line per event, tagged with the run it belongs to.

    {"ts": "...", "run": "3f2a91c0-7b1e", "event": "tool", "agent": "email_agent", "tool": "draft_email", "ms": 12, "ok": true}

View a run as a timeline:  python trace_view.py            (latest run)
                           python trace_view.py <run-id>
Alternatives: LangSmith / Langfuse (hosted dashboards, auto-instrument LangChain) or OpenTelemetry.
"""
import contextvars
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

TRACE_FILE = Path(os.environ.get("TRACE_LOG", Path(__file__).resolve().parent / "data" / "logs" / "trace.jsonl"))
TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)

# The run id travels with the code via a context variable (like a thread-local that also follows async code and
# LangGraph's worker threads), so deep code (a tool, an LLM call) can log without the id being passed around.
RUN_ID = contextvars.ContextVar("run_id", default="-")
_lock = threading.Lock()


def trace(event: str, **fields) -> None:
    line = {"ts": datetime.now().isoformat(timespec="milliseconds"), "run": RUN_ID.get(), "event": event, **fields}
    text = json.dumps(line, ensure_ascii=False, default=str)
    with _lock:  # several agents may log at the same moment (parallel steps)
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


class Timer:
    """with Timer() as t: ...   then t.ms = elapsed milliseconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self.start) * 1000)


def read_run(run_id: str | None = None) -> list[dict]:
    """All events of one run (default: the most recent run)."""
    if not TRACE_FILE.exists():
        return []
    events = [json.loads(line) for line in TRACE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [e["run"] for e in events if e["event"] == "run_start"]
    run_id = run_id or (runs[-1] if runs else None)
    return [e for e in events if e["run"] == run_id]
