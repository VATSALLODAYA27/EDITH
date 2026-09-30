"""Show one traced run as a timeline.   python trace_view.py [run-id]   (default: the latest run)"""
import sys
from collections import Counter
from datetime import datetime

from tracing import TRACE_FILE, read_run

sys.stdout.reconfigure(encoding="utf-8")
events = read_run(sys.argv[1] if len(sys.argv) > 1 else None)
if not events:
    sys.exit(f"No traced runs in {TRACE_FILE}")

t0 = datetime.fromisoformat(events[0]["ts"])
for e in events:
    at = (datetime.fromisoformat(e["ts"]) - t0).total_seconds()
    ok = "" if "ok" not in e else ("✓" if e["ok"] else "✗")
    extra = {k: v for k, v in e.items() if k not in ("ts", "run", "event", "ok")}
    print(f"+{at:7.2f}s  {e['event']:<18} {ok} {extra}")

llm = [e for e in events if e["event"] == "llm"]
print(f"\nrun {events[0]['run']}: {len(llm)} LLM calls "
      f"({sum(e['ok'] for e in llm)} ok, failures by model: {dict(Counter(e['model'] for e in llm if not e['ok']))}), "
      f"{sum(e['event'] == 'tool' for e in events)} tool calls")
