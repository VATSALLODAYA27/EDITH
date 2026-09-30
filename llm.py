"""Shared LLM access with fallbacks, per-model cooldowns (circuit breaker) and retry-with-backoff.

Needs GROQ_API_KEY and GOOGLE_API_KEY in the environment.

    get_llm().invoke(messages)
      -> try each model in order, SKIPPING models that are cooling down after a rate limit / outage
      -> all failed but only briefly (429/503)? wait for the soonest cooldown to end, try again (bounded)
      -> still nothing? raise AllModelsFailed listing EVERY model's error (not just the first)
"""
import re
import threading
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from tracing import Timer, trace

# temperature=0 -> same question gives the same routing decision (predictable, testable).
# max_retries=0: WE decide retries (cooldowns + backoff), instead of each SDK silently retrying for ages.
models = [
    ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=0, timeout=30),  # primary
    ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_retries=0, timeout=30),   # own quota
    # VERSIONED names, not "-latest" aliases: Gemini 3 rejects tool calls made by ANOTHER provider (Groq) unless they
    # carry a placeholder "thought signature", and langchain-google-genai only adds it when the name contains
    # "gemini-3". Aliases hid that (Phase 9 bug) - and they can also silently switch models.
    ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=0, timeout=30),
    ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=0, timeout=30),  # ignores temperature
]

MAX_WAIT = 45        # seconds we'll wait in total for a cooling model before giving up on a call
MAX_ROUNDS = 3       # passes over the model list per call
_cooldown_until: dict[str, float] = {}  # model name -> time.monotonic() when it may be used again
_lock = threading.Lock()


class AllModelsFailed(Exception):
    """Every model failed for one call. The message lists each model's error."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        super().__init__("; ".join(f"{name}: {type(e).__name__}: {str(e)[:120]}" for name, e in errors) or
                         "every model is cooling down after rate limits")


def classify(error: Exception) -> tuple[str, float]:
    """(kind, cooldown seconds). Only rate limits / outages cool a model down; a bad request is about THIS call."""
    text = f"{type(error).__name__} {error}".lower()
    if "per day" in text or "(tpd)" in text or "tokens per day" in text:
        return "daily_limit", 1800  # won't come back soon: stop wasting a round trip on it for 30 minutes
    if "429" in text or "ratelimit" in text or "rate limit" in text or "resource_exhausted" in text:
        wait = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text)
        return "rate_limit", (int(wait.group(1) or 0) * 60 + float(wait.group(2))) if wait else 20
    if "503" in text or "unavailable" in text or "overloaded" in text or "timeout" in text or "timed out" in text:
        return "unavailable", 15
    if "401" in text or "authentication" in text or "invalid api key" in text:
        return "auth", 600  # a broken key won't fix itself
    return "bad_request", 0  # e.g. 400 "tool choice required": another model may cope, no cooldown


def cooldown_left(name: str) -> float:
    with _lock:
        return max(0.0, _cooldown_until.get(name, 0) - time.monotonic())


def _cool(name: str, seconds: float) -> None:
    with _lock:
        _cooldown_until[name] = max(_cooldown_until.get(name, 0), time.monotonic() + seconds)


class ResilientLLM:
    """Same .invoke() as a LangChain model, over several models. Holds (name, runnable) pairs."""

    def __init__(self, options: list[tuple[str, object]]):
        self.options = options

    def invoke(self, messages, config=None, **kwargs):
        errors, waited = [], 0.0
        given_up = set()  # models whose failure was about THIS request (400, bad key): retrying it won't help
        for _ in range(MAX_ROUNDS):
            transient = False
            for name, runnable in self.options:
                if name in given_up or cooldown_left(name):
                    continue  # circuit open: don't waste a round trip on a model we know is failing
                try:
                    with Timer() as t:
                        result = runnable.invoke(messages, config, **kwargs)
                    trace("llm", model=name, ok=True, ms=t.ms)
                    return result
                except Exception as e:  # noqa: BLE001 - every provider error means "try the next one"
                    kind, seconds = classify(e)
                    errors.append((name, e))
                    transient |= kind in ("rate_limit", "unavailable", "daily_limit")
                    if kind in ("bad_request", "auth"):
                        given_up.add(name)
                    if seconds:
                        _cool(name, seconds)
                    trace("llm", model=name, ok=False, ms=t.ms, error=kind, detail=str(e)[:200], cooldown_s=seconds)
                    print(f"[llm] {name} failed ({kind}{f', cooling {seconds:.0f}s' if seconds else ''}), trying next")
            # Nobody answered. Worth waiting only if the failures were temporary and a model frees up soon.
            candidates = [n for n, _ in self.options if n not in given_up]
            soonest = min((cooldown_left(n) for n in candidates), default=float("inf"))
            if not transient or waited + soonest > MAX_WAIT:
                break
            wait = max(soonest, 1.0)  # backoff = until the first model's cooldown ends
            print(f"[llm] all models busy, waiting {wait:.0f}s")
            time.sleep(wait)
            waited += wait
        raise AllModelsFailed(errors)


def get_llm(schema=None, tools=None) -> ResilientLLM:
    """Plain chat LLM, one forced to reply with `schema` (a Pydantic model), or one that can call `tools`.

    Structured output / tools are applied to EACH model, because they're model-specific wrappers.
    """
    if schema:
        bound = [m.with_structured_output(schema) for m in models]
    elif tools:
        bound = [m.bind_tools(tools) for m in models]
    else:
        bound = models
    return ResilientLLM([(m.model, b) for m, b in zip(models, bound)])
