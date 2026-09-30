"""Safe JSON files for small per-user stores (mailbox, calendar, approval queue).

Two problems with "read -> change -> write" on a plain file (both reproduced: 20 drafts written at once
CORRUPTED the mailbox with interleaved bytes):
  1. lost updates: two writers read the same old version; the second write erases the first
  2. torn writes: a crash (or another writer) mid-write leaves half a file

Fixes:
  locked(path)  - one read-change-write at a time per file (re-entrant, so helpers can nest)
  write_json()  - write a temp file, then os.replace() it over the real one: all-or-nothing

ponytail: in-process locks = one server process. Several processes/servers need a file lock (e.g. portalocker)
or, better, a real database (SQLite/Postgres transactions do both jobs).
"""
import json
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
_guard = threading.Lock()


@contextmanager
def locked(path: Path):
    with _guard:  # creating the per-file lock must itself be thread-safe
        lock = _locks[str(Path(path).resolve())]
    with lock:
        yield


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: readers see either the old file or the new one, never a mix
