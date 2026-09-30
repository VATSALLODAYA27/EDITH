"""Phase 8: persistence and memory.

- SHORT-TERM memory (a conversation): LangGraph's SqliteSaver checkpointer saves the whole graph state after
  every step, keyed by thread_id. Same thread_id next time = the conversation continues.
- LONG-TERM memory (about the user, across all threads): a small profile (name, email, sign-off, preferences).
- TASK HISTORY: an index of threads (id, title, last update) so past conversations can be listed and reopened.

Everything lives in data/memory.sqlite (one file, no server).
"""
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from userdata import CURRENT_USER

# MEMORY_DB lets tests use a throwaway database, so they never pollute the user's real history/profile.
DB_PATH = Path(os.environ.get("MEMORY_DB", Path(__file__).resolve().parent / "data" / "memory.sqlite"))

# check_same_thread=False: FastAPI serves requests from several worker threads. SqliteSaver locks internally;
# our own small tables use _lock. ponytail: one SQLite file = single server process; use PostgresSaver to scale out.
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
_lock = threading.Lock()

def _columns(table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


with _lock:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS profile (user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                                            PRIMARY KEY (user_id, key));
        CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, title TEXT NOT NULL, updated TEXT NOT NULL,
                                            user_id TEXT NOT NULL DEFAULT 'local');
    """)
    # MIGRATIONS from the single-user version (Phase 8): existing data belongs to the "local" user.
    if "user_id" not in _columns("profile"):  # the primary key changes, so the table has to be rebuilt
        conn.executescript("""
            ALTER TABLE profile RENAME TO profile_single_user;
            CREATE TABLE profile (user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                                  PRIMARY KEY (user_id, key));
            INSERT INTO profile SELECT 'local', key, value FROM profile_single_user;
            DROP TABLE profile_single_user;
        """)
    if "user_id" not in _columns("threads"):
        conn.execute("ALTER TABLE threads ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'")
    conn.commit()

PROFILE_FIELDS = ("name", "email", "role", "sign_off", "preferences")


# --- long-term memory: the CURRENT user's profile ---
def get_profile() -> dict:
    with _lock:
        rows = conn.execute("SELECT key, value FROM profile WHERE user_id = ?", (CURRENT_USER.get(),)).fetchall()
    return {k: v for k, v in rows if k in PROFILE_FIELDS}


def save_profile(values: dict) -> dict:
    user = CURRENT_USER.get()
    with _lock:
        for key, value in values.items():
            if key not in PROFILE_FIELDS:
                continue
            if value:
                conn.execute("INSERT OR REPLACE INTO profile VALUES (?, ?, ?)", (user, key, value.strip()))
            else:  # empty = forget this fact
                conn.execute("DELETE FROM profile WHERE user_id = ? AND key = ?", (user, key))
        conn.commit()
    return get_profile()


def profile_prompt() -> str:
    """Text added to agent prompts so they know who they work for (signatures, "my", preferences)."""
    p = get_profile()
    if not p:
        return ""
    return ("\n\nAbout the user you work for (use it; never invent missing details. If sign_off is set, end "
            "emails with it exactly, word for word):\n" + json.dumps(p, ensure_ascii=False))


# --- task history: which conversations exist, and WHO owns each one ---
def thread_owner(thread_id: str) -> str | None:
    with _lock:
        row = conn.execute("SELECT user_id FROM threads WHERE id = ?", (thread_id,)).fetchone()
    return row[0] if row else None


def touch_thread(thread_id: str, first_message: str) -> None:
    """Create the thread (owned by the current user, titled by its first message); later just bump 'updated'.
    Callers must check thread_owner() first: this never changes an existing thread's owner."""
    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        conn.execute("INSERT INTO threads (id, title, updated, user_id) VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET updated = excluded.updated",
                     (thread_id, first_message[:80], now, CURRENT_USER.get()))
        conn.commit()


def list_threads(limit: int = 30) -> list[dict]:
    with _lock:
        rows = conn.execute("SELECT id, title, updated FROM threads WHERE user_id = ? ORDER BY updated DESC LIMIT ?",
                            (CURRENT_USER.get(), limit)).fetchall()
    return [{"id": i, "title": t, "updated": u} for i, t, u in rows]
