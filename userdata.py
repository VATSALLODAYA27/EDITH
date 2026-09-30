"""Multi-user: WHO is the current request for, and WHERE does that user's data live.

CURRENT_USER is a context variable (like tracing.RUN_ID): the API sets it once per request, and deep code
(a tool, the mailbox, the profile) reads it without the user being passed through every function.

    user "local" (the first account; owns the pre-accounts data) -> <base>/workspace/, <base>/data/mailbox.json
    any other user                                               -> <base>/data/users/<id>/workspace/, ...

<base> is the project folder, or APP_DATA_DIR if set. Tests call use_temp_data(), so they never touch a real
user's files. Shared, read-only things (RAG documents, the mailbox seed, sample inputs) always come from the project.
"""
import contextvars
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL = "local"
CURRENT_USER = contextvars.ContextVar("current_user", default=LOCAL)
SAMPLE_INPUTS = ("sales.xlsx", "survey_report.pdf")  # every new workspace starts with the demo inputs


@contextmanager
def as_user(user_id: str):
    """Run a block of code on behalf of one user."""
    token = CURRENT_USER.set(user_id)
    try:
        yield
    finally:
        CURRENT_USER.reset(token)


def use_temp_data() -> Path:
    """Tests: send all per-user files to a fresh temporary folder."""
    folder = Path(tempfile.mkdtemp())
    os.environ["APP_DATA_DIR"] = str(folder)
    return folder


def base_dir() -> Path:
    return Path(os.environ.get("APP_DATA_DIR", ROOT))  # read on every call, so tests can switch it


def user_root() -> Path:
    uid = CURRENT_USER.get()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", uid):  # user ids become folder names: never allow ../ tricks
        raise ValueError(f"Invalid user id {uid!r}")
    return base_dir() / "data" if uid == LOCAL else base_dir() / "data" / "users" / uid


def workspace_dir() -> Path:
    path = (base_dir() / "workspace" if CURRENT_USER.get() == LOCAL else user_root() / "workspace").resolve()
    if not path.exists():
        path.mkdir(parents=True)
        for name in SAMPLE_INPUTS:  # seed a brand-new workspace with the demo inputs
            if (ROOT / "workspace" / name).exists():
                shutil.copy(ROOT / "workspace" / name, path / name)
    return path


def data_file(name: str) -> Path:
    """A per-user data file, e.g. data_file("mailbox.json")."""
    root = user_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / name
