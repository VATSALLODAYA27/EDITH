"""Accounts and sessions (username + password -> session token).

- Passwords: salted scrypt hash (Python stdlib). scrypt is deliberately slow + memory-hungry, so a stolen
  database can't be brute-forced cheaply. Never store or log the password itself.
- Sessions: a random token given to the client; the DB stores only its SHA-256, so a stolen DB can't be used
  to log in either. Tokens expire after SESSION_DAYS and are deleted on logout.
- The FIRST account ever registered becomes user "local" and takes over the data from before accounts existed.

Alternatives: OAuth/OIDC "Sign in with Google" (no passwords to store at all), JWT (stateless, but can't be
revoked before it expires), a hosted auth service (Auth0, Clerk, Supabase Auth).
"""
import hashlib
import hmac
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta

from memory import _lock, conn
from userdata import LOCAL

SESSION_DAYS = 7
SCRYPT = {"n": 2**14, "r": 8, "p": 1}  # ~16 MB and tens of ms per hash: slow for attackers, fine for a login

with _lock:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                                          salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires TEXT NOT NULL);
    """)
    conn.commit()


class AuthError(Exception):
    pass


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def register(username: str, password: str) -> str:
    """Create an account; returns the new user id."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        raise AuthError("Username: 3-32 letters, digits, '.', '_' or '-'.")
    if len(password) < 8:
        raise AuthError("Password: at least 8 characters.")
    salt = secrets.token_bytes(16)
    with _lock:
        first = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        user_id = LOCAL if first else uuid.uuid4().hex[:12]  # first account owns the pre-accounts data
        try:
            conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (user_id, username, salt,
                         _hash_password(password, salt), datetime.now().isoformat(timespec="seconds")))
        except Exception as e:  # sqlite3.IntegrityError: username taken
            raise AuthError("That username is taken.") from e
        conn.commit()
    return user_id


MAX_FAILURES, LOCKOUT_SECONDS = 5, 300
_failures: dict[str, tuple[int, float]] = {}  # username -> (failed attempts, locked until); ponytail: per process


def login(username: str, password: str) -> tuple[str, str]:
    """Check the password; returns (session token, user id). Same error for 'no such user' and 'wrong password',
    so the login form can't be used to find out which usernames exist."""
    key = username.lower()
    count, locked_until = _failures.get(key, (0, 0.0))
    if time.time() < locked_until:  # brute-force protection: too many wrong passwords -> wait
        raise AuthError(f"Too many failed attempts. Try again in {int(locked_until - time.time()) + 1}s.")
    with _lock:
        row = conn.execute("SELECT id, salt, pw_hash FROM users WHERE username = ?", (username,)).fetchone()
    # hash even when the user doesn't exist, so the response time doesn't reveal it either
    salt, expected = (row[1], row[2]) if row else (b"0" * 16, b"")
    if not hmac.compare_digest(_hash_password(password, salt), expected) or not row:
        count += 1
        _failures[key] = (count, time.time() + LOCKOUT_SECONDS if count >= MAX_FAILURES else 0.0)
        raise AuthError("Wrong username or password.")
    _failures.pop(key, None)
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
    with _lock:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (_hash_token(token), row[0], expires))
        conn.commit()
    return token, row[0]


def user_for_token(token: str) -> tuple[str, str] | None:
    """(user id, username) for a valid, unexpired token; None otherwise."""
    with _lock:
        row = conn.execute("SELECT s.user_id, u.username, s.expires FROM sessions s JOIN users u ON u.id = s.user_id "
                           "WHERE s.token_hash = ?", (_hash_token(token),)).fetchone()
    if not row or row[2] < datetime.now().isoformat(timespec="seconds"):
        return None
    return row[0], row[1]


def logout(token: str) -> None:
    with _lock:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash_token(token),))
        conn.commit()
