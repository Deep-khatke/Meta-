"""
RoboForge User Authentication & Persistence Database
Uses SQLite with PBKDF2-HMAC-SHA256 salted password hashing and secure session tokens.
"""

import sqlite3
import hashlib
import secrets
import time
import os
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("roboforge.db")

DB_PATH = os.environ.get("ROBOFORGE_DB_PATH", os.path.join(os.path.dirname(__file__), "roboforge.db"))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables for persistent users and sessions."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT,
                picture TEXT,
                provider TEXT NOT NULL DEFAULT 'email',
                created_at REAL NOT NULL,
                last_login_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                thought_steps TEXT,
                actions_executed TEXT,
                response_data TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_user_time ON chat_messages(user_id, created_at)")
        conn.commit()
    logger.info("RoboForge database initialized at %s", DB_PATH)
    try:
        from backend.billing.db import init_billing_db
        init_billing_db()
    except Exception as e:
        logger.warning("Billing tables initialization error: %s", e)


def hash_password(password: str) -> str:
    """Generates a salted PBKDF2-HMAC-SHA256 hash."""
    salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return f"{salt}:{pw_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verifies a password against the stored salt:hash string."""
    try:
        salt, expected_hash = stored_hash.split(":", 1)
        actual_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
        return secrets.compare_digest(actual_hash, expected_hash)
    except Exception as e:
        logger.warning("Error verifying password: %s", e)
        return False


def create_session(user_id: str, duration_days: int = 30) -> str:
    """Creates a persistent session token for the user."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    expires_at = now + (duration_days * 86400)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now, expires_at),
        )
        conn.commit()
    return token


def get_user_by_session(token: str) -> Optional[Dict[str, Any]]:
    """Retrieves authenticated user from active session token."""
    if not token:
        return None
    now = time.time()
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, u.name, u.picture, u.provider, u.created_at, u.last_login_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now),
        ).fetchone()
        if row:
            return dict(row)
    return None


def delete_session(token: str) -> None:
    """Deletes an active session (logout)."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Finds user by case-insensitive email."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)).fetchone()
        if row:
            return dict(row)
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Finds public user profile by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, email, name, picture, provider, created_at, last_login_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row:
            return dict(row)
    return None


def register_user(name: str, email: str, password: str) -> Dict[str, Any]:
    """Registers a new user with email and password into the persistent database."""
    name_clean = name.strip()
    email_clean = email.strip().lower()
    if not name_clean:
        raise ValueError("Name is required.")
    if not email_clean or "@" not in email_clean:
        raise ValueError("A valid email address is required.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    if get_user_by_email(email_clean):
        raise ValueError("An account with this email already exists. Please log in.")

    user_id = f"usr_{secrets.token_hex(8)}"
    now = time.time()
    pw_hash = hash_password(password)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, name, password_hash, picture, provider, created_at, last_login_at)
            VALUES (?, ?, ?, ?, NULL, 'email', ?, ?)
            """,
            (user_id, email_clean, name_clean, pw_hash, now, now),
        )
        conn.commit()

    user = get_user_by_id(user_id)
    token = create_session(user_id)
    return {"user": user, "token": token}


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticates email and password against the database."""
    email_clean = email.strip().lower()
    user_row = get_user_by_email(email_clean)
    if not user_row or not user_row.get("password_hash"):
        raise ValueError("Invalid email or password.")

    if not verify_password(password, user_row["password_hash"]):
        raise ValueError("Invalid email or password.")

    now = time.time()
    with get_db() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_row["id"]))
        conn.commit()

    user = get_user_by_id(user_row["id"])
    token = create_session(user_row["id"])
    return {"user": user, "token": token}


def upsert_google_user(google_sub: str, email: str, name: str, picture: Optional[str] = None) -> Dict[str, Any]:
    """Inserts or updates a Google OAuth user into the database and returns session."""
    email_clean = email.strip().lower()
    now = time.time()
    user_row = get_user_by_email(email_clean)

    with get_db() as conn:
        if user_row:
            user_id = user_row["id"]
            conn.execute(
                """
                UPDATE users
                SET name = COALESCE(?, name), picture = COALESCE(?, picture), last_login_at = ?, provider = 'google'
                WHERE id = ?
                """,
                (name, picture, now, user_id),
            )
        else:
            user_id = f"goog_{google_sub}"
            conn.execute(
                """
                INSERT INTO users (id, email, name, password_hash, picture, provider, created_at, last_login_at)
                VALUES (?, ?, ?, NULL, ?, 'google', ?, ?)
                """,
                (user_id, email_clean, name, picture, now, now),
            )
        conn.commit()

    user = get_user_by_id(user_id)
    token = create_session(user_id)
    return {"user": user, "token": token}


def save_chat_message(
    user_id: str,
    role: str,
    content: str,
    model: Optional[str] = None,
    thought_steps: Optional[str] = None,
    actions_executed: Optional[str] = None,
    response_data: Optional[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Saves a user or assistant chat message associated with a user in the database."""
    msg_id = f"msg_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
    now = time.time()
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (
                id, user_id, session_id, role, content, model, thought_steps, actions_executed, response_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, user_id, session_id, role, content, model, thought_steps, actions_executed, response_data, now)
        )
        conn.commit()
    return {
        "id": msg_id,
        "user_id": user_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "model": model,
        "created_at": now
    }


def get_user_chat_history(user_id: str, limit: int = 50, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves chat messages for a specific user from the database."""
    with get_db() as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT id, user_id, session_id, role, content, model, thought_steps, actions_executed, response_data, created_at
                FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (user_id, session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, session_id, role, content, model, thought_steps, actions_executed, response_data, created_at
                FROM chat_messages
                WHERE user_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()

        results = []
        for r in rows:
            row_dict = dict(r)
            if row_dict.get("response_data"):
                try:
                    row_dict["response"] = json.loads(row_dict["response_data"])
                except Exception:
                    row_dict["response"] = None
            if row_dict.get("actions_executed"):
                try:
                    row_dict["actions_executed"] = json.loads(row_dict["actions_executed"])
                except Exception:
                    pass
            if row_dict.get("thought_steps"):
                try:
                    row_dict["thought_steps"] = json.loads(row_dict["thought_steps"])
                except Exception:
                    pass
            results.append(row_dict)
        return results


def clear_user_chat_history(user_id: str, session_id: Optional[str] = None) -> int:
    """Deletes chat messages for a specific user from the database."""
    with get_db() as conn:
        if session_id:
            cur = conn.execute("DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?", (user_id, session_id))
        else:
            cur = conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount


# Automatically create tables on import
init_db()
