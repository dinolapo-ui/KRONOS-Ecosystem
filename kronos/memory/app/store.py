"""Session store: ordered conversation history in SQLite (WAL)."""
import sqlite3
import threading
import time


class SessionStore:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS messages(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       session_id TEXT NOT NULL,
                       role TEXT NOT NULL,
                       agent TEXT,
                       content TEXT NOT NULL,
                       ts REAL NOT NULL)"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )
            self._conn.commit()

    def add(self, session_id: str, role: str, content: str, agent: str | None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages(session_id, role, agent, content, ts) VALUES (?,?,?,?,?)",
                (session_id, role, agent, content, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def history(self, session_id: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, agent, content, ts FROM messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r, "agent": a, "content": c, "ts": t} for r, a, c, t in reversed(rows)]

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
