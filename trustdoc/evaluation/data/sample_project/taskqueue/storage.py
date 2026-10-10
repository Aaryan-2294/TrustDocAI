"""Durable task storage on SQLite."""

from __future__ import annotations

import json
import sqlite3
import time


class SqliteBackend:
    """Stores tasks in a SQLite database so they survive restarts."""

    def __init__(self, db_path: str = "tasks.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT, "
            "priority INTEGER, status TEXT DEFAULT 'pending', finished_at REAL)"
        )

    def save_task(self, task_id: str, payload: dict, priority: int = 0) -> None:
        """Insert or replace a pending task row."""
        self._conn.execute(
            "INSERT OR REPLACE INTO tasks (id, payload, priority) VALUES (?, ?, ?)",
            (task_id, json.dumps(payload), priority),
        )
        self._conn.commit()

    def fetch_pending(self, limit: int = 100) -> list[dict]:
        """Return up to ``limit`` pending tasks ordered by priority."""
        rows = self._conn.execute(
            "SELECT id, payload FROM tasks WHERE status = 'pending' ORDER BY priority LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "payload": json.loads(r[1])} for r in rows]

    def compact(self, vacuum: bool = False) -> int:
        """Delete completed tasks older than seven days; optionally VACUUM the file to reclaim disk space."""
        cutoff = time.time() - 7 * 86400
        cur = self._conn.execute(
            "DELETE FROM tasks WHERE status = 'done' AND finished_at < ?", (cutoff,)
        )
        self._conn.commit()
        if vacuum:
            self._conn.execute("VACUUM")
        return cur.rowcount
