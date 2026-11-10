"""SQLite persistence layer for conversation history and memory mirror.

This module is the single source of truth for structured records. It deliberately
has no knowledge of Markdown mirrors or of mem0 — those live in
``markdown_vault.py`` and ``history_service.py`` respectively, so each layer can
be reasoned about (and replaced) on its own.

Concurrency model: the backend writes from several threads (``asyncio.to_thread``
workers, WebSocket handlers), so every statement goes through one connection
guarded by a re-entrant lock. SQLite in WAL mode then handles the on-disk side.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
VALID_ROLES = {ROLE_USER, ROLE_ASSISTANT}


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SessionRecord:
    id: str
    persona_id: str
    title: str
    source: str
    created_at: float
    updated_at: float
    deleted: int = 0
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "title": self.title,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted": bool(self.deleted),
            "message_count": self.message_count,
        }


@dataclass
class MessageRecord:
    id: str
    session_id: str
    turn_id: str
    role: str
    content: str
    created_at: float
    edited: int = 0
    deleted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "edited": bool(self.edited),
            "deleted": bool(self.deleted),
        }


@dataclass
class MemoryRecord:
    id: str
    mem0_id: str
    owner_id: str
    user_id: str
    content: str
    created_at: float
    updated_at: float
    edited: int = 0
    deleted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mem0_id": self.mem0_id,
            "owner_id": self.owner_id,
            "user_id": self.user_id,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "edited": bool(self.edited),
            "deleted": bool(self.deleted),
        }


def _session_from_row(row: sqlite3.Row) -> SessionRecord:
    keys = row.keys()
    return SessionRecord(
        id=row["id"],
        persona_id=row["persona_id"],
        title=row["title"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted=row["deleted"],
        message_count=row["message_count"] if "message_count" in keys else 0,
    )


def _message_from_row(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        edited=row["edited"],
        deleted=row["deleted"],
    )


def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        mem0_id=row["mem0_id"] or "",
        owner_id=row["owner_id"],
        user_id=row["user_id"],
        content=row["content"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        edited=row["edited"],
        deleted=row["deleted"],
    )


class HistoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    # ------------------------------------------------------------------ schema

    def _migrate(self) -> None:
        with self._lock:
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if current >= SCHEMA_VERSION:
                return
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'voice',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_updated
                    ON sessions(deleted, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    edited INTEGER NOT NULL DEFAULT 0,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(turn_id);

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    mem0_id TEXT UNIQUE,
                    owner_id TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    edited INTEGER NOT NULL DEFAULT 0,
                    deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memories_owner
                    ON memories(owner_id, user_id);

                CREATE TABLE IF NOT EXISTS md_sync (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- sessions

    def ensure_session(self, session_id: str, persona_id: str, source: str = "voice") -> SessionRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is not None:
                return _session_from_row(row)
            now = _now()
            self._conn.execute(
                "INSERT INTO sessions(id, persona_id, title, source, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, persona_id or "", "", source, now, now),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            return _session_from_row(row)

    def set_session_persona(self, session_id: str, persona_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET persona_id=?, updated_at=? WHERE id=?",
                (persona_id or "", _now(), session_id),
            )
            self._conn.commit()

    def touch_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (_now(), session_id),
            )
            self._conn.commit()

    def list_sessions(self, include_deleted: bool = False, limit: int = 200) -> list[SessionRecord]:
        with self._lock:
            where = "" if include_deleted else "WHERE s.deleted=0"
            rows = self._conn.execute(
                f"""
                SELECT s.*, (
                    SELECT COUNT(*) FROM messages m
                    WHERE m.session_id = s.id AND m.deleted = 0
                ) AS message_count
                FROM sessions s
                {where}
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [_session_from_row(row) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT s.*, (
                    SELECT COUNT(*) FROM messages m
                    WHERE m.session_id = s.id AND m.deleted = 0
                ) AS message_count
                FROM sessions s WHERE s.id=?
                """,
                (session_id,),
            ).fetchone()
            return _session_from_row(row) if row is not None else None

    def set_session_title(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                (title, _now(), session_id),
            )
            self._conn.commit()

    def soft_delete_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET deleted=1, updated_at=? WHERE id=? AND deleted=0",
                (_now(), session_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def restore_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET deleted=0, updated_at=? WHERE id=? AND deleted=1",
                (_now(), session_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # ---------------------------------------------------------------- messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn_id: str = "",
    ) -> MessageRecord | None:
        value = str(content or "").strip()
        if not value:
            return None
        if role not in VALID_ROLES:
            raise ValueError(f"unsupported role: {role!r}")
        message_id = _new_id("msg")
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(id, session_id, turn_id, role, content, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (message_id, session_id, turn_id or "", role, value, now),
            )
            self._conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
            return _message_from_row(row)

    def list_messages(self, session_id: str, include_deleted: bool = False) -> list[MessageRecord]:
        with self._lock:
            where = "" if include_deleted else "AND deleted=0"
            rows = self._conn.execute(
                f"SELECT * FROM messages WHERE session_id=? {where} ORDER BY created_at ASC, id ASC",
                (session_id,),
            ).fetchall()
            return [_message_from_row(row) for row in rows]

    def get_message(self, message_id: str) -> MessageRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
            return _message_from_row(row) if row is not None else None

    def update_message(self, message_id: str, content: str) -> bool:
        value = str(content or "").strip()
        if not value:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE messages SET content=?, edited=1 WHERE id=? AND deleted=0",
                (value, message_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def soft_delete_message(self, message_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE messages SET deleted=1 WHERE id=? AND deleted=0",
                (message_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def restore_message(self, message_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE messages SET deleted=0 WHERE id=? AND deleted=1",
                (message_id,),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # ---------------------------------------------------------------- memories

    def upsert_memory(
        self,
        mem0_id: str,
        owner_id: str,
        user_id: str,
        content: str,
    ) -> MemoryRecord | None:
        value = str(content or "").strip()
        if not value:
            return None
        mem0_key = str(mem0_id or "").strip()
        now = _now()
        with self._lock:
            row = None
            if mem0_key:
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE mem0_id=?", (mem0_key,)
                ).fetchone()
            if row is None:
                memory_id = _new_id("mem")
                self._conn.execute(
                    "INSERT INTO memories(id, mem0_id, owner_id, user_id, content, "
                    "created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                    (memory_id, mem0_key or None, owner_id or "", user_id or "", value, now, now),
                )
                self._conn.commit()
                row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                return _memory_from_row(row)
            if row["content"] == value:
                return _memory_from_row(row)
            self._conn.execute(
                "UPDATE memories SET content=?, updated_at=? WHERE id=?",
                (value, now, row["id"]),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (row["id"],)).fetchone()
            return _memory_from_row(row)

    def list_memories(
        self,
        owner_id: str | None = None,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_deleted:
            clauses.append("deleted=0")
        if owner_id is not None:
            clauses.append("owner_id=?")
            params.append(owner_id)
        if user_id is not None:
            clauses.append("user_id=?")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories {where} ORDER BY created_at ASC", params
            ).fetchall()
            return [_memory_from_row(row) for row in rows]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            return _memory_from_row(row) if row is not None else None

    def find_memory_by_mem0_id(self, mem0_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE mem0_id=?", (str(mem0_id or ""),)
            ).fetchone()
            return _memory_from_row(row) if row is not None else None

    def link_memory_mem0_id(self, memory_id: str, mem0_id: str) -> bool:
        """Attach a mem0 id to a locally-created memory row (after pushing it)."""
        key = str(mem0_id or "").strip()
        if not key:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET mem0_id=?, updated_at=? WHERE id=?",
                (key, _now(), memory_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def update_memory(self, memory_id: str, content: str) -> bool:
        value = str(content or "").strip()
        if not value:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET content=?, edited=1, updated_at=? WHERE id=? AND deleted=0",
                (value, _now(), memory_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def soft_delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET deleted=1, updated_at=? WHERE id=? AND deleted=0",
                (_now(), memory_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def soft_delete_memory_by_mem0_id(self, mem0_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET deleted=1, updated_at=? WHERE mem0_id=? AND deleted=0",
                (_now(), str(mem0_id or "")),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def restore_memory(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE memories SET deleted=0, updated_at=? WHERE id=? AND deleted=1",
                (_now(), memory_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # ----------------------------------------------------------- md sync state

    def get_sync(self, path: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM md_sync WHERE path=?", (path,)).fetchone()
            if row is None:
                return None
            return {"path": row["path"], "content_hash": row["content_hash"], "updated_at": row["updated_at"]}

    def set_sync(self, path: str, content_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO md_sync(path, content_hash, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET content_hash=excluded.content_hash, "
                "updated_at=excluded.updated_at",
                (path, content_hash, _now()),
            )
            self._conn.commit()

    def clear_sync(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM md_sync WHERE path=?", (path,))
            self._conn.commit()

    def list_sync_paths(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT path FROM md_sync").fetchall()
            return [row["path"] for row in rows]

    # -------------------------------------------------------------- diagnostics

    def is_connected(self) -> bool:
        """True when the SQLite connection answers a trivial query."""
        with self._lock:
            try:
                self._conn.execute("SELECT 1").fetchone()
                return True
            except sqlite3.Error:
                return False

    def stats(self) -> dict[str, int]:
        """Row counts per table, split into live and soft-deleted."""
        with self._lock:
            return {
                "sessions": self._count("sessions", 0),
                "sessions_deleted": self._count("sessions", 1),
                "messages": self._count("messages", 0),
                "messages_deleted": self._count("messages", 1),
                "memories": self._count("memories", 0),
                "memories_deleted": self._count("memories", 1),
            }

    def _count(self, table: str, deleted: int) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE deleted=?", (deleted,)
        ).fetchone()
        return int(row["n"])
