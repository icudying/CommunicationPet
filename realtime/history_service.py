"""Orchestration layer tying SQLite, the Markdown mirror and mem0 together.

``HistoryService`` is the only thing the backend routes and the WebSocket runtime
talk to. It owns the ordering that matters:

* a turn is written to SQLite first, then mirrored to Markdown;
* a user edit made in the Markdown is reconciled into SQLite, then pushed on to
  mem0 so semantic recall stays in sync with what the user sees;
* deleting a memory in the UI/vault removes it from both SQLite and mem0.

mem0 is optional: when the memory module is disabled the service still persists
conversations and mirrors them, it just has no semantic backend to push to.
"""

from __future__ import annotations

import logging
from typing import Any

from .history_store import HistoryStore, MessageRecord, MemoryRecord, ROLE_ASSISTANT, ROLE_USER
from .markdown_vault import MarkdownVault, ReconcileReport

logger = logging.getLogger(__name__)

_TITLE_MAX_CHARS = 24


def _make_title(text: str) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= _TITLE_MAX_CHARS:
        return collapsed
    return collapsed[:_TITLE_MAX_CHARS] + "…"


def _extract_mem0_id(result: Any) -> str:
    """Pull the first stored memory id out of a mem0 ``add`` result."""
    if not isinstance(result, dict):
        return ""
    results = result.get("results")
    for item in results if isinstance(results, list) else []:
        if isinstance(item, dict) and str(item.get("id") or "").strip():
            return str(item["id"]).strip()
    return ""


class HistoryService:
    def __init__(
        self,
        store: HistoryStore,
        vault: MarkdownVault,
        memory: Any | None = None,
        owner_id: str = "default",
        user_id: str = "local_user",
    ):
        self.store = store
        self.vault = vault
        self.memory = memory
        self.owner_id = owner_id
        self.user_id = user_id

    # ------------------------------------------------------------ turn capture

    def ensure_session(self, session_id: str, persona_id: str) -> None:
        self.store.ensure_session(session_id, persona_id)

    def set_session_persona(self, session_id: str, persona_id: str) -> None:
        self.store.set_session_persona(session_id, persona_id)
        self.vault.export_session(session_id)

    def record_turn(
        self,
        session_id: str,
        persona_id: str,
        turn_id: str,
        user_text: str,
        assistant_text: str,
    ) -> dict[str, Any]:
        self.store.ensure_session(session_id, persona_id)
        if persona_id:
            self.store.set_session_persona(session_id, persona_id)
        user_message = self.store.add_message(session_id, ROLE_USER, user_text, turn_id)
        assistant_message = self.store.add_message(session_id, ROLE_ASSISTANT, assistant_text, turn_id)

        session = self.store.get_session(session_id)
        if session is not None and not session.title and user_text:
            self.store.set_session_title(session_id, _make_title(user_text))

        self.vault.export_session(session_id)
        return {
            "session_id": session_id,
            "user_message_id": user_message.id if user_message else None,
            "assistant_message_id": assistant_message.id if assistant_message else None,
        }

    # ------------------------------------------------------------ memory mirror

    def refresh_memories(self) -> int:
        """Pull mem0's memories into SQLite and rewrite the Markdown mirror."""
        if self.memory is not None and getattr(self.memory, "available", False):
            items = self.memory.list_memories()
            for item in items:
                self.store.upsert_memory(
                    item.get("id", ""),
                    self.owner_id,
                    self.user_id,
                    item.get("memory", ""),
                )
            self.vault.export_memories(self.owner_id, self.user_id)
            return len(items)
        # Memory backend off: only refresh a mirror that already exists so we do
        # not create a misleading empty memory file on every startup.
        if self.store.list_memories(
            owner_id=self.owner_id, user_id=self.user_id, include_deleted=True
        ):
            self.vault.export_memories(self.owner_id, self.user_id)
        return 0

    def memory_status(self) -> dict[str, Any]:
        if self.memory is None:
            return {"enabled": False, "available": False}
        return self.memory.status_dict()

    # ---------------------------------------------------------------- reconcile

    def sync(self) -> dict[str, Any]:
        """Reconcile the Markdown vault into SQLite, then apply mem0 side effects."""
        report = self.vault.reconcile_all()
        self._apply_memory_side_effects(report)

        # Re-export session mirrors that the reconcile altered so the file matches
        # the database again (e.g. a deleted block stays deleted).
        if report.updated_messages or report.deleted_messages:
            for session in self.store.list_sessions():
                self.vault.export_session(session.id)

        if report.created_memory_ids or report.updated_memory_ids:
            self.vault.export_memories(self.owner_id, self.user_id)

        if report.changed:
            logger.info("[history] sync applied: %s", report.to_dict())
        return report.to_dict()

    def _apply_memory_side_effects(self, report: ReconcileReport) -> None:
        if self.memory is None or not getattr(self.memory, "available", False):
            return
        for memory_id in report.deleted_memory_ids:
            record = self.store.get_memory(memory_id)
            if record is not None and record.mem0_id:
                self.memory.delete_memory(record.mem0_id)
        for memory_id in report.updated_memory_ids:
            record = self.store.get_memory(memory_id)
            if record is None:
                continue
            if record.mem0_id:
                self.memory.update_memory(record.mem0_id, record.content)
            else:
                self._push_new_memory(record)
        for memory_id in report.created_memory_ids + report.restored_memory_ids:
            record = self.store.get_memory(memory_id)
            if record is not None and not record.mem0_id:
                self._push_new_memory(record)

    def _push_new_memory(self, record: MemoryRecord) -> None:
        if self.memory is None or not getattr(self.memory, "available", False):
            return
        result = self.memory.add_memory(
            record.content,
            metadata={"agent_id": self.owner_id, "source": "vault"},
        )
        mem0_id = _extract_mem0_id(result)
        if mem0_id:
            self.store.link_memory_mem0_id(record.id, mem0_id)

    # -------------------------------------------------------------- sessions UI

    def list_sessions(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.store.list_sessions(include_deleted=include_deleted)]

    def get_session_detail(self, session_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        session = self.store.get_session(session_id)
        if session is None:
            return None
        messages = [m.to_dict() for m in self.store.list_messages(session_id, include_deleted=include_deleted)]
        return {"session": session.to_dict(), "messages": messages}

    def edit_message(self, message_id: str, content: str) -> bool:
        if not self.store.update_message(message_id, content):
            return False
        message = self.store.get_message(message_id)
        if message is not None:
            self.vault.export_session(message.session_id)
        return True

    def delete_message(self, message_id: str) -> bool:
        message = self.store.get_message(message_id)
        if not self.store.soft_delete_message(message_id):
            return False
        if message is not None:
            self.vault.export_session(message.session_id)
        return True

    def restore_message(self, message_id: str) -> bool:
        message = self.store.get_message(message_id)
        if not self.store.restore_message(message_id):
            return False
        if message is not None:
            self.vault.export_session(message.session_id)
        return True

    def delete_session(self, session_id: str) -> bool:
        if not self.store.soft_delete_session(session_id):
            return False
        self.vault.remove_session_file(session_id)
        return True

    def restore_session(self, session_id: str) -> bool:
        if not self.store.restore_session(session_id):
            return False
        self.vault.export_session(session_id)
        return True

    # -------------------------------------------------------------- memories UI

    def list_memories(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        records = self.store.list_memories(
            owner_id=self.owner_id,
            user_id=self.user_id,
            include_deleted=include_deleted,
        )
        return [record.to_dict() for record in records]

    def add_memory(self, content: str) -> dict[str, Any] | None:
        record = self.store.upsert_memory("", self.owner_id, self.user_id, content)
        if record is None:
            return None
        self._push_new_memory(record)
        self.vault.export_memories(self.owner_id, self.user_id)
        refreshed = self.store.get_memory(record.id)
        return (refreshed or record).to_dict()

    def edit_memory(self, memory_id: str, content: str) -> bool:
        if not self.store.update_memory(memory_id, content):
            return False
        record = self.store.get_memory(memory_id)
        if record is not None:
            if record.mem0_id and self.memory is not None and getattr(self.memory, "available", False):
                self.memory.update_memory(record.mem0_id, record.content)
            else:
                self._push_new_memory(record)
        self.vault.export_memories(self.owner_id, self.user_id)
        return True

    def delete_memory(self, memory_id: str) -> bool:
        record = self.store.get_memory(memory_id)
        if not self.store.soft_delete_memory(memory_id):
            return False
        if (
            record is not None
            and record.mem0_id
            and self.memory is not None
            and getattr(self.memory, "available", False)
        ):
            self.memory.delete_memory(record.mem0_id)
        self.vault.export_memories(self.owner_id, self.user_id)
        return True

    def restore_memory(self, memory_id: str) -> bool:
        if not self.store.restore_memory(memory_id):
            return False
        record = self.store.get_memory(memory_id)
        if record is not None and not record.mem0_id:
            self._push_new_memory(record)
        self.vault.export_memories(self.owner_id, self.user_id)
        return True

    # ------------------------------------------------------------------- info

    def vault_info(self) -> dict[str, Any]:
        return {
            "vault_root": str(self.vault.root),
            "owner_id": self.owner_id,
            "user_id": self.user_id,
        }

    def database_info(self) -> dict[str, Any]:
        """Connection status, on-disk size and row counts for the SQLite store."""
        info: dict[str, Any] = {
            "db_path": str(self.store.db_path),
            "connected": False,
            "db_exists": False,
            "db_size_bytes": 0,
            "error": None,
            "sessions": 0,
            "sessions_deleted": 0,
            "messages": 0,
            "messages_deleted": 0,
            "memories": 0,
            "memories_deleted": 0,
        }
        try:
            info["connected"] = self.store.is_connected()
            info.update(self.store.stats())
        except Exception as exc:
            info["error"] = str(exc)
            return info
        path = self.store.db_path
        try:
            if path.exists():
                info["db_exists"] = True
                total = path.stat().st_size
                # WAL mode keeps recent writes in sidecar files; count them so the
                # reported size matches what actually occupies disk.
                for suffix in ("-wal", "-shm"):
                    sidecar = path.with_name(path.name + suffix)
                    if sidecar.exists():
                        total += sidecar.stat().st_size
                info["db_size_bytes"] = total
        except OSError as exc:
            info["error"] = str(exc)
        return info


__all__ = ["HistoryService", "MessageRecord"]
