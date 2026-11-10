"""Markdown mirror of the conversation history and long-term memory.

The SQLite store (``history_store.py``) stays the source of truth; this module
renders it to plain ``.md`` files so a human can read, edit and delete records in
any editor (including Obsidian), then reconciles those edits back.

Reconciliation is one-directional and explicit: SQLite wins unless the file on
disk differs from what we last wrote (tracked via a content hash in ``md_sync``).
That means hand edits are never silently overwritten, and a file/line/block the
user removes really disappears from the database.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .history_store import HistoryStore

logger = logging.getLogger(__name__)

SESSIONS_DIR = "sessions"
MEMORIES_DIR = "memories"

_HEADER_RE = re.compile(r"^##\s+(user|assistant)\b(.*)$")
_ID_MARKER_RE = re.compile(r"<!--\s*id:\s*([A-Za-z0-9_\-]+)\s*-->")
_MEM_LINE_MARKER_RE = re.compile(r"\s*<!--\s*id:\s*([A-Za-z0-9_\-]+)\s*-->\s*$")

_SESSION_HINT = (
    "<!-- 修改某条消息的文字即可更新它；删除整条（连同 ## 标题行）即可删除该条；"
    "删除本文件即可删除整个会话。 -->"
)
_MEMORY_HINT = (
    "<!-- 每行一条记忆：修改文字即更新；删除整行即让 AI 忘记；新增一行即新增一条记忆。 -->"
)


@dataclass
class ReconcileReport:
    deleted_session_ids: list[str] = field(default_factory=list)
    deleted_memory_ids: list[str] = field(default_factory=list)
    updated_memory_ids: list[str] = field(default_factory=list)
    created_memory_ids: list[str] = field(default_factory=list)
    restored_memory_ids: list[str] = field(default_factory=list)
    updated_messages: int = 0
    deleted_messages: int = 0

    def to_dict(self) -> dict:
        return {
            "deleted_session_ids": self.deleted_session_ids,
            "deleted_memory_ids": self.deleted_memory_ids,
            "updated_memory_ids": self.updated_memory_ids,
            "created_memory_ids": self.created_memory_ids,
            "restored_memory_ids": self.restored_memory_ids,
            "updated_messages": self.updated_messages,
            "deleted_messages": self.deleted_messages,
        }

    @property
    def changed(self) -> bool:
        return bool(
            self.deleted_session_ids
            or self.deleted_memory_ids
            or self.updated_memory_ids
            or self.created_memory_ids
            or self.restored_memory_ids
            or self.updated_messages
            or self.deleted_messages
        )


def _fmt_time(value: float) -> str:
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _safe_component(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."}:
        return fallback
    if any(sep in text for sep in ("/", "\\", os.sep)):
        return fallback
    # Windows forbids a handful of characters in file names; replace them so a
    # persona/user id can never make the vault path unwritable.
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', "_", text)
    return cleaned or fallback


class MarkdownVault:
    def __init__(self, root: str | Path, store: HistoryStore):
        self.root = Path(root)
        self.store = store
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths

    def session_path(self, session_id: str) -> Path:
        return self.root / SESSIONS_DIR / f"{_safe_component(session_id, 'session')}.md"

    def memory_path(self, owner_id: str, user_id: str) -> Path:
        return (
            self.root
            / MEMORIES_DIR
            / _safe_component(owner_id, "default")
            / f"{_safe_component(user_id, 'user')}.md"
        )

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # --------------------------------------------------------------- rendering

    def _render_session(self, session_id: str) -> str:
        session = self.store.get_session(session_id)
        messages = self.store.list_messages(session_id)
        lines: list[str] = []
        if session is None:
            return ""
        lines.append(f"# 会话 {session.id}")
        lines.append("")
        lines.append(f"- 人格: {session.persona_id or '-'}")
        lines.append(f"- 开始: {_fmt_time(session.created_at)}")
        lines.append(f"- 消息数: {len(messages)}")
        lines.append("")
        lines.append(_SESSION_HINT)
        for message in messages:
            lines.append("")
            lines.append(f"## {message.role} · {_fmt_time(message.created_at)}")
            lines.append(f"<!-- id: {message.id} -->")
            lines.append(message.content)
        lines.append("")
        return "\n".join(lines)

    def _render_memories(self, owner_id: str, user_id: str) -> str:
        memories = self.store.list_memories(owner_id=owner_id, user_id=user_id)
        lines: list[str] = []
        lines.append(f"# 长期记忆 · {owner_id or '-'} · {user_id or '-'}")
        lines.append("")
        lines.append(_MEMORY_HINT)
        lines.append("")
        for memory in memories:
            lines.append(f"- {memory.content} <!-- id: {memory.id} -->")
        lines.append("")
        return "\n".join(lines)

    def _write_file(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        self.store.set_sync(self._rel(path), _hash_text(text))

    # ----------------------------------------------------------------- exports

    def export_session(self, session_id: str) -> None:
        path = self.session_path(session_id)
        self._reconcile_path_if_touched(path)
        text = self._render_session(session_id)
        if not text:
            return
        self._write_file(path, text)

    def export_memories(self, owner_id: str, user_id: str) -> None:
        path = self.memory_path(owner_id, user_id)
        self._reconcile_path_if_touched(path)
        text = self._render_memories(owner_id, user_id)
        self._write_file(path, text)

    def remove_session_file(self, session_id: str) -> None:
        path = self.session_path(session_id)
        self._unlink(path)

    def remove_memory_file(self, owner_id: str, user_id: str) -> None:
        path = self.memory_path(owner_id, user_id)
        self._unlink(path)

    def _unlink(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("[vault] failed to remove %s: %s", path, exc)
        self.store.clear_sync(self._rel(path))

    # ------------------------------------------------------------- reconcile

    def reconcile_all(self) -> ReconcileReport:
        report = ReconcileReport()

        # 1) Files we previously wrote but that no longer exist → user deleted them.
        for rel in self.store.list_sync_paths():
            path = self.root / rel
            if not path.exists():
                self._apply_file_deletion(rel, report)

        # 2) Existing managed files: reconcile when the content differs from what
        #    we wrote, and import memory files we have never written before. Only
        #    the two managed subfolders are scanned, so pointing the vault at a
        #    large Obsidian folder never triggers a full-vault read.
        for base in (self.root / SESSIONS_DIR, self.root / MEMORIES_DIR):
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.md")):
                rel = self._rel(path)
                state = self.store.get_sync(rel)
                text = _read_text(path)
                if text is None:
                    continue
                if state is not None and state["content_hash"] == _hash_text(text):
                    continue
                self._reconcile_file(path, text, report, known=state is not None)

        return report

    def _reconcile_path_if_touched(self, path: Path) -> None:
        state = self.store.get_sync(self._rel(path))
        if state is None or not path.exists():
            return
        text = _read_text(path)
        if text is None or state["content_hash"] == _hash_text(text):
            return
        report = ReconcileReport()
        self._reconcile_file(path, text, report, known=True)

    def _apply_file_deletion(self, rel: str, report: ReconcileReport) -> None:
        parts = rel.split("/")
        if len(parts) == 2 and parts[0] == SESSIONS_DIR:
            session_id = Path(parts[1]).stem
            if self.store.soft_delete_session(session_id):
                report.deleted_session_ids.append(session_id)
        elif len(parts) == 3 and parts[0] == MEMORIES_DIR:
            owner_id = parts[1]
            user_id = Path(parts[2]).stem
            for memory in self.store.list_memories(owner_id=owner_id, user_id=user_id):
                if self.store.soft_delete_memory(memory.id):
                    report.deleted_memory_ids.append(memory.id)
        self.store.clear_sync(rel)

    def _reconcile_file(
        self,
        path: Path,
        text: str,
        report: ReconcileReport,
        *,
        known: bool,
    ) -> None:
        parts = path.relative_to(self.root).parts
        if len(parts) == 2 and parts[0] == SESSIONS_DIR:
            self._reconcile_session_file(Path(parts[1]).stem, text, report)
        elif len(parts) == 3 and parts[0] == MEMORIES_DIR:
            self._reconcile_memory_file(parts[1], Path(parts[2]).stem, text, report, known=known)
        else:
            # Not a file this vault manages; leave it alone and stop tracking it.
            self.store.clear_sync(self._rel(path))
            return
        self.store.set_sync(self._rel(path), _hash_text(text))

    def _reconcile_session_file(self, session_id: str, text: str, report: ReconcileReport) -> None:
        if self.store.get_session(session_id) is None:
            # An unknown session file (user-authored) cannot be attributed to a
            # session row, so it is left untouched.
            return
        parsed: dict[str, str] = {}
        for block_id, _role, content in _parse_session_blocks(text):
            if block_id:
                parsed[block_id] = content
        for message in self.store.list_messages(session_id):
            if message.id not in parsed:
                if self.store.soft_delete_message(message.id):
                    report.deleted_messages += 1
                continue
            new_content = parsed[message.id]
            if new_content and new_content != message.content:
                if self.store.update_message(message.id, new_content):
                    report.updated_messages += 1

    def _reconcile_memory_file(
        self,
        owner_id: str,
        user_id: str,
        text: str,
        report: ReconcileReport,
        *,
        known: bool,
    ) -> None:
        entries = _parse_memory_lines(text)
        seen_ids: set[str] = set()
        for block_id, content in entries:
            if block_id:
                record = self.store.get_memory(block_id)
                if record is None:
                    continue
                seen_ids.add(block_id)
                if record.deleted:
                    if self.store.restore_memory(block_id):
                        report.restored_memory_ids.append(block_id)
                    continue
                if content and content != record.content:
                    if self.store.update_memory(block_id, content):
                        report.updated_memory_ids.append(block_id)
            else:
                created = self.store.upsert_memory("", owner_id, user_id, content)
                if created is not None:
                    report.created_memory_ids.append(created.id)
                    # Track it so the "missing from file" sweep below does not
                    # immediately soft-delete the row we just created.
                    seen_ids.add(created.id)

        if known:
            for memory in self.store.list_memories(owner_id=owner_id, user_id=user_id):
                if memory.id in seen_ids:
                    continue
                if self.store.soft_delete_memory(memory.id):
                    report.deleted_memory_ids.append(memory.id)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("[vault] cannot read %s: %s", path, exc)
        return None


def _parse_session_blocks(text: str) -> list[tuple[str | None, str, str]]:
    """Return ``(message_id, role, content)`` for every ``## role`` block."""
    lines = text.splitlines()
    blocks: list[tuple[str | None, str, str]] = []
    index = 0
    while index < len(lines):
        header = _HEADER_RE.match(lines[index])
        if header is None:
            index += 1
            continue
        role = header.group(1)
        cursor = index + 1
        block_id: str | None = None
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines):
            marker = _ID_MARKER_RE.fullmatch(lines[cursor].strip())
            if marker is not None:
                block_id = marker.group(1)
                cursor += 1
        content_lines: list[str] = []
        while cursor < len(lines) and _HEADER_RE.match(lines[cursor]) is None:
            content_lines.append(lines[cursor])
            cursor += 1
        content = "\n".join(content_lines).strip()
        if content:
            blocks.append((block_id, role, content))
        index = cursor
    return blocks


def _parse_memory_lines(text: str) -> list[tuple[str | None, str]]:
    entries: list[tuple[str | None, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].strip()
        marker = _MEM_LINE_MARKER_RE.search(body)
        if marker is not None:
            block_id: str | None = marker.group(1)
            content = body[: marker.start()].strip()
        else:
            block_id = None
            content = body
        if content:
            entries.append((block_id, content))
    return entries
