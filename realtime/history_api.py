"""HTTP routes for browsing and controlling conversation history and memory.

Kept in its own module (mounted with ``include_router``) so the existing
WebSocket/realtime routes in ``backend_main.py`` stay untouched.

Everything here is blocking DB/file work, so handlers hand off to a worker
thread via ``asyncio.to_thread`` and never block the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from .history_service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])


def _service(request: Request) -> HistoryService:
    runtime = getattr(request.app.state, "realtime_runtime", None)
    service = getattr(runtime, "history", None) if runtime is not None else None
    if service is None:
        raise HTTPException(status_code=503, detail="history service is unavailable")
    return service


def _content(payload: dict[str, Any] | None) -> str:
    value = str((payload or {}).get("content") or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="content is required")
    return value


@router.get("/info")
async def history_info(request: Request) -> dict[str, Any]:
    service = _service(request)
    info = await asyncio.to_thread(service.vault_info)
    info["database"] = await asyncio.to_thread(service.database_info)
    info["memory"] = await asyncio.to_thread(service.memory_status)
    return info


@router.post("/sync")
async def history_sync(request: Request) -> dict[str, Any]:
    service = _service(request)
    report = await asyncio.to_thread(service.sync)
    return {"ok": True, "report": report}


# ------------------------------------------------------------------- sessions


@router.get("/sessions")
async def list_sessions(
    request: Request,
    include_deleted: bool = Query(False),
) -> dict[str, Any]:
    service = _service(request)
    sessions = await asyncio.to_thread(service.list_sessions, include_deleted)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(
    request: Request,
    session_id: str,
    include_deleted: bool = Query(False),
) -> dict[str, Any]:
    service = _service(request)
    detail = await asyncio.to_thread(service.get_session_detail, session_id, include_deleted)
    if detail is None:
        raise HTTPException(status_code=404, detail="session not found")
    return detail


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.delete_session, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@router.post("/sessions/{session_id}/restore")
async def restore_session(request: Request, session_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.restore_session, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


# ------------------------------------------------------------------- messages


@router.patch("/messages/{message_id}")
async def edit_message(
    request: Request,
    message_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service = _service(request)
    content = _content(payload)
    if not await asyncio.to_thread(service.edit_message, message_id, content):
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True}


@router.delete("/messages/{message_id}")
async def delete_message(request: Request, message_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.delete_message, message_id):
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True}


@router.post("/messages/{message_id}/restore")
async def restore_message(request: Request, message_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.restore_message, message_id):
        raise HTTPException(status_code=404, detail="message not found")
    return {"ok": True}


# ------------------------------------------------------------------- memories


@router.get("/memories")
async def list_memories(
    request: Request,
    include_deleted: bool = Query(False),
) -> dict[str, Any]:
    service = _service(request)
    memories = await asyncio.to_thread(service.list_memories, include_deleted)
    return {"memories": memories}


@router.post("/memories/refresh")
async def refresh_memories(request: Request) -> dict[str, Any]:
    service = _service(request)
    count = await asyncio.to_thread(service.refresh_memories)
    return {"ok": True, "count": count}


@router.post("/memories")
async def add_memory(request: Request, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    service = _service(request)
    content = _content(payload)
    record = await asyncio.to_thread(service.add_memory, content)
    if record is None:
        raise HTTPException(status_code=400, detail="could not add memory")
    return {"ok": True, "memory": record}


@router.patch("/memories/{memory_id}")
async def edit_memory(
    request: Request,
    memory_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service = _service(request)
    content = _content(payload)
    if not await asyncio.to_thread(service.edit_memory, memory_id, content):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}


@router.delete("/memories/{memory_id}")
async def delete_memory(request: Request, memory_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.delete_memory, memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}


@router.post("/memories/{memory_id}/restore")
async def restore_memory(request: Request, memory_id: str) -> dict[str, Any]:
    service = _service(request)
    if not await asyncio.to_thread(service.restore_memory, memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}
