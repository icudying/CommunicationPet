from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .asr_service import ASRRuntimeConfig, FunAsrService
from .history_api import router as history_router
from .history_service import HistoryService
from .history_store import HistoryStore
from .llm_client import (
    TALKATIVENESS_PROFILES,
    LLMRuntimeConfig,
    LocalLLMClient,
    resolve_talkativeness_profile,
)
from .local_modules import REPO_ROOT, default_asr_model_path
from .markdown_vault import MarkdownVault
from .memory_service import MemoryRuntimeConfig, RealtimeMemoryService
from .motion_intent import FORMS, MotionIntentService

import logging
logger = logging.getLogger("uvicorn.error.realtime")
from .persona_loader import PersonaLoader
from .session_manager import ConversationState
from .text_segmenter import StreamingTextSegmenter
from .turn_capture import TurnCaptureRecorder
from .tts_types import TTSRequestOptions, TTSRuntimeConfig
from .vad_session import RealtimeSession, VADConfig, VADTurnTiming

if TYPE_CHECKING:
    from .gpt_sovits_tts_service import GptSoVitsTtsService


def _next_stream_item(stream):
    try:
        return False, next(stream)
    except StopIteration:
        return True, None


def _encode_pcm_f32(audio: np.ndarray) -> tuple[str, int]:
    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
    return base64.b64encode(chunk.tobytes()).decode("ascii"), int(chunk.size)


def _elapsed_ms(started_at: float | None, ended_at: float | None) -> float | None:
    if started_at is None or ended_at is None:
        return None
    return (ended_at - started_at) * 1000.0


def _normalize_log_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    return value


def _log_turn_timing(
    session_id: str,
    turn_id: str | None,
    stage: str,
    *,
    event_at: float | None = None,
    speech_finished_at: float | None = None,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "stage": stage,
    }
    if turn_id:
        payload["turn_id"] = turn_id
    if event_at is not None and speech_finished_at is not None:
        # since_user_done_ms: elapsed time since the user actually finished
        # speaking (estimated as VAD speech-end detection time minus the VAD
        # trailing-silence window).
        payload["since_user_done_ms"] = round((event_at - speech_finished_at) * 1000.0, 2)
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = _normalize_log_value(value)
    logger.info("[TURN_TIMING] %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


@dataclass
class RealtimeAppConfig:
    host: str = "127.0.0.1"
    port: int = 8011
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str | None = "deepseek-chat"
    llm_api_key: str | None = None
    max_history_turns: int = 6
    asr_model_path: str = str(default_asr_model_path())
    asr_device: str = "cuda"
    vad_path: str = str(REPO_ROOT / "model" / "vad" / "silero_vad.onnx")
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 128
    vad_min_silence_ms: int = 800
    tts_enabled: bool = True
    # VoxCPM-era fields retained for CLI compatibility; the GPT-SoVITS service
    # ignores them and resolves voice from the active persona instead.
    tts_model_path: str | None = None
    tts_lora_root: str | None = None
    gpt_sovits_url: str = "http://127.0.0.1:9880"
    persona_root: str | None = None
    persona: str | None = "niya"
    debug_turns: bool = False
    debug_output_dir: str | None = None
    memory_enabled: bool = False
    # SQLite is the source of truth; the Markdown vault is the human-editable
    # mirror of it. Both are user-overridable (env / CLI / UI).
    history_db_path: str = str(REPO_ROOT / "runtime" / "data" / "whispera.db")
    history_vault_root: str = str(REPO_ROOT / "runtime" / "memory_vault")
    # 化身服务：把每轮用户话语解析成动作/表情计划，经 WS 下发给前端的 VRM 角色。
    # 与主回复并行发起，不占用 TTS/LLM 关键路径；失败只影响动作，不影响对话。
    motion_enabled: bool = False
    motion_timeout: float = 8.0


def _build_tts_service(config: RealtimeAppConfig) -> tuple["GptSoVitsTtsService | None", str | None]:
    if not config.tts_enabled:
        return None, None

    try:
        from .gpt_sovits_tts_service import GptSoVitsTtsService
    except Exception as exc:
        return None, f"TTS dependencies are unavailable: {exc}"

    try:
        service = GptSoVitsTtsService(
            TTSRuntimeConfig(),
            base_url=config.gpt_sovits_url,
            persona_root=config.persona_root,
        )
    except Exception as exc:
        return None, f"failed to initialize GPT-SoVITS TTS service: {exc}"

    return service, None


class RealtimeRuntime:
    def __init__(self, config: RealtimeAppConfig):
        self.config = config
        self.llm = LocalLLMClient(
            LLMRuntimeConfig(
                base_url=config.llm_base_url,
                model=config.llm_model,
                api_key=config.llm_api_key,
            )
        )
        self.asr = FunAsrService(
            ASRRuntimeConfig(
                model_path=config.asr_model_path,
                device=config.asr_device,
            )
        )
        # 动作意图：与主回复共用同一个 DeepSeek 端点，独立短请求、不流式。
        self.motion = (
            MotionIntentService(
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                model=config.llm_model,
                timeout=config.motion_timeout,
            )
            if config.motion_enabled
            else None
        )
        self.tts_requested = bool(config.tts_enabled)
        self.tts, self.tts_error = _build_tts_service(config)
        # torch.compile(mode="reduce-overhead") stores its CUDA-graph manager in
        # thread-local storage, so every call that touches the compiled TTS model
        # (warmup, sample_rate, each streaming next()) must run on the SAME thread.
        # asyncio.to_thread uses the default multi-threaded pool, which routes calls
        # to arbitrary threads and trips `assert _is_key_in_tls(...)` in
        # cudagraph_trees. Pinning all TTS work to one dedicated thread fixes it.
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        # NOTE(llm-api-mode): when the chat LLM runs against an external
        # OpenAI-compatible API (config.llm_api_key set), the mem0 inference LLM
        # is still pointed at llm_base_url WITHOUT the api_key. Memory is disabled
        # by default so this is deferred, but if memory is later enabled in API
        # mode, thread config.llm_api_key through MemoryRuntimeConfig here too.
        self.memory = RealtimeMemoryService(
            MemoryRuntimeConfig.from_env(
                llm_base_url=config.llm_base_url,
                llm_model=config.llm_model,
                enabled=config.memory_enabled,
            )
        )
        logger.info("[MEMORY-BOOT] enabled=%s available=%s last_error=%s",
                    self.memory.config.enabled, self.memory.available,
                    self.memory._last_error)
        # Persona system: a persona bundles (a) the character system prompt for
        # the LLM and (b) the GPT-SoVITS reference audio + prompt text for TTS
        # voice cloning. Switching persona at runtime updates both atomically.
        self.persona_loader = PersonaLoader(config.persona_root)
        self.current_persona: str | None = config.persona
        self.default_tts_options = TTSRequestOptions()
        fallback = self.persona_loader.build_fallback()
        persona = self.persona_loader.load(config.persona) if config.persona else None
        if persona is not None:
            self.llm.config.system_prompt = persona.system_prompt
            self.default_tts_options = TTSRequestOptions(
                prompt_audio_path=persona.voice.ref_audio_path,
                prompt_text=persona.voice.prompt_text,
                aux_ref_audio_paths=persona.voice.aux_ref_audio_paths or None,
                gpt_weights_path=persona.voice.gpt_weights_path or None,
                sovits_weights_path=persona.voice.sovits_weights_path or None,
            )
            print(
                f"[VOICE-BE] persona loaded id={persona.id} name={persona.name} "
                f"ref={persona.voice.ref_audio_path} prompt_text_set={bool(persona.voice.prompt_text)}",
                flush=True,
            )
            logger.info("[PERSONA-BOOT] id=%s name=%s ref=%s",
                        persona.id, persona.name, persona.voice.ref_audio_path)
        else:
            self.llm.config.system_prompt = fallback.system_prompt
            print(
                f"[VOICE-BE] persona NOT loaded id={config.persona!r} using fallback "
                f"(check personas/<id>/voice.json + SKILL.md exist)",
                flush=True,
            )
            logger.info("[PERSONA-BOOT] no persona matched '%s', using fallback prompt", config.persona)

        # Conversation history + memory mirror. SQLite is authoritative; the
        # Markdown vault is reconciled into it on boot so hand edits/deletions
        # made while the app was closed are not lost.
        history_store = HistoryStore(config.history_db_path)
        self.history = HistoryService(
            history_store,
            MarkdownVault(config.history_vault_root, history_store),
            memory=self.memory,
            owner_id=self.memory.config.agent_id or "default",
            user_id=self.memory.config.user_id or "local_user",
        )
        try:
            report = self.history.sync()
            self.history.refresh_memories()
            logger.info("[HISTORY-BOOT] db=%s vault=%s sync=%s",
                        config.history_db_path, config.history_vault_root, report)
        except Exception as exc:
            logger.warning("[HISTORY-BOOT] initial history sync failed: %s", exc, exc_info=True)

    @property
    def tts_enabled(self) -> bool:
        return self.tts is not None

    def set_persona(self, persona_id: str | None) -> dict[str, Any] | None:
        """Switch the active persona: updates the LLM system prompt and the
        default TTS reference audio/prompt text used for voice cloning."""
        persona = self.persona_loader.load(persona_id)
        if persona is None:
            return None
        self.current_persona = persona.id
        self.llm.config.system_prompt = persona.system_prompt
        self.default_tts_options = TTSRequestOptions(
            prompt_audio_path=persona.voice.ref_audio_path,
            prompt_text=persona.voice.prompt_text,
            aux_ref_audio_paths=persona.voice.aux_ref_audio_paths or None,
            gpt_weights_path=persona.voice.gpt_weights_path or None,
            sovits_weights_path=persona.voice.sovits_weights_path or None,
        )
        logger.info("[PERSONA-SWITCH] id=%s name=%s ref=%s",
                    persona.id, persona.name, persona.voice.ref_audio_path)
        return {
            "id": persona.id,
            "name": persona.name,
            "source": persona.source,
            "ref_audio_path": persona.voice.ref_audio_path,
            "prompt_text": persona.voice.prompt_text,
            "aux_ref_audio_paths": list(persona.voice.aux_ref_audio_paths or []),
            "gpt_weights_path": persona.voice.gpt_weights_path,
            "sovits_weights_path": persona.voice.sovits_weights_path,
        }


def _decode_pcm16(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _number_or_none(value: Any, kind: type[int] | type[float]):
    if value is None or value == "":
        return None
    return kind(value)


def _build_tts_options(payload: dict[str, Any], base: TTSRequestOptions | None = None) -> TTSRequestOptions:
    nested = payload.get("tts_options")
    source = nested if isinstance(nested, dict) else payload
    current = base or TTSRequestOptions()
    # IMPORTANT: prompt_audio_path / prompt_text / reference_wav_path must
    # fall back to `current` (the persona's voice) when the client sends null
    # or empty strings. The Electron frontend ships tts_options with these
    # fields set to null by default; without the `or current.X` guard, the
    # null would OVERRIDE the persona ref audio that was configured at boot,
    # and every TTS request would raise "needs both reference audio and
    # prompt text". model_version already used this pattern; the voice fields
    # were missing it.
    return TTSRequestOptions(
        model_version=_clean_text(source.get("model_version", current.model_version)) or current.model_version,
        lora_selection=_clean_text(source.get("lora_selection", current.lora_selection)),
        prompt_audio_path=_clean_text(source.get("prompt_audio_path")) or current.prompt_audio_path,
        prompt_text=_clean_text(source.get("prompt_text")) or current.prompt_text,
        reference_wav_path=_clean_text(source.get("reference_wav_path")) or current.reference_wav_path,
        aux_ref_audio_paths=(
            [str(p) for p in source["aux_ref_audio_paths"] if str(p).strip()]
            if isinstance(source.get("aux_ref_audio_paths"), list)
            else current.aux_ref_audio_paths
        ),
        gpt_weights_path=_clean_text(source.get("gpt_weights_path")) or current.gpt_weights_path,
        sovits_weights_path=_clean_text(source.get("sovits_weights_path")) or current.sovits_weights_path,
        cfg_value=_number_or_none(source.get("cfg_value", current.cfg_value), float),
        inference_timesteps=_number_or_none(source.get("inference_timesteps", current.inference_timesteps), int),
        seed=_number_or_none(source.get("seed", current.seed), int),
    )


def _bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def create_app(config: RealtimeAppConfig | None = None):
    config = config or RealtimeAppConfig()
    runtime = RealtimeRuntime(config)
    app = FastAPI(title="MiniMind Realtime Backend", version="0.1.0")
    app.state.realtime_runtime = runtime
    app.include_router(history_router)

    @app.get("/health")
    async def health():
        models = []
        upstream_ok = True
        error_message = None
        # In API mode (explicit model name configured), SKIP the slow upstream
        # /v1/models probe entirely. Remote endpoints (e.g. DeepSeek) can take
        # ~5s to respond or 401 on /v1/models, which exceeds Electron's 3s
        # fetchJson timeout and blocks startVoiceServices forever in an
        # infinite ready-poll loop. The configured model name is enough;
        # chat completions will surface a bad key loudly on the first turn.
        if config.llm_model:
            models = [config.llm_model]
            error_message = "model configured, skipping slow upstream models probe"
        else:
            try:
                models = runtime.llm.list_models()
            except Exception as exc:
                upstream_ok = False
                error_message = str(exc)
        notes = [
            "text.input is available in the scaffold",
            "binary audio streaming is now supported",
        ]
        if runtime.tts_enabled:
            notes.append("tts streaming is enabled when the optional VoxCPM stack is available")
        elif runtime.tts_requested:
            notes.append("tts was requested but the optional VoxCPM stack is unavailable, so the backend is running text-only")
        else:
            notes.append("tts is disabled by default in the slim runtime path")
        return {
            "status": "ok",
            "service": "minimind-realtime",
            "ws_path": "/ws/realtime",
            "upstream_ok": upstream_ok,
            "upstream_models": models,
            "llm_base_url": config.llm_base_url,
            "llm_model": config.llm_model,
            "error_message": error_message,
            "asr_model_path": config.asr_model_path,
            "asr_device": config.asr_device,
            "asr_warmed": runtime.asr.is_warmed,
            "vad_path": config.vad_path,
            "tts_requested": runtime.tts_requested,
            "tts_enabled": runtime.tts_enabled,
            "tts_warmed": runtime.tts.is_warmed if runtime.tts is not None else False,
            "tts_error": runtime.tts_error,
            "tts_model_path": config.tts_model_path,
            "tts_lora_root": config.tts_lora_root,
            "memory": runtime.memory.status_dict(),
            "notes": notes,
        }

    @app.post("/warmup")
    async def warmup(payload: dict[str, Any] | None = None):
        request = payload or {}
        force = bool(request.get("force", False))
        started_at = perf_counter()
        # Base this on the active persona so warmup preloads its checkpoints;
        # the frontend ships nulls for the voice fields (see _build_tts_options).
        tts_options = _build_tts_options(request, runtime.default_tts_options)
        asr_result = await asyncio.to_thread(runtime.asr.warmup, force)
        tts_result: dict[str, Any] = {}
        if runtime.tts is not None:
            loop = asyncio.get_running_loop()
            tts_result = await loop.run_in_executor(
                runtime.tts_executor, runtime.tts.warmup, tts_options, force
            )
        elif runtime.tts_requested and runtime.tts_error:
            tts_result = {
                "tts_skipped": True,
                "tts_error": runtime.tts_error,
            }
        cached = bool(asr_result.get("cached", False))
        if tts_result:
            cached = cached and bool(tts_result.get("cached", False))
        return {
            "ok": True,
            "cached": cached,
            **asr_result,
            **tts_result,
            "total_ms": round((perf_counter() - started_at) * 1000.0, 2),
        }

    @app.get("/tts/lora/catalog")
    async def tts_lora_catalog():
        if runtime.tts is None:
            if runtime.tts_error:
                raise HTTPException(status_code=503, detail=runtime.tts_error)
            raise HTTPException(status_code=404, detail="TTS is disabled")
        return {"models": runtime.tts.list_lora_checkpoints()}

    @app.websocket("/ws/realtime")
    async def websocket_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            client = websocket.client
            client_addr = f"{client.host}:{client.port}" if client else "unknown"
        except Exception:
            client_addr = "unknown"
        print(f"[VOICE-BE] ws.connected client={client_addr}", flush=True)
        send_lock = asyncio.Lock()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        state = ConversationState(session_id=session_id, max_turns=config.max_history_turns)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                runtime.history.ensure_session, session_id, runtime.current_persona or ""
            )
        vad_session = RealtimeSession(
            config.vad_path,
            VADConfig(
                threshold=config.vad_threshold,
                min_speech_ms=config.vad_min_speech_ms,
                min_silence_ms=config.vad_min_silence_ms,
            ),
        )
        active_task: asyncio.Task | None = None
        interrupt_event = asyncio.Event()
        # New sessions inherit the runtime's currently active persona voice so
        # the LLM prompt and TTS reference audio stay in sync from turn one.
        session_tts_options = TTSRequestOptions(
            prompt_audio_path=runtime.default_tts_options.prompt_audio_path,
            prompt_text=runtime.default_tts_options.prompt_text,
            aux_ref_audio_paths=runtime.default_tts_options.aux_ref_audio_paths,
            gpt_weights_path=runtime.default_tts_options.gpt_weights_path,
            sovits_weights_path=runtime.default_tts_options.sovits_weights_path,
        )
        turn_capture = TurnCaptureRecorder(
            config.debug_output_dir,
            sample_rate=vad_session.config.sample_rate,
            enabled=config.debug_turns,
            session_id=session_id,
        )

        async def send_json(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def receiver() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        queue.put_nowait({"type": "socket.closed"})
                        break
                    if message.get("text") is not None:
                        queue.put_nowait(json.loads(message["text"]))
                    elif message.get("bytes") is not None:
                        queue.put_nowait(message["bytes"])
            except WebSocketDisconnect:
                queue.put_nowait({"type": "socket.closed"})

        async def notify_vad() -> None:
            await send_json({"type": "vad", "speaking": vad_session.speaking})

        async def run_text_request(
            user_text: str,
            turn_id: str | None = None,
            tts_options: TTSRequestOptions | None = None,
            turn_timing: dict[str, float | None] | None = None,
        ) -> None:
            stream = None
            full_text = ""
            motion_task: asyncio.Task[None] | None = None
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            segmenter = StreamingTextSegmenter()
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            request_tts_options = tts_options or session_tts_options
            timing = dict(turn_timing or {})
            speech_finished_at = timing.get("speech_finished_at")
            vad_session.generating = True
            state.active_request_id = request_id
            interrupt_event.clear()

            def log_timing(stage: str, event_at: float | None = None, **fields: Any) -> None:
                _log_turn_timing(
                    session_id,
                    turn_id,
                    stage,
                    event_at=event_at,
                    speech_finished_at=speech_finished_at,
                    assistant_request_id=request_id,
                    **fields,
                )

            async def emit_motion() -> None:
                """把本轮用户话语解析成动作/表情计划下发给前端化身。

                与主回复流并行，不占用 LLM/TTS 关键路径；任何失败都只是让角色
                少一个动作，绝不影响对话本身。
                """
                service = runtime.motion
                if service is None:
                    print("[MOTION] disabled (motion_enabled=false)", flush=True)
                    return
                try:
                    plan = await asyncio.to_thread(
                        service.plan,
                        user_text,
                        state.history,
                        state.avatar_form,
                    )
                    if plan is None:
                        # 必须显式记录：静默失败会让"角色不动"无从排查。
                        print(f"[MOTION] no plan turn={turn_id} reason={service.last_error}", flush=True)
                        logger.warning("[MOTION] no plan turn=%s reason=%s", turn_id, service.last_error)
                        return
                    if interrupt_event.is_set():
                        print(f"[MOTION] dropped (interrupted) turn={turn_id}", flush=True)
                        return
                    suggested_form = plan["presentation"].get("form")
                    # 这里只记录建议值，不立即写入 state.avatar_form：前端可能因为
                    # 冷却或手动选择而没有真正切换。等前端切换成功后用 avatar.form
                    # 回报真实形态，下一轮才以真实形态作为 current_form。
                    keys = [i["key"] for i in plan["intents"]]
                    print(
                        f"[MOTION] turn={turn_id} op={plan['operation']} keys={keys} "
                        f"expr={plan['presentation']['expression']} form={suggested_form} "
                        f"body={plan['presentation'].get('body')} latency={plan['seconds']}s",
                        flush=True,
                    )
                    await send_json(
                        {
                            "type": "assistant.motion",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "plan": plan,
                        }
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # 动作是附加能力：解析或下发失败都只记录，不打断对话。
                    print(f"[MOTION] failed turn={turn_id} {type(exc).__name__}: {exc}", flush=True)
                    logger.warning("[MOTION] failed turn=%s", turn_id, exc_info=True)

            async def run_tts_worker() -> None:
                if runtime.tts is None:
                    return
                loop = asyncio.get_running_loop()
                while True:
                    text = await tts_queue.get()
                    try:
                        if text is None or interrupt_event.is_set():
                            return
                        tts_request_id = f"tts-{uuid.uuid4().hex[:8]}"
                        tts_request_started_at = perf_counter()
                        if timing.get("tts_request_started_at") is None:
                            timing["tts_request_started_at"] = tts_request_started_at
                            log_timing(
                                "tts.request_started",
                                tts_request_started_at,
                                tts_request_id=tts_request_id,
                                chunk_chars=len(text),
                                llm_first_text_to_tts_request_ms=_elapsed_ms(
                                    timing.get("llm_first_text_at"),
                                    tts_request_started_at,
                                ),
                                first_segment_to_tts_request_ms=_elapsed_ms(
                                    timing.get("tts_first_segment_ready_at"),
                                    tts_request_started_at,
                                ),
                            )
                        sample_rate = await loop.run_in_executor(
                            runtime.tts_executor, runtime.tts.sample_rate, request_tts_options
                        )
                        await send_json(
                            {
                                "type": "assistant.audio.start",
                                "request_id": tts_request_id,
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "sample_rate": sample_rate,
                                "audio_format": "pcm_f32le",
                                "text": text,
                            }
                        )
                        audio_start_sent_at = perf_counter()
                        if timing.get("tts_audio_start_sent_at") is None:
                            timing["tts_audio_start_sent_at"] = audio_start_sent_at
                            log_timing(
                                "tts.audio_start_sent",
                                audio_start_sent_at,
                                tts_request_id=tts_request_id,
                                tts_request_to_audio_start_ms=_elapsed_ms(
                                    timing.get("tts_request_started_at"),
                                    audio_start_sent_at,
                                ),
                            )
                        audio_stream = runtime.tts.stream_tts(text, options=request_tts_options)
                        chunk_index = 0
                        total_samples = 0
                        captured_chunks: list[np.ndarray] | None = [] if turn_capture.enabled else None
                        while not interrupt_event.is_set():
                            done, audio = await loop.run_in_executor(
                                runtime.tts_executor, _next_stream_item, audio_stream
                            )
                            if done:
                                break
                            encoded, num_samples = _encode_pcm_f32(audio)
                            total_samples += num_samples
                            if captured_chunks is not None:
                                captured_chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                            await send_json(
                                {
                                    "type": "assistant.audio.chunk",
                                    "request_id": tts_request_id,
                                    "assistant_request_id": request_id,
                                    "turn_id": turn_id,
                                    "index": chunk_index,
                                    "sample_rate": sample_rate,
                                    "audio_format": "pcm_f32le",
                                    "num_samples": num_samples,
                                    "data": encoded,
                                }
                            )
                            chunk_sent_at = perf_counter()
                            if timing.get("tts_first_audio_chunk_at") is None:
                                timing["tts_first_audio_chunk_at"] = chunk_sent_at
                                log_timing(
                                    "tts.first_audio_chunk_sent",
                                    chunk_sent_at,
                                    tts_request_id=tts_request_id,
                                    num_samples=num_samples,
                                    tts_request_to_first_audio_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        chunk_sent_at,
                                    ),
                                )
                                log_timing(
                                    "turn.first_audio_summary",
                                    chunk_sent_at,
                                    # vad_tail_ms: user-finished-speech -> VAD speech_end.
                                    vad_tail_ms=timing.get("vad_tail_ms"),
                                    # speech_end_to_turn_start_ms: VAD speech_end -> backend
                                    # turn task actually starts.
                                    speech_end_to_turn_start_ms=_elapsed_ms(
                                        timing.get("speech_end_detected_at"),
                                        timing.get("turn_started_at"),
                                    ),
                                    # asr_ms: ASR transcription duration.
                                    asr_ms=_elapsed_ms(
                                        timing.get("asr_started_at"),
                                        timing.get("asr_completed_at"),
                                    ),
                                    asr_done_to_llm_request_ms=_elapsed_ms(
                                        timing.get("asr_completed_at"),
                                        timing.get("llm_request_started_at"),
                                    ),
                                    memory_search_ms=_elapsed_ms(
                                        timing.get("memory_search_started_at"),
                                        timing.get("memory_search_completed_at"),
                                    ),
                                    # llm_request_to_first_text_ms: LLM request start ->
                                    # first speakable text delta from the model.
                                    llm_request_to_first_text_ms=_elapsed_ms(
                                        timing.get("llm_request_started_at"),
                                        timing.get("llm_first_text_at"),
                                    ),
                                    # llm_first_text_to_first_segment_ms: first text delta ->
                                    # first segment that the segmenter is willing to hand to TTS.
                                    llm_first_text_to_first_segment_ms=_elapsed_ms(
                                        timing.get("llm_first_text_at"),
                                        timing.get("tts_first_segment_ready_at"),
                                    ),
                                    first_segment_to_tts_request_ms=_elapsed_ms(
                                        timing.get("tts_first_segment_ready_at"),
                                        timing.get("tts_request_started_at"),
                                    ),
                                    tts_request_to_audio_start_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        timing.get("tts_audio_start_sent_at"),
                                    ),
                                    # tts_request_to_first_audio_ms: TTS request start ->
                                    # first emitted audio chunk.
                                    tts_request_to_first_audio_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        chunk_sent_at,
                                    ),
                                    speech_end_to_first_audio_ms=_elapsed_ms(
                                        timing.get("speech_end_detected_at"),
                                        chunk_sent_at,
                                    ),
                                    # total_to_first_audio_ms: user-finished-speech estimate ->
                                    # first emitted audio chunk.
                                    total_to_first_audio_ms=_elapsed_ms(
                                        speech_finished_at,
                                        chunk_sent_at,
                                    ),
                                )
                            chunk_index += 1
                        if captured_chunks:
                            full_audio = np.concatenate(captured_chunks)
                            await asyncio.to_thread(
                                turn_capture.capture_tts_turn,
                                turn_id or request_id,
                                full_audio,
                                text,
                                request_id=tts_request_id,
                                sample_rate=sample_rate,
                                interrupted=interrupt_event.is_set(),
                                extra={"chunk_count": chunk_index, "assistant_request_id": request_id},
                            )
                        await send_json(
                            {
                                "type": "assistant.audio.completed",
                                "request_id": tts_request_id,
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "chunk_count": chunk_index,
                                "total_samples": total_samples,
                                "interrupted": interrupt_event.is_set(),
                            }
                        )
                    except Exception as exc:
                        logger.exception(
                            "tts generation failed | request_id=%s | turn_id=%s | text=%r",
                            request_id,
                            turn_id,
                            text,
                        )
                        await send_json(
                            {
                                "type": "assistant.audio.error",
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "message": str(exc),
                            }
                        )
                    finally:
                        tts_queue.task_done()

            tts_task = asyncio.create_task(run_tts_worker())
            await send_json({"type": "assistant.started", "request_id": request_id, "turn_id": turn_id})
            # 动作意图与主回复并行发起：它通常比首个 TTS 音频块更快返回，
            # 于是角色在开口前就开始动，不额外增加端到端延迟。
            motion_task = asyncio.create_task(emit_motion())
            try:
                memory_context = ""
                if runtime.memory.available:
                    timing["memory_search_started_at"] = perf_counter()
                    memory_context = await asyncio.to_thread(runtime.memory.search_context, user_text)
                    timing["memory_search_completed_at"] = perf_counter()
                    log_timing(
                        "memory.search_completed",
                        timing["memory_search_completed_at"],
                        memory_search_ms=_elapsed_ms(
                            timing.get("memory_search_started_at"),
                            timing.get("memory_search_completed_at"),
                        ),
                        memory_context_found=bool(memory_context),
                    )
                    if memory_context:
                        await send_json(
                            {
                                "type": "memory.context",
                                "request_id": request_id,
                                "turn_id": turn_id,
                                "available": True,
                            }
                        )
                messages = state.build_messages(user_text)
                timing["llm_request_started_at"] = perf_counter()
                log_timing("llm.request_started", timing["llm_request_started_at"])
                stream = runtime.llm.start_stream(messages, memory_context=memory_context)
                reasoning_text = ""
                while True:
                    done, delta = await asyncio.to_thread(_next_stream_item, stream)
                    if done:
                        break
                    if interrupt_event.is_set():
                        break
                    text_delta = str(getattr(delta, "text", "") or "")
                    reasoning_delta = str(getattr(delta, "reasoning_text", "") or "")
                    if reasoning_delta:
                        reasoning_text += reasoning_delta
                    if not text_delta:
                        continue
                    if timing.get("llm_first_text_at") is None:
                        timing["llm_first_text_at"] = perf_counter()
                        log_timing(
                            "llm.first_text_delta",
                            timing["llm_first_text_at"],
                            llm_request_to_first_text_ms=_elapsed_ms(
                                timing.get("llm_request_started_at"),
                                timing.get("llm_first_text_at"),
                            ),
                            text_chars=len(text_delta),
                        )
                    full_text += text_delta
                    await send_json({"type": "assistant.delta", "request_id": request_id, "turn_id": turn_id, "text": text_delta})
                    for chunk in segmenter.feed(text_delta):
                        if timing.get("tts_first_segment_ready_at") is None:
                            timing["tts_first_segment_ready_at"] = perf_counter()
                            log_timing(
                                "tts.first_segment_ready",
                                timing["tts_first_segment_ready_at"],
                                chunk_chars=len(chunk),
                                llm_request_to_first_segment_ms=_elapsed_ms(
                                    timing.get("llm_request_started_at"),
                                    timing.get("tts_first_segment_ready_at"),
                                ),
                                llm_first_text_to_first_segment_ms=_elapsed_ms(
                                    timing.get("llm_first_text_at"),
                                    timing.get("tts_first_segment_ready_at"),
                                ),
                            )
                        await tts_queue.put(chunk)
                if not full_text.strip() and reasoning_text.strip() and not interrupt_event.is_set():
                    # DeepSeek thinking mode consumed the entire max_tokens
                    # budget with reasoning before any speakable content was
                    # produced. Rather than erroring out (which leaves the
                    # user with no audio response and can stall the session
                    # after a context.clear), inject a short natural fallback
                    # so the turn still completes: TTS plays something, the
                    # VAD returns to listening, and the user can try again.
                    fallback_text = "嗯，让我想一下，你能再说一遍吗？"
                    full_text = fallback_text
                    if timing.get("llm_first_text_at") is None:
                        timing["llm_first_text_at"] = perf_counter()
                    await send_json(
                        {
                            "type": "assistant.delta",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "text": fallback_text,
                        }
                    )
                    for chunk in segmenter.feed(fallback_text):
                        if timing.get("tts_first_segment_ready_at") is None:
                            timing["tts_first_segment_ready_at"] = perf_counter()
                        await tts_queue.put(chunk)
                final_tts_chunk = segmenter.flush()
                if final_tts_chunk:
                    if timing.get("tts_first_segment_ready_at") is None:
                        timing["tts_first_segment_ready_at"] = perf_counter()
                        log_timing(
                            "tts.first_segment_ready",
                            timing["tts_first_segment_ready_at"],
                            chunk_chars=len(final_tts_chunk),
                            llm_request_to_first_segment_ms=_elapsed_ms(
                                timing.get("llm_request_started_at"),
                                timing.get("tts_first_segment_ready_at"),
                            ),
                            llm_first_text_to_first_segment_ms=_elapsed_ms(
                                timing.get("llm_first_text_at"),
                                timing.get("tts_first_segment_ready_at"),
                            ),
                            flushed_at_stream_end=True,
                        )
                    await tts_queue.put(final_tts_chunk)
                await tts_queue.put(None)
                await tts_task
                state.add_user_turn(user_text)
                state.add_assistant_turn(full_text)

                async def record_history_turn() -> None:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(
                            runtime.history.record_turn,
                            state.session_id,
                            runtime.current_persona or "",
                            turn_id or "",
                            user_text,
                            full_text,
                        )

                if interrupt_event.is_set():
                    await send_json(
                        {
                            "type": "assistant.completed",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "interrupted": True,
                            "text": full_text,
                        }
                    )
                    asyncio.create_task(record_history_turn())
                    return
                await send_json(
                    {
                        "type": "assistant.completed",
                        "request_id": request_id,
                        "turn_id": turn_id,
                        "interrupted": False,
                        "text": full_text,
                    }
                )
                asyncio.create_task(record_history_turn())
                if runtime.memory.available:
                    async def save_memory_turn() -> None:
                        result = await asyncio.to_thread(
                            runtime.memory.save_turn,
                            user_text,
                            full_text,
                            state.session_id,
                        )
                        # Mirror what mem0 now remembers into the Markdown vault so
                        # the user can read, edit or delete it.
                        await asyncio.to_thread(runtime.history.refresh_memories)
                        with contextlib.suppress(Exception):
                            await send_json(
                                {
                                    "type": "memory.saved",
                                    "request_id": request_id,
                                    "turn_id": turn_id,
                                    "saved": bool(result),
                                }
                            )

                    asyncio.create_task(save_memory_turn())
            except Exception as exc:
                await send_json({"type": "assistant.error", "request_id": request_id, "turn_id": turn_id, "message": str(exc)})
            finally:
                if not tts_task.done():
                    await tts_queue.put(None)
                    tts_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await tts_task
                if stream is not None:
                    runtime.llm.close_stream(stream)
                # 只在被打断时放弃动作解析。正常收尾时若它还没回来, 就让它继续跑完:
                # 动作晚到一点仍然应该播出来, 强行取消会让角色整轮不动。
                if motion_task is not None and not motion_task.done() and interrupt_event.is_set():
                    motion_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await motion_task
                vad_session.generating = False
                vad_session.interrupt = False
                state.active_request_id = None

        async def run_audio_turn(
            samples: np.ndarray,
            vad_timing: VADTurnTiming | None = None,
            tts_options: TTSRequestOptions | None = None,
        ) -> None:
            handed_off_to_llm = False
            turn_id = f"turn-{uuid.uuid4().hex[:8]}"
            user_text = ""
            capture_error: str | None = None
            timing: dict[str, float | None] = {}
            speech_finished_at = None
            if vad_timing is not None:
                speech_finished_at = vad_timing.speech_finished_at_est
                timing["speech_finished_at"] = speech_finished_at
                timing["speech_end_detected_at"] = vad_timing.speech_end_at
                timing["vad_tail_ms"] = vad_timing.vad_tail_ms

            def log_timing(stage: str, event_at: float | None = None, **fields: Any) -> None:
                _log_turn_timing(
                    session_id,
                    turn_id,
                    stage,
                    event_at=event_at,
                    speech_finished_at=speech_finished_at,
                    **fields,
                )

            try:
                vad_session.generating = True
                timing["turn_started_at"] = perf_counter()
                log_timing(
                    "turn.started",
                    timing["turn_started_at"],
                    vad_tail_ms=timing.get("vad_tail_ms"),
                    speech_duration_ms=vad_timing.speech_duration_ms if vad_timing is not None else None,
                    buffered_audio_ms=vad_timing.buffered_audio_ms if vad_timing is not None else None,
                    speech_end_to_turn_start_ms=_elapsed_ms(
                        timing.get("speech_end_detected_at"),
                        timing.get("turn_started_at"),
                    ),
                )
                timing["asr_started_at"] = perf_counter()
                log_timing(
                    "asr.started",
                    timing["asr_started_at"],
                    speech_end_to_asr_start_ms=_elapsed_ms(
                        timing.get("speech_end_detected_at"),
                        timing.get("asr_started_at"),
                    ),
                )
                await send_json({"type": "asr.started", "turn_id": turn_id})
                user_text = await asyncio.to_thread(runtime.asr.transcribe, samples)
                timing["asr_completed_at"] = perf_counter()
                log_timing(
                    "asr.completed",
                    timing["asr_completed_at"],
                    asr_ms=_elapsed_ms(
                        timing.get("asr_started_at"),
                        timing.get("asr_completed_at"),
                    ),
                    user_text_chars=len(user_text),
                )
                if interrupt_event.is_set():
                    return
                if not user_text:
                    await send_json({"type": "asr.completed", "turn_id": turn_id, "text": ""})
                    await send_json({"type": "user_text", "turn_id": turn_id, "text": ""})
                    return
                await send_json({"type": "asr.completed", "turn_id": turn_id, "text": user_text})
                await send_json({"type": "user_text", "turn_id": turn_id, "text": user_text})
                handed_off_to_llm = True
                await run_text_request(
                    user_text,
                    turn_id=turn_id,
                    tts_options=tts_options,
                    turn_timing=timing,
                )
            except Exception as exc:
                capture_error = str(exc)
                vad_session.generating = False
                await send_json({"type": "asr.error", "turn_id": turn_id, "message": str(exc)})
            finally:
                if turn_capture.enabled:
                    await asyncio.to_thread(
                        turn_capture.capture_user_turn,
                        turn_id,
                        samples,
                        user_text,
                        interrupted=interrupt_event.is_set(),
                        error_message=capture_error,
                    )
                if not handed_off_to_llm:
                    vad_session.generating = False

        # Diagnostic counter for audio chunks received from the client. Logged
        # via the [VOICE-BE] channel (forwarded by main.js attachProcessEvents)
        # so we can confirm the backend is actually receiving PCM frames and
        # see their signal energy, independent of the service-log file.
        audio_stats = {"chunks": 0, "bytes": 0}

        def push_audio(audio_bytes: bytes) -> str:
            audio_stats["chunks"] += 1
            audio_stats["bytes"] += len(audio_bytes)
            n = audio_stats["chunks"]
            try:
                arr = np.frombuffer(audio_bytes, dtype=np.int16)
                max_abs = int(np.abs(arr).max()) if arr.size else 0
            except Exception:
                max_abs = -1
            status = vad_session.push_chunk(_decode_pcm16(audio_bytes))
            # PRIMARY VAD diagnostic channel. push_chunk's own print was
            # silently dropped when a stale __pycache__ masked the source
            # edit, so we also surface prob/threshold/action here — this
            # print is guaranteed to reach the terminal via main.js
            # attachProcessEvents. Log first 10 chunks + every 10 after.
            if n <= 10 or n % 10 == 0:
                print(
                    f"[VOICE-BE] push_audio chunk#{n} bytes={len(audio_bytes)} "
                    f"maxAbs={max_abs} prob={vad_session.last_prob:.3f} "
                    f"thr={vad_session.config.threshold} action={vad_session.last_action} "
                    f"status={status} speaking={vad_session.speaking} "
                    f"spk_samples={vad_session.speech_samples}",
                    flush=True,
                )
            return status

        await send_json(
            {
                "type": "server.ready",
                "protocol_version": "0.1.0",
                "session_id": session_id,
                "supported_client_messages": [
                    "session.start",
                    "text.input",
                    "tts.configure",
                    "persona.configure",
                    "logging.configure",
                    "avatar.form",
                    "audio.chunk",
                    "interrupt",
                    "context.clear",
                    "ping",
                    "session.stop",
                ],
            }
        )

        receiver_task = asyncio.create_task(receiver())
        try:
            while True:
                message = await queue.get()

                if isinstance(message, (bytes, bytearray)):
                    if active_task is not None and active_task.done():
                        active_task = None
                    status = push_audio(bytes(message))
                    await notify_vad()
                    if status == "interrupt":
                        if not interrupt_event.is_set():
                            interrupt_event.set()
                            await send_json(
                                {
                                    "type": "interrupt.ack",
                                    "session_id": session_id,
                                    "request_id": f"interrupt-{uuid.uuid4().hex[:8]}",
                                    "interrupted_request_id": state.active_request_id,
                                    "accepted": bool(active_task and not active_task.done()),
                                    "reason": "vad_barge_in",
                                }
                            )
                        continue
                    if status != "speech_end":
                        continue
                    print(
                        f"[VOICE-BE] speech_end detected total_chunks={audio_stats['chunks']} "
                        f"speech_samples={vad_session.speech_samples} silence_samples={vad_session.silence_samples}",
                        flush=True,
                    )
                    if active_task is not None and not active_task.done():
                        if interrupt_event.is_set():
                            try:
                                await asyncio.wait_for(active_task, timeout=5.0)
                            except asyncio.TimeoutError:
                                active_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, Exception):
                                    await active_task
                            finally:
                                active_task = None
                        else:
                            await send_json({"type": "error", "message": "assistant request is still running"})
                            continue
                    samples = vad_session.get_audio()
                    vad_timing = vad_session.consume_completed_timing()
                    if samples.size == 0:
                        continue
                    interrupt_event.clear()
                    vad_session.interrupt = False
                    active_task = asyncio.create_task(
                        run_audio_turn(samples, vad_timing=vad_timing, tts_options=session_tts_options)
                    )
                    continue

                if not isinstance(message, dict):
                    await send_json({"type": "error", "message": "received unsupported queue payload"})
                    continue

                message_type = str(message.get("type") or "").strip()

                if message_type == "socket.closed":
                    print(f"[VOICE-BE] socket.closed total_chunks={audio_stats['chunks']}", flush=True)
                    break

                if message_type == "ping":
                    await send_json({"type": "pong", "session_id": session_id})
                    continue

                if message_type == "session.start":
                    print(
                        f"[VOICE-BE] session.start logging_enabled={message.get('logging_enabled')} "
                        f"tts_options_keys={list((message.get('tts_options') or {}).keys())}",
                        flush=True,
                    )
                    requested = str(message.get("session_id") or "").strip()
                    if requested:
                        session_id = requested
                        state.session_id = requested
                        turn_capture.set_session_id(requested)
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(
                                runtime.history.ensure_session,
                                session_id,
                                runtime.current_persona or "",
                            )
                    session_tts_options = _build_tts_options(message, session_tts_options)
                    print(
                        f"[VOICE-BE] session_tts resolved ref={session_tts_options.prompt_audio_path!r} "
                        f"prompt_text_set={bool(session_tts_options.prompt_text)}",
                        flush=True,
                    )
                    turn_capture.set_enabled(_bool_flag(message.get("logging_enabled"), config.debug_turns))
                    await send_json({"type": "session.ready", "session_id": session_id, "state": "ready"})
                    await send_json(
                        {
                            "type": "logging.configured",
                            "session_id": session_id,
                            "enabled": turn_capture.enabled,
                            "capture_dir": config.debug_output_dir,
                        }
                    )
                    continue

                if message_type == "tts.configure":
                    session_tts_options = _build_tts_options(message, session_tts_options)
                    await send_json(
                        {
                            "type": "tts.configured",
                            "session_id": session_id,
                            "model_version": session_tts_options.model_version,
                            "lora_selection": session_tts_options.lora_selection,
                            "reference_enabled": bool(session_tts_options.prompt_audio_path and session_tts_options.prompt_text),
                            "reference_wav_enabled": bool(session_tts_options.reference_wav_path),
                            "reference_ignored": bool(session_tts_options.prompt_audio_path and not session_tts_options.prompt_text),
                            "cfg_value": session_tts_options.cfg_value,
                            "inference_timesteps": session_tts_options.inference_timesteps,
                            "seed": session_tts_options.seed,
                        }
                    )
                    continue

                if message_type == "llm.configure":
                    # 话痨程度运行时切换：前端滑动开关发送档位 key，这里更新
                    # runtime.llm.config.talkativeness，下一轮 start_stream 立即派生
                    # 新的 max_tokens/temperature 并追加 hint 到 system prompt。
                    requested = str(message.get("talkativeness") or "").strip().lower()
                    if requested and requested not in TALKATIVENESS_PROFILES:
                        await send_json(
                            {
                                "type": "error",
                                "message": f"unknown talkativeness level: {requested}",
                            }
                        )
                        continue
                    if requested:
                        runtime.llm.config.talkativeness = requested
                    profile = resolve_talkativeness_profile(runtime.llm.config.talkativeness)
                    print(
                        f"[VOICE-BE] llm.configure talkativeness={runtime.llm.config.talkativeness} "
                        f"label={profile['label']} max_tokens={profile['max_tokens']}",
                        flush=True,
                    )
                    logger.info(
                        "[LLM-CONFIG] talkativeness=%s label=%s max_tokens=%s",
                        runtime.llm.config.talkativeness, profile["label"], profile["max_tokens"],
                    )
                    await send_json(
                        {
                            "type": "llm.configured",
                            "session_id": session_id,
                            "talkativeness": runtime.llm.config.talkativeness,
                            "label": profile["label"],
                            "max_tokens": profile["max_tokens"],
                        }
                    )
                    continue

                if message_type == "persona.configure":
                    requested_persona = str(message.get("persona") or "").strip()
                    info = runtime.set_persona(requested_persona or None)
                    if info is None:
                        await send_json(
                            {
                                "type": "error",
                                "message": f"unknown persona: {requested_persona or '(empty)'}",
                            }
                        )
                        continue
                    # Apply this persona's voice to the active session so the
                    # very next TTS request uses the new reference audio.
                    session_tts_options = TTSRequestOptions(
                        prompt_audio_path=info.get("ref_audio_path") or "",
                        prompt_text=info.get("prompt_text") or "",
                        aux_ref_audio_paths=info.get("aux_ref_audio_paths") or None,
                        gpt_weights_path=info.get("gpt_weights_path") or None,
                        sovits_weights_path=info.get("sovits_weights_path") or None,
                    )
                    # A fine-tuned persona switches the server's global weights;
                    # do it here so the reload is not paid mid-turn.
                    weights_error: str | None = None
                    if runtime.tts is not None:
                        try:
                            await asyncio.get_running_loop().run_in_executor(
                                runtime.tts_executor, runtime.tts.load_weights, session_tts_options
                            )
                        except Exception as exc:
                            weights_error = str(exc)
                            print(
                                f"[VOICE-BE] persona weights load failed: {weights_error}",
                                flush=True,
                            )
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(
                            runtime.history.set_session_persona,
                            session_id,
                            str(info.get("id") or runtime.current_persona or ""),
                        )
                    await send_json(
                        {
                            "type": "persona.configured",
                            "session_id": session_id,
                            "persona": info,
                            "weights_error": weights_error,
                        }
                    )
                    continue

                if message_type == "logging.configure":
                    turn_capture.set_enabled(_bool_flag(message.get("enabled"), False))
                    await send_json(
                        {
                            "type": "logging.configured",
                            "session_id": session_id,
                            "enabled": turn_capture.enabled,
                            "capture_dir": config.debug_output_dir,
                        }
                    )
                    continue

                if message_type == "avatar.form":
                    form = str(message.get("form") or "").strip()
                    if form not in FORMS:
                        await send_json({"type": "error", "message": f"invalid avatar form: {form}"})
                        continue
                    # 前端切换成功后回报真实形态，避免手动选择后模型仍按旧形态判断。
                    state.avatar_form = form
                    await send_json(
                        {
                            "type": "avatar.form.updated",
                            "session_id": session_id,
                            "form": form,
                        }
                    )
                    continue

                if message_type == "session.stop":
                    interrupt_event.set()
                    if active_task is not None:
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await active_task
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                        await send_json({"type": "session.closed", "session_id": session_id})
                    break

                if message_type == "interrupt":
                    interrupt_event.set()
                    vad_session.interrupt = True
                    await send_json(
                        {
                            "type": "interrupt.ack",
                            "session_id": session_id,
                            "request_id": message.get("request_id"),
                            "interrupted_request_id": state.active_request_id,
                            "accepted": bool(state.active_request_id),
                        }
                    )
                    continue

                if message_type == "context.clear":
                    # Interrupt any in-flight turn (LLM/TTS), wait for it to wind
                    # down, then wipe the short-term conversation context so the
                    # next turn starts fresh. Long-term memory is left untouched.
                    interrupt_event.set()
                    vad_session.interrupt = True
                    if active_task is not None:
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await active_task
                        active_task = None
                    state.clear()
                    vad_session.reset()
                    interrupt_event.clear()
                    await send_json(
                        {
                            "type": "context.cleared",
                            "session_id": session_id,
                            "request_id": message.get("request_id"),
                        }
                    )
                    continue

                if message_type == "audio.chunk":
                    await send_json({"type": "error", "message": "send raw websocket binary frames for audio, not JSON audio.chunk"})
                    continue

                if message_type == "text.input":
                    text = str(message.get("text") or "").strip()
                    if not text:
                        await send_json({"type": "error", "message": "text.input requires a non-empty text field"})
                        continue
                    if active_task is not None and not active_task.done():
                        await send_json({"type": "error", "message": "assistant request is still running"})
                        continue
                    request_options = _build_tts_options(message, session_tts_options)
                    active_task = asyncio.create_task(run_text_request(text, tts_options=request_options))
                    continue

                await send_json({"type": "error", "message": f"unsupported message type: {message_type}"})
                continue
        finally:
            interrupt_event.set()
            if active_task is not None:
                active_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await active_task
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver_task

    return app


def parse_args() -> RealtimeAppConfig:
    parser = argparse.ArgumentParser(description="CommunicationPet realtime backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--llm-model", default="deepseek-chat")
    # API key is read from MINIMIND_LLM_API_KEY (env) rather than a CLI flag so
    # the secret does not leak into process listings or service launch logs.
    parser.add_argument("--max-history-turns", default=6, type=int)
    parser.add_argument("--asr-model-path", default=str(default_asr_model_path()),
                        help="ASR model dir (default: GPT-SoVITS bundled paraformer)")
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--vad-path", default=str(REPO_ROOT / "model" / "vad" / "silero_vad.onnx"))
    parser.add_argument("--vad-threshold", default=0.5, type=float)
    parser.add_argument("--vad-min-speech-ms", default=128, type=int)
    parser.add_argument("--vad-min-silence-ms", default=800, type=int)
    tts_group = parser.add_mutually_exclusive_group()
    tts_group.add_argument("--enable-tts", action="store_true")
    tts_group.add_argument("--disable-tts", action="store_true")
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument("--tts-lora-root", default=None)
    parser.add_argument("--gpt-sovits-url", default="http://127.0.0.1:9880",
                        help="base URL of the external GPT-SoVITS api_v2 server")
    parser.add_argument("--persona-root", default=str(REPO_ROOT / "personas"),
                        help="directory holding persona subfolders (SKILL.md + voice.json)")
    parser.add_argument("--persona", default="niya",
                        help="persona id to load at startup (niya | peri | ...)")
    parser.add_argument("--debug-turns", action="store_true")
    parser.add_argument("--debug-output-dir", default=None)
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--enable-memory", action="store_true")
    memory_group.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--history-db-path", default=None,
                        help="SQLite file for conversation history (env: MINIMIND_DB_PATH)")
    parser.add_argument("--history-vault-root", default=None,
                        help="root of the editable Markdown mirror (env: MINIMIND_VAULT_PATH)")
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument("--enable-motion", action="store_true",
                              help="把用户话语解析成动作/表情计划下发给 VRM 化身")
    motion_group.add_argument("--disable-motion", action="store_true",
                              help="关闭化身动作解析 (env: MINIMIND_MOTION_ENABLED)")
    parser.add_argument("--motion-timeout", default=8.0, type=float,
                        help="动作意图请求超时秒数")
    args = parser.parse_args()
    memory_enabled = _bool_flag(os.getenv("MINIMIND_MEMORY_ENABLED"), False)
    if args.enable_memory:
        memory_enabled = True
    if args.disable_memory:
        memory_enabled = False
    motion_enabled = _bool_flag(os.getenv("MINIMIND_MOTION_ENABLED"), False)
    if args.enable_motion:
        motion_enabled = True
    if args.disable_motion:
        motion_enabled = False
    return RealtimeAppConfig(
        host=args.host,
        port=args.port,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=(os.getenv("MINIMIND_LLM_API_KEY") or "").strip() or None,
        max_history_turns=args.max_history_turns,
        asr_model_path=args.asr_model_path,
        asr_device=args.asr_device,
        vad_path=args.vad_path,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        tts_enabled=(not args.disable_tts),
        tts_model_path=args.tts_model_path,
        tts_lora_root=args.tts_lora_root,
        gpt_sovits_url=args.gpt_sovits_url,
        persona_root=args.persona_root,
        persona=args.persona,
        debug_turns=bool(args.debug_turns),
        debug_output_dir=args.debug_output_dir,
        memory_enabled=memory_enabled,
        motion_enabled=motion_enabled,
        motion_timeout=args.motion_timeout,
        history_db_path=(
            args.history_db_path
            or os.getenv("MINIMIND_DB_PATH")
            or str(REPO_ROOT / "runtime" / "data" / "whispera.db")
        ),
        history_vault_root=(
            args.history_vault_root
            or os.getenv("MINIMIND_VAULT_PATH")
            or str(REPO_ROOT / "runtime" / "memory_vault")
        ),
    )


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run realtime/app.py") from exc

    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
