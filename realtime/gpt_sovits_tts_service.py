from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generator

import numpy as np

from .tts_types import TTSRequestOptions, TTSRuntimeConfig


# GPT-SoVITS api_v2 default output sample rate (v2ProPlus = 32 kHz). The real
# value is parsed from the streaming WAV header on the first successful call;
# this is just the fallback used before warmup completes.
GPTSOVITS_DEFAULT_SAMPLE_RATE = 32000


def _parse_wav_header(header: bytes) -> tuple[int, int, int]:
    """Extract (sample_rate, num_channels, bits_per_sample) from a WAV header.

    GPT-SoVITS streaming_mode=2 emits a canonical 44-byte WAV header first
    (with a placeholder data-size), then raw little-endian PCM samples.
    We only need the fmt fields; the data chunk size is ignored because the
    stream is open-ended.
    """
    if len(header) < 44:
        raise ValueError(f"WAV header too short: {len(header)} bytes")
    if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    sample_rate = struct.unpack("<I", header[24:28])[0]
    num_channels = struct.unpack("<H", header[22:24])[0]
    bits_per_sample = struct.unpack("<H", header[34:36])[0]
    return int(sample_rate), int(num_channels), int(bits_per_sample)


def _pcm_bytes_to_float32(pcm: bytes, bits_per_sample: int) -> np.ndarray:
    """Decode little-endian PCM bytes into a float32 ndarray in [-1, 1)."""
    if bits_per_sample == 16:
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if bits_per_sample == 32:
        return np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    # 24-bit and 8-bit are rare for GPT-SoVITS; decode 24-bit best-effort.
    if bits_per_sample == 24:
        raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3)
        ints = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        # sign-extend 24-bit
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        return ints.astype(np.float32) / 8388608.0
    raise ValueError(f"unsupported bits_per_sample: {bits_per_sample}")


class GptSoVitsTtsService:
    """Streaming TTS backed by an external GPT-SoVITS api_v2 HTTP server.

    This replaces VoxCpmTtsService. Instead of loading a local VoxCPM model,
    it talks to a running GPT-SoVITS api_v2 server (default
    http://127.0.0.1:9880). Voice cloning is driven per request by the
    reference audio + prompt text carried in TTSRequestOptions
    (prompt_audio_path / prompt_text).

    The VoxCPM service kept a local torch model and yielded chunks from
    generate_streaming(); here we mirror that interface but pull PCM chunks
    out of GPT-SoVITS' streaming_mode=2 HTTP response so the existing realtime
    pipeline (segmenter -> TTS queue -> ws audio chunks) is unchanged.
    """

    def __init__(
        self,
        config: TTSRuntimeConfig | None = None,
        *,
        base_url: str = "http://127.0.0.1:9880",
        persona_root: str | Path | None = None,
    ):
        self.config = config or TTSRuntimeConfig()
        self.base_url = str(base_url or "http://127.0.0.1:9880").rstrip("/")
        self.persona_root = Path(persona_root).expanduser() if persona_root else None
        self._sample_rate = GPTSOVITS_DEFAULT_SAMPLE_RATE
        self._warmup_result: dict[str, float | bool | int | None] | None = None
        # GPT-SoVITS keeps weights in global server state, so a fine-tuned
        # persona needs a /set_gpt_weights + /set_sovits_weights round trip.
        # _loaded_weights caches the pair currently loaded so we only pay that
        # cost when the persona actually changes; the base pair below is what
        # the server booted with (tts_infer.yaml's `custom:` section) and is
        # restored when a zero-shot persona is selected again.
        gsv_root = os.environ.get("MINIMIND_GSV_ROOT", "").strip()
        self._base_gpt_weights = (
            str(Path(gsv_root) / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt") if gsv_root else ""
        )
        self._base_sovits_weights = (
            str(Path(gsv_root) / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth")
            if gsv_root
            else ""
        )
        self._loaded_weights: tuple[str, str] | None = None

    # ------------------------------------------------------------------ utils
    @property
    def is_warmed(self) -> bool:
        return self._warmup_result is not None

    def list_lora_checkpoints(self) -> list[dict[str, str | None]]:
        # GPT-SoVITS has no VoxCPM-style LoRA catalog. Personas are surfaced to
        # the UI through the persona system, so this returns an empty list and
        # the /tts/lora/catalog endpoint simply reports none.
        return []

    def _build_params(self, text: str, options: TTSRequestOptions | None) -> dict[str, object]:
        request_options = options or TTSRequestOptions()
        ref_audio = request_options.prompt_audio_path
        prompt_text = request_options.prompt_text
        if not ref_audio or not prompt_text:
            raise ValueError(
                "GPT-SoVITS needs both reference audio and prompt text. "
                "Configure a persona or send tts.configure with prompt_audio_path + prompt_text."
            )
        params: dict[str, object] = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text,
            "prompt_lang": "zh",
            "media_type": "wav",
            "streaming_mode": "2",
            "text_split_method": "cut0",
        }
        # Multi-reference tone fusion: only send aux_ref_audio_paths when the
        # persona actually configured auxiliary references. doseq=True in
        # _open_stream encodes the list as repeated query keys, which FastAPI
        # parses back into the list that api_v2's tts_get_endpoint expects.
        aux = [p for p in (request_options.aux_ref_audio_paths or []) if str(p).strip()]
        if aux:
            params["aux_ref_audio_paths"] = aux
        return params

    def _open_stream(self, params: dict[str, object]):
        # doseq=True so aux_ref_audio_paths (a list) is encoded as repeated
        # query keys (aux_ref_audio_paths=a&aux_ref_audio_paths=b), which
        # FastAPI parses into the list api_v2's tts_get_endpoint expects.
        url = f"{self.base_url}/tts?{urllib.parse.urlencode(params, doseq=True)}"
        request = urllib.request.Request(url, headers={"Accept": "audio/wav"}, method="GET")
        return urllib.request.urlopen(request, timeout=180)

    # ------------------------------------------------------------- weights
    def _set_server_weights(self, endpoint: str, weights_path: str) -> None:
        """Call api_v2's /set_gpt_weights or /set_sovits_weights (both GET)."""
        query = urllib.parse.urlencode({"weights_path": weights_path})
        request = urllib.request.Request(f"{self.base_url}{endpoint}?{query}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"GPT-SoVITS {endpoint} failed: {exc.code} {exc.reason} | {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"failed to reach GPT-SoVITS at {self.base_url}: {exc.reason}"
            ) from exc
        # api_v2 answers 200 + {"message": "success"}; anything else means the
        # checkpoint was rejected (wrong version, missing file, OOM...).
        if "success" not in body:
            raise RuntimeError(f"GPT-SoVITS {endpoint} unexpected response: {body}")

    def load_weights(self, options: TTSRequestOptions | None = None) -> tuple[str, str]:
        """Load the weights this persona needs, skipping the reload when the
        same pair is already active.

        A fine-tuned persona carries checkpoint paths and switches them via
        /set_gpt_weights + /set_sovits_weights; a zero-shot persona leaves them
        empty and restores the base pretrained pair the server booted with.
        Returns the active (gpt, sovits) pair.
        """
        request_options = options or TTSRequestOptions()
        gpt = str(request_options.gpt_weights_path or "").strip() or self._base_gpt_weights
        sovits = str(request_options.sovits_weights_path or "").strip() or self._base_sovits_weights
        desired = (gpt, sovits)
        if desired == self._loaded_weights:
            return desired
        if not gpt and not sovits:
            # Neither the persona nor MINIMIND_GSV_ROOT gave us a checkpoint:
            # keep whatever the server loaded at startup rather than guessing.
            print(
                "[TTS-WEIGHTS] no checkpoint configured and base weights unknown "
                "(MINIMIND_GSV_ROOT unset?); keeping server default",
                flush=True,
            )
            self._loaded_weights = desired
            return desired
        started_at = perf_counter()
        if gpt:
            self._set_server_weights("/set_gpt_weights", gpt)
        if sovits:
            self._set_server_weights("/set_sovits_weights", sovits)
        self._loaded_weights = desired
        print(
            f"[TTS-WEIGHTS] switched in {(perf_counter() - started_at) * 1000.0:.0f}ms "
            f"gpt={gpt or '(server default)'} sovits={sovits or '(server default)'}",
            flush=True,
        )
        return desired

    # ------------------------------------------------------------- public API
    def sample_rate(self, options: TTSRequestOptions | None = None) -> int:
        return int(self._sample_rate)

    def warmup(
        self,
        options: TTSRequestOptions | None = None,
        force: bool = False,
    ) -> dict[str, float | bool | int | None]:
        if self._warmup_result is not None and not force:
            return {**self._warmup_result, "cached": True}

        total_started_at = perf_counter()
        # We do NOT synthesize audio at warmup (that would need a valid persona
        # ref which may not be selected yet). Instead we probe the server with a
        # cheap GET / and record the round-trip. The real sample rate is latched
        # from the streaming WAV header on the first synthesis call below.
        probe_ms: float | None = None
        try:
            probe_started_at = perf_counter()
            with urllib.request.urlopen(f"{self.base_url}/", timeout=10) as resp:
                # any response (even 404) means the server is up
                _ = resp.read(1)
            probe_ms = (perf_counter() - probe_started_at) * 1000.0
        except Exception:
            # Server not reachable yet is not fatal at warmup; the first real
            # request will surface a clearer error.
            probe_ms = None

        # Preload this persona's checkpoints so the first spoken turn after a
        # persona switch does not pay the weight-switch latency mid-conversation.
        weights_error: str | None = None
        loaded_weights: tuple[str, str] = ("", "")
        try:
            loaded_weights = self.load_weights(options)
        except Exception as exc:
            # Non-fatal here for the same reason the probe above is: the first
            # real synthesis call will raise the same error with more context.
            weights_error = str(exc)
            print(f"[TTS-WEIGHTS] warmup load failed: {weights_error}", flush=True)

        result = {
            "cached": False,
            "gpt_sovits_url": self.base_url,
            "gpt_sovits_probe_ms": round(probe_ms, 2) if probe_ms is not None else None,
            "tts_sample_rate": int(self._sample_rate),
            "weights_gpt": loaded_weights[0] or None,
            "weights_sovits": loaded_weights[1] or None,
            "weights_error": weights_error,
            "total_ms": round((perf_counter() - total_started_at) * 1000.0, 2),
        }
        self._warmup_result = result
        return result

    def stream_tts(
        self,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        options: TTSRequestOptions | None = None,
    ) -> Generator[np.ndarray, None, None]:
        request_options = options or TTSRequestOptions()
        # Allow direct prompt_wav_path/prompt_text args to override options
        # (mirrors the VoxCPM service signature).
        if prompt_wav_path and not request_options.prompt_audio_path:
            request_options = TTSRequestOptions(
                **{**request_options.__dict__, "prompt_audio_path": prompt_wav_path}
            )
        if prompt_text and not request_options.prompt_text:
            request_options = TTSRequestOptions(
                **{**request_options.__dict__, "prompt_text": prompt_text}
            )

        # Safety net: if a turn starts before warmup (or the persona changed
        # since warmup), make sure the server is holding these checkpoints.
        self.load_weights(request_options)
        params = self._build_params(text, request_options)

        try:
            response = self._open_stream(params)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"failed to connect to GPT-SoVITS at {self.base_url}: {exc.reason}. "
                "Make sure the api_v2 server is running (default port 9880)."
            ) from exc

        produced = False
        try:
            # 1) read & parse the WAV header
            header = response.read(44)
            if not header:
                raise RuntimeError("GPT-SoVITS returned an empty response")
            sample_rate, _num_channels, bits_per_sample = _parse_wav_header(header)
            self._sample_rate = sample_rate

            # 2) stream the remaining raw PCM and emit float32 chunks
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                audio = _pcm_bytes_to_float32(chunk, bits_per_sample)
                if audio.size == 0:
                    continue
                produced = True
                yield audio

            if not produced:
                raise RuntimeError("GPT-SoVITS streamed a header but no PCM samples")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GPT-SoVITS /tts failed: {exc.code} {exc.reason} | {body}") from exc
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
