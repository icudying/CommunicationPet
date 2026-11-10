from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .local_modules import default_asr_model_path, ensure_repo_root_on_path


# funasr 1.0.27 (GPT-SoVITS runtime 自带版本) 的 SenseVoice 实现会从 checkpoint
# 元数据里读 dims 构建 ModelDimensions；但 ModelScope 上的 SenseVoiceSmall 快照
# 没带这些 dims 元数据，导致 ModelDimensions(**{}) 缺 10 个必填参数而报错。这
# 是硬性不兼容，所以本项目改用 GPT-SoVITS 自带且日常在用的 paraformer-large
# 快照（中文 ASR，单句直接返回纯文本，无需 rich_transcription_postprocess）。
# 下面按 config.yaml 的 model 字段自动识别两类模型，保留 SenseVoice 分支以便
# 日后 funasr 修复兼容性时可平滑切回。
def _detect_model_kind(model_path: str) -> str:
    cfg = Path(model_path) / "config.yaml"
    if not cfg.is_file():
        return "unknown"
    try:
        text = cfg.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"
    m = re.search(r"(?m)^\s*model\s*:\s*(\S+)", text)
    if not m:
        return "unknown"
    name = m.group(1)
    if "SenseVoice" in name:
        return "sense_voice"
    if "Paraformer" in name:
        return "paraformer"
    return "unknown"


@dataclass
class ASRRuntimeConfig:
    model_path: str = str(default_asr_model_path())
    device: str = "cuda"
    language: str = "auto"
    use_itn: bool = True
    strip_emotion: bool = True
    # Peak-normalize each utterance before inference so quiet segments reach a
    # consistent level. Amplification is capped to avoid blowing up noise/silence.
    normalize_audio: bool = True
    normalize_target_peak: float = 0.95
    normalize_max_gain: float = 8.0


_EMOTION_TAG_RE = re.compile(r"<\|[^|]+?\|>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emotion_markup(text: str) -> str:
    cleaned = _EMOTION_TAG_RE.sub("", text or "")
    cleaned = _EMOJI_RE.sub("", cleaned)
    return cleaned.strip()


def _peak_normalize(samples: np.ndarray, target_peak: float, max_gain: float) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-4:
        # Effectively silence; leave it alone to avoid amplifying noise.
        return samples
    gain = min(target_peak / peak, max_gain)
    if gain <= 1.0:
        return samples
    return samples * gain


class FunAsrService:
    def __init__(self, config: ASRRuntimeConfig | None = None):
        self.config = config or ASRRuntimeConfig()
        self._model = None
        self._kind: str | None = None
        # 仅 SenseVoice 需要 rich_transcription_postprocess 把 token id 序列解码
        # 成带 <|zh|> 等标记的文本；paraformer 直接返回纯文本字符串。
        self._postprocess = None
        self._warmup_result: dict[str, float | bool] | None = None
        # 串行化对 funasr 模型的访问（原因见 transcribe 的注释）。用 RLock 是因为
        # warmup() 持锁时会再调用 transcribe()。
        self._lock = threading.RLock()

    @property
    def is_warmed(self) -> bool:
        return self._warmup_result is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ensure_repo_root_on_path()
        from funasr import AutoModel

        kind = _detect_model_kind(self.config.model_path)
        self._kind = kind

        # paraformer 是 funasr 内置模型，无需 trust_remote_code；SenseVoice 需要。
        kwargs: dict = {
            "model": self.config.model_path,
            "device": self.config.device,
            "disable_update": True,
        }
        if kind == "sense_voice":
            kwargs["trust_remote_code"] = True
            from funasr.utils.postprocess_utils import rich_transcription_postprocess

            self._postprocess = rich_transcription_postprocess
        else:
            # paraformer / 未知：默认按 paraformer 协议处理（纯文本直出）。
            self._postprocess = None

        self._model = AutoModel(**kwargs)

    def transcribe(self, samples: np.ndarray) -> str:
        # funasr 的 AutoModel.generate() 内部持有活跃生成器（pipeline / 流式解码），
        # 不可重入：两个线程同时进入会抛 RuntimeError: generator already executing。
        # 本项目有多个 ASR 入口共享同一个 runtime.asr 实例——每一轮对话、/warmup，
        # 以及每个 WebSocket 会话；而 asyncio.to_thread 用的是多线程默认池，且
        # 对 asyncio task 执行 cancel() 并不会终止已进入的底层线程（它会把当前
        # transcribe 跑完）。所以中断/打断后新开的一轮，可能与尚未结束的上一轮
        # 同时进入模型。这里用锁把访问串行化，保证同一时刻只有一个线程进入模型。
        with self._lock:
            self._ensure_loaded()
            samples = np.asarray(samples, dtype=np.float32).reshape(-1)
            if self.config.normalize_audio:
                samples = _peak_normalize(
                    samples,
                    self.config.normalize_target_peak,
                    self.config.normalize_max_gain,
                )

            # SenseVoice 接受 language/use_itn 并产出带标记的文本；
            # paraformer 是中文专用模型，这两个参数不适用，generate 仅需 input+cache。
            if self._kind == "sense_voice":
                result = self._model.generate(
                    input=samples,
                    cache={},
                    language=self.config.language,
                    use_itn=self.config.use_itn,
                )
            else:
                result = self._model.generate(input=samples, cache={})
            if not result:
                return ""

            raw = result[0].get("text", "")
            if self._postprocess is not None:
                text = str(self._postprocess(raw)).strip()
            else:
                text = str(raw).strip()
            if self.config.strip_emotion:
                text = _strip_emotion_markup(text)
            return text

    def warmup(self, force: bool = False) -> dict[str, float | bool]:
        # 与 transcribe 共用同一把锁：/warmup 是独立 HTTP 路由，前端每次启动语音
        # 服务都会触发，可能在某一轮对话正在识别时并发进入模型。
        with self._lock:
            if self._warmup_result is not None and not force:
                return {
                    **self._warmup_result,
                    "cached": True,
                }

            total_started_at = perf_counter()
            model_loaded_before = self._model is not None

            load_started_at = perf_counter()
            self._ensure_loaded()
            asr_model_load_ms = (perf_counter() - load_started_at) * 1000.0
            if model_loaded_before:
                asr_model_load_ms = 0.0

            probe_audio = np.zeros(int(0.32 * 16000), dtype=np.float32)
            inference_started_at = perf_counter()
            self.transcribe(probe_audio)
            asr_inference_ms = (perf_counter() - inference_started_at) * 1000.0

            result = {
                "cached": False,
                "asr_model_load_ms": round(asr_model_load_ms, 2),
                "asr_inference_ms": round(asr_inference_ms, 2),
                "total_ms": round((perf_counter() - total_started_at) * 1000.0, 2),
            }
            self._warmup_result = result
            return result
