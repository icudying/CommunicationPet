from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .local_modules import ensure_llm_module_on_path


ensure_llm_module_on_path()

from llm_module import LLMConfig, LlamaServerLLM  # type: ignore  # noqa: E402
from llm_module.llm import VOICE_STRICT_PROMPT_PRESET, resolve_system_prompt  # type: ignore  # noqa: E402


# 话痨程度档位：从左到右滑动，AI 回复越来越长。
# "normal" 为默认档位，保留 max_tokens=2048 的硬约束（兼顾推理 token 与可朗读文本，
# 避免思考耗尽预算导致 "no speakable text"）。低档位 floor 设为 1280（远高于曾出
# 问题的 512），并主要依赖 system prompt hint 控制简短度；高档位提升预算与温度，
# 鼓励更详尽的展开。
TALKATIVENESS_PROFILES = {
    "minimal": {
        "key": "minimal",
        "label": "极简",
        "max_tokens": 1280,
        "temperature": 0.55,
        "hint": "请极其简短地回答，只说最核心的一点，通常一两句话即可。",
    },
    "concise": {
        "key": "concise",
        "label": "简洁",
        "max_tokens": 1536,
        "temperature": 0.60,
        "hint": "请简明扼要地回答，直奔主题，不要展开多余细节。",
    },
    "normal": {
        "key": "normal",
        "label": "普通",
        "max_tokens": 2048,
        "temperature": 0.65,
        "hint": "",
    },
    "talkative": {
        "key": "talkative",
        "label": "健谈",
        "max_tokens": 3072,
        "temperature": 0.72,
        "hint": "回答可以适当展开，补充一些细节和背景说明。",
    },
    "chatty": {
        "key": "chatty",
        "label": "话痨",
        "max_tokens": 4096,
        "temperature": 0.78,
        "hint": "请尽情展开回答，内容丰富详实，可以多说一些。",
    },
}


def resolve_talkativeness_profile(level) -> dict:
    """Return the talkativeness profile dict for ``level`` (falls back to normal)."""
    key = str(level or "normal").strip().lower()
    return TALKATIVENESS_PROFILES.get(key) or TALKATIVENESS_PROFILES["normal"]


@dataclass
class LLMRuntimeConfig:
    # This project talks to DeepSeek's OpenAI-compatible API by default; the
    # local llama-server + GGUF path was removed. base_url/model are still
    # overridden at runtime from RealtimeAppConfig.
    base_url: str = "https://api.deepseek.com"
    model: str | None = "deepseek-chat"
    api_key: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.65
    top_p: float = 0.88
    # 2048 leaves room for BOTH DeepSeek thinking/reasoning tokens AND the
    # actual content. The previous 512 was too small — thinking alone could
    # consume the entire budget, leaving no content and triggering the
    # "LLM produced reasoning tokens but no speakable text" error, especially
    # right after a context.clear when the model "thinks" more on a fresh
    # conversation.
    max_tokens: int = 2048
    timeout: float = 300.0
    # reasoning_budget / enable_thinking are llama.cpp chat-template kwargs; keep
    # them None so we never send those fields to DeepSeek (which would reject or
    # ignore them). They can still be opted into per-request via generate_stream.
    reasoning_budget: int | None = None
    enable_thinking: bool | None = None
    # 话痨程度档位 key (minimal | concise | normal | talkative | chatty)。
    # 由前端滑动开关通过 WebSocket llm.configure 消息运行时更新，下一轮 start_stream
    # 立即生效；start_stream 据此派生 max_tokens/temperature 并追加 hint 到 system prompt。
    talkativeness: str = "normal"


class LocalLLMClient:
    def __init__(self, config: LLMRuntimeConfig | None = None):
        self.config = config or LLMRuntimeConfig()
        self._client = LlamaServerLLM(
            LLMConfig(
                base_url=self.config.base_url,
                model=self.config.model,
                api_key=self.config.api_key,
                system_prompt=resolve_system_prompt(self.config.system_prompt, VOICE_STRICT_PROMPT_PRESET),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                reasoning_budget=self.config.reasoning_budget,
                enable_thinking=self.config.enable_thinking,
            )
        )

    def _resolve_system_prompt(self) -> str:
        return resolve_system_prompt(self.config.system_prompt, VOICE_STRICT_PROMPT_PRESET)

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def start_stream(self, messages: Iterable[dict[str, str]], memory_context: str | None = None):
        system_prompt = self._resolve_system_prompt()
        context = str(memory_context or "").strip()
        if context:
            system_prompt = f"{system_prompt}\n\n{context}"
        # Apply the talkativeness profile: append a verbosity hint to the
        # system prompt (after persona + memory) and override max_tokens /
        # temperature per-request. LlamaServerLLM.generate_stream accepts
        # these kwargs directly, so the persona's system_prompt is preserved
        # and only the derived generation budget/style changes per turn.
        profile = resolve_talkativeness_profile(getattr(self.config, "talkativeness", "normal"))
        hint = str(profile.get("hint") or "").strip()
        if hint:
            system_prompt = f"{system_prompt}\n\n{hint}"
        return self._client.generate_stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=profile["temperature"],
            top_p=self.config.top_p,
            max_tokens=profile["max_tokens"],
        )

    @staticmethod
    def close_stream(stream) -> None:
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            close()
        except (ValueError, RuntimeError) as exc:
            # The stream is iterated with next() on a worker thread (see the
            # asyncio.to_thread(_next_stream_item, ...) call in backend_main).
            # Cancelling that asyncio task does NOT stop the underlying thread,
            # so when a turn is interrupted (barge-in / context.clear / a new
            # turn timing out the old one) the reader may still be inside
            # next(). Calling close() then re-enters the running generator and
            # CPython raises ValueError("generator already executing") from
            # gen_close() [RuntimeError is listed only as a defensive alias].
            #
            # Swallowing it is safe: close() is only an early release of the
            # response. The generator is finalized as soon as the in-flight
            # next() returns and the last reference drops, which runs its
            # GeneratorExit path and closes the underlying HTTP response.
            if "generator already executing" not in str(exc):
                raise
