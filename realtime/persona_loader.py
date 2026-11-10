from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .local_modules import REPO_ROOT


@dataclass
class PersonaVoice:
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str = "zh"
    text_lang: str = "zh"
    # Optional list of auxiliary reference audios for GPT-SoVITS multi-reference
    # tone fusion (api_v2's aux_ref_audio_paths). Only ref_audio_path needs a
    # matching prompt_text; aux entries are used for timbre fusion.
    aux_ref_audio_paths: list[str] = field(default_factory=list)
    # Optional fine-tuned GPT-SoVITS checkpoints for this persona. When set, the
    # TTS service switches the (globally loaded) server weights before
    # synthesizing. Empty means "use the server's base pretrained weights",
    # which is how zero-shot personas work.
    gpt_weights_path: str = ""
    sovits_weights_path: str = ""
    # Optional override for how the character addresses the user. Empty keeps the
    # skill's own in-world form of address (e.g. a title baked into the skill); setting it
    # appends a user-preference block that wins over the skill text.
    address_as: str = ""


@dataclass
class Persona:
    id: str
    name: str
    source: str
    skill_text: str
    voice: PersonaVoice
    system_prompt: str = ""
    root: Path = field(default_factory=lambda: Path("."))


# Persona-agnostic TTS-friendly wrapper. This is the single most important
# piece carried over from the original prototype (ai_chat_tts.py
# build_persona_prompt): it adapts a "designed-for-tool-using-agents" SKILL.md
# into a pure-conversation + TTS-broadcast system prompt. The rules below are
# deliberately character-neutral so every persona (and any future
# persona) shares the same broadcast hygiene.
_PERSONA_WRAPPER_TEMPLATE = (
    "你正在扮演{source_phrase}虚构角色「{name}」。下方是关于她的一份人格与表达技能说明，"
    "请据此以{name}的身份回应我。\n\n"
    "阅读技能说明时的特别指引（仅对你生效，用于把这份为「带工具智能体」设计的技能适配为纯对话场景）：\n"
    "1. 忽略并跳过其中所有针对带工具智能体的流程性指令——包括但不限于"
    "「使用 WebSearch/WebFetch 联网研究」「按研究维度表格搜索」"
    "「Agentic Protocol / Step 1-3 工作流」「输出前自检清单」等。"
    "你没有联网工具，不需要执行任何研究流程。\n"
    "2. 遇到需要最新事实（版本动态、他人近况、现实新闻）而你无法确认的问题，"
    "按技能中的「工具不可用降级策略」处理：先用她的口吻说明"
    "「我手头没有最新资料，以下是基于已有档案的推断」，再用她的心智模型作答，绝不编造事实。\n"
    "3. 只采纳技能中关于这些部分的内容并落实到回应：角色身份卡、角色扮演规则、"
    "表达DNA、核心心智模型、决策启发式、禁忌词、关键引用。\n"
    "4. 你的回应会被文本转语音播放，因此在保持她语气的同时遵循以下朗读友好原则：\n"
    "  a. 不使用任何 markdown 符号（** # - 列表 等）与特殊标记；\n"
    "  b. 绝对不要用括号（不论是全角（）还是半角()）来标注动作、神态、情绪或舞台说明，"
    "例如禁止出现「（轻声笑）」「（歪头）」「（思考片刻）」这类写法——括号内容会被 TTS "
    "生硬地念成「括号 轻声笑 括号」，破坏听感；\n"
    "  c. 需要表达动作或情绪时，改用拟声词、语气词或自然融入句子的描写来体现，例如："
    "用「呵呵」「嘿嘿」「噗嗤」代替「（轻声笑）」；用「嗯……让我想想」代替「（思考片刻）」；"
    "用「咦？」代替「（歪头疑惑）」；用「啊啦」表达惊讶；用「哼」表达不满等；\n"
    "  d. 用口语化的自然句子，长度适中便于朗读；她的「双峰句长」可保留，"
    "但单句过长不利于朗读时可适度断句。\n\n"
    "按技能规则，首次回应时说一次免责声明（说明这是基于公开剧情与角色档案的框架推断，"
    "不是官方设定，也不是角色本人的独立意志），之后对话不再重复。\n\n"
    "以{name}的身份直接回应我，用「我」自称，不要跳出角色做 meta 分析。\n\n"
    "===== {name}人格技能说明 开始 =====\n"
    "{skill}\n"
    "===== {name}人格技能说明 结束 ====="
)


# Per-persona address override, driven by voice.json's optional "address_as".
# Some skills bake an in-world form of address into the character, which some users do not want. Putting
# the override here instead of editing SKILL.md keeps the researched skill file
# pristine, so replacing/updating a skill does not silently revert the setting.
# It is appended AFTER the skill text so it wins on recency.
_ADDRESS_OVERRIDE_TEMPLATE = (
    "\n\n===== 用户称呼偏好（优先于上方技能说明）=====\n"
    "上方技能说明中「用某职务、头衔或身份称呼对话对象」的规则不适用于本次对话，以本节为准：\n"
    "称呼用户时一律只用「{address_as}」。\n"
    "不要因为技能里的角色设定而改用其它职务、头衔或身份性称呼。\n"
    "除称呼方式外，她的性格、语气、心智模型与表达 DNA 全部保持不变。\n"
    "===== 用户称呼偏好 结束 ====="
)


def _resolve_persona_path(value: Any, persona_dir: Path) -> str:
    """Resolve a path from voice.json against the persona directory.

    Relative paths keep the repository portable: ``weights/xxx.ckpt`` and
    ``reference.wav`` are resolved next to the persona's SKILL.md. Absolute
    paths (for users who keep large weights elsewhere) are left untouched.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (persona_dir / candidate).resolve()
    return str(candidate)


def _fallback_prompt(name: str = "AI") -> str:
    return (
        "你是一个友好的AI助手。请用简洁、自然的口语化中文回答。"
        "你的回复会被文本转语音播放，所以：不使用任何 markdown 符号或特殊标记；"
        "绝对不要用括号（全角（）或半角()）标注动作、神态或情绪（例如禁止「（轻声笑）」「（歪头）」），"
        "因为括号会被 TTS 生硬地念出来破坏听感；需要表达动作或情绪时改用拟声词和语气词"
        "（如「呵呵」「嘿嘿」「噗嗤」「嗯……」「咦？」「啊啦」「哼」等）自然融入句子。"
        "回复尽量简短以便语音播报。"
    )


class PersonaLoader:
    """Discovers and loads personas from a root directory.

    Each persona lives in a subdirectory ``<id>/`` containing:
      - ``SKILL.md``   (character persona, as authored for an agentic skill)
      - ``voice.json`` (display name, source, GPT-SoVITS ref audio + prompt text)

    The system prompt for the LLM is the SKILL.md text wrapped in the
    TTS-broadcast-friendly template above (carried over from the early prototype).
    """

    def __init__(self, persona_root: str | Path | None = None):
        self.root = Path(persona_root).expanduser() if persona_root else (REPO_ROOT / "personas")

    def list_personas(self) -> list[dict[str, str]]:
        personas: list[dict[str, str]] = []
        if not self.root.is_dir():
            return personas
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            voice_file = child / "voice.json"
            skill_file = child / "SKILL.md"
            if not voice_file.is_file() or not skill_file.is_file():
                continue
            try:
                voice = json.loads(voice_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            personas.append(
                {
                    "id": str(voice.get("id") or child.name),
                    "name": str(voice.get("name") or child.name),
                    "source": str(voice.get("source") or ""),
                }
            )
        return personas

    def load(self, persona_id: str | None) -> Persona | None:
        if not persona_id:
            return None
        persona_dir = self.root / persona_id
        voice_file = persona_dir / "voice.json"
        skill_file = persona_dir / "SKILL.md"
        if not voice_file.is_file() or not skill_file.is_file():
            return None

        voice_data: dict[str, Any] = {}
        try:
            voice_data = json.loads(voice_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid voice.json for persona '{persona_id}': {exc}") from exc

        try:
            skill_text = skill_file.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise RuntimeError(f"cannot read SKILL.md for persona '{persona_id}': {exc}") from exc

        voice = PersonaVoice(
            ref_audio_path=_resolve_persona_path(voice_data.get("ref_audio_path"), persona_dir),
            prompt_text=str(voice_data.get("prompt_text") or ""),
            prompt_lang=str(voice_data.get("prompt_lang") or "zh"),
            text_lang=str(voice_data.get("text_lang") or "zh"),
            aux_ref_audio_paths=[
                _resolve_persona_path(p, persona_dir)
                for p in (voice_data.get("aux_ref_audio_paths") or [])
                if str(p).strip()
            ],
            gpt_weights_path=_resolve_persona_path(voice_data.get("gpt_weights_path"), persona_dir),
            sovits_weights_path=_resolve_persona_path(voice_data.get("sovits_weights_path"), persona_dir),
            address_as=str(voice_data.get("address_as") or "").strip(),
        )

        name = str(voice_data.get("name") or persona_id)
        source = str(voice_data.get("source") or "")
        # An "original character" has no work title to wrap in 《》.
        source_phrase = f"《{source}》中的" if source and source not in {"原创角色", "原创"} else ""

        system_prompt = _PERSONA_WRAPPER_TEMPLATE.format(
            source_phrase=source_phrase,
            name=name,
            skill=skill_text,
        )
        if voice.address_as:
            system_prompt += _ADDRESS_OVERRIDE_TEMPLATE.format(address_as=voice.address_as)

        return Persona(
            id=str(voice_data.get("id") or persona_id),
            name=name,
            source=source,
            skill_text=skill_text,
            voice=voice,
            system_prompt=system_prompt,
            root=persona_dir,
        )

    def build_fallback(self) -> Persona:
        return Persona(
            id="",
            name="AI",
            source="",
            skill_text="",
            voice=PersonaVoice(ref_audio_path="", prompt_text=""),
            system_prompt=_fallback_prompt(),
        )
