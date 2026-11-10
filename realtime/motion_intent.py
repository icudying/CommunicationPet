"""把用户话语解析成 化身服务 能直接执行的动作计划。

这是方案 B 的核心：原本由 化身服务 自带的 Qwen3.5-0.8B 做这件事，现改由
DeepSeek 统一承担（同一套 API Key / base_url，不额外引入本地模型进程）。

输出的 plan 形状与 化身服务 ``/v13-api/plan`` 的响应一致，由前端经
``window.languageV13.applyPlan(plan)`` 直接驱动角色，因此不需要它的
``service/qwen_*`` 与 llama-server。

本模块只依赖标准库（与项目既有的 llm-module 一样走 urllib）。
动作是锦上添花，任何失败都不应影响对话主链路，故 :meth:`MotionIntentService.plan`
只返回 ``None`` 而不抛异常。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

# 与 化身服务 models/skills/manifest.json + models/control/skills.json 对齐的 26 个动作键。
_ARM_MOTIONS = (
    "向上抬高",
    "向上抬到头部高度",
    "向上举过头部",
    "向下放低",
    "向前移动",
    "向后移动",
    "向左移动",
    "向右移动",
    "摆动手臂",
    "小幅自然活动",
)
MOTION_KEYS: tuple[str, ...] = (
    *(f"{side}-{motion}" for side in ("left", "right") for motion in _ARM_MOTIONS),
    "left-挥手打招呼",
    "right-挥手打招呼",
    "left-抬腿",
    "right-抬腿",
    "body-自然静止",
    "body-自然活动",
)
_MOTION_KEY_SET = frozenset(MOTION_KEYS)

# face/v9_expressions.mjs 的 EXPRESSION_LABELS 键。
# 前 13 个是通用语义表情；后面是给妮娅自定义表情（にこり/じと目/はぁと 等）
# 新增的细分情绪，由 configs/avatars.json 映射到具体形态键。
EXPRESSIONS: tuple[str, ...] = (
    "neutral",
    "friendly",
    "happy",
    "surprised",
    "confused",
    "shy",
    "angry",
    "sad",
    "pout",
    "admire",
    "loving",
    "heart",
    "star",
    "serious",
    "smirk",
    "wink",
    "playful",
    "sullen",
    "panic",
    "disgust",
    "tender",
    "cry",
    "excited",
    "cold",
    "melancholy",
)
_GESTURES = ("none", "left_wave", "right_wave")
_ENGAGEMENTS = ("brief", "conversation")
_OPERATIONS = ("chat", "command", "release")
# 两种妮娅形态：red=布景之形（伪装/好学生），blue=幻灭之形（鸣式侵蚀/真实）。
FORMS: tuple[str, ...] = ("red", "blue")
# 对话驱动的身体语言。动作库只有方向/高度类动作，所以这里做的是近似编排。
_BODY_GESTURES = ("none", "think", "hand_forward", "open_arms", "raise_both", "hands_low", "lively", "still")
_BODY_INTENTS: dict[str, list[dict[str, Any]]] = {
    "think": [
        {"key": "right-向上抬到头部高度", "strength": 0.55, "speed": 0.8, "hold": 4.0, "delay": 0.0},
    ],
    "hand_forward": [
        {"key": "right-向前移动", "strength": 0.5, "speed": 0.8, "hold": 3.0, "delay": 0.0},
    ],
    "open_arms": [
        {"key": "left-向左移动", "strength": 0.45, "speed": 0.8, "hold": 3.0, "delay": 0.0},
        {"key": "right-向右移动", "strength": 0.45, "speed": 0.8, "hold": 3.0, "delay": 0.0},
    ],
    "raise_both": [
        {"key": "left-向上抬高", "strength": 0.5, "speed": 0.9, "hold": 3.0, "delay": 0.0},
        {"key": "right-向上抬高", "strength": 0.5, "speed": 0.9, "hold": 3.0, "delay": 0.0},
    ],
    "hands_low": [
        {"key": "left-向下放低", "strength": 0.5, "speed": 0.8, "hold": 4.0, "delay": 0.0},
        {"key": "right-向下放低", "strength": 0.5, "speed": 0.8, "hold": 4.0, "delay": 0.0},
    ],
    "lively": [
        {"key": "left-小幅自然活动", "strength": 0.55, "speed": 1.0, "hold": 4.0, "delay": 0.0},
        {"key": "right-小幅自然活动", "strength": 0.55, "speed": 1.0, "hold": 4.0, "delay": 0.0},
    ],
    "still": [
        {"key": "body-自然静止", "strength": 0.5, "speed": 1.0, "hold": 4.0, "delay": 0.0},
    ],
}

# 保持当前底座状态（不切换姿态原型）。
_BASE_HOLD = {"change": False, "name": "保持", "motion": 0.3, "rest": False}

_SYSTEM_PROMPT = """你是虚拟角色妮娅的动作、表情与形态意图解析器。只输出一个 JSON 对象，不要任何解释文字。

需要输出的字段：
- operation: "chat"(普通聊天) | "command"(用户明确要求角色做动作) | "release"(要求停止或放下)
- base: {"change": 布尔, "name": 中文短语, "motion": 0~1, "rest": 布尔}
- intents: 数组，最多 4 项，每项 {"key": 动作键, "strength": 0~1, "speed": 0.4~2, "hold": 0~20, "delay": 0~30}
- timing: "同时" | "依次"
- presentation: {
    "expression": 表情,
    "gesture": "none"|"left_wave"|"right_wave",
    "engagement": "brief"|"conversation",
    "form": "red"|"blue",
    "form_immediate": 布尔,
    "body": "none"|"think"|"hand_forward"|"open_arms"|"raise_both"|"hands_low"|"lively"|"still"
  }

动作键必须严格从这个列表中选择，不得编造：
left-向上抬高, right-向上抬高, left-向上抬到头部高度, right-向上抬到头部高度,
left-向上举过头部, right-向上举过头部, left-向下放低, right-向下放低,
left-向前移动, right-向前移动, left-向后移动, right-向后移动,
left-向左移动, right-向左移动, left-向右移动, right-向右移动,
left-摆动手臂, right-摆动手臂, left-小幅自然活动, right-小幅自然活动,
left-挥手打招呼, right-挥手打招呼, left-抬腿, right-抬腿,
body-自然静止, body-自然活动

expression 只能是：
neutral, friendly, happy, surprised, confused, shy, angry, sad, pout, admire, loving,
heart, star, serious, smirk, wink, playful, sullen, panic, disgust, tender, cry,
excited, cold, melancholy

形态规则（角色的双形态）：
- red = 布景之形（伪装成好学生、泡泡、温柔礼貌）。默认形态；日常闲聊、打招呼、安慰对方、维持伪装时都用它。
- blue = 幻灭之形（鸣式侵蚀后的真实形貌、寂静虚无、负面情绪）。只在话题触及孤独、虚无、死亡、被利用/被抛弃、身份与记忆、残星会/鸣式/阿列夫一，或她真正生气、冷淡、认真、难过、不再伪装时使用。安慰对方时不要用 blue。
- 没有明显情绪变化时保持当前形态（current_form）；不要每轮都换。只有用户明确要求换形态时 form_immediate=true，否则 false。

身体语言规则（body，只能选一个，且 gesture 挥手时用 none）：
- think：困惑、思考、装傻时，右手抬到头部附近。
- hand_forward：解释、邀请、把话题递给你时，右手前伸。
- open_arms：开心、展示、欢迎时，双臂向外打开。
- raise_both：惊讶、撒娇、投降、兴奋时，双手抬高。
- hands_low：难过、道歉、泄气、收敛时，双手放低。
- lively：轻松愉快、聊嗨了时的小幅自然活动。
- still：严肃、生气、冷淡、不想动时，收回动作保持静止。
- none：不需要额外身体语言。

其他规则：
1. 普通聊天用 operation="chat"。若 presentation.body 不是 none，由后端自动转换成身体动作。
2. 只有用户明确要求角色做动作时才用 operation="command"，并把动作放进 intents。手指、手臂、胳膊都算手臂；腿、膝盖、脚算腿，腿目前只支持抬起。
3. 用户要求停止或放下当前动作时用 operation="release"、intents=[]。
4. 左/右指的是角色自身的左右，不是观察者的。
5. 打招呼、问候、道别时可用 presentation.gesture 挥手；其余一律 "none"。
6. 预期回复较短时 engagement="brief"，要展开聊时用 "conversation"。
7. 用户没有要求改变站姿时 base 用 {"change": false, "name": "保持", "motion": 0.3, "rest": false}。
8. strength 与 speed 未特别说明时都用 1。
9. hold 是维持该姿态的秒数，务必给足，否则角色刚抬起就放下了：抬起、举高、前伸这类保持姿态用 6~10；挥手、摆臂这类往复动作 2~4。
10. delay 是相对本次请求的开始时间，未说明用 0。
11. 结合最近对话和本轮用户话语判断情绪；表情要贴合妮娅此刻的台词，而不是机械地跟随用户情绪。"""


def _resolve_endpoint(base_url: str) -> str:
    """把 base_url 规范成 OpenAI 风格的 /v1/chat/completions（与 llm-module 保持一致）。"""
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.rsplit("/", 1)[-1].lower() != "v1":
        normalized = f"{normalized}/v1"
    return f"{normalized}/chat/completions"


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return max(low, min(high, number))


def _pick(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else fallback


def _normalize_intent(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    key = str(raw.get("key") or "").strip()
    if key not in _MOTION_KEY_SET:
        return None
    return {
        "key": key,
        "strength": _clamp(raw.get("strength"), 0.0, 1.0, 1.0),
        "speed": _clamp(raw.get("speed"), 0.4, 2.0, 1.0),
        # 缺省给足 6 秒：默认值太短会让角色刚抬起就放下，用户根本来不及看见。
        "hold": _clamp(raw.get("hold"), 0.0, 20.0, 6.0),
        "delay": _clamp(raw.get("delay"), 0.0, 30.0, 0.0),
    }


def normalize_plan(raw: Any, latency_seconds: float = 0.0) -> dict[str, Any] | None:
    """把模型输出收敛成 化身服务 的动作计划；无法使用时返回 None。"""
    if not isinstance(raw, dict):
        return None

    operation = _pick(raw.get("operation"), _OPERATIONS, "chat")
    intents: list[dict[str, Any]] = []
    if operation == "command":
        for item in raw.get("intents") or []:
            intent = _normalize_intent(item)
            if intent is not None:
                intents.append(intent)
            if len(intents) >= 4:
                break
        # command 没有任何可用动作就退化成纯聊天，避免空动作指令。
        if not intents:
            operation = "chat"
    if operation != "command":
        intents = []

    base_raw = raw.get("base") if isinstance(raw.get("base"), dict) else {}
    base = {
        "change": bool(base_raw.get("change", False)),
        "name": str(base_raw.get("name") or "保持")[:40] or "保持",
        "motion": _clamp(base_raw.get("motion"), 0.0, 1.0, _BASE_HOLD["motion"]),
        "rest": bool(base_raw.get("rest", False)),
    }
    if not base["change"]:
        base = dict(_BASE_HOLD)

    presentation_raw = raw.get("presentation") if isinstance(raw.get("presentation"), dict) else {}
    body = _pick(presentation_raw.get("body"), _BODY_GESTURES, "none")
    # form 缺省时保持 None：前端会沿用当前形态，避免每轮都被默认值拉回蓝色。
    form_raw = str(presentation_raw.get("form") or "").strip()
    form = form_raw if form_raw in FORMS else None
    presentation = {
        "expression": _pick(presentation_raw.get("expression"), EXPRESSIONS, "friendly"),
        "gesture": _pick(presentation_raw.get("gesture"), _GESTURES, "none"),
        "engagement": _pick(presentation_raw.get("engagement"), _ENGAGEMENTS, "brief"),
        "form": form,
        "form_immediate": bool(presentation_raw.get("form_immediate", False)),
        "body": body,
    }
    # 聊天时把情绪对应的身体语言转换成动作；它和用户显式命令是两回事，
    # 所以只加 1~2 个低强度动作，且不覆盖 command 的 intents。
    if operation == "chat" and body != "none":
        for intent in _BODY_INTENTS.get(body, [])[:2]:
            intents.append(dict(intent))

    timing = "依次" if len({intent["delay"] for intent in intents}) > 1 else "同时"
    return {
        "operation": operation,
        "base": base,
        "intents": intents,
        "timing": timing,
        "presentation": presentation,
        "seconds": round(float(latency_seconds), 3),
    }


class MotionIntentService:
    """调用 DeepSeek 把一句话解析成动作计划。线程安全（无共享可变状态）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str | None,
        *,
        timeout: float = 8.0,
        # DeepSeek 是推理模型，思考 token 与正文共用这个预算，且推理长度随请求难度
        # 剧烈变化：实测"抬起右手"只花 54 个推理 token，而"将右手放在胸口上"要 457 个。
        # 曾经设为 240，结果后者被截断（finish_reason=length）→ JSON 残缺 → 动作静默丢失。
        # 1536 约为实测峰值的 3 倍，留足余量；本次调用与主回复并行，慢一点不影响体感。
        max_tokens: int = 1536,
        temperature: float = 0.2,
    ) -> None:
        self._endpoint = _resolve_endpoint(base_url)
        self._api_key = str(api_key or "").strip()
        self._model = str(model or "").strip()
        self._timeout = float(timeout)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        # 最近一次失败的原因。plan() 对调用方只返回 None（动作是附加能力，不该
        # 影响对话），但静默失败会让"角色不动"无从排查，所以由调用方读它并记日志。
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self._endpoint and self._model and self._api_key)

    def plan(
        self,
        text: str,
        context: list[dict[str, str]] | None = None,
        current_form: str | None = None,
    ) -> dict[str, Any] | None:
        """解析本轮用户话语 + 最近双方对话；任何失败都返回 None。

        ``context`` 是最近的会话历史（user/assistant 交替），用于让表情、身体语言
        和形态跟随整段对话，而不是只看最后一句。失败原因写入 :attr:`last_error`。
        """
        self.last_error = None
        cleaned = str(text or "").strip()
        if not cleaned:
            self.last_error = "用户文本为空"
            return None
        if not self.available:
            missing = [
                name
                for name, present in (
                    ("endpoint", bool(self._endpoint)),
                    ("model", bool(self._model)),
                    ("api_key", bool(self._api_key)),
                )
                if not present
            ]
            self.last_error = "服务未就绪，缺少 " + "、".join(missing)
            return None

        history_lines: list[str] = []
        for item in (context or [])[-6:]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            role = "用户" if item.get("role") == "user" else "妮娅"
            history_lines.append(f"{role}：{content[:240]}")
        parts: list[str] = []
        if history_lines:
            parts.append("【最近对话】\n" + "\n".join(history_lines))
        if current_form in FORMS:
            parts.append(f"【当前形态】{current_form}")
        parts.append(f"【本轮用户】\n{cleaned[:300]}")
        user_content = "\n\n".join(parts)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:200]
            except Exception:  # noqa: BLE001
                pass
            self.last_error = f"HTTP {exc.code} {exc.reason} {detail}".strip()
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.last_error = f"请求失败（{self._timeout:g} 秒超时）：{exc}"
            return None
        except ValueError as exc:
            self.last_error = f"响应不是合法 JSON：{exc}"
            return None

        latency = time.perf_counter() - started_at
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            self.last_error = f"响应结构异常：{exc}"
            return None

        if not str(content or "").strip():
            finish = choice.get("finish_reason")
            usage = body.get("usage") or {}
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            self.last_error = (
                f"模型返回空内容（finish_reason={finish}, completion_tokens={usage.get('completion_tokens')}, "
                f"reasoning_tokens={reasoning}）——若是 length 请调大 max_tokens"
            )
            return None

        try:
            parsed = json.loads(content)
        except ValueError as exc:
            # 最常见的原因是 max_tokens 被推理占满后正文被截断，把 finish_reason 一并带上，
            # 否则"JSON 残缺"这个现象看不出是被截断的。
            self.last_error = (
                f"内容不是合法 JSON（finish_reason={choice.get('finish_reason')}）：{exc}；"
                f"原文前 120 字：{str(content)[:120]}"
            )
            return None

        plan = normalize_plan(parsed, latency)
        if plan is None:
            self.last_error = f"计划未通过校验：{str(parsed)[:160]}"
        return plan
