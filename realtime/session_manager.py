from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    session_id: str
    max_turns: int = 6
    history: list[dict[str, str]] = field(default_factory=list)
    active_request_id: str | None = None
    # 当前化身形态（red/blue）。red=布景之形是对话默认形态；
    # 前端切换成功后通过 avatar.form 同步，模型据此判断是否真的需要换形态。
    avatar_form: str = "red"

    def add_user_turn(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        self.history.append({"role": "user", "content": value})
        self._trim()

    def add_assistant_turn(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        self.history.append({"role": "assistant", "content": value})
        self._trim()

    def clear(self) -> None:
        """Reset the short-term conversation context for a fresh dialogue."""
        self.history = []
        self.active_request_id = None

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = list(self.history)
        value = str(user_text or "").strip()
        if value:
            messages.append({"role": "user", "content": value})
        return messages

    def _trim(self) -> None:
        if self.max_turns <= 0:
            self.history = []
            return
        self.history = self.history[-(self.max_turns * 2) :]

