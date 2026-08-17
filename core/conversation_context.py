"""
core.conversation_context
~~~~~~~~~~~~~~~~~~~~~~~~~

Bounded, in-memory turn history for the conversation manager.

Design
------
* Each :class:`ConversationTurn` is one user or assistant utterance,
  tagged with role + content + an optional ``meta`` dict that the
  conversation manager fills in (tool name, success flag, etc.).
* The list is capped at ``Config.conversation_history`` (default 20).
  When the cap is hit the oldest turn is dropped - this keeps the
  LLM prompt bounded.
* ``to_llm_messages()`` returns the turns in the OpenAI /
  LangChain ``[{"role": ..., "content": ...}]`` shape that
  ``ChatOllama`` expects. ``tool`` and assistant tool-call fields
  are intentionally NOT generated - arvis does not stream tool
  calls into the conversation history (the executor already ran
  them; the manager only needs the human-readable summary).

The context is NOT thread-safe; the conversation manager holds a
lock around every mutation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.config import get_config


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant" | "system"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_llm_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationContext:
    """Bounded turn history + helpers for the conversation manager."""

    def __init__(self, max_turns: Optional[int] = None) -> None:
        cfg = get_config()
        cap = max_turns if max_turns is not None else cfg.conversation_history
        self._turns: Deque[ConversationTurn] = deque(maxlen=max(1, cap))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def append(self, role: str, content: str, *, meta: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        import time

        turn = ConversationTurn(
            role=role,
            content=content,
            meta=dict(meta or {}),
            ts=time.time(),
        )
        self._turns.append(turn)
        return turn

    def clear(self) -> None:
        self._turns.clear()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def turns(self) -> List[ConversationTurn]:
        return list(self._turns)

    def last_user(self) -> Optional[ConversationTurn]:
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn
        return None

    def last_assistant(self) -> Optional[ConversationTurn]:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn
        return None

    def to_llm_messages(self) -> List[Dict[str, str]]:
        return [t.to_llm_message() for t in self._turns]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "turn_count": len(self._turns),
            "turns": [t.to_dict() for t in self._turns],
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._turns)


__all__ = ["ConversationContext", "ConversationTurn"]
