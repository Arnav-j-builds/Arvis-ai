"""
core.conversation_manager
~~~~~~~~~~~~~~~~~~~~~~~~~

Replaces the legacy "wake word → 30s timeout" loop with a real
session.

State machine
-------------
::

    IDLE  ── begin_session ──▶ LISTENING ── classify ─┐
                              ▲   │   │   │           │
                              │   │   │   ▼           ▼
                              │   │   │  THINKING    CANCEL_SESSION
                              │   │   │     │
                              │   │   │     ▼
                              │   │   │  SPEAKING
                              │   │   │     │
                              │   │   │     ▼
                              │   │   └─ WAITING_FOLLOWUP
                              │   │           │
                              │   │           ▼ (timeout / end phrase)
                              │   └────── ENDING
                              │                │
                              └──── end_session ┘

Public surface
--------------
* ``begin_session()`` / ``end_session()`` / ``cancel_session()``
* ``classify(utterance) -> UtteranceKind`` - rule-based, no LLM
* ``record(role, content, **meta)`` - append to bounded context
* ``set_state(state)`` - state transition helper that also
  broadcasts
* ``snapshot()`` - JSON-friendly view for the web API

Classification (no LLM)
-----------------------
Order of checks (first match wins):

1. Explicit cancel phrases ("stop", "cancel", "never mind", ...) -> ``CANCEL``
2. Explicit wait phrases ("wait", "hold on", "pause", ...)      -> ``WAIT``
3. Explicit end phrases ("goodbye", "that's all", "thanks bye", ...) -> ``END_SESSION``
4. Bare trigger word alone ("arvis")                            -> ``WAIT`` (standby)
5. Pronoun / deictic markers ("it", "that", "the first one")   -> ``FOLLOWUP``
6. Question marks / "what / how / when / who / why"            -> ``QUESTION``
7. Anything else                                                -> ``NEW_COMMAND``

We keep this classification rule-based so the mic loop stays fast
and offline. The LLM is reserved for the planner and the fallback
agent.
"""

from __future__ import annotations

import enum
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.config import get_config
from core.conversation_context import ConversationContext, ConversationTurn
from core.logger import get_logger
from core.speech import is_speaking, speak, stop_speaking
from core.task_context import TaskContext

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class ConversationState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WAITING_FOLLOWUP = "waiting_followup"
    ENDING = "ending"


class UtteranceKind(str, enum.Enum):
    CANCEL = "cancel"
    WAIT = "wait"
    END_SESSION = "end_session"
    NEW_COMMAND = "new_command"
    FOLLOWUP = "followup"
    QUESTION = "question"


_CANCEL_PHRASES = (
    "cancel",
    "cancel that",
    "stop",
    "stop it",
    "never mind",
    "forget it",
    "abort",
    "no stop",
    "halt",
)

_WAIT_PHRASES = (
    "wait",
    "hold on",
    "hold up",
    "pause",
    "one second",
    "one moment",
    "give me a second",
    "standby",
    "stand by",
)

_END_PHRASES = (
    "goodbye",
    "bye",
    "bye bye",
    "that's all",
    "that is all",
    "thanks bye",
    "thank you bye",
    "see you",
    "see ya",
    "go to sleep",
    "shut down jarvis",
    "that's it for today",
)

# Pronouns / deictics that strongly suggest a follow-up to the
# previous turn rather than a brand-new command.
_PRONOUN_TOKENS = (
    "it",
    "that",
    "this",
    "them",
    "those",
    "there",
    "the first",
    "the second",
    "the third",
    "the previous",
    "the last",
    "the one",
    "again",
    "too",
    "as well",
)

_QUESTION_START = (
    "what",
    "how",
    "when",
    "where",
    "who",
    "why",
    "which",
    "can you",
    "could you",
    "do you",
    "is there",
    "are there",
    "should",
)


def _norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_phrase(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
@dataclass
class ConversationSnapshot:
    state: str = ConversationState.IDLE.value
    session_started_at: float = 0.0
    last_turn_at: float = 0.0
    turn_count: int = 0
    last_user: Optional[str] = None
    last_assistant: Optional[str] = None
    barge_in_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "session_started_at": self.session_started_at,
            "last_turn_at": self.last_turn_at,
            "turn_count": self.turn_count,
            "last_user": self.last_user,
            "last_assistant": self.last_assistant,
            "barge_in_available": self.barge_in_available,
        }


class ConversationManager:
    """Stateful session manager wrapping the bounded turn context.

    Threading
    ---------
    Public methods are thread-safe. Internal state lives behind a
    single lock so the mic loop and the executor thread never race.
    """

    def __init__(
        self,
        ctx: Optional[TaskContext] = None,
        followup_timeout_s: Optional[float] = None,
        max_turns: Optional[int] = None,
        state_callback: Optional[Callable[[ConversationSnapshot], None]] = None,
    ) -> None:
        cfg = get_config()
        self._lock = threading.Lock()
        self._ctx = ctx or TaskContext()
        self._followup_timeout_s = (
            followup_timeout_s if followup_timeout_s is not None else cfg.followup_timeout_s
        )
        self._max_turns = max_turns if max_turns is not None else cfg.conversation_max_turns
        self._turns = ConversationContext(max_turns=max_turns)
        self._state = ConversationState.IDLE
        self._session_started = 0.0
        self._last_turn_at = 0.0
        self._state_cb = state_callback
        self._cancel_session_flag = threading.Event()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def begin_session(self) -> None:
        with self._lock:
            self._state = ConversationState.LISTENING
            self._session_started = time.time()
            self._cancel_session_flag.clear()
            self._last_turn_at = time.time()
            self._broadcast()
        log.info("[CONV] session begun")

    def end_session(self, *, reason: str = "user ended") -> None:
        with self._lock:
            if self._state == ConversationState.IDLE:
                return
            self._state = ConversationState.ENDING
            self._broadcast()
            self._state = ConversationState.IDLE
            self._session_started = 0.0
            self._cancel_session_flag.set()
            self._broadcast()
        log.info("[CONV] session ended: %s", reason)

    def cancel_session(self, *, reason: str = "user cancelled") -> None:
        """Hard cancel: stop TTS, mark session ended."""
        try:
            stop_speaking()
        except Exception:
            pass
        self.end_session(reason=reason)

    def is_active(self) -> bool:
        return self._state != ConversationState.IDLE

    # ------------------------------------------------------------------
    # Utterance classification
    # ------------------------------------------------------------------
    def classify(self, utterance: str) -> UtteranceKind:
        """Rule-based classification. No LLM call."""
        text = _norm(utterance)
        if not text:
            return UtteranceKind.WAIT

        # 1. Cancel.
        if _has_phrase(text, _CANCEL_PHRASES):
            return UtteranceKind.CANCEL

        # 2. End session.
        if _has_phrase(text, _END_PHRASES):
            return UtteranceKind.END_SESSION

        # 3. Wait / pause.
        if _has_phrase(text, _WAIT_PHRASES):
            return UtteranceKind.WAIT

        # 4. Bare trigger word.
        cfg = get_config()
        trigger = (cfg.trigger_word or "arvis").lower()
        if text == trigger:
            return UtteranceKind.WAIT

        # 5. Pronoun / deictic follow-up if the previous turn set up
        #    something we can refer to.
        if self._ctx.has_referent() and self._looks_like_followup(text):
            return UtteranceKind.FOLLOWUP

        # 6. Question.
        if "?" in utterance or text.startswith(_QUESTION_START):
            return UtteranceKind.QUESTION

        # 7. Default: new command.
        return UtteranceKind.NEW_COMMAND

    def _looks_like_followup(self, text: str) -> bool:
        if _has_phrase(text, _PRONOUN_TOKENS):
            return True
        # Very short utterances after a successful tool call are
        # almost always follow-ups ("open it", "type again").
        if len(text.split()) <= 3 and self._ctx.last_tool_result is not None:
            return True
        return False

    # ------------------------------------------------------------------
    # Turn recording
    # ------------------------------------------------------------------
    def record(self, role: str, content: str, **meta: Any) -> ConversationTurn:
        with self._lock:
            turn = self._turns.append(role, content, meta=meta)
            self._last_turn_at = time.time()
            self._broadcast()
            return turn

    @property
    def context(self) -> ConversationContext:
        return self._turns

    @property
    def task_context(self) -> TaskContext:
        return self._ctx

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------
    def set_state(self, state: ConversationState) -> None:
        with self._lock:
            self._state = state
            self._broadcast()

    def state(self) -> ConversationState:
        with self._lock:
            return self._state

    def snapshot(self) -> ConversationSnapshot:
        with self._lock:
            last_user_turn = self._turns.last_user()
            last_assistant_turn = self._turns.last_assistant()
            return ConversationSnapshot(
                state=self._state.value,
                session_started_at=self._session_started,
                last_turn_at=self._last_turn_at,
                turn_count=len(self._turns),
                last_user=last_user_turn.content if last_user_turn else None,
                last_assistant=last_assistant_turn.content if last_assistant_turn else None,
                barge_in_available=is_speaking(),
            )

    # ------------------------------------------------------------------
    # Follow-up timeout
    # ------------------------------------------------------------------
    def followup_expired(self) -> bool:
        """Return True if we have been waiting for a follow-up past the timeout."""
        with self._lock:
            if self._state != ConversationState.WAITING_FOLLOWUP:
                return False
            if not self._last_turn_at:
                return False
            return (time.time() - self._last_turn_at) > self._followup_timeout_s

    def wait_for_followup(self) -> None:
        with self._lock:
            self._state = ConversationState.WAITING_FOLLOWUP
            self._last_turn_at = time.time()
            self._broadcast()

    # ------------------------------------------------------------------
    # Barge-in
    # ------------------------------------------------------------------
    def interrupt_speech(self) -> None:
        """Stop TTS without ending the session."""
        try:
            stop_speaking()
        except Exception:
            pass
        self.set_state(ConversationState.LISTENING)

    # ------------------------------------------------------------------
    # Broadcast helper
    # ------------------------------------------------------------------
    def _broadcast(self) -> None:
        if self._state_cb is None:
            return
        try:
            self._state_cb(self.snapshot())
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("[CONV] state callback raised: %s", exc)


__all__ = [
    "ConversationManager",
    "ConversationState",
    "ConversationSnapshot",
    "UtteranceKind",
]
