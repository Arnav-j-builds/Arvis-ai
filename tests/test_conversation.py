"""
tests.test_conversation
~~~~~~~~~~~~~~~~~~~~~~~

Pure-unit tests for the conversation manager + bounded context.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conversation_context import ConversationContext, ConversationTurn  # noqa: E402
from core.conversation_manager import (  # noqa: E402
    ConversationManager,
    ConversationState,
    UtteranceKind,
)
from core.task_context import TaskContext  # noqa: E402


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------
def test_context_caps_at_max_turns() -> None:
    ctx = ConversationContext(max_turns=3)
    for i in range(10):
        ctx.append("user", f"msg {i}")
    assert len(ctx) == 3
    assert ctx.to_llm_messages()[0]["content"] == "msg 7"


def test_context_last_user_and_assistant() -> None:
    ctx = ConversationContext()
    ctx.append("user", "hi")
    ctx.append("assistant", "hello")
    ctx.append("user", "bye")
    assert ctx.last_user().content == "bye"
    assert ctx.last_assistant().content == "hello"


def test_context_clear_empties_history() -> None:
    ctx = ConversationContext()
    ctx.append("user", "hi")
    ctx.clear()
    assert len(ctx) == 0


def test_context_snapshot_is_serialisable() -> None:
    ctx = ConversationContext()
    ctx.append("user", "hi")
    snap = ctx.snapshot()
    assert "turns" in snap
    assert snap["turns"][0]["role"] == "user"
    assert "ts" in snap["turns"][0]


# ---------------------------------------------------------------------------
# ConversationManager.classify
# ---------------------------------------------------------------------------
@pytest.fixture
def conversation() -> ConversationManager:
    return ConversationManager()


@pytest.mark.parametrize(
    "phrase",
    ["cancel", "stop", "never mind", "cancel that", "abort", "stop it"],
)
def test_classify_cancel_phrases(conversation: ConversationManager, phrase: str) -> None:
    assert conversation.classify(phrase) is UtteranceKind.CANCEL


@pytest.mark.parametrize("phrase", ["goodbye", "bye", "that's all", "see ya"])
def test_classify_end_session_phrases(conversation: ConversationManager, phrase: str) -> None:
    assert conversation.classify(phrase) is UtteranceKind.END_SESSION


@pytest.mark.parametrize("phrase", ["wait", "hold on", "pause", "one second", "standby"])
def test_classify_wait_phrases(conversation: ConversationManager, phrase: str) -> None:
    assert conversation.classify(phrase) is UtteranceKind.WAIT


def test_classify_question(conversation: ConversationManager) -> None:
    assert conversation.classify("what time is it?") is UtteranceKind.QUESTION
    assert conversation.classify("how are you") is UtteranceKind.QUESTION


def test_classify_bare_trigger_word(conversation: ConversationManager) -> None:
    assert conversation.classify("arvis") is UtteranceKind.WAIT


def test_classify_new_command(conversation: ConversationManager) -> None:
    assert conversation.classify("open chrome and search for cats") is UtteranceKind.NEW_COMMAND


def test_classify_followup_with_pronoun(conversation: ConversationManager) -> None:
    ctx = TaskContext(current_app="chrome")
    conversation._ctx = ctx  # wire the fixture
    assert conversation.classify("open it again") is UtteranceKind.FOLLOWUP


def test_classify_pronoun_without_context_is_new_command(
    conversation: ConversationManager,
) -> None:
    # No referent -> even with "it" we treat it as a new command.
    assert conversation.classify("do it") is UtteranceKind.NEW_COMMAND


def test_classify_empty_utterance(conversation: ConversationManager) -> None:
    assert conversation.classify("") is UtteranceKind.WAIT
    assert conversation.classify("   ") is UtteranceKind.WAIT


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
def test_session_begin_and_end(conversation: ConversationManager) -> None:
    assert conversation.is_active() is False
    conversation.begin_session()
    assert conversation.is_active() is True
    assert conversation.state() is ConversationState.LISTENING
    conversation.end_session()
    assert conversation.is_active() is False
    assert conversation.state() is ConversationState.IDLE


def test_session_record_updates_state(conversation: ConversationManager) -> None:
    conversation.begin_session()
    conversation.record("user", "hello")
    snap = conversation.snapshot()
    assert snap.last_user == "hello"
    assert snap.turn_count == 1


def test_session_followup_expired_only_when_waiting() -> None:
    conversation = ConversationManager(followup_timeout_s=0.0)
    conversation.begin_session()
    # Not in WAITING_FOLLOWUP yet -> never expired.
    assert conversation.followup_expired() is False
    conversation.wait_for_followup()
    time.sleep(0.05)
    assert conversation.followup_expired() is True


def test_session_cancel_resets(conversation: ConversationManager) -> None:
    conversation.begin_session()
    conversation.record("user", "do something")
    conversation.cancel_session()
    assert conversation.is_active() is False


def test_session_state_callback_fires() -> None:
    captured = []

    def cb(snap):
        captured.append(snap.state)

    conv = ConversationManager(state_callback=cb)
    conv.begin_session()
    conv.record("user", "hi")
    conv.end_session()
    assert captured[0] == "listening"
    assert captured[-1] == "idle"
