"""
tests.run_phase2_tests
~~~~~~~~~~~~~~~~~~~~~~

Lightweight test harness for the autonomous-task + conversation
upgrade. Runs without pytest so we don't have to install an extra
dependency on Windows. Invoke with:

    .venv\\Scripts\\python.exe tests\\run_phase2_tests.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PASS = "PASS"
FAIL = "FAIL"


def _imports():
    from core.base import ToolResult
    from core.confirmation import ConfirmationManager, RiskLevel
    from core.conversation_context import ConversationContext
    from core.conversation_manager import (
        ConversationManager,
        ConversationState,
        UtteranceKind,
    )
    from core.planner import TaskPlanner, _build_plan_from_obj, _strip_fences
    from core.router import CommandRouter
    from core.task_context import TaskContext
    from core.task_executor import TaskExecutor
    from core.task_plan import Task, TaskStep, hint_to_router_text
    from core.task_state import TaskState, TaskStateInfo
    from core.verification import Verifier, VerifyOutcome
    # Assign explicitly to module globals so the test functions below
    # can reference these names as bare identifiers.
    g = globals()
    g["ToolResult"] = ToolResult
    g["RiskLevel"] = RiskLevel
    g["ConfirmationManager"] = ConfirmationManager
    g["ConversationManager"] = ConversationManager
    g["ConversationState"] = ConversationState
    g["ConversationContext"] = ConversationContext
    g["UtteranceKind"] = UtteranceKind
    g["TaskPlanner"] = TaskPlanner
    g["_strip_fences"] = _strip_fences
    g["_build_plan_from_obj"] = _build_plan_from_obj
    g["CommandRouter"] = CommandRouter
    g["TaskContext"] = TaskContext
    g["TaskExecutor"] = TaskExecutor
    g["Task"] = Task
    g["TaskStep"] = TaskStep
    g["hint_to_router_text"] = hint_to_router_text
    g["TaskState"] = TaskState
    g["TaskStateInfo"] = TaskStateInfo
    g["Verifier"] = Verifier
    g["VerifyOutcome"] = VerifyOutcome
    return g


# Populate module globals immediately so the test functions below can
# reference bare names without going through main(). Without this the
# closures fail with NameError when invoked standalone.
_imports()


def _run(test_func):
    name = test_func.__name__
    try:
        test_func()
    except Exception:
        print(f"[{FAIL}] {name}", flush=True)
        traceback.print_exc()
        return False
    print(f"[{PASS}] {name}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------
class StubTool:
    name = "stub_tool"

    def __init__(self, claimed: bool = True, success: bool = True, message: str = "ok"):
        self._claimed = claimed
        self._success = success
        self._message = message
        self.calls = []

    def can_handle(self, text: str) -> bool:
        return self._claimed

    def execute(self, command: str, context=None):
        self.calls.append(command)
        return ToolResult(success=self._success, message=self._message)


class StubRouter:
    def __init__(self, *, success: bool = True, message: str = "ok"):
        self.calls = []
        self._success = success
        self._message = message

    def dispatch(self, command: str, context=None, default=None):
        self.calls.append(command)
        return ToolResult(success=self._success, message=self._message)


class StubPlanner:
    def __init__(self, task):
        self._task = task
        self.plan_calls = []

    def plan(self, goal: str, ctx=None):
        self.plan_calls.append(goal)
        return self._task


class StubConfirm:
    def __init__(self, *, deny_risky: bool = False):
        self.calls = 0
        self._deny_risky = deny_risky

    def classify(self, step):
        self.calls += 1
        if step.tool_hint == "open_app":
            return RiskLevel.CONFIRMATION_REQUIRED
        return RiskLevel.SAFE

    def ask(self, step):
        return not self._deny_risky


def _make_planner(router, payload):
    class _StaticPlanner(TaskPlanner):
        def _invoke(self, prompt: str) -> str:
            return payload

    return _StaticPlanner(router=router)


def _fake_config(**overrides):
    from core.config import get_config as real

    base = real()
    cfg = type("Cfg", (), {})()
    for attr in dir(base):
        if attr.startswith("_"):
            continue
        setattr(cfg, attr, getattr(base, attr))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------
def test_planner_parses_single_step():
    payload = (
        '{"single_step": {"description": "Opening Chrome", '
        '"tool_hint": "open_app", "arguments": {"app": "chrome"}}}'
    )
    planner = _make_planner(None, payload)
    task = planner.plan("open chrome")
    assert len(task.steps) == 1
    assert task.steps[0].tool_hint == "open_app"
    assert task.steps[0].arguments.get("app") == "chrome"


def test_planner_parses_multi_step():
    payload = (
        '{"steps": ['
        '{"description": "Open VS Code", "tool_hint": "open_app", "arguments": {"app": "vscode"}}, '
        '{"description": "Type hello", "tool_hint": "type", "arguments": {"text": "hello"}}, '
        '{"description": "Press enter", "tool_hint": "press_key", "arguments": {"key": "enter"}}'
        "]}"
    )
    planner = _make_planner(None, payload)
    task = planner.plan("multi")
    assert len(task.steps) == 3
    assert [s.tool_hint for s in task.steps] == ["open_app", "type", "press_key"]


def test_planner_handles_clarification():
    payload = '{"clarification": "Which application?"}'
    planner = _make_planner(None, payload)
    task = planner.plan("open it")
    assert task.steps[0].tool_hint == "__clarify__"


def test_planner_strips_markdown_fences():
    assert _strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _strip_fences("```\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _strip_fences("{\"a\": 1}") == '{"a": 1}'


def test_planner_fallback_on_garbage():
    planner = _make_planner(None, "this is not json at all")
    task = planner.plan("open notepad")
    assert len(task.steps) == 1
    assert task.steps[0].description == "open notepad"


def test_planner_unknown_hint_is_kept():
    payload = '{"single_step": {"description": "do a thing", "tool_hint": "frobnicate", "arguments": {}}}'
    planner = _make_planner(None, payload)
    task = planner.plan("frobnicate the widget")
    assert task.steps[0].tool_hint == "frobnicate"


def test_build_plan_from_obj_rejects_unknown_shapes():
    assert _build_plan_from_obj({"random": "shape"}) is None
    assert _build_plan_from_obj([]) is None
    assert _build_plan_from_obj(None) is None
    assert _build_plan_from_obj({"steps": []}) is None


# ---------------------------------------------------------------------------
# hint_to_router_text
# ---------------------------------------------------------------------------
def test_hint_to_router_text_open_app():
    step = TaskStep(id=1, description="Opening VS Code", tool_hint="open_app", arguments={"app": "vscode"})
    assert hint_to_router_text(step) == "open vscode"


def test_hint_to_router_text_type():
    step = TaskStep(id=1, description="Typing hello", tool_hint="type", arguments={"text": "hello world"})
    assert hint_to_router_text(step) == "type hello world"


def test_hint_to_router_text_press_key():
    step = TaskStep(id=1, description="Press enter", tool_hint="press_key", arguments={"key": "enter"})
    assert hint_to_router_text(step) == "press enter"


def test_hint_to_router_text_falls_back_to_description():
    step = TaskStep(id=1, description="Do something weird", tool_hint="", arguments={})
    assert hint_to_router_text(step) == "Do something weird"


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------
def test_confirmation_classify_safe():
    step = TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={})
    assert ConfirmationManager().classify(step) is RiskLevel.SAFE


def test_confirmation_classify_high_risk_via_name():
    step = TaskStep(id=1, description="Shut down", tool_hint="pc_control", arguments={"name": "shutdown"})
    assert ConfirmationManager().classify(step) is RiskLevel.HIGH_RISK


def test_confirmation_classify_high_risk_via_hint_suffix():
    step = TaskStep(id=1, description="Delete file", tool_hint="file_delete", arguments={})
    assert ConfirmationManager().classify(step) is RiskLevel.HIGH_RISK


def test_confirmation_high_risk_requires_explicit_yes():
    assert ConfirmationManager._is_yes("okay", risk=RiskLevel.HIGH_RISK) is False
    assert ConfirmationManager._is_yes("yes please", risk=RiskLevel.HIGH_RISK) is True


def test_confirmation_low_risk_accepts_okay():
    assert ConfirmationManager._is_yes("okay", risk=RiskLevel.CONFIRMATION_REQUIRED) is True


def test_confirmation_negative_words_deny():
    assert ConfirmationManager._is_yes("no thanks", risk=RiskLevel.CONFIRMATION_REQUIRED) is False


def test_confirmation_empty_reply_is_no():
    assert ConfirmationManager._is_yes("", risk=RiskLevel.CONFIRMATION_REQUIRED) is False


def test_confirmation_safe_step_short_circuits():
    mgr = ConfirmationManager()
    mgr._confirm_callback = lambda *_: True
    step = TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={})
    assert mgr.ask(step) is True


# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------
def test_context_resolves_ordinal():
    ctx = TaskContext(last_search_results=["one.com", "two.com", "three.com"])
    assert "two.com" in ctx.resolve("open the second result")


def test_context_resolves_pronoun_to_current_file():
    ctx = TaskContext(current_file="notes.txt")
    assert "notes.txt" in ctx.resolve("open it again")


def test_context_resolves_pronoun_to_current_app():
    ctx = TaskContext(current_app="chrome")
    assert "chrome" in ctx.resolve("close it")


def test_context_has_referent():
    assert TaskContext().has_referent() is False
    assert TaskContext(current_app="chrome").has_referent() is True


def test_context_reset_clears_state():
    ctx = TaskContext(current_app="chrome", current_file="x.txt")
    ctx.reset()
    assert ctx.current_app is None
    assert ctx.current_file is None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def test_verifier_unknown_hint_trusts_router():
    step = TaskStep(id=1, description="do", tool_hint="unknown_thing", arguments={})
    assert Verifier().verify(step, ToolResult(success=True, message="ok")).verified is True


def test_verifier_type_trusts_router():
    step = TaskStep(id=1, description="type", tool_hint="type", arguments={})
    assert Verifier().verify(step, ToolResult(success=False, message="fail")).verified is False


def test_verifier_open_app_passes_with_running_process(monkeypatch=None):
    from core import verification

    orig = verification._running_process_names
    verification._running_process_names = lambda: {"chrome.exe"}
    try:
        step = TaskStep(id=1, description="open chrome", tool_hint="open_app", arguments={"app": "chrome"})
        out = verification.Verifier().verify(step, ToolResult(success=True, message="ok"))
        assert out.verified is True
        assert out.data and out.data.get("process") == "chrome.exe"
    finally:
        verification._running_process_names = orig


def test_verifier_open_app_fails_when_process_missing():
    from core import verification

    orig = verification._running_process_names
    verification._running_process_names = lambda: set()
    try:
        step = TaskStep(id=1, description="open chrome", tool_hint="open_app", arguments={"app": "chrome"})
        out = verification.Verifier().verify(step, ToolResult(success=True, message="ok"))
        assert out.verified is False
    finally:
        verification._running_process_names = orig


# ---------------------------------------------------------------------------
# TaskExecutor
# ---------------------------------------------------------------------------
def _simple_task():
    return Task(
        goal="open chrome",
        steps=[TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={"app": "chrome"})],
        state=TaskState.PLANNING,
    )


def _multi_task():
    return Task(
        goal="multi",
        steps=[
            TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={"app": "chrome"}),
            TaskStep(id=2, description="Type hi", tool_hint="type", arguments={"text": "hi"}),
        ],
        state=TaskState.PLANNING,
    )


def test_executor_runs_single_step_task():
    from core import verification
    from core.config import get_config

    orig = verification._running_process_names
    verification._running_process_names = lambda: {"chrome.exe"}
    try:
        # Patch the high-risk pause so we don't sleep in tests.
        cfg = _fake_config(confirmation_high_risk_pause_s=0.0)
        # The executor imports ``get_config`` lazily; patch the symbol
        # in the executor's namespace.
        import core.task_executor as tex
        orig_get_config = tex.get_config
        tex.get_config = lambda: cfg
        try:
            router = StubRouter()
            executor = TaskExecutor(router=router, ctx=TaskContext())
            thread = executor.run_async(_simple_task())
            thread.join(timeout=2)
            assert not thread.is_alive()
            snap = executor.state_snapshot()
            assert snap.state == "completed"
            assert snap.total_steps == 1
            assert router.calls == ["open chrome"]
        finally:
            tex.get_config = orig_get_config
    finally:
        verification._running_process_names = orig


def test_executor_retries_then_fails():
    from core import verification
    import core.task_executor as tex

    orig = verification._running_process_names
    verification._running_process_names = lambda: set()
    try:
        cfg = _fake_config(task_max_retries=1, confirmation_high_risk_pause_s=0.0)
        orig_get_config = tex.get_config
        tex.get_config = lambda: cfg
        try:
            router = StubRouter(success=False, message="nope")
            executor = TaskExecutor(router=router, ctx=TaskContext())
            thread = executor.run_async(_simple_task())
            thread.join(timeout=2)
            snap = executor.state_snapshot()
            assert snap.state == "failed"
            assert len(router.calls) == 2  # 1 initial + 1 retry
        finally:
            tex.get_config = orig_get_config
    finally:
        verification._running_process_names = orig


def test_executor_cancel_marks_cancelled():
    import core.task_executor as tex

    cfg = _fake_config(confirmation_high_risk_pause_s=0.0)
    orig_get_config = tex.get_config
    tex.get_config = lambda: cfg
    try:
        router = StubRouter()
        executor = TaskExecutor(router=router, ctx=TaskContext())

        # Cancel on first dispatch call.
        original_dispatch = router.dispatch

        def slow_dispatch(command, context=None, default=None):
            executor.cancel()
            return original_dispatch(command, context=context, default=default)

        router.dispatch = slow_dispatch
        thread = executor.run_async(_multi_task())
        thread.join(timeout=2)
        snap = executor.state_snapshot()
        assert snap.state in {"cancelled", "failed"}
    finally:
        tex.get_config = orig_get_config


def test_executor_high_risk_decline_skips_and_continues():
    from core import verification
    import core.task_executor as tex

    orig = verification._running_process_names
    verification._running_process_names = lambda: {"chrome.exe"}
    try:
        cfg = _fake_config(confirmation_high_risk_pause_s=0.0)
        orig_get_config = tex.get_config
        tex.get_config = lambda: cfg
        try:
            router = StubRouter()
            executor = TaskExecutor(
                router=router, ctx=TaskContext(), confirmation=StubConfirm(deny_risky=True)
            )
            thread = executor.run_async(_multi_task())
            thread.join(timeout=2)
            snap = executor.state_snapshot()
            assert snap.state == "completed"
            # Only the type step actually ran (open_app was denied).
            assert router.calls == ["type hi"]
        finally:
            tex.get_config = orig_get_config
    finally:
        verification._running_process_names = orig


def test_executor_runs_replan_when_planner_says_yes():
    from core import verification
    import core.task_executor as tex

    orig = verification._running_process_names
    verification._running_process_names = lambda: set()
    try:
        cfg = _fake_config(task_max_retries=0, task_max_replans=1, confirmation_high_risk_pause_s=0.0)
        orig_get_config = tex.get_config
        tex.get_config = lambda: cfg
        try:
            # Router fails the open_app step but succeeds the replanned
            # type step - we want to assert replan happened AND the
            # replacement ran.
            router = _ScriptedRouter([
                ("open chrome", False, "not found"),
                ("type ok", True, "ok"),
            ])
            replacement = Task(
                goal="recovered",
                steps=[TaskStep(id=99, description="Just type", tool_hint="type", arguments={"text": "ok"})],
            )
            executor = TaskExecutor(
                router=router, ctx=TaskContext(), planner=StubPlanner(replacement)
            )
            thread = executor.run_async(_simple_task())
            thread.join(timeout=2)
            snap = executor.state_snapshot()
            assert snap.state == "completed"
            assert snap.total_steps >= 1
            # The replan should have triggered a second dispatch with
            # the replacement's command.
            assert any(call == "type ok" for call in router.calls)
        finally:
            tex.get_config = orig_get_config
    finally:
        verification._running_process_names = orig


class _ScriptedRouter:
    """Router that returns per-command canned (success, message)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def dispatch(self, command, context=None, default=None):
        self.calls.append(command)
        for needle, ok, msg in self._script:
            if needle in command:
                return ToolResult(success=ok, message=msg)
        return ToolResult(success=True, message="ok")


# ---------------------------------------------------------------------------
# TaskState enum
# ---------------------------------------------------------------------------
def test_task_state_is_terminal():
    assert TaskState.COMPLETED.is_terminal() is True
    assert TaskState.FAILED.is_terminal() is True
    assert TaskState.CANCELLED.is_terminal() is True
    assert TaskState.EXECUTING.is_terminal() is False
    assert TaskState.IDLE.is_terminal() is False


# ---------------------------------------------------------------------------
# ConversationContext
# ---------------------------------------------------------------------------
def test_context_caps_at_max_turns():
    ctx = ConversationContext(max_turns=3)
    for i in range(10):
        ctx.append("user", f"msg {i}")
    assert len(ctx) == 3
    assert ctx.to_llm_messages()[0]["content"] == "msg 7"


def test_context_last_user_and_assistant():
    ctx = ConversationContext()
    ctx.append("user", "hi")
    ctx.append("assistant", "hello")
    ctx.append("user", "bye")
    assert ctx.last_user().content == "bye"
    assert ctx.last_assistant().content == "hello"


def test_context_clear_empties_history():
    ctx = ConversationContext()
    ctx.append("user", "hi")
    ctx.clear()
    assert len(ctx) == 0


def test_context_snapshot_is_serialisable():
    ctx = ConversationContext()
    ctx.append("user", "hi")
    snap = ctx.snapshot()
    assert snap["turns"][0]["role"] == "user"
    assert "ts" in snap["turns"][0]


# ---------------------------------------------------------------------------
# ConversationManager.classify + lifecycle
# ---------------------------------------------------------------------------
def test_classify_cancel_phrases():
    cm = ConversationManager()
    for phrase in ("cancel", "stop", "never mind", "cancel that", "abort"):
        assert cm.classify(phrase) is UtteranceKind.CANCEL, phrase


def test_classify_end_session_phrases():
    cm = ConversationManager()
    for phrase in ("goodbye", "bye", "that's all", "see ya"):
        assert cm.classify(phrase) is UtteranceKind.END_SESSION, phrase


def test_classify_wait_phrases():
    cm = ConversationManager()
    for phrase in ("wait", "hold on", "pause", "one second", "standby"):
        assert cm.classify(phrase) is UtteranceKind.WAIT, phrase


def test_classify_question():
    cm = ConversationManager()
    assert cm.classify("what time is it?") is UtteranceKind.QUESTION
    assert cm.classify("how are you") is UtteranceKind.QUESTION


def test_classify_bare_trigger_word():
    cm = ConversationManager()
    assert cm.classify("arvis") is UtteranceKind.WAIT


def test_classify_new_command():
    cm = ConversationManager()
    assert cm.classify("open chrome and search for cats") is UtteranceKind.NEW_COMMAND


def test_classify_followup_with_pronoun():
    cm = ConversationManager()
    cm._ctx = TaskContext(current_app="chrome")
    assert cm.classify("open it again") is UtteranceKind.FOLLOWUP


def test_classify_pronoun_without_context_is_new_command():
    cm = ConversationManager()
    assert cm.classify("do it") is UtteranceKind.NEW_COMMAND


def test_classify_empty_utterance():
    cm = ConversationManager()
    assert cm.classify("") is UtteranceKind.WAIT


def test_session_begin_and_end():
    cm = ConversationManager()
    assert cm.is_active() is False
    cm.begin_session()
    assert cm.is_active() is True
    assert cm.state() is ConversationState.LISTENING
    cm.end_session()
    assert cm.is_active() is False


def test_session_record_updates_state():
    cm = ConversationManager()
    cm.begin_session()
    cm.record("user", "hello")
    snap = cm.snapshot()
    assert snap.last_user == "hello"
    assert snap.turn_count == 1


def test_session_followup_expired_only_when_waiting():
    cm = ConversationManager(followup_timeout_s=0.0)
    cm.begin_session()
    assert cm.followup_expired() is False
    cm.wait_for_followup()
    time.sleep(0.05)
    assert cm.followup_expired() is True


def test_session_cancel_resets():
    cm = ConversationManager()
    cm.begin_session()
    cm.record("user", "do something")
    cm.cancel_session()
    assert cm.is_active() is False


def test_session_state_callback_fires():
    captured = []

    def cb(snap):
        captured.append(snap.state)

    cm = ConversationManager(state_callback=cb)
    cm.begin_session()
    cm.record("user", "hi")
    cm.end_session()
    assert captured[0] == "listening"
    assert captured[-1] == "idle"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TESTS = [
    # Planner
    test_planner_parses_single_step,
    test_planner_parses_multi_step,
    test_planner_handles_clarification,
    test_planner_strips_markdown_fences,
    test_planner_fallback_on_garbage,
    test_planner_unknown_hint_is_kept,
    test_build_plan_from_obj_rejects_unknown_shapes,
    # hint translation
    test_hint_to_router_text_open_app,
    test_hint_to_router_text_type,
    test_hint_to_router_text_press_key,
    test_hint_to_router_text_falls_back_to_description,
    # confirmation
    test_confirmation_classify_safe,
    test_confirmation_classify_high_risk_via_name,
    test_confirmation_classify_high_risk_via_hint_suffix,
    test_confirmation_high_risk_requires_explicit_yes,
    test_confirmation_low_risk_accepts_okay,
    test_confirmation_negative_words_deny,
    test_confirmation_empty_reply_is_no,
    test_confirmation_safe_step_short_circuits,
    # context
    test_context_resolves_ordinal,
    test_context_resolves_pronoun_to_current_file,
    test_context_resolves_pronoun_to_current_app,
    test_context_has_referent,
    test_context_reset_clears_state,
    # verifier
    test_verifier_unknown_hint_trusts_router,
    test_verifier_type_trusts_router,
    test_verifier_open_app_passes_with_running_process,
    test_verifier_open_app_fails_when_process_missing,
    # executor
    test_executor_runs_single_step_task,
    test_executor_retries_then_fails,
    test_executor_cancel_marks_cancelled,
    test_executor_high_risk_decline_skips_and_continues,
    test_executor_runs_replan_when_planner_says_yes,
    # state machine
    test_task_state_is_terminal,
    # conversation context
    test_context_caps_at_max_turns,
    test_context_last_user_and_assistant,
    test_context_clear_empties_history,
    test_context_snapshot_is_serialisable,
    # conversation manager
    test_classify_cancel_phrases,
    test_classify_end_session_phrases,
    test_classify_wait_phrases,
    test_classify_question,
    test_classify_bare_trigger_word,
    test_classify_new_command,
    test_classify_followup_with_pronoun,
    test_classify_pronoun_without_context_is_new_command,
    test_classify_empty_utterance,
    test_session_begin_and_end,
    test_session_record_updates_state,
    test_session_followup_expired_only_when_waiting,
    test_session_cancel_resets,
    test_session_state_callback_fires,
]


def main() -> int:
    # Imports already resolved at module-load time via the _imports()
    # call below. Nothing more to do here.
    pass

    passed = 0
    failed = 0
    for test in TESTS:
        ok = _run(test)
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
