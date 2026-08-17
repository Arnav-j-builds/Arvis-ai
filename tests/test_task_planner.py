"""
tests.test_task_planner
~~~~~~~~~~~~~~~~~~~~~~~

Pure-unit tests for the autonomous-task layer.

We avoid hitting the real Ollama / LLM by stubbing the planner's
``_invoke`` and the executor's ``router.dispatch``. This keeps the
suite fast (well under 1s) and runnable on any environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure the project root is on sys.path so ``core`` imports resolve
# when pytest is run from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.base import ToolResult  # noqa: E402
from core.confirmation import ConfirmationManager, RiskLevel  # noqa: E402
from core.planner import TaskPlanner, _build_plan_from_obj, _strip_fences  # noqa: E402
from core.router import CommandRouter  # noqa: E402
from core.task_context import TaskContext  # noqa: E402
from core.task_executor import TaskExecutor  # noqa: E402
from core.task_plan import Task, TaskStep, hint_to_router_text  # noqa: E402
from core.task_state import TaskState  # noqa: E402
from core.verification import Verifier  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _StubTool:
    """Minimal ``BaseTool``-shaped stand-in used for router tests."""

    name = "stub_tool"

    def __init__(self, *, claimed: bool = True) -> None:
        self._claimed = claimed
        self.calls: int = 0

    def can_handle(self, text: str) -> bool:
        return self._claimed

    def execute(self, command: str, context=None):  # noqa: D401
        self.calls += 1
        return ToolResult(success=True, message=f"stub ok: {command}")


@pytest.fixture
def router() -> CommandRouter:
    r = CommandRouter()
    r.register(_StubTool(), keywords=("open ", "launch "), priority=100)
    return r


@pytest.fixture
def ctx() -> TaskContext:
    return TaskContext()


def _make_planner(router: CommandRouter, payload: str) -> TaskPlanner:
    """Build a planner whose ``_invoke`` returns *payload* verbatim."""

    class _StaticPlanner(TaskPlanner):
        def _invoke(self, prompt: str) -> str:
            return payload

    return _StaticPlanner(router=router)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def test_planner_parses_single_step(router: CommandRouter) -> None:
    payload = (
        '{"single_step": {"description": "Opening Chrome", '
        '"tool_hint": "open_app", "arguments": {"app": "chrome"}}}'
    )
    planner = _make_planner(router, payload)
    task = planner.plan("open chrome")
    assert len(task.steps) == 1
    step = task.steps[0]
    assert step.tool_hint == "open_app"
    assert step.arguments.get("app") == "chrome"


def test_planner_parses_multi_step(router: CommandRouter) -> None:
    payload = (
        '{"steps": ['
        '{"description": "Open VS Code", "tool_hint": "open_app", "arguments": {"app": "vscode"}}, '
        '{"description": "Type hello", "tool_hint": "type", "arguments": {"text": "hello"}}, '
        '{"description": "Press enter", "tool_hint": "press_key", "arguments": {"key": "enter"}}'
        "]}"
    )
    planner = _make_planner(router, payload)
    task = planner.plan("open vscode, type hello, press enter")
    assert len(task.steps) == 3
    assert [s.tool_hint for s in task.steps] == ["open_app", "type", "press_key"]


def test_planner_handles_clarification(router: CommandRouter) -> None:
    payload = '{"clarification": "Which application should I open?"}'
    planner = _make_planner(router, payload)
    task = planner.plan("open it")
    assert task.steps[0].tool_hint == "__clarify__"
    assert "Which application" in task.steps[0].description


def test_planner_strips_markdown_fences() -> None:
    assert _strip_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _strip_fences("```\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _strip_fences("{\"a\": 1}") == '{"a": 1}'


def test_planner_fallback_on_garbage(router: CommandRouter) -> None:
    """Bad JSON should produce a single-step plan with the raw goal."""
    planner = _make_planner(router, "this is not json at all")
    task = planner.plan("open notepad")
    assert len(task.steps) == 1
    # The fallback uses the raw goal as the description so the
    # executor's router dispatch still has a chance to match.
    assert task.steps[0].description == "open notepad"


def test_planner_retry_then_fallback(router: CommandRouter) -> None:
    """Two consecutive bad parses -> fallback plan."""

    class _Planner(TaskPlanner):
        def _invoke(self, prompt: str) -> str:
            return "totally not json"

    planner = _Planner(router=router)
    task = planner.plan("open chrome")
    assert len(task.steps) == 1
    assert task.steps[0].description == "open chrome"


def test_planner_unknown_hint_is_kept(router: CommandRouter) -> None:
    """Unknown tool_hint is preserved; executor falls back to description."""
    payload = (
        '{"single_step": {"description": "do a thing", '
        '"tool_hint": "frobnicate", "arguments": {}}}'
    )
    planner = _make_planner(router, payload)
    task = planner.plan("frobnicate the widget")
    assert task.steps[0].tool_hint == "frobnicate"


def test_build_plan_from_obj_rejects_unknown_shapes() -> None:
    assert _build_plan_from_obj({"random": "shape"}) is None
    assert _build_plan_from_obj([]) is None
    assert _build_plan_from_obj(None) is None
    assert _build_plan_from_obj({"steps": []}) is None


# ---------------------------------------------------------------------------
# hint_to_router_text
# ---------------------------------------------------------------------------
def test_hint_to_router_text_open_app() -> None:
    step = TaskStep(
        id=1,
        description="Opening VS Code",
        tool_hint="open_app",
        arguments={"app": "vscode"},
    )
    assert hint_to_router_text(step) == "open vscode"


def test_hint_to_router_text_type() -> None:
    step = TaskStep(
        id=1,
        description="Typing hello",
        tool_hint="type",
        arguments={"text": "hello world"},
    )
    assert hint_to_router_text(step) == "type hello world"


def test_hint_to_router_text_press_key() -> None:
    step = TaskStep(
        id=1,
        description="Press enter",
        tool_hint="press_key",
        arguments={"key": "enter"},
    )
    assert hint_to_router_text(step) == "press enter"


def test_hint_to_router_text_falls_back_to_description() -> None:
    step = TaskStep(id=1, description="Do something weird", tool_hint="", arguments={})
    assert hint_to_router_text(step) == "Do something weird"


# ---------------------------------------------------------------------------
# ConfirmationManager
# ---------------------------------------------------------------------------
def test_confirmation_classify_safe() -> None:
    step = TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={})
    assert ConfirmationManager().classify(step) is RiskLevel.SAFE


def test_confirmation_classify_high_risk_via_name() -> None:
    step = TaskStep(
        id=1,
        description="Shut down the PC",
        tool_hint="pc_control",
        arguments={"name": "shutdown"},
    )
    assert ConfirmationManager().classify(step) is RiskLevel.HIGH_RISK


def test_confirmation_classify_high_risk_via_hint_suffix() -> None:
    step = TaskStep(
        id=1,
        description="Delete file",
        tool_hint="file_delete",
        arguments={},
    )
    assert ConfirmationManager().classify(step) is RiskLevel.HIGH_RISK


def test_confirmation_high_risk_requires_explicit_yes() -> None:
    # "okay" alone should NOT count as HIGH_RISK approval.
    assert ConfirmationManager._is_yes("okay", risk=RiskLevel.HIGH_RISK) is False
    assert ConfirmationManager._is_yes("yes please", risk=RiskLevel.HIGH_RISK) is True


def test_confirmation_low_risk_accepts_okay() -> None:
    assert ConfirmationManager._is_yes("okay", risk=RiskLevel.CONFIRMATION_REQUIRED) is True


def test_confirmation_negative_words_deny() -> None:
    assert ConfirmationManager._is_yes("no thanks", risk=RiskLevel.CONFIRMATION_REQUIRED) is False
    assert ConfirmationManager._is_yes("cancel", risk=RiskLevel.HIGH_RISK) is False


def test_confirmation_empty_reply_is_no() -> None:
    assert ConfirmationManager._is_yes("", risk=RiskLevel.CONFIRMATION_REQUIRED) is False


def test_confirmation_ask_safe_step_returns_true_without_callback() -> None:
    mgr = ConfirmationManager()
    mgr._confirm_callback = lambda *_: True  # bypass voice
    step = TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={})
    assert mgr.ask(step) is True


# ---------------------------------------------------------------------------
# TaskContext pronoun resolution
# ---------------------------------------------------------------------------
def test_context_resolves_ordinal() -> None:
    ctx = TaskContext(last_search_results=["one.com", "two.com", "three.com"])
    out = ctx.resolve("open the second result")
    assert "two.com" in out


def test_context_resolves_pronoun_to_current_file() -> None:
    ctx = TaskContext(current_file="notes.txt")
    out = ctx.resolve("open it again")
    assert "notes.txt" in out


def test_context_resolves_pronoun_to_current_app() -> None:
    ctx = TaskContext(current_app="chrome")
    out = ctx.resolve("close it")
    assert "chrome" in out


def test_context_has_referent() -> None:
    assert TaskContext().has_referent() is False
    assert TaskContext(current_app="chrome").has_referent() is True


def test_context_reset_clears_state() -> None:
    ctx = TaskContext(current_app="chrome", current_file="x.txt")
    ctx.reset()
    assert ctx.current_app is None
    assert ctx.current_file is None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def test_verifier_unknown_hint_trusts_router() -> None:
    step = TaskStep(id=1, description="do", tool_hint="unknown_thing", arguments={})
    out = Verifier().verify(step, ToolResult(success=True, message="ok"))
    assert out.verified is True


def test_verifier_type_trusts_router() -> None:
    step = TaskStep(id=1, description="type", tool_hint="type", arguments={})
    out = Verifier().verify(step, ToolResult(success=False, message="fail"))
    assert out.verified is False


def test_verifier_open_app_passes_with_running_process(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend chrome.exe is running.
    from core import verification

    monkeypatch.setattr(verification, "_running_process_names", lambda: {"chrome.exe"})
    step = TaskStep(id=1, description="open chrome", tool_hint="open_app", arguments={"app": "chrome"})
    out = verification.Verifier().verify(step, ToolResult(success=True, message="ok"))
    assert out.verified is True
    assert out.data and out.data.get("process") == "chrome.exe"


def test_verifier_open_app_fails_when_process_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import verification

    monkeypatch.setattr(verification, "_running_process_names", lambda: set())
    step = TaskStep(id=1, description="open chrome", tool_hint="open_app", arguments={"app": "chrome"})
    out = verification.Verifier().verify(step, ToolResult(success=True, message="ok"))
    assert out.verified is False
    assert "looked_for" in (out.data or {})


# ---------------------------------------------------------------------------
# TaskExecutor
# ---------------------------------------------------------------------------
class _RecordingRouter:
    """Stub router that records dispatch calls and returns success."""

    def __init__(self, *, success: bool = True, message: str = "ok") -> None:
        self.calls: list = []
        self._success = success
        self._message = message

    def dispatch(self, command: str, context=None, default=None):
        self.calls.append(command)
        return ToolResult(success=self._success, message=self._message)


class _RecordingPlanner:
    """Stub planner that returns a pre-built Task on demand."""

    def __init__(self, task: Task) -> None:
        self._task = task
        self.plan_calls: list = []

    def plan(self, goal: str, ctx: TaskContext = None) -> Task:
        self.plan_calls.append(goal)
        return self._task


def _simple_task() -> Task:
    return Task(
        goal="open chrome",
        steps=[TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={"app": "chrome"})],
        state=TaskState.PLANNING,
    )


def _multi_task() -> Task:
    return Task(
        goal="multi",
        steps=[
            TaskStep(id=1, description="Open Chrome", tool_hint="open_app", arguments={"app": "chrome"}),
            TaskStep(id=2, description="Type hi", tool_hint="type", arguments={"text": "hi"}),
        ],
        state=TaskState.PLANNING,
    )


def test_executor_runs_single_step_task(monkeypatch: pytest.MonkeyPatch, ctx: TaskContext) -> None:
    from core import verification

    # Pretend chrome is running so the verifier accepts the open_app step.
    monkeypatch.setattr(verification, "_running_process_names", lambda: {"chrome.exe"})

    router = _RecordingRouter()
    executor = TaskExecutor(router=router, ctx=ctx)  # type: ignore[arg-type]
    task = _simple_task()
    thread = executor.run_async(task)
    thread.join(timeout=2)
    assert not thread.is_alive()
    snap = executor.state_snapshot()
    assert snap.state == "completed"
    assert snap.total_steps == 1
    assert router.calls == ["open chrome"]


def test_executor_retries_then_fails(monkeypatch: pytest.MonkeyPatch, ctx: TaskContext) -> None:
    from core import verification

    monkeypatch.setattr(verification, "_running_process_names", lambda: set())
    monkeypatch.setattr("core.config.get_config", lambda: _FakeConfig(max_retries=1))

    router = _RecordingRouter(success=False, message="nope")
    executor = TaskExecutor(router=router, ctx=ctx)  # type: ignore[arg-type]
    task = _simple_task()
    thread = executor.run_async(task)
    thread.join(timeout=2)
    snap = executor.state_snapshot()
    assert snap.state == "failed"
    # retries+1 attempts (initial + 1 retry)
    assert len(router.calls) == 2


def test_executor_cancel_marks_cancelled(ctx: TaskContext) -> None:
    router = _RecordingRouter()
    executor = TaskExecutor(router=router, ctx=ctx)  # type: ignore[arg-type]

    # Schedule the cancel right after the first dispatch attempt.
    original = router.dispatch

    def _slow_dispatch(command, context=None, default=None):
        executor.cancel()
        return original(command, context=context, default=default)

    router.dispatch = _slow_dispatch  # type: ignore[assignment]
    task = _multi_task()
    thread = executor.run_async(task)
    thread.join(timeout=2)
    snap = executor.state_snapshot()
    assert snap.state in {"cancelled", "failed"}


def test_executor_high_risk_decline_skips_and_continues(
    monkeypatch: pytest.MonkeyPatch, ctx: TaskContext
) -> None:
    """Confirm denial on CONFIRMATION_REQUIRED should mark the step
    skipped and continue (NOT cancel the whole task)."""
    from core import verification

    monkeypatch.setattr(verification, "_running_process_names", lambda: {"chrome.exe"})

    # Stub the confirmation manager to deny the risky step but allow
    # subsequent steps to run.
    class _StubConfirm:
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, step: TaskStep) -> RiskLevel:
            self.calls += 1
            # Only the open_app step is risky in this fixture.
            if step.tool_hint == "open_app":
                return RiskLevel.CONFIRMATION_REQUIRED
            return RiskLevel.SAFE

        def ask(self, step: TaskStep) -> bool:
            return False  # user said no

    router = _RecordingRouter()
    executor = TaskExecutor(router=router, ctx=ctx, confirmation=_StubConfirm())  # type: ignore[arg-type]
    task = _multi_task()
    thread = executor.run_async(task)
    thread.join(timeout=2)
    snap = executor.state_snapshot()
    # The task should still finish (the skipped step doesn't fail it).
    assert snap.state == "completed"
    # Both steps should have been visited (open_app denied -> skipped,
    # type step ran).
    assert len(router.calls) == 1  # only the type step ran


def test_executor_runs_replan_when_planner_says_yes(
    monkeypatch: pytest.MonkeyPatch, ctx: TaskContext
) -> None:
    from core import verification

    monkeypatch.setattr(verification, "_running_process_names", lambda: set())
    monkeypatch.setattr("core.config.get_config", lambda: _FakeConfig(max_retries=0, max_replans=1))

    router = _RecordingRouter(success=False, message="not found")
    new_task = Task(
        goal="recovered",
        steps=[TaskStep(id=99, description="Just type", tool_hint="type", arguments={"text": "ok"})],
    )
    executor = TaskExecutor(
        router=router,  # type: ignore[arg-type]
        ctx=ctx,
        planner=_RecordingPlanner(new_task),  # type: ignore[arg-type]
    )
    task = _simple_task()
    thread = executor.run_async(task)
    thread.join(timeout=2)
    snap = executor.state_snapshot()
    # After the original step fails (verification misses process), the
    # executor should replan and successfully run the replacement step.
    assert snap.state == "completed"
    assert snap.total_steps >= 1


class _FakeConfig:
    """Drop-in for ``core.config.get_config()`` with overrides."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        max_replans: int = 1,
        max_steps: int = 30,
        max_duration_s: float = 120.0,
        confirmation_high_risk_pause_s: float = 0.0,
        confirmation_listen_timeout_s: float = 0.5,
    ) -> None:
        from core.config import get_config as _real_get_config
        base = _real_get_config()
        # Copy fields we care about; preserve the rest.
        self.task_max_retries = max_retries
        self.task_max_replans = max_replans
        self.task_max_steps = max_steps
        self.task_max_duration_s = max_duration_s
        self.confirmation_high_risk_pause_s = confirmation_high_risk_pause_s
        self.confirmation_listen_timeout_s = confirmation_listen_timeout_s
        self.confirm_extra_tools = getattr(base, "confirm_extra_tools", [])
        self.planner_model = base.planner_model
        self.planner_temperature = base.planner_temperature
        self.ollama_base_url = base.ollama_base_url
        self.trigger_word = base.trigger_word
        self.conversation_history = base.conversation_history
        self.followup_timeout_s = base.followup_timeout_s
        self.conversation_max_turns = base.conversation_max_turns
        self.confirmation_high_risk_pause_s = confirmation_high_risk_pause_s


# ---------------------------------------------------------------------------
# Snapshot / state machine
# ---------------------------------------------------------------------------
def test_task_state_is_terminal() -> None:
    assert TaskState.COMPLETED.is_terminal() is True
    assert TaskState.FAILED.is_terminal() is True
    assert TaskState.CANCELLED.is_terminal() is True
    assert TaskState.EXECUTING.is_terminal() is False
    assert TaskState.IDLE.is_terminal() is False


def test_hint_to_router_text_handles_email_and_unknown() -> None:
    step = TaskStep(
        id=1,
        description="Send email",
        tool_hint="send_email",
        arguments={"to": "a@b.com", "subject": "hi"},
    )
    out = hint_to_router_text(step)
    assert "a@b.com" in out
    assert "send email" in out
