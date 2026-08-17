"""
core.task_plan
~~~~~~~~~~~~~~

The structured ``Task`` / ``TaskStep`` representation produced by the
planner and consumed by the executor.

A planner never executes anything directly - it returns a ``Task``.
The executor walks ``Task.steps`` in order, dispatches each one to
``CommandRouter``, verifies the result, retries / replans on failure,
and updates ``core.task_context.TaskContext`` with what happened.

The dataclasses are deliberately simple: no validation logic lives
here, only data. Validation lives in :mod:`core.planner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.base import ToolResult
from core.task_state import TaskState


@dataclass
class TaskStep:
    """One step in a plan.

    Attributes
    ----------
    id:
        1-based position in the plan.
    description:
        Human-readable description - used for TTS progress ("Opening
        VS Code", "Typing hello world", ...).
    tool_hint:
        Free-form hint for the executor. The executor converts it to
        a router command via :func:`_hint_to_router_text`; if the hint
        is unknown the executor still tries to call the router with
        ``description`` as a last resort.
    arguments:
        Free-form dict with whatever the LLM decided. The executor
        uses ``arguments`` to build the synthetic command string the
        router sees.
    status:
        ``pending`` / ``in_progress`` / ``done`` / ``failed`` /
        ``skipped``. Updated in-place by the executor.
    retries:
        Number of retry attempts the executor has already spent on
        this step. Capped by ``Config.task_max_retries``.
    result:
        The :class:`ToolResult` returned by the most recent attempt.
    error:
        Human-readable error string for the dashboard.
    """

    id: int
    description: str
    tool_hint: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    retries: int = 0
    result: Optional[ToolResult] = None
    error: Optional[str] = None


@dataclass
class Task:
    """A complete plan: a goal + ordered steps + execution state."""

    goal: str
    steps: List[TaskStep] = field(default_factory=list)
    state: TaskState = TaskState.PLANNING
    current_step: int = 0
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def pending_step(self) -> Optional[TaskStep]:
        for step in self.steps:
            if step.status in {"pending", "in_progress"}:
                return step
        return None


def hint_to_router_text(step: TaskStep) -> str:
    """Translate a ``TaskStep`` into a string ``CommandRouter.dispatch``
    can route.

    The executor calls this. ``arguments`` are flattened into the
    command string so the router's keyword matchers (which are
    ``needle in lowered`` checks) still fire.
    """
    hint = (step.tool_hint or "").strip().lower()
    args = step.arguments or {}

    if hint in {"open_app", "launch_app", "app"}:
        name = args.get("app") or args.get("name") or args.get("query")
        if not name:
            return step.description or "open"
        return f"open {name}"

    if hint in {"open_url", "website"}:
        url = args.get("url") or args.get("query")
        return f"go to {url}" if url else (step.description or "open url")

    if hint in {"google_search", "search_google", "search"}:
        q = args.get("query") or args.get("q")
        return f"search google for {q}" if q else (step.description or "search")

    if hint in {"youtube_search", "search_youtube"}:
        q = args.get("query") or args.get("q")
        return f"search youtube for {q}" if q else (step.description or "search youtube")

    if hint in {"type", "type_text", "typing"}:
        text = args.get("text") or args.get("content") or args.get("query")
        return f"type {text}" if text else (step.description or "type")

    if hint in {"press_key", "press", "key"}:
        key = args.get("key") or args.get("name")
        return f"press {key}" if key else (step.description or "press enter")

    if hint in {"wait", "sleep"}:
        secs = args.get("seconds") or args.get("duration") or 1
        return f"wait {secs}"

    if hint in {"screenshot", "capture_screen"}:
        return "take a screenshot"

    if hint in {"read_screen", "ocr"}:
        return "read text"

    if hint in {"send_email", "email"}:
        recipient = args.get("to") or args.get("recipient")
        subject = args.get("subject") or ""
        return f"send email to {recipient} subject {subject}".strip()

    if hint in {"send_whatsapp", "whatsapp"}:
        contact = args.get("contact") or args.get("to")
        return f"send whatsapp to {contact}"

    if hint in {"send_telegram", "telegram"}:
        return "send telegram"

    if hint in {"send_discord", "discord"}:
        return "send discord"

    if hint in {"send_slack", "slack"}:
        return "send slack"

    if hint in {"terminal", "run_command"}:
        cmd = args.get("command") or args.get("cmd")
        return f"run terminal {cmd}" if cmd else (step.description or "terminal")

    if hint in {"smart_room", "room"}:
        action = args.get("action") or "status"
        return f"room {action}"

    if hint in {"shutdown", "power"}:
        return "shutdown"

    if hint in {"reboot"}:
        return "reboot"

    if hint in {"volume", "brightness"}:
        level = args.get("level")
        verb = args.get("verb", "set")
        return f"{hint} {verb} {level}" if level else hint

    # Fallback: feed the description into the router so its keyword
    # matchers get a shot. ``CommandRouter.dispatch`` will try every
    # registered tool's ``can_handle`` on whatever string we give it.
    return step.description or hint or "do it"


__all__ = ["Task", "TaskStep", "hint_to_router_text"]
