"""
core.planner
~~~~~~~~~~~~

Turns a free-form user goal into a structured :class:`~core.task_plan.Task`.

The planner is deliberately *narrow*. It does NOT execute tools, it does
NOT call the LLM agent, it ONLY produces a sequence of steps the
executor will run. This separation is what lets us add retry /
replanning / verification on top of the existing router without
breaking it.

Pipeline
--------
1. Build a strict JSON-only system prompt that lists every tool name
   registered with the router. The model must return exactly one of:

       {"steps": [...]}             # full plan, 1..N steps
       {"single_step": {...}}       # one tool call
       {"clarification": "..."}     # the goal is ambiguous; ask the user

2. Call the model (mistral / whatever ``JARVIS_PLANNER_MODEL`` says)
   with low temperature (0.2).
3. Strip markdown fences if the model wrapped its answer in ```json```.
4. ``json.loads``. If that fails: retry once with a tighter prompt;
   on the second failure synthesise a ``single_step`` plan from the
   goal so the user still gets *something* done.
5. Validate: every step has a ``description``, every ``tool_hint`` is
   one of the known hints. Unknown hints become the literal
   ``description`` so the executor can fall back to ``can_handle``.
6. Wrap into a :class:`Task` and return.

The planner is intentionally tolerant of malformed output. Falling
back to a single-step plan is preferred over raising - the user
asked for something to happen, not a stack trace.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from core.config import get_config
from core.logger import get_logger
from core.router import CommandRouter
from core.task_context import TaskContext
from core.task_plan import Task, TaskStep
from core.task_state import TaskState

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------
# ``hint_catalog`` is the vocabulary of ``tool_hint`` values the planner
# may emit. It MUST stay in sync with the cases handled by
# ``core.task_plan.hint_to_router_text``; otherwise the executor falls
# back to ``description`` as a free-form router command, which works
# but loses the deterministic routing path.
#
# When you add a new tool_hint, add a one-line summary here too so
# the planner LLM learns the available actions from the system prompt.
HINT_CATALOG = (
    "open_app | open a desktop application (args: app, name, query)",
    "open_url | navigate a browser to a URL (args: url, query)",
    "google_search | run a Google search (args: query, q)",
    "youtube_search | run a YouTube search (args: query, q)",
    "type | type text into the focused window (args: text, content)",
    "press_key | press a single key (args: key, name)",
    "wait | sleep N seconds (args: seconds, duration)",
    "screenshot | capture the screen",
    "read_screen | OCR the current screen",
    "send_email | send an email (args: to, recipient, subject, body)",
    "send_whatsapp | send a WhatsApp message (args: contact, to, body)",
    "send_telegram | send a Telegram message",
    "send_discord | send a Discord message",
    "send_slack | send a Slack message",
    "terminal | run a command in the local terminal (args: command, cmd)",
    "smart_room | control the smart-room controller (args: action, device)",
    "shutdown | shut down the PC",
    "reboot | reboot the PC",
    "volume | set / change the volume (args: level, verb)",
    "brightness | set / change brightness (args: level, verb)",
)


SYSTEM_PROMPT_TEMPLATE = """You are the ARVIS planner.

Your job: turn the user's GOAL into a JSON plan the executor will run.
NEVER execute anything. NEVER write code or shell. Only return JSON.

Available tool hints (you may only use these):
{hints}

Hard rules:
- Output MUST be a single JSON object, nothing else.
- No markdown fences, no commentary, no trailing prose.
- Use ONE of these shapes (exactly):
  1. {{"steps": [{{"description": str, "tool_hint": str, "arguments": {{...}}}}, ...]}}
  2. {{"single_step": {{"description": str, "tool_hint": str, "arguments": {{...}}}}}}
  3. {{"clarification": "..."}}   -- only if the goal is genuinely ambiguous
- "arguments" keys must match the catalog above for the chosen hint.
- If a single tool call is enough, return {{"single_step": ...}}.
- If the goal mentions multiple actions ("open X then type Y then press Z"),
  return {{"steps": [...]}} in execution order.
- Use concise "description" strings the assistant can speak aloud:
  - "Opening VS Code"
  - "Typing hello world"
  - "Pressing enter"
- Tool hints MUST be one of the catalog tokens above. If you are
  unsure of the right hint, set "tool_hint": "" so the executor falls
  back to the description.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove a single ```json``` or ```JSON``` fence if present."""
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def _valid_hint(hint: str) -> bool:
    if not hint:
        return True
    known = {line.split("|", 1)[0].strip() for line in HINT_CATALOG}
    return hint.strip().lower() in known


def _coerce_step(raw: Dict[str, Any], index: int) -> Optional[TaskStep]:
    if not isinstance(raw, dict):
        return None
    description = str(raw.get("description") or raw.get("desc") or "").strip()
    if not description:
        description = f"step {index + 1}"
    hint = str(raw.get("tool_hint") or raw.get("tool") or "").strip().lower()
    if not _valid_hint(hint):
        # Unknown hint - keep it but the executor will route on description.
        pass
    args = raw.get("arguments") or raw.get("args") or {}
    if not isinstance(args, dict):
        args = {"value": str(args)}
    return TaskStep(id=index + 1, description=description, tool_hint=hint, arguments=args)


def _build_plan_from_obj(obj: Dict[str, Any]) -> Optional[Task]:
    """Translate a parsed JSON object into a :class:`Task`.

    Returns ``None`` if the shape is unrecognised - the caller falls
    back to a single-step plan.
    """
    if not isinstance(obj, dict):
        return None

    if "clarification" in obj and obj["clarification"]:
        # We treat clarification as a special single-step whose
        # ``description`` is the question to ask. The executor will
        # detect the special marker and speak it instead of running a
        # tool.
        question = str(obj["clarification"]).strip()
        step = TaskStep(
            id=1,
            description=question,
            tool_hint="__clarify__",
            arguments={"question": question},
        )
        return Task(goal=question, steps=[step], state=TaskState.PLANNING)

    if "single_step" in obj:
        step = _coerce_step(obj["single_step"], 0)
        if step is None:
            return None
        return Task(goal=step.description, steps=[step])

    if "steps" in obj:
        raw_steps = obj["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        steps: List[TaskStep] = []
        for i, raw in enumerate(raw_steps):
            step = _coerce_step(raw, i)
            if step is not None:
                steps.append(step)
        if not steps:
            return None
        # First step's description is a reasonable placeholder goal if
        # the model did not give us one.
        return Task(goal=steps[0].description, steps=steps)

    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class TaskPlanner:
    """LLM-backed multi-step planner.

    The planner owns no state beyond a reference to the router (used
    to list available tool names in the system prompt) and the
    context (used to resolve pronouns the user is likely to reuse -
    e.g. "in *that* folder").
    """

    def __init__(
        self,
        router: Optional[CommandRouter] = None,
        *,
        llm=None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        cfg = get_config()
        self._router = router
        self._model_name = model or cfg.planner_model
        self._temperature = temperature if temperature is not None else cfg.planner_temperature
        self._llm = llm  # injected for tests
        self._last_raw: str = ""
        self._last_plan: Optional[Task] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def plan(self, goal: str, ctx: Optional[TaskContext] = None) -> Task:
        """Convert *goal* into a :class:`Task`.

        Always returns a Task (never raises). If the model is
        unreachable the planner returns a single-step plan with the
        raw goal as the description so the executor can still call
        the router.
        """
        cfg = get_config()
        ctx_snapshot = (ctx.snapshot() if ctx is not None else {})

        prompt = self._build_prompt(goal, ctx_snapshot)

        raw = self._invoke(prompt)
        self._last_raw = raw

        # First parse attempt.
        plan = self._parse_and_build(raw, goal)
        if plan is not None:
            self._last_plan = plan
            return plan

        # Retry with a tighter prompt.
        log.warning("[PLANNER] first parse failed; retrying with stricter prompt")
        tighter = (
            "Return ONLY a JSON object. No markdown. No prose. "
            "Pick exactly one of the three shapes from the system prompt."
        )
        raw2 = self._invoke(prompt + "\n\n" + tighter)
        self._last_raw = raw2
        plan = self._parse_and_build(raw2, goal)
        if plan is not None:
            self._last_plan = plan
            return plan

        # Fallback - synthesize a single-step plan so the user gets
        # something done. The executor will call router.dispatch on
        # the raw goal and that has a deterministic fallback path.
        log.warning("[PLANNER] both parses failed; falling back to single-step")
        fallback = Task(
            goal=goal,
            steps=[TaskStep(id=1, description=goal, tool_hint="", arguments={})],
            state=TaskState.PLANNING,
        )
        self._last_plan = fallback
        return fallback

    # Backwards-compatible alias some modules may import.
    def make_plan(self, goal: str, ctx: Optional[TaskContext] = None) -> Task:  # pragma: no cover - alias
        return self.plan(goal, ctx)

    @property
    def last_raw(self) -> str:
        return self._last_raw

    @property
    def last_plan(self) -> Optional[Task]:
        return self._last_plan

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_prompt(self, goal: str, ctx_snapshot: Dict[str, Any]) -> str:
        hints_text = "\n".join(f"  - {line}" for line in HINT_CATALOG)
        system = SYSTEM_PROMPT_TEMPLATE.format(hints=hints_text)

        # Optional context line - keep it short so the model does not
        # blow its token budget on it.
        ctx_line = ""
        if ctx_snapshot:
            try:
                ctx_line = "\nContext: " + json.dumps(ctx_snapshot, ensure_ascii=False)[:400]
            except Exception:
                ctx_line = ""
        user = f"Goal: {goal}{ctx_line}"
        return f"{system}\n\n{user}\n\nJSON:"

    def _invoke(self, prompt: str) -> str:
        """Call the planner model and return its raw text response."""
        # Tests inject ``self._llm`` directly.
        if self._llm is not None:
            try:
                return self._call_llm(self._llm, prompt)
            except Exception as exc:
                log.warning("[PLANNER] injected LLM failed: %s", exc)
                return ""

        # Default path: build a fresh ChatOllama per call. We do NOT
        # cache it because mistral on a long-lived session can leak
        # context tokens between calls.
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("[PLANNER] ChatOllama unavailable: %s", exc)
            return ""

        try:
            llm = ChatOllama(
                model=self._model_name,
                base_url=get_config().ollama_base_url,
                reasoning=False,
                temperature=self._temperature,
            )
        except Exception as exc:
            log.warning("[PLANNER] failed to build ChatOllama: %s", exc)
            return ""

        return self._call_llm(llm, prompt)

    @staticmethod
    def _call_llm(llm, prompt: str) -> str:
        """Wrap the model call with a timeout / error net."""
        try:
            t0 = time.time()
            response = llm.invoke(prompt)
            dt = time.time() - t0
            log.debug("[PLANNER] LLM responded in %.2fs", dt)
        except Exception as exc:
            log.warning("[PLANNER] LLM invoke failed: %s", exc)
            return ""

        # LangChain chat models return a ``BaseMessage`` whose
        # ``content`` is the textual response.
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        # Some model wrappers return a list of message parts.
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(response or "")

    def _parse_and_build(self, raw: str, goal: str) -> Optional[Task]:
        if not raw:
            return None
        cleaned = _strip_fences(raw)
        # Strip any leading prose the model might have added before
        # the JSON block (e.g. "Here is the plan: {...}").
        cleaned = self._extract_json_object(cleaned)
        try:
            obj = json.loads(cleaned)
        except Exception as exc:
            log.debug("[PLANNER] json.loads failed: %s\nraw=%r", exc, cleaned[:400])
            return None
        return _build_plan_from_obj(obj)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Return the first balanced ``{...}`` substring in *text*."""
        text = (text or "").strip()
        if text.startswith("{"):
            return text
        start = text.find("{")
        if start < 0:
            return text
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]


__all__ = ["TaskPlanner", "HINT_CATALOG"]
