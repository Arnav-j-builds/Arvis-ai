"""
core.intent_tools
~~~~~~~~~~~~~~~~~

Wires the four new features (Universal Screen Context, Visual Action
Engine, ARVIS Skill Builder, ARVIS Browser Agent) into the existing
:mod:`core.router` as a single :class:`BaseTool` so they share the
same voice loop, the same TaskExecutor confirmation flow, and the
same routine plumbing.

Why one tool, not four?
-----------------------

* All four features can answer a single user utterance - "find a
  MediaPipe tutorial and save it as a skill" needs the browser agent
  and the skill builder.
* The CommandRouter already has deterministic keyword + can_handle
  dispatch.  Putting the four features behind one router entry keeps
  the wiring minimal and avoids fighting the existing priority
  scheme.
* Routing inside the tool is also deterministic: screen questions go
  to :func:`_handle_screen_question`, click-style utterances go to
  :func:`_handle_visual_action`, etc.  The LLM is only consulted when
  there is no rule that fires (fallback path).

The module is intentionally small - the heavy lifting lives in
:mod:`core.context_engine`, :mod:`core.visual_actions`,
:mod:`core.skill_manager`, and :mod:`core.browser_agent`.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.context_engine import (
    ScreenContext,
    get_browser_cache,
    get_screen_cache,
)
from core.logger import get_logger
from core.skill_manager import SkillManager
from core.visual_actions import (
    VisualTarget,
    build_screen_context,
    execute_visual,
    looks_like_visual_action,
    parse_target,
    type_into,
)
from core.browser_agent import (
    looks_like_browser_intent,
    open_ordinal,
    open_url,
    parse_search_query,
    research,
    search_web,
    summarise_results,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Phrase detection helpers
# ---------------------------------------------------------------------------
_SCREEN_TRIGGERS = (
    "what is on my screen", "what's on my screen", "what is on screen",
    "what's on screen", "on my screen", "read this page", "read the screen",
    "read the error", "explain this error", "explain the error",
    "what am i looking at", "this window", "active window",
    "summarise this page", "summarize this page", "summarise this",
    "summarize this", "what should i click", "what should i do",
)

_SCREEN_QUESTION_TRIGGERS = (
    "what does this error mean", "what does this say",
    "what is the error", "what's the error", "tell me the error",
)

_VISUAL_VERBS = (
    "click ", "tap ", "press ", "double click", "right click", "right-click",
    "scroll down", "scroll up", "scroll left", "scroll right",
)

_SKILL_VERBS = (
    "create a skill", "create skill", "new skill", "make a skill",
    "learn this", "teach yourself", "save this as a skill",
    "save this as a routine", "save this skill",
    "list my skills", "show my skills", "show skills",
    "delete skill", "remove skill",
    "run skill", "run my skill", "use skill",
    "rename skill",
    "what does the skill", "show skill",
    "update this skill", "update skill",
)

_BROWSER_VERBS = (
    "search for ", "look up ", "find me ", "find ",
    "google ", "search the web", "duckduckgo",
    "research ", "compare ", "summarise the results",
    "summarize the results", "compare these", "compare them",
    "open the first result", "open the second result",
    "open the third result", "use the first website",
    "use the first result", "use the second result",
    "go to the first", "go to the second", "go to the third",
    "open the last result", "use the last result",
    "what did you find", "show me the results",
)


def _contains(text: str, phrases) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in phrases)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class IntentTool(BaseTool):
    """Single router entry point for the four new capabilities."""

    name = "intent_tool"
    description = (
        "Screen questions ('what is on my screen'), visual actions "
        "('click Download', 'scroll down'), skill creation / running "
        "('learn this', 'run coding setup'), and browser research "
        "('search for MediaPipe', 'open the second result')."
    )

    def __init__(self, skill_manager: Optional[SkillManager] = None) -> None:
        self._skills = skill_manager or SkillManager()

    # ------------------------------------------------------------------
    # BaseTool
    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        if _contains(text, _SCREEN_TRIGGERS):
            return True
        if _contains(text, _SCREEN_QUESTION_TRIGGERS):
            return True
        if _contains(text, _VISUAL_VERBS):
            return True
        if _contains(text, _SKILL_VERBS):
            return True
        if _contains(text, _BROWSER_VERBS):
            return True
        return False

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()
        try:
            if _contains(lowered, _SKILL_VERBS):
                return self._handle_skill(text, context)
            if _contains(lowered, _VISUAL_VERBS) or looks_like_visual_action(text):
                return self._handle_visual(text, context)
            if _contains(lowered, _BROWSER_VERBS) or looks_like_browser_intent(text):
                return self._handle_browser(text, context)
            if _contains(lowered, _SCREEN_TRIGGERS) or _contains(lowered, _SCREEN_QUESTION_TRIGGERS):
                return self._handle_screen(text, context)
        except Exception as exc:
            log.exception("Intent tool failed: %s", exc)
            return ToolResult(success=False, message=f"I could not complete that request: {exc}")
        return ToolResult(success=False, message="I am not sure how to handle that, sir.")

    # ------------------------------------------------------------------
    # Universal Screen Context
    # ------------------------------------------------------------------
    def _handle_screen(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        cfg = get_config()
        # Force re-capture for explicit screen requests; the user
        # asked NOW so we do not want a stale 2-second-old frame.
        ctx: ScreenContext = build_screen_context(force=True)
        active = ctx.active_window or ctx.title or ctx.application
        ocr = ctx.ocr_text.strip()
        message = self._summarise_screen(command, ctx, active=active, ocr=ocr)
        return ToolResult(
            success=True,
            message=message,
            data={"context": ctx.to_dict(), "path": ctx.source_path},
        )

    @staticmethod
    def _summarise_screen(command: str, ctx: ScreenContext, *, active: str, ocr: str) -> str:
        lowered = command.lower()
        # Error-specific shortcut.
        if "error" in lowered or "what does this" in lowered:
            if ocr:
                snippet = ocr if len(ocr) <= 400 else ocr[:400] + "..."
                return f"I can see the following text on your screen: {snippet}"
            return "I do not see any text on the screen to read."

        if not active and not ocr:
            return "I captured the screen but did not find any text or a clear active window."

        app = active or "an application"
        if ocr:
            # Show the first 3 distinct non-empty lines so the response
            # stays short.
            lines = [ln.strip() for ln in ocr.splitlines() if ln.strip()]
            preview = "; ".join(lines[:3])
            if len(preview) > 200:
                preview = preview[:197] + "..."
            return f"You have {app} open. I can see: {preview}."
        return f"You have {app} open."

    # ------------------------------------------------------------------
    # Visual Action
    # ------------------------------------------------------------------
    def _handle_visual(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        target = parse_target(command)
        if target.action == "scroll":
            result = execute_visual(target)
            return ToolResult(success=result.get("success", False), message=result.get("message", ""))
        if target.action == "type_into":
            out = type_into(target, target.text)
            return ToolResult(success=out.get("success", False), message=out.get("message", ""))
        result = execute_visual(target)
        return ToolResult(
            success=result.get("success", False),
            message=result.get("message", "I could not complete that click."),
            data={"target": target.describe(), "element": result.get("element")},
        )

    # ------------------------------------------------------------------
    # Skill Builder
    # ------------------------------------------------------------------
    def _handle_skill(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        lowered = command.lower()
        sm = self._skills

        if "list" in lowered and "skill" in lowered:
            return self._skill_list()
        if "delete" in lowered or "remove" in lowered:
            return self._skill_delete(command)
        if "rename" in lowered:
            return self._skill_rename(command)
        if "show" in lowered or "what does the skill" in lowered or "what does the" in lowered:
            return self._skill_show(command)
        if any(kw in lowered for kw in ("create", "new skill", "make a skill", "save this as a skill", "teach yourself", "learn this")):
            return self._skill_create(command, context)
        if "run" in lowered or "use skill" in lowered:
            return self._skill_run(command, context)
        return ToolResult(success=False, message="I am not sure what to do with that skill request.")

    def _skill_list(self) -> ToolResult:
        names = self._skills.names()
        if not names:
            return ToolResult(success=True, message="You have no skills saved yet, sir.")
        return ToolResult(success=True, message="Your skills: " + ", ".join(names), data={"skills": names})

    def _skill_show(self, command: str) -> ToolResult:
        name = _extract_name(command, ("show skill ", "show the skill ", "what does the skill ", "what does ", "show "))
        rec = self._skills.get(name) if name else None
        if rec is None:
            return ToolResult(success=False, message=f"No skill called '{name}'.")
        steps = "\n".join(f"- {s.get('action')} {s.get('args')}" for s in rec.steps)
        body = f"{rec.name}: {rec.description or '(no description)'}\n{steps or '(no steps)'}"
        return ToolResult(success=True, message=body, data={"skill": rec.to_dict()})

    def _skill_delete(self, command: str) -> ToolResult:
        name = _extract_name(command, ("delete skill ", "remove skill "))
        if not name:
            return ToolResult(success=False, message="Tell me which skill to delete, sir.")
        ok = self._skills.delete(name)
        if not ok:
            return ToolResult(success=False, message=f"No skill called '{name}'.")
        return ToolResult(success=True, message=f"Deleted skill '{name}'.")

    def _skill_rename(self, command: str) -> ToolResult:
        m = re.search(r"rename\s+(?:skill\s+)?([\w\s]+?)\s+to\s+([\w\s]+)$", command, flags=re.IGNORECASE)
        if not m:
            return ToolResult(success=False, message="Use: rename skill <old> to <new>.")
        old, new = m.group(1).strip(), m.group(2).strip()
        if not self._skills.rename(old, new):
            return ToolResult(success=False, message=f"No skill called '{old}'.")
        return ToolResult(success=True, message=f"Renamed '{old}' to '{new}'.")

    def _skill_create(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        from core.context_engine import SkillRecord

        # Pull the name from "create a skill called <name>".
        lowered = command.lower()
        name = ""
        for prefix in ("create a skill called ", "create skill called ", "new skill called ", "make a skill called ", "save this as a skill called "):
            if prefix in lowered:
                name = command[lowered.index(prefix) + len(prefix):].strip().strip("'\"")
                break
        if not name:
            # Try "save this as a skill" / "learn this skill" / "remember this as a skill"
            # without an explicit name - synthesize a timestamped placeholder so
            # the user can flesh it out next turn instead of being told the
            # command was malformed.
            if (
                "save this as a skill" in lowered
                or "save this skill" in lowered
                or "learn this skill" in lowered
                or "learn this" in lowered
                or "remember this as a skill" in lowered
                or "remember this skill" in lowered
                or "teach yourself" in lowered
                or "remember what i did" in lowered
            ):
                name = f"skill_{int(time.time())}"
            else:
                return ToolResult(
                    success=False,
                    message="Use: create a skill called <name>.",
                )

        # Seed an empty skill with placeholder steps the user can fill
        # in.  If the context has a previous task plan, lift it.
        steps: List[Dict[str, Any]] = []
        variables: List[str] = []
        if context:
            plan = context.get("last_plan") if isinstance(context, dict) else None
            if plan and hasattr(plan, "steps"):
                for s in plan.steps:  # type: ignore[attr-defined]
                    step = _plan_step_to_skill_step(s)
                    if step:
                        steps.append(step)
        if not steps:
            steps = [
                {"action": "open_app", "args": {"app": "vscode"}},
                {"action": "wait", "args": {"seconds": "1"}},
                {"action": "say", "args": {"text": "Skill ready."}},
            ]
        rec = SkillRecord(
            name=name,
            description=f"Created from voice on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            steps=steps,
            variables=variables,
        )
        rec.variables = SkillManager.extract_variables(steps)
        self._skills.upsert(rec)
        return ToolResult(
            success=True,
            message=f"Skill '{name}' saved with {len(steps)} step(s). Say 'run skill {name}' to try it.",
            data={"skill": rec.to_dict()},
        )

    def _skill_run(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        lowered = command.lower()
        for prefix in ("run skill ", "run my skill ", "use skill "):
            if prefix in lowered:
                name = command[lowered.index(prefix) + len(prefix):].strip().strip("'\"")
                break
        else:
            # Plain "run <name>" if the name matches a saved skill.
            tokens = command.split()
            name = tokens[-1].strip("'\"") if tokens else ""
        if not name:
            return ToolResult(success=False, message="Which skill should I run, sir?")

        # Variable substitution: "run skill open_project with F:\\Projects\\Nova"
        variables: Dict[str, str] = {}
        m = re.search(r"\bwith\s+(.+)$", command, flags=re.IGNORECASE)
        if m:
            payload = m.group(1).strip()
            # Single value - assign to the first declared variable, if any.
            rec = self._skills.get(name)
            if rec and rec.variables:
                variables[rec.variables[0]] = payload
            else:
                variables["VALUE"] = payload

        results = self._skills.run(name, variables=variables, context=context or {})
        if not results:
            return ToolResult(success=False, message=f"No skill called '{name}'.")
        last = results[-1]
        ok = all(r.success for r in results)
        msg = f"Skill '{name}' finished." if ok else f"Skill '{name}' stopped early: {last.message}"
        return ToolResult(success=ok, message=msg, data={"results": [r.message for r in results]})

    # ------------------------------------------------------------------
    # Browser Agent
    # ------------------------------------------------------------------
    def _handle_browser(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        lowered = command.lower()
        if "compare" in lowered or "summarise the results" in lowered or "summarize the results" in lowered or "what did you find" in lowered or "show me the results" in lowered:
            return ToolResult(success=True, message=summarise_results())
        if any(p in lowered for p in ("open the first", "open the second", "open the third", "open the last", "use the first", "use the second", "use the third", "go to the first", "go to the second", "go to the third")):
            ok, msg = open_ordinal(command)
            return ToolResult(success=ok, message=msg)
        if lowered.startswith("research ") or "research " in lowered:
            query = parse_search_query(command) or ""
            report = research(query)
            return ToolResult(success=True, message=report.summary, data=report.to_dict())

        query = parse_search_query(command) or ""
        if not query:
            return ToolResult(success=False, message="I need a search query, sir.")
        hits = search_web(query)
        if not hits:
            return ToolResult(success=False, message=f"I could not find any results for {query!r}.")
        return ToolResult(success=True, message=summarise_results(hits), data={"results": [h.to_dict() for h in hits]})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_name(command: str, prefixes) -> str:
    lowered = command.lower()
    for prefix in prefixes:
        if prefix in lowered:
            return command[lowered.index(prefix) + len(prefix):].strip().strip("'\"")
    return ""


def _plan_step_to_skill_step(step: Any) -> Optional[Dict[str, Any]]:
    """Convert a :class:`core.task_plan.TaskStep` into a skill step dict.

    Unknown step shapes are skipped silently - the skill builder is
    best-effort.
    """
    try:
        hint = (getattr(step, "tool_hint", "") or "").lower()
        args = getattr(step, "arguments", {}) or {}
        if not hint:
            return None
        action = hint
        return {"action": action, "args": {k: str(v) for k, v in args.items()}}
    except Exception:
        return None


def register_intent_tool(router, skill_manager: Optional[SkillManager] = None) -> List[BaseTool]:
    """Register the unified intent tool with *router*."""
    sm = skill_manager or SkillManager()
    # The skill manager is also passed to the router so SkillRunner
    # steps can dispatch through the same router.
    from core.router import CommandRouter

    if isinstance(router, CommandRouter):
        sm.attach_router(router)
    tool = IntentTool(skill_manager=sm)
    router.register(
        tool,
        keywords=(
            "what is on my screen", "what's on my screen", "on my screen",
            "click ", "scroll ", "double click", "right click",
            "create a skill", "create skill", "new skill", "run skill",
            "list my skills", "delete skill", "rename skill",
            "search for ", "research ", "look up ",
            "open the first", "open the second", "open the third",
            "use the first", "use the second", "use the third",
        ),
        priority=70,
    )
    return [tool]


__all__ = ["IntentTool", "register_intent_tool"]
