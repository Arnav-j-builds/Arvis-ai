"""
routines.commands
~~~~~~~~~~~~~~~~

The :class:`RoutinesTool` and the :func:`register_routines_tool` helper.

Voice patterns
--------------

* ``"I'm starting work."`` - run the routine whose trigger matches.
* ``"Create a routine called Movie Time."`` - start the interactive builder.
* ``"List my routines."`` - enumerate routines.
* ``"Delete routine Movie Time."`` - remove a routine.
* ``"Edit routine Movie Time."`` - replace its actions.
* ``"Show routine Movie Time."`` - dump a routine.

The interactive builder is intentionally simple: it relies on the
:class:`core.speech.speak` helper to ask the user questions and on the
existing STT pipeline (handled by ``main.py``) to capture the answers. To
avoid coupling, the tool returns a ``ToolResult`` whose ``data`` carries a
``"questions"`` list and ``main.py`` forwards that list to the microphone.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.base import BaseTool, ToolResult
from core.logger import get_logger
from routines.manager import Routine, RoutineManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Interactive builder
# ---------------------------------------------------------------------------
@dataclass
class RoutineDraft:
    """State of an in-progress interactive routine creation."""

    name: str = ""
    description: str = ""
    trigger: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    step: str = "trigger"  # trigger -> action_kind -> action_value -> save

    def to_routine(self) -> Routine:
        from core.base import RoutineAction

        return Routine(
            name=self.name,
            description=self.description,
            trigger=self.trigger,
            actions=[RoutineAction.from_dict(a) for a in self.actions],
        )


@dataclass
class BuilderSession:
    draft: RoutineDraft = field(default_factory=RoutineDraft)
    prompt: str = ""


class InteractiveRoutineBuilder:
    """A small state machine used while creating a routine."""

    def __init__(self) -> None:
        self._sessions: Dict[int, BuilderSession] = {}
        self._lock = threading.Lock()

    def begin(self, name: str) -> BuilderSession:
        session = BuilderSession(draft=RoutineDraft(name=name, step="trigger"), prompt="trigger")
        with self._lock:
            self._sessions[id(session)] = session
        return session

    def get(self, session_id: int) -> Optional[BuilderSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def discard(self, session_id: int) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


def _next_prompt(draft: RoutineDraft) -> tuple[str, str]:
    """Return the (step_name, question_text) tuple for the next question."""
    if draft.step == "trigger":
        return draft.step, "What phrase should trigger this routine? Say it now."
    if draft.step == "actions":
        return draft.step, (
            "Tell me the actions one by one. For each say "
            "'open app <name>', 'open url <address>', 'say <text>', "
            "'wait <seconds>', or 'done' to finish."
        )
    if draft.step == "description":
        return draft.step, "Add a short description, or say 'skip'."
    if draft.step == "save":
        return draft.step, "Saving your routine now."
    return "done", ""


def _parse_action_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single action description into a dict.

    Recognised forms::

        "open app code"
        "open url https://github.com"
        "google search python decorators"
        "say everything is ready"
        "wait 2 seconds"
    """
    text = line.strip()
    if not text:
        return None
    lowered = text.lower()

    if lowered.startswith("open app ") or lowered.startswith("launch app "):
        value = text.split(" ", 2)[-1]
        return {"action": "open_app", "value": value}
    if lowered.startswith("open url ") or lowered.startswith("open website "):
        value = text.split(" ", 2)[-1]
        return {"action": "open_url", "value": value}
    if lowered.startswith("google search ") or lowered.startswith("search google "):
        value = text.split(" ", 2)[-1]
        return {"action": "google_search", "value": value}
    if lowered.startswith("youtube search ") or lowered.startswith("search youtube "):
        value = text.split(" ", 2)[-1]
        return {"action": "youtube_search", "value": value}
    if lowered.startswith("say ") or lowered.startswith("speak "):
        return {"action": "say", "value": text.split(" ", 1)[1]}
    if lowered.startswith("wait "):
        return {"action": "wait", "value": text.split(" ", 1)[1]}

    # Fallback: treat the line as a free-form URL if it looks like one
    if lowered.startswith("http"):
        return {"action": "open_url", "value": text}

    return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class RoutinesTool(BaseTool):
    """Voice tool that runs and edits routines."""

    name = "routines_tool"
    description = (
        "Run, create, list, edit, or delete routines. Examples: 'I'm starting work', "
        "'create a routine called movie time', 'list my routines', 'delete routine movie time'."
    )

    # The manager is supplied by the application, not via config.
    def __init__(self, manager: RoutineManager) -> None:
        self.manager = manager
        self.builder = InteractiveRoutineBuilder()
        self._pending_session: Optional[int] = None

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        lowered = (command or "").lower()
        if self.manager.matches(lowered) is not None:
            return True
        return any(
            token in lowered
            for token in (
                "create a routine",
                "create routine",
                "new routine",
                "list routines",
                "show my routines",
                "delete routine",
                "edit routine",
                "show routine",
                "list my routines",
                "routine called",
            )
        )

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()

        # If we are mid-way through an interactive builder, route to it.
        if self._pending_session is not None:
            return self._continue_builder(text)

        # Trigger match - run the routine.
        match = self.manager.matches(lowered)
        if match is not None:
            results = self.manager.run_routine(match, context)
            failures = [r for r in results if not r.success]
            if failures:
                return ToolResult(
                    success=False,
                    message=f"Routine '{match.name}' finished with {len(failures)} failure(s).",
                    data={"results": results},
                )
            return ToolResult(
                success=True,
                message=f"Routine '{match.name}' finished successfully.",
                data={"results": results},
            )

        if "create a routine" in lowered or "create routine" in lowered or "new routine" in lowered:
            return self._begin_builder(text)
        if "list" in lowered and "routine" in lowered:
            return self._list()
        if "show routine" in lowered:
            return self._show(text)
        if "delete routine" in lowered:
            return self._delete(text)
        if "edit routine" in lowered:
            return self._edit(text)

        return ToolResult(success=False, message="I am not sure what to do with that routine request.")

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------
    def _begin_builder(self, command: str) -> ToolResult:
        lowered = command.lower()
        for prefix in ("create a routine called ", "create routine called ", "new routine called "):
            if prefix in lowered:
                name = command[lowered.index(prefix) + len(prefix):].strip().strip("'\"")
                break
        else:
            name = ""

        if not name:
            return ToolResult(
                success=False,
                message="What would you like to call the new routine? Please say 'create a routine called <name>'.",
            )

        session = self.builder.begin(name)
        self._pending_session = id(session)
        _, prompt = _next_prompt(session.draft)
        return ToolResult(
            success=True,
            message=prompt,
            data={"builder": {"session_id": id(session), "step": "trigger", "expected_input": "voice"}},
        )

    def _continue_builder(self, response: str) -> ToolResult:
        session = self.builder.get(self._pending_session)
        if session is None:
            self._pending_session = None
            return ToolResult(success=False, message="The routine builder session expired, sir.")

        draft = session.draft
        lowered = response.lower().strip()

        if draft.step == "trigger":
            if lowered in {"cancel", "stop", "exit"}:
                self.builder.discard(self._pending_session)
                self._pending_session = None
                return ToolResult(success=True, message="Routine creation cancelled.")
            draft.trigger = [response.strip()] if response.strip() else []
            draft.step = "actions"
            _, prompt = _next_prompt(draft)
            return ToolResult(
                success=True,
                message=prompt,
                data={"builder": {"session_id": self._pending_session, "step": "actions"}},
            )

        if draft.step == "actions":
            if lowered in {"done", "finish", "that's all", "thats all"}:
                if not draft.actions:
                    return ToolResult(success=False, message="Add at least one action before finishing.")
                draft.step = "description"
                _, prompt = _next_prompt(draft)
                return ToolResult(
                    success=True,
                    message=prompt,
                    data={"builder": {"session_id": self._pending_session, "step": "description"}},
                )
            action = _parse_action_line(response)
            if action is None:
                return ToolResult(
                    success=False,
                    message="I did not understand that action. Try 'open app code', 'open url https://github.com', or 'say everything is ready'.",
                )
            draft.actions.append(action)
            return ToolResult(
                success=True,
                message=f"Added {action['action']} {action['value']}. Add another or say 'done'.",
                data={"builder": {"session_id": self._pending_session, "step": "actions"}},
            )

        if draft.step == "description":
            draft.description = "" if lowered in {"skip", "none"} else response.strip()
            routine = draft.to_routine()
            self.manager.upsert(routine)
            self.builder.discard(self._pending_session)
            self._pending_session = None
            return ToolResult(
                success=True,
                message=f"Routine '{routine.name}' saved with {len(routine.actions)} action(s).",
                data={"routine": routine.to_dict()},
            )

        return ToolResult(success=True, message="Builder step complete.")

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def _list(self) -> ToolResult:
        routines = self.manager.list_routines()
        if not routines:
            return ToolResult(success=True, message="You have no routines yet, sir.")
        lines = []
        for routine in routines:
            triggers = ", ".join(f"'{t}'" for t in routine.trigger) or "(no trigger)"
            lines.append(f"- {routine.name} [trigger {triggers}] ({len(routine.actions)} action(s))")
        return ToolResult(success=True, message="\n".join(lines), data={"routines": [r.to_dict() for r in routines]})

    def _show(self, command: str) -> ToolResult:
        name = _extract_name(command, prefixes=("show routine ", "show my routine ", "display routine "))
        routine = self.manager.get(name) if name else None
        if routine is None:
            return ToolResult(success=False, message=f"I could not find a routine called '{name}'.")
        return ToolResult(success=True, message=str(routine.to_dict()), data={"routine": routine.to_dict()})

    def _delete(self, command: str) -> ToolResult:
        name = _extract_name(command, prefixes=("delete routine ", "remove routine "))
        if not name:
            return ToolResult(success=False, message="Tell me which routine to delete, sir.")
        deleted = self.manager.delete(name)
        if not deleted:
            return ToolResult(success=False, message=f"No routine called '{name}' to delete.")
        return ToolResult(success=True, message=f"Routine '{name}' deleted.")

    def _edit(self, command: str) -> ToolResult:
        name = _extract_name(command, prefixes=("edit routine ", "modify routine "))
        routine = self.manager.get(name) if name else None
        if routine is None:
            return ToolResult(success=False, message=f"No routine called '{name}' to edit.")

        # Seed an in-progress builder with the routine's existing state so the
        # user can extend or replace the actions one by one.
        session = self.builder.begin(name)
        session.draft = RoutineDraft(
            name=routine.name,
            description=routine.description,
            trigger=list(routine.trigger),
            actions=[{"action": a.action, "value": a.value, "metadata": dict(a.metadata)} for a in routine.actions],
            step="actions",
        )
        self._pending_session = id(session)
        return ToolResult(
            success=True,
            message=(
                f"Editing '{name}'. Add more actions, or say 'done' to save your changes. "
                f"Current actions: {len(routine.actions)}."
            ),
            data={"builder": {"session_id": id(session), "step": "actions", "existing": routine.to_dict()}},
        )


def _extract_name(command: str, prefixes: tuple[str, ...]) -> str:
    lowered = command.lower()
    for prefix in prefixes:
        if prefix in lowered:
            return command[lowered.index(prefix) + len(prefix):].strip().strip("'\"")
    return ""


def register_routines_tool(router, manager: RoutineManager) -> List[BaseTool]:
    tool = RoutinesTool(manager)
    manager.attach_router(router)
    router.register(tool, keywords=("routine",), priority=60)
    return [tool]


__all__ = ["RoutinesTool", "register_routines_tool", "InteractiveRoutineBuilder"]
