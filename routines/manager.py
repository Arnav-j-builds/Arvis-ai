"""
routines.manager
~~~~~~~~~~~~~~~

Stores and executes user-defined routines.

The manager is intentionally framework-agnostic: it knows nothing about
voice, STT, or TTS. It only deals with CRUD + execution. Voice handling
lives in :mod:`routines.commands`.

Schema
------

A routine is a dict with this shape::

    {
        "description": "What this routine does",
        "trigger": ["start work", "starting work"],   # optional spoken aliases
        "actions": [
            {"action": "open_app", "value": "code"},
            {"action": "open_url", "value": "https://github.com"}
        ]
    }

The ``storage/routines.json`` file is the single source of truth. If the
file is missing or empty, :class:`RoutineManager` boots with an empty
dictionary; if it is corrupted the manager logs a warning and starts over
after backing up the broken file to ``routines.json.bak``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.base import RoutineAction, ToolResult
from core.config import get_config
from core.logger import get_logger
from core.router import CommandRouter

log = get_logger(__name__)


@dataclass
class Routine:
    name: str
    description: str = ""
    trigger: List[str] = field(default_factory=list)
    actions: List[RoutineAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger": list(self.trigger),
            "actions": [{"action": a.action, "value": a.value, "metadata": a.metadata} for a in self.actions],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Routine":
        return cls(
            name=str(payload.get("name", "")).strip(),
            description=str(payload.get("description", "")),
            trigger=[str(t).strip() for t in payload.get("trigger", []) if str(t).strip()],
            actions=[RoutineAction.from_dict(a) for a in payload.get("actions", [])],
        )


class RoutineManager:
    """Thread-safe CRUD + executor for routines."""

    def __init__(self, path: Optional[Path] = None) -> None:
        cfg = get_config()
        self._path = path or cfg.routines_file
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._routines: Dict[str, Routine] = {}
        self._router: Optional[CommandRouter] = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._save_locked()
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                backup = self._path.with_suffix(".bak")
                log.warning("routines.json is corrupt (%s) - backing up to %s", exc, backup)
                backup.write_text(self._path.read_text(encoding="utf-8"), encoding="utf-8")
                self._save_locked()
                return

            if isinstance(raw, dict):
                # Two layouts supported:
                #   { "<name>": { actions: [...] } }
                #   { "<name>": Routine(...) }
                for name, body in raw.items():
                    if isinstance(body, dict):
                        body.setdefault("name", name)
                        routine = Routine.from_dict(body)
                        self._routines[name.lower()] = routine
            elif isinstance(raw, list):
                for body in raw:
                    if not isinstance(body, dict):
                        continue
                    name = str(body.get("name", "")).strip()
                    if name:
                        routine = Routine.from_dict(body)
                        self._routines[name.lower()] = routine

    def _save_locked(self) -> None:
        payload = {name: routine.to_dict() for name, routine in self._routines.items()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save(self) -> None:
        with self._lock:
            self._save_locked()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def attach_router(self, router: CommandRouter) -> None:
        self._router = router

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._routines.keys())

    def list_routines(self) -> List[Routine]:
        with self._lock:
            return list(self._routines.values())

    def get(self, name: str) -> Optional[Routine]:
        with self._lock:
            return self._routines.get(name.lower())

    def upsert(self, routine: Routine) -> None:
        if not routine.name:
            raise ValueError("Routine name is required")
        with self._lock:
            self._routines[routine.name.lower()] = routine
            self._save_locked()

    def delete(self, name: str) -> bool:
        with self._lock:
            existed = self._routines.pop(name.lower(), None) is not None
            if existed:
                self._save_locked()
            return existed

    def matches(self, command: str) -> Optional[Routine]:
        """Return the routine triggered by *command* (case-insensitive).

        A command matches when one of the routine's triggers appears as a
        substring. The longest trigger wins (so ``"start work"`` matches
        before ``"work"``).
        """
        lowered = (command or "").lower().strip()
        if not lowered:
            return None
        with self._lock:
            candidates: List[tuple[int, Routine]] = []
            for routine in self._routines.values():
                for trigger in routine.trigger:
                    if trigger and trigger in lowered:
                        candidates.append((len(trigger), routine))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, name: str, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        routine = self.get(name)
        if routine is None:
            return [ToolResult(success=False, message=f"No routine called '{name}'.")]
        return self.run_routine(routine, context)

    def run_routine(self, routine: Routine, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        if self._router is None:
            log.warning("Routine executed without router; only built-in actions will work.")
            return [ToolResult(success=False, message="Routine router is not configured.")]
        ctx = dict(context or {})
        ctx.setdefault("routine_manager", self)
        log.info("Running routine '%s' (%d actions)", routine.name, len(routine.actions))
        results = self._router.run_actions(routine.actions, ctx)
        log.info("Routine '%s' finished (%d steps, %d succeeded)", routine.name, len(results), sum(1 for r in results if r.success))
        return results

    # ------------------------------------------------------------------
    # Import helpers (used by interactive builder)
    # ------------------------------------------------------------------
    @staticmethod
    def actions_from_text(actions: Iterable[Dict[str, Any]]) -> List[RoutineAction]:
        out: List[RoutineAction] = []
        for entry in actions:
            out.append(RoutineAction.from_dict(entry))
        return out


__all__ = ["Routine", "RoutineManager"]
