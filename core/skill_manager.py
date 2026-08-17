"""
core.skill_manager
~~~~~~~~~~~~~~~~~~

Reusable, named "skills" that ARVIS can learn and replay.

A skill is a JSON-serialisable record of the *semantic* actions a
user performs.  Coordinates are stored as a fallback only - we
prefer to remember ``click "Download"`` over ``click 480,320``.

Storage
-------

``storage/skills.json`` is the single source of truth.  The format is::

    {
      "<name>": {
        "description": "...",
        "version": 1,
        "variables": ["PROJECT_PATH"],
        "steps": [
          {"action": "open_app", "args": {"app": "vscode"}},
          {"action": "open_url", "args": {"url": "{{PROJECT_PATH}}"}}
        ]
      }
    }

Action vocabulary is intentionally small and aligned with
:mod:`routines.manager` + :mod:`core.task_executor` so the executor
can dispatch each step through the existing :class:`CommandRouter`
without writing a second command pipeline.

Supported step actions::

    open_app        args: app/name
    open_url        args: url
    google_search   args: query
    youtube_search  args: query
    type            args: text
    press_key       args: key
    click_visible   args: text (visual target)
    wait            args: seconds
    say             args: text
    run_routine     args: name
    run_command     args: command (terminal)
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.context_engine import SkillRecord
from core.logger import get_logger
from core.router import CommandRouter

log = get_logger(__name__)


_VAR_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class SkillManager:
    """Thread-safe CRUD + lightweight runner for skills."""

    def __init__(self, path: Optional[Path] = None, router: Optional[CommandRouter] = None) -> None:
        cfg = get_config()
        self._path = path or (cfg.storage_dir / "skills.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._skills: Dict[str, SkillRecord] = {}
        self._router = router
        self._load()

    # ------------------------------------------------------------------
    def attach_router(self, router: CommandRouter) -> None:
        self._router = router

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
                log.warning("skills.json corrupt (%s) - backing up to %s", exc, backup)
                try:
                    backup.write_text(self._path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass
                self._save_locked()
                return
            if isinstance(raw, dict):
                for name, body in raw.items():
                    if not isinstance(body, dict):
                        continue
                    body.setdefault("name", name)
                    try:
                        rec = SkillRecord.from_dict(body)
                    except Exception as exc:
                        log.warning("Skipping bad skill %r: %s", name, exc)
                        continue
                    if rec.name:
                        self._skills[rec.name.lower()] = rec

    def _save_locked(self) -> None:
        payload = {rec.name: rec.to_dict() for rec in self._skills.values()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save(self) -> None:
        with self._lock:
            self._save_locked()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._skills.keys())

    def list_skills(self) -> List[SkillRecord]:
        with self._lock:
            return list(self._skills.values())

    def get(self, name: str) -> Optional[SkillRecord]:
        with self._lock:
            return self._skills.get(name.lower())

    def upsert(self, record: SkillRecord) -> None:
        if not record.name:
            raise ValueError("Skill name is required")
        record.updated_at = time.time()
        with self._lock:
            self._skills[record.name.lower()] = record
            self._save_locked()

    def delete(self, name: str) -> bool:
        with self._lock:
            existed = self._skills.pop(name.lower(), None) is not None
            if existed:
                self._save_locked()
            return existed

    def rename(self, old: str, new: str) -> bool:
        if not new:
            return False
        with self._lock:
            rec = self._skills.pop(old.lower(), None)
            if rec is None:
                return False
            rec.name = new
            rec.updated_at = time.time()
            self._skills[new.lower()] = rec
            self._save_locked()
            return True

    # ------------------------------------------------------------------
    # Variable substitution
    # ------------------------------------------------------------------
    @staticmethod
    def extract_variables(steps: List[Dict[str, Any]]) -> List[str]:
        seen: List[str] = []
        for step in steps:
            for value in (step.get("args") or {}).values():
                if not isinstance(value, str):
                    continue
                for match in _VAR_RE.finditer(value):
                    name = match.group(1)
                    if name not in seen:
                        seen.append(name)
        return seen

    @staticmethod
    def render(text: str, variables: Dict[str, str]) -> str:
        def _sub(match: "re.Match[str]") -> str:
            key = match.group(1)
            return str(variables.get(key, match.group(0)))
        return _VAR_RE.sub(_sub, text or "")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(
        self,
        name: str,
        variables: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        rec = self.get(name)
        if rec is None:
            return [ToolResult(success=False, message=f"No skill called '{name}'.")]
        return self.run_skill(rec, variables=variables, context=context)

    def run_skill(
        self,
        record: SkillRecord,
        variables: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        if self._router is None:
            return [ToolResult(success=False, message="Skill router is not configured.")]
        cfg = get_config()
        vars_merged: Dict[str, str] = {v: "" for v in record.variables}
        if variables:
            vars_merged.update({k: str(v) for k, v in variables.items()})
        ctx = dict(context or {})
        log.info("Running skill '%s' (%d steps)", record.name, len(record.steps))
        results: List[ToolResult] = []
        for index, step in enumerate(record.steps):
            if index >= cfg.max_skill_steps:
                results.append(ToolResult(success=False, message="Skill truncated at max steps."))
                break
            text = self._step_to_router_text(step, vars_merged)
            if text is None:
                results.append(
                    ToolResult(success=False, message=f"Skipping unknown step {step.get('action')!r}.")
                )
                continue
            try:
                result = self._router.dispatch(text, context=ctx)
            except Exception as exc:  # pragma: no cover - defensive
                result = ToolResult(success=False, message=f"Skill step error: {exc}")
            results.append(result)
            if not result.success and step.get("stop_on_error", True):
                break
        log.info("Skill '%s' done: %d/%d ok", record.name, sum(1 for r in results if r.success), len(results))
        return results

    def _step_to_router_text(self, step: Dict[str, Any], variables: Dict[str, str]) -> Optional[str]:
        action = (step.get("action") or "").strip().lower()
        args = step.get("args") or {}
        # Render every string value through the variable substitution.
        rendered = {k: (self.render(v, variables) if isinstance(v, str) else v) for k, v in args.items()}

        if action in {"open_app", "launch_app", "app"}:
            name = rendered.get("app") or rendered.get("name") or ""
            return f"open app {name}" if name else None
        if action in {"open_url", "open_website", "url"}:
            url = rendered.get("url") or ""
            return f"open url {url}" if url else None
        if action in {"google_search", "search_google"}:
            q = rendered.get("query") or ""
            return f"google search {q}" if q else None
        if action in {"youtube_search", "search_youtube"}:
            q = rendered.get("query") or ""
            return f"youtube search {q}" if q else None
        if action == "type":
            text = rendered.get("text") or ""
            return f"type {text}" if text else None
        if action == "press_key":
            key = rendered.get("key") or ""
            return f"press {key}" if key else None
        if action == "click_visible":
            text = rendered.get("text") or ""
            return f"click {text}" if text else None
        if action == "wait":
            seconds = rendered.get("seconds") or "1"
            return f"wait {seconds}"
        if action == "say":
            text = rendered.get("text") or ""
            return f"say {text}" if text else None
        if action == "run_routine":
            name = rendered.get("name") or ""
            return f"run routine {name}" if name else None
        if action == "run_command":
            cmd = rendered.get("command") or ""
            return f"run {cmd}" if cmd else None
        if action == "scroll":
            direction = rendered.get("direction") or "down"
            return f"scroll {direction}"
        return None


# ---------------------------------------------------------------------------
# Public helpers used by tests / other modules
# ---------------------------------------------------------------------------
def _step_dispatch_text(action: str, args: Dict[str, Any]) -> str:
    """Helper for tests: turn a single step into a router command string."""
    skill = SkillManager()
    return skill._step_to_router_text({"action": action, "args": args}, {}) or ""


__all__ = [
    "SkillManager",
    "_step_dispatch_text",
]
