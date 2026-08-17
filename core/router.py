"""
core.router
~~~~~~~~~~~

A thin registry that:

* Holds the list of registered :class:`~core.base.BaseTool` instances.
* Exposes them as LangChain tools for the existing agent executor.
* Provides a deterministic string-matching fallback for commands that the
  LLM does not route correctly.
* Runs routines (lists of :class:`~core.base.RoutineAction`).

The router is intentionally decoupled from voice/audio code so it can be
unit-tested in isolation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.base import BaseTool, RoutineAction, ToolResult
from core.logger import get_logger
from core.config import get_config

log = get_logger(__name__)


@dataclass
class RouterRegistration:
    """Metadata for a registered tool."""

    tool: BaseTool
    keywords: tuple[str, ...] = field(default_factory=tuple)
    priority: int = 100


class CommandRouter:
    """Routes commands to the most appropriate tool."""

    def __init__(self) -> None:
        self._registry: List[RouterRegistration] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(
        self,
        tool: BaseTool,
        keywords: Iterable[str] | None = None,
        priority: int = 100,
    ) -> None:
        """Register *tool* with optional :attr:`keywords` for fast dispatch."""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool, got {type(tool).__name__}")
        key = getattr(tool, "name", tool.__class__.__name__)
        log.info("Registering tool %s (priority=%d, keywords=%s)", key, priority, list(keywords or ()))
        self._registry.append(
            RouterRegistration(
                tool=tool,
                keywords=tuple(k.lower() for k in (keywords or ())),
                priority=priority,
            )
        )

    def tools(self) -> List[BaseTool]:
        """Return the registered tools in registration order."""
        return [reg.tool for reg in self._registry]

    def langchain_tools(self, exclude_predicate=None) -> List[Any]:
        """Expose registered tools to the LangChain agent executor.

        ``exclude_predicate`` is an optional callable ``(tool_name) -> bool``
        that returns ``True`` for tools that should NOT be exposed to the
        LLM (typically the legacy @tool-decorated functions that need
        structured input). The deterministic router still uses them.
        """
        langchain_tools = []
        for reg in self._registry:
            if exclude_predicate is not None:
                try:
                    if exclude_predicate(reg.tool.name):
                        continue
                except Exception:  # pragma: no cover
                    pass
            try:
                langchain_tools.append(reg.tool.as_langchain_tool())
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Tool %s could not be converted to LangChain tool: %s", reg.tool.name, exc)
        return langchain_tools

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def dispatch(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        default: Optional[Callable[[str], ToolResult]] = None,
    ) -> ToolResult:
        """Find a tool via keyword match or via ``tool.can_handle`` fallback.

        Returns a :class:`ToolResult` from the first matching tool. If no
        tool claims the command and *default* is provided, *default* is
        invoked.
        """
        text = (command or "").strip()
        if not text:
            return ToolResult(success=False, message="Empty command received, sir.")

        lowered = text.lower()

        # 1. Highest priority: exact-ish keyword match
        candidates = sorted(self._registry, key=lambda r: r.priority)
        for reg in candidates:
            for keyword in reg.keywords:
                if keyword and keyword in lowered:
                    log.debug("Dispatching %r to %s via keyword %r", text, reg.tool.name, keyword)
                    return reg.tool.safe_execute(text, context)

        # 2. can_handle fallback
        for reg in candidates:
            try:
                if reg.tool.can_handle(text, context):
                    log.debug("Dispatching %r to %s via can_handle", text, reg.tool.name)
                    return reg.tool.safe_execute(text, context)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("can_handle failed on %s: %s", reg.tool.name, exc)

        if default is not None:
            return default(text)

        log.info("No tool handled command: %r", text)
        return ToolResult(success=False, message=f"I am not sure how to handle '{text}', sir.")

    # ------------------------------------------------------------------
    # Routines
    # ------------------------------------------------------------------
    def run_actions(
        self,
        actions: Iterable[RoutineAction],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        """Run a sequence of routine actions sequentially.

        This is intentionally synchronous because each action tends to be
        I/O-bound but not blocking on long-running tasks. The existing
        agent already runs on the main thread; routines follow the same
        pattern.
        """
        cfg = get_config()
        results: List[ToolResult] = []
        for index, action in enumerate(actions):
            if index >= cfg.router_max_routine_actions:
                log.warning("Routine truncated at %d actions", cfg.router_max_routine_actions)
                break
            result = self._run_single_action(action, context)
            results.append(result)
            if not result.success and action.metadata.get("stop_on_error", True):
                log.info("Routine halted because step %d failed: %s", index, result.message)
                break
        return results

    async def run_actions_async(
        self,
        actions: Iterable[RoutineAction],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ToolResult]:
        """Async variant for future use - currently mirrors the sync version."""
        return await asyncio.to_thread(self.run_actions, list(actions), context)

    def _run_single_action(
        self,
        action: RoutineAction,
        context: Optional[Dict[str, Any]] | None,
    ) -> ToolResult:
        action_name = action.action.strip().lower()
        value = action.value

        try:
            if action_name in {"open_app", "launch_app", "app"}:
                return self._open_app(value)
            if action_name in {"open_url", "open_website", "url"}:
                return self._open_url(value)
            if action_name in {"google_search", "search_google"}:
                return self._google_search(value)
            if action_name in {"youtube_search", "search_youtube"}:
                return self._youtube_search(value)
            if action_name in {"play_youtube", "play"}:
                return self._youtube_search(value)
            if action_name in {"say", "speak"}:
                # Speak is handled by the speech bridge in main.py; here we
                # simply return a ToolResult so callers can echo it.
                return ToolResult(success=True, message=f"say:{value}")
            if action_name in {"wait", "sleep"}:
                import time
                try:
                    seconds = float(value)
                except ValueError:
                    seconds = 1.0
                time.sleep(seconds)
                return ToolResult(success=True, message=f"waited {seconds}s")
            if action_name in {"run_routine", "routine"}:
                # Lazy import to avoid a cycle.
                from routines.manager import RoutineManager

                manager: RoutineManager = context.get("routine_manager") if context else None  # type: ignore[assignment]
                if manager is None:
                    return ToolResult(success=False, message="Routine manager unavailable in context.")
                results = manager.run(value, context)
                last = results[-1] if results else None
                return (
                    ToolResult(success=True, message=f"Routine '{value}' finished.")
                    if last is None
                    else last
                )
            if action_name == "custom":
                handler_name = action.metadata.get("handler")
                handler = (context or {}).get(handler_name) if handler_name else None
                if callable(handler):
                    res = handler(value)
                    if isinstance(res, ToolResult):
                        return res
                    return ToolResult(success=True, message=str(res))
                return ToolResult(success=False, message=f"No handler registered for '{handler_name}'.")
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Routine action %s failed: %s", action_name, exc)
            return ToolResult(success=False, message=f"Action {action_name} failed: {exc}")

        return ToolResult(success=False, message=f"Unknown routine action '{action_name}'.")

    # ------------------------------------------------------------------
    # Action helpers - delegate to existing opener helpers where possible.
    # ------------------------------------------------------------------
    def _open_app(self, name: str) -> ToolResult:
        try:
            from tools.opener import APP_SHORTCUTS, _launch_app  # local import to keep core decoupled

        except Exception as exc:  # pragma: no cover
            return ToolResult(success=False, message=f"Cannot import opener: {exc}")
        if not name:
            return ToolResult(success=False, message="open_app requires a value")
        key = name.lower().strip()
        shortcut = APP_SHORTCUTS.get(key)
        command = shortcut[0] if shortcut else name
        ok, msg = _launch_app(command)
        return ToolResult(success=ok, message=msg)

    def _open_url(self, url: str) -> ToolResult:
        if not url:
            return ToolResult(success=False, message="open_url requires a value")
        try:
            from tools.opener import _open_url as opener_open_url
        except Exception as exc:  # pragma: no cover
            return ToolResult(success=False, message=f"Cannot import opener: {exc}")
        opener_open_url(url)
        return ToolResult(success=True, message=f"Opened {url}")

    def _google_search(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(success=False, message="google_search requires a value")
        try:
            from tools.opener import google_search
        except Exception as exc:  # pragma: no cover
            return ToolResult(success=False, message=f"Cannot import opener: {exc}")
        return ToolResult(success=True, message=google_search(query))

    def _youtube_search(self, query: str) -> ToolResult:
        if not query:
            return ToolResult(success=False, message="youtube_search requires a value")
        try:
            from tools.opener import youtube_search
        except Exception as exc:  # pragma: no cover
            return ToolResult(success=False, message=f"Cannot import opener: {exc}")
        return ToolResult(success=True, message=youtube_search(query))


# Singleton accessor ----------------------------------------------------------
_router: Optional[CommandRouter] = None


def get_router() -> CommandRouter:
    """Return the singleton router used by ``main.py``."""
    global _router
    if _router is None:
        _router = CommandRouter()
    return _router
