"""
core.base
~~~~~~~~~

Abstract base classes and dataclasses every feature module should use.

The contract is intentionally small: a tool must expose ``can_handle`` and
``execute`` so that the router (:mod:`core.router`) can dispatch either by
the LangChain agent (function-calling) or by simple string matching.

Two abstractions are exposed:

* :class:`BaseTool` - a leaf tool that handles a single intent (for example
  "take screenshot", "send email").
* :class:`RoutineAction` - a single step inside a routine.

Tools never hold state - any state they need is supplied through ``context``
or read from environment variables at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ToolResult:
    """The return type every tool yields.

    Attributes
    ----------
    success:
        Whether the tool ran without raising.
    message:
        Human-readable summary suitable for TTS.
    data:
        Optional structured payload. The router may forward this to the LLM
        so it can produce a natural-language answer.
    """

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class RoutineAction:
    """A single step inside a user-defined routine.

    ``action`` is one of:

    * ``open_app``      - ``value`` is the executable / app name
    * ``open_url``      - ``value`` is an absolute URL
    * ``google_search`` - ``value`` is the query string
    * ``youtube_search`` - ``value`` is the query string
    * ``say``           - ``value`` is the literal utterance
    * ``wait``          - ``value`` is the number of seconds (string)
    * ``run_routine``   - ``value`` is the routine name
    * ``custom``        - ``value`` is a free-form string passed to a custom
      handler registered in :mod:`routines.manager`.
    """

    action: str
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutineAction":
        action = str(payload.get("action", "")).strip().lower()
        value = str(payload.get("value", ""))
        metadata_raw = payload.get("metadata", {}) or {}
        if not isinstance(metadata_raw, dict):
            metadata_raw = {"raw": metadata_raw}
        return cls(action=action, value=value, metadata=dict(metadata_raw))


class BaseTool(ABC):
    """Contract every feature tool must implement.

    A tool is intentionally tiny - it never owns I/O resources or global
    state. Sub-classes should accept their dependencies through the
    constructor and re-implement :meth:`can_handle` and :meth:`execute`.
    """

    #: Identifier used in logs and the router registry.
    name: str = "base_tool"

    @abstractmethod
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Return ``True`` if this tool can handle *command*.

        Implementations should be lightweight (no I/O, no model calls) so the
        router can quickly dispatch on a string-match fallback.
        """

    @abstractmethod
    def execute(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Run the tool and return a :class:`ToolResult`."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def safe_execute(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Wrap :meth:`execute` with consistent error handling."""
        try:
            log.debug("%s handling %r", self.name, command)
            result = self.execute(command, context)
            if not isinstance(result, ToolResult):
                # Tools that return bare strings are common in legacy code;
                # wrap them so the router can rely on a stable type.
                result = ToolResult(success=True, message=str(result))
            return result
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Tool %s failed: %s", self.name, exc)
            return ToolResult(success=False, message=f"{self.name} failed: {exc}")

    # ------------------------------------------------------------------
    # LangChain glue
    # ------------------------------------------------------------------
    def as_langchain_tool(self):
        """Convert this tool to a LangChain tool for the agent executor.

        Sub-classes may override this if they need richer descriptions.
        """
        try:
            from langchain.tools import Tool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LangChain is not installed. Install langchain-core to expose tools."
            ) from exc

        description = (
            getattr(self, "description", "")
            or f"Run the {self.name} tool. Input is a natural-language command."
        )

        def _run(payload: str) -> str:
            result = self.safe_execute(payload)
            return result.message

        return Tool.from_function(
            func=_run,
            name=self.name,
            description=description,
        )


__all__ = ["BaseTool", "RoutineAction", "ToolResult"]
