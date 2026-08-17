"""
core.legacy_bridge
~~~~~~~~~~~~~~~~~~

Adapt the legacy LangChain ``@tool``-decorated functions (the original
``tools/time.py``, ``tools/opener.py`` etc.) into :class:`BaseTool`
instances so they can be registered with :class:`CommandRouter`.

The bridge is **not** a dumb pass-through. Each adapter sets a real
``can_handle`` that looks for the keywords the underlying tool actually
understands, so the router no longer dumps every unknown command into the
first registered tool. We also forward the structured-input tools
correctly (some legacy tools require a JSON object, others a plain string).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain.tools import Tool  # local import - LangChain is required anyway

from core.base import BaseTool, ToolResult
from core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _invoke_legacy(tool, payload: str) -> Any:
    """Call a LangChain tool with the right argument shape.

    * If the tool wants a structured schema (it exposes ``args`` and the
      top-level field is not ``"input"``), pass the raw string as ``input``
      so the LLM-style prompt parser still receives a usable payload.
    * Otherwise, invoke with ``{"input": payload}`` which works for plain
      ``@tool`` functions.
    """
    try:
        # Newer LangChain tools expose ``args_schema``; if so we cannot call
        # them with arbitrary strings. We fall back to ``.run(payload)``.
        schema = getattr(tool, "args_schema", None)
        args = getattr(tool, "args", None) or {}
        if schema is not None and args and "input" not in args:
            # Structured tool - the LLM is the only thing that should be
            # invoking it. As a deterministic fallback, call ``.run`` with
            # the raw command and let the tool's own defaults fill any
            # missing fields.
            return tool.run(payload)
    except Exception:
        pass

    if hasattr(tool, "invoke"):
        try:
            return tool.invoke({"input": payload})
        except Exception:
            return tool.run(payload)
    return tool.run(payload)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class _LangChainToolAdapter(BaseTool):
    """Wraps a legacy LangChain tool as a :class:`BaseTool`.

    Parameters
    ----------
    tool:
        The LangChain ``BaseTool`` / ``StructuredTool`` returned by
        ``@tool``.
    keywords:
        Iterable of substrings that mark the command as a candidate for
        this tool. The first keyword that is found inside the command wins.
    fallback_handler:
        Optional callable used as a last resort when no keyword matches
        and we are unsure whether this tool is the right one. Returning
        ``None`` from the callable signals "I cannot handle this command"
        and the router tries the next registered tool.
    """

    def __init__(
        self,
        tool,
        keywords: tuple[str, ...] = (),
        fallback_handler=None,
        description: str = "",
    ) -> None:
        self._tool = tool
        self.name = getattr(tool, "name", "legacy_tool") or "legacy_tool"
        # Prefer the tool's own description unless we override it.
        self.description = description or getattr(tool, "description", "") or ""
        self._keywords = tuple(k.lower() for k in keywords)
        self._fallback_handler = fallback_handler

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        if any(kw in text for kw in self._keywords):
            return True
        if self._fallback_handler is not None:
            try:
                return bool(self._fallback_handler(text))
            except Exception:  # pragma: no cover
                return False
        return False

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        try:
            raw = _invoke_legacy(self._tool, command)
            return ToolResult(success=True, message=str(raw))
        except Exception as exc:
            log.exception("Legacy tool %s failed: %s", self.name, exc)
            return ToolResult(success=False, message=f"{self.name} failed: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def langchain_tool_to_base(
    tool,
    keywords: tuple[str, ...] = (),
    fallback_handler=None,
    description: str = "",
) -> BaseTool:
    """Convert a LangChain tool into a :class:`BaseTool`.

    See :class:`_LangChainToolAdapter` for parameter semantics.
    """
    return _LangChainToolAdapter(
        tool=tool,
        keywords=keywords,
        fallback_handler=fallback_handler,
        description=description,
    )


def _is_open_intent(text: str) -> bool:
    """Heuristic matching for the legacy ``open_anything`` tool."""
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    starters = (
        "open ",
        "launch ",
        "start ",
        "go to ",
        "google ",
        "search google ",
        "youtube ",
        "search youtube ",
        "play ",
        "youtube play ",
        "open app ",
        "launch app ",
        "start app ",
    )
    if any(lowered.startswith(s) for s in starters):
        return True
    # Bare domain or site keyword
    if "open " in lowered or "launch " in lowered:
        return True
    return False


def _is_time_query(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "what time",
            "what's the time",
            "whats the time",
            "current time",
            "tell me the time",
            "time in ",
            "time at ",
        )
    )


def _is_arp_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in ("arp scan", "scan the network", "who is on my network", "scan network"))


def _is_search_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in ("search for", "look up", "duckduckgo", "search the web"))


def _is_matrix(text: str) -> bool:
    return "matrix mode" in (text or "").lower()


def _is_screenshot(text: str) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in ("take a screenshot", "screenshot this", "capture the screen"))


__all__ = ["langchain_tool_to_base"]
