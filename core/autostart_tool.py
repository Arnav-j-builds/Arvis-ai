"""
core.autostart_tool
~~~~~~~~~~~~~~~~~~~

Voice entry point for the Windows autostart mechanism implemented in
:mod:`core.autostart`. The user can say things like:

* "start at boot" / "run at startup" / "launch on windows startup"
* "stop at boot"  / "do not run at startup"
* "am I set to start at boot?" / "is autostart on?"

The tool never raises - every error becomes a friendly :class:`ToolResult`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.autostart import disable, enable, is_enabled
from core.base import BaseTool, ToolResult
from core.logger import get_logger

log = get_logger(__name__)


_ENABLE_PHRASES = (
    "start at boot",
    "start at startup",
    "start at windows startup",
    "start at windows boot",
    "run at boot",
    "run at startup",
    "run at windows startup",
    "run on startup",
    "run on boot",
    "run on windows startup",
    "launch at boot",
    "launch at startup",
    "launch on startup",
    "launch on boot",
    "launch on windows startup",
    "auto start at boot",
    "auto start on boot",
    "autostart on",
    "enable autostart",
    "enable auto start",
    "enable start at boot",
    "register for startup",
    "add to startup",
    "add me to startup",
)

_DISABLE_PHRASES = (
    "stop at boot",
    "stop at startup",
    "do not run at startup",
    "don't run at startup",
    "do not start at boot",
    "don't start at boot",
    "remove from startup",
    "remove me from startup",
    "remove from autostart",
    "disable autostart",
    "disable auto start",
    "disable start at boot",
    "autostart off",
    "turn off autostart",
    "turn off auto start",
)

_STATUS_PHRASES = (
    "is autostart on",
    "is autostart enabled",
    "am i set to start at boot",
    "am i starting at boot",
    "autostart status",
    "startup status",
    "do you start at boot",
    "do you start with windows",
)


def _contains_any(text: str, needles: tuple) -> bool:
    lowered = (text or "").lower()
    return any(needle in lowered for needle in needles)


class AutostartTool(BaseTool):
    """Toggle the Windows autostart registration for arvis."""

    name = "autostart_tool"
    description = (
        "Toggle the Windows autostart registration for arvis. "
        "Use 'start at boot' to register arvis to launch automatically "
        "when Windows starts, 'stop at boot' to remove the registration, "
        "and 'autostart status' to check the current state."
    )

    def can_handle(
        self, command: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        if _contains_any(text, _ENABLE_PHRASES):
            return True
        if _contains_any(text, _DISABLE_PHRASES):
            return True
        if _contains_any(text, _STATUS_PHRASES):
            return True
        if "startup" in text or "autostart" in text or "start at boot" in text:
            return True
        return False

    def execute(
        self, command: str, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()
        try:
            if _contains_any(lowered, _DISABLE_PHRASES):
                ok = disable()
                if ok:
                    return ToolResult(
                        success=True,
                        message=(
                            "Done, sir. arvis will no longer launch automatically "
                            "when Windows starts."
                        ),
                        data={"enabled": False},
                    )
                return ToolResult(
                    success=False,
                    message=(
                        "I could not remove the autostart entry, sir. This "
                        "feature is only available on Windows."
                    ),
                    data={"enabled": is_enabled()},
                )

            if _contains_any(lowered, _STATUS_PHRASES):
                enabled = is_enabled()
                msg = (
                    "Yes sir, arvis is set to start automatically when Windows boots."
                    if enabled
                    else "No sir, arvis is not registered to start with Windows."
                )
                return ToolResult(success=True, message=msg, data={"enabled": enabled})

            if _contains_any(lowered, _ENABLE_PHRASES):
                ok = enable()
                if ok:
                    return ToolResult(
                        success=True,
                        message=(
                            "Done, sir. arvis will now launch automatically "
                            "the next time Windows starts."
                        ),
                        data={"enabled": True},
                    )
                return ToolResult(
                    success=False,
                    message=(
                        "I could not register the autostart entry, sir. "
                        "This feature is only available on Windows."
                    ),
                    data={"enabled": is_enabled()},
                )

            # Bare "autostart" or "startup" word - report current state.
            if "autostart" in lowered or "startup" in lowered:
                enabled = is_enabled()
                return ToolResult(
                    success=True,
                    message=(
                        "Autostart is currently " + ("on" if enabled else "off") + ", sir."
                    ),
                    data={"enabled": enabled},
                )

            return ToolResult(
                success=False,
                message="I did not understand that startup command, sir.",
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("AutostartTool failed: %s", exc)
            return ToolResult(success=False, message=f"Autostart error: {exc}")


def register_autostart_tool(router) -> list:
    """Register the autostart tool with *router* and return the new tools."""
    tool = AutostartTool()
    router.register(
        tool,
        keywords=(
            "autostart",
            "startup",
            "start at boot",
            "run at startup",
            "run at boot",
            "launch on startup",
            "launch at boot",
        ),
        priority=70,
    )
    return [tool]


__all__ = ["AutostartTool", "register_autostart_tool"]
