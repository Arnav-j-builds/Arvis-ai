"""
smart_room.exceptions
~~~~~~~~~~~~~~~~~~~~~

Typed error hierarchy for the smart-room module. The :class:`BaseTool`
subclass in :mod:`smart_room.commands` relies on these to translate
low-level failures (timeouts, refused connections, malformed payloads)
into a single :class:`~core.base.ToolResult` that arvis can speak
without leaking implementation details.
"""

from __future__ import annotations


class SmartRoomError(Exception):
    """Base class for every error raised by the smart-room module."""


class SmartRoomNotConfiguredError(SmartRoomError):
    """The ESP32 IP has not been set (``ESP32_ROOM_IP`` is empty)."""


class SmartRoomUnavailableError(SmartRoomError):
    """The ESP32 refused the connection or did not respond at all."""


class SmartRoomTimeoutError(SmartRoomError):
    """The ESP32 took longer than the configured timeout to answer."""


class SmartRoomInvalidCommandError(SmartRoomError):
    """The requested device or action is not in the whitelist."""


class SmartRoomResponseError(SmartRoomError):
    """The ESP32 answered, but the body was not usable."""


__all__ = [
    "SmartRoomError",
    "SmartRoomNotConfiguredError",
    "SmartRoomUnavailableError",
    "SmartRoomTimeoutError",
    "SmartRoomInvalidCommandError",
    "SmartRoomResponseError",
]
