"""
smart_room.devices
~~~~~~~~~~~~~~~~~~

The single source of truth for *what* arvis is allowed to ask the room
controller to do. Nothing in this module talks to the network - it only
defines the whitelist and the validator.

The whitelist is intentionally tiny in the first version (``light`` plus
``on``/``off``/``toggle``) so the surface area for abuse is small. The
shapes here also leave room for a future ``fan``, ``rgb_light``,
``sensor`` etc. without rewriting the validator.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from smart_room.exceptions import SmartRoomInvalidCommandError

# Whitelist of device ids the ESP32 firmware understands. Keeping them
# lowercase means callers can normalise once and never again.
ALLOWED_DEVICES: Tuple[str, ...] = ("light",)

# Whitelist of action verbs understood by the firmware.
ALLOWED_ACTIONS: Tuple[str, ...] = ("on", "off", "toggle")


def normalise(value: str) -> str:
    """Trim whitespace and lowercase a value. Empty strings are rejected."""
    if value is None:  # type: ignore[unreachable]
        raise SmartRoomInvalidCommandError("Missing value.")
    text = str(value).strip().lower()
    if not text:
        raise SmartRoomInvalidCommandError("Value is empty.")
    return text


def is_allowed_device(device: str) -> bool:
    return normalise(device) in ALLOWED_DEVICES


def is_allowed_action(action: str) -> bool:
    return normalise(action) in ALLOWED_ACTIONS


def validate_command(device: str, action: str, *, allowed_devices: Iterable[str] = ALLOWED_DEVICES,
                     allowed_actions: Iterable[str] = ALLOWED_ACTIONS) -> Tuple[str, str]:
    """Validate *device* and *action* against the whitelists.

    Returns the normalised ``(device, action)`` pair. Raises
    :class:`SmartRoomInvalidCommandError` if either is unknown.

    Accepting ``allowed_devices`` / ``allowed_actions`` as parameters
    keeps the function trivial to unit-test with custom whitelists.
    """
    device_n = normalise(device)
    action_n = normalise(action)
    if device_n not in tuple(allowed_devices):
        raise SmartRoomInvalidCommandError(
            f"Unknown device {device!r}. Allowed devices: {', '.join(allowed_devices)}."
        )
    if action_n not in tuple(allowed_actions):
        raise SmartRoomInvalidCommandError(
            f"Unknown action {action!r}. Allowed actions: {', '.join(allowed_actions)}."
        )
    return device_n, action_n


__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_DEVICES",
    "is_allowed_action",
    "is_allowed_device",
    "normalise",
    "validate_command",
]
