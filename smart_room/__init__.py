"""
smart_room
~~~~~~~~~~

Local-network IoT integration for arvis. The current version talks to a
single ESP32 controller that exposes two HTTP endpoints:

* ``GET  /status``  - returns the controller's state and Wi-Fi info.
* ``POST /command`` - accepts a small JSON document that names a device
  and an action.

The package is intentionally split into small modules so it can grow
without touching existing code:

* :mod:`smart_room.exceptions` - typed errors for the module.
* :mod:`smart_room.devices`    - the device / action whitelists and the
  validation helpers. *No* device is ever sent to the ESP32 unless it has
  been validated here first.
* :mod:`smart_room.client`     - the only place that opens a socket to
  the ESP32. All other modules go through this client.
* :mod:`smart_room.status`     - a typed view of the controller's status
  payload.
* :mod:`smart_room.commands`   - the :class:`~core.base.BaseTool` that
  exposes the room to arvis, plus the registration helper used by
  :mod:`app`.

The room controller's IP address is **never hardcoded** - it is read
from the ``ESP32_ROOM_IP`` environment variable (see
:mod:`core.config`). This module never executes shell commands, never
opens arbitrary URLs, and never exposes GPIO numbers.
"""

from smart_room.commands import SmartRoomTool, register_smart_room_tools
from smart_room.devices import ALLOWED_DEVICES, ALLOWED_ACTIONS, validate_command
from smart_room.client import ESP32RoomClient, SmartRoomSettings
from smart_room.status import RoomStatus
from smart_room.exceptions import (
    SmartRoomError,
    SmartRoomInvalidCommandError,
    SmartRoomNotConfiguredError,
    SmartRoomResponseError,
    SmartRoomTimeoutError,
    SmartRoomUnavailableError,
)

__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_DEVICES",
    "ESP32RoomClient",
    "RoomStatus",
    "SmartRoomError",
    "SmartRoomInvalidCommandError",
    "SmartRoomNotConfiguredError",
    "SmartRoomResponseError",
    "SmartRoomSettings",
    "SmartRoomTimeoutError",
    "SmartRoomTool",
    "SmartRoomUnavailableError",
    "register_smart_room_tools",
    "validate_command",
]
