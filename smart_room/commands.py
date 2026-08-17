"""
smart_room.commands
~~~~~~~~~~~~~~~~~~~

The voice entry point for the smart-room / IoT controller. The
:class:`SmartRoomTool` is the deterministic fast-path used by
:mod:`core.router` (it runs *before* the LLM for obvious phrases such
as ``"turn on my room light"``). On top of that, two AI-callable tools
- ``control_room_device`` and ``get_room_status`` - are registered
with the LangChain agent through thin :class:`BaseTool` adapters so
the LLM can call them by name when the user phrasing is unclear.

Voice patterns the tool recognises
---------------------------------

``"Turn on my room light."``        -> ``control_room_device("light", "on")``
``"Turn off my room light."``       -> ``control_room_device("light", "off")``
``"Toggle my room light."``         -> ``control_room_device("light", "toggle")``
``"What's the status of my room?"`` -> ``get_room_status()``
``"Is my room light on?"``          -> ``get_room_status()``
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.logger import get_logger
from smart_room.client import ESP32RoomClient
from smart_room.devices import ALLOWED_ACTIONS, ALLOWED_DEVICES, validate_command
from smart_room.exceptions import (
    SmartRoomError,
    SmartRoomInvalidCommandError,
    SmartRoomNotConfiguredError,
    SmartRoomResponseError,
    SmartRoomTimeoutError,
    SmartRoomUnavailableError,
)
from smart_room.status import RoomStatus

log = get_logger(__name__)


# Deterministic router triggers. The first list is for commands; the
# second is for status queries. Both run before the LLM is consulted.
_TRIGGERS_COMMAND = (
    "turn on my room light",
    "turn on the room light",
    "turn on room light",
    "turn my room light on",
    "turn the room light on",
    "switch on the room light",
    "switch on my room light",
    "turn off my room light",
    "turn off the room light",
    "turn off room light",
    "turn my room light off",
    "turn the room light off",
    "switch off the room light",
    "switch off my room light",
    "toggle my room light",
    "toggle the room light",
    "toggle room light",
    "flip the room light",
    "flip my room light",
)

_TRIGGERS_STATUS = (
    "room status",
    "status of my room",
    "status of the room",
    "what is the status of my room",
    "what's the status of my room",
    "is my room light on",
    "is the room light on",
    "is my room light off",
    "is the room light off",
    "room light status",
    "check the room",
    "check my room",
)


# ----------------------------------------------------------------------
# Small parsing helpers
# ----------------------------------------------------------------------
def _is_command_intent(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    if any(trigger in lowered for trigger in _TRIGGERS_COMMAND):
        return True
    if ("room light" in lowered or "room's light" in lowered) and any(
        token in lowered for token in ("on", "off", "toggle", "switch", "flip")
    ):
        return True
    return False


def _is_status_intent(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    if any(trigger in lowered for trigger in _TRIGGERS_STATUS):
        return True
    return "room" in lowered and any(token in lowered for token in ("status", "state", "check"))


def _normalise_action_from_command(command: str) -> Optional[str]:
    """Best-effort extraction of an action from a free-form phrase.

    Only used for the deterministic fast-path. The LLM is still allowed
    to call the AI tool directly with a structured payload.
    """
    lowered = (command or "").lower()
    if not lowered:
        return None
    if "toggle" in lowered or "flip" in lowered:
        return "toggle"
    if "turn off" in lowered or "switch off" in lowered:
        return "off"
    if "turn on" in lowered or "switch on" in lowered:
        return "on"
    # Bare "room light on" / "room light off" variants.
    if "room light" in lowered or "room's light" in lowered:
        if re.search(r"\boff\b", lowered):
            return "off"
        if re.search(r"\bon\b", lowered):
            return "on"
    return None


def _device_token(device: str) -> str:
    return {"light": "room light"}.get(device, device)


def _action_token(action: str) -> str:
    return {"on": "on", "off": "off", "toggle": "toggled"}.get(action, action)


def _natural_command_message(device: str, action: str, response: dict) -> str:
    new_state = (response or {}).get("state") or _action_token(action)
    friendly = _device_token(device)
    if action == "toggle":
        return f"The {friendly} is now {new_state}."
    if action == "on":
        return f"The {friendly} is on."
    if action == "off":
        return f"The {friendly} is off."
    return f"The {friendly} is {new_state}."


def _status_message(status: RoomStatus) -> str:
    if not status.wifi:
        return "The room controller is online but its Wi-Fi link is down."
    light_state = "on" if status.light == "on" else "off"
    ip = status.ip or "an unknown address"
    return f"The room light is {light_state}. The controller is connected at {ip}."


# ----------------------------------------------------------------------
# The user-facing tool used by the deterministic router
# ----------------------------------------------------------------------
class SmartRoomTool(BaseTool):
    """Single :class:`BaseTool` for the deterministic voice loop.

    It dispatches to either :meth:`control_room_device` or
    :meth:`get_room_status` based on the user's command. The two AI
    tools exposed to the LangChain agent are *separate* small
    :class:`BaseTool` adapters (see :func:`register_smart_room_tools`)
    so the LLM can call them by their natural names.
    """

    name = "smart_room_tool"
    description = (
        "Control a smart room controller (an ESP32 on the local network). "
        "Use this for room-light on / off / toggle and room status queries."
    )

    def __init__(self, client: Optional[ESP32RoomClient] = None) -> None:
        self._client = client or ESP32RoomClient.from_config()

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        return _is_command_intent(command) or _is_status_intent(command)

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").strip()
        if _is_command_intent(text):
            action = _normalise_action_from_command(text)
            if action is None:
                return ToolResult(
                    success=False,
                    message="I am not sure whether to turn the room light on or off, sir.",
                )
            return self.control_room_device(device="light", action=action)
        if _is_status_intent(text):
            return self.get_room_status()
        return ToolResult(success=False, message="I did not understand that room command, sir.")

    # ------------------------------------------------------------------
    # Public API used by the AI-tool adapters
    # ------------------------------------------------------------------
    def control_room_device(self, *, device: str, action: str) -> ToolResult:
        """Send a validated command to the room controller.

        Validates the device / action pair against the whitelist before
        opening a socket. All network errors are translated to friendly
        sentences - the user never sees a stack trace.
        """
        try:
            device_n, action_n = validate_command(device, action)
        except SmartRoomInvalidCommandError as exc:
            log.warning("Rejected room command: %s", exc)
            return ToolResult(success=False, message=str(exc))

        try:
            response = self._client.send_command(device_n, action_n)
        except SmartRoomNotConfiguredError as exc:
            log.warning("Smart room not configured: %s", exc)
            return ToolResult(
                success=False,
                message=(
                    "The room controller is not configured yet, sir. "
                    "Set ESP32_ROOM_IP in the .env file."
                ),
            )
        except SmartRoomTimeoutError:
            log.warning("Smart room timeout")
            return ToolResult(success=False, message="I can't reach the room controller right now.")
        except SmartRoomUnavailableError:
            log.warning("Smart room unavailable")
            return ToolResult(success=False, message="I can't reach the room controller right now.")
        except SmartRoomResponseError as exc:
            log.warning("Smart room rejected the command: %s", exc)
            return ToolResult(
                success=False,
                message="The room controller rejected the command, sir.",
            )
        except SmartRoomError as exc:
            log.exception("Unexpected smart room error: %s", exc)
            return ToolResult(success=False, message="I can't reach the room controller right now.")

        return ToolResult(
            success=True,
            message=_natural_command_message(device_n, action_n, response),
            data={
                "device": device_n,
                "action": action_n,
                "state": response.get("state"),
                "raw": response,
            },
        )

    def get_room_status(self) -> ToolResult:
        """Return the current room status to the LLM / the user."""
        try:
            status = self._client.get_status()
        except SmartRoomNotConfiguredError as exc:
            log.warning("Smart room not configured: %s", exc)
            return ToolResult(
                success=False,
                message=(
                    "The room controller is not configured yet, sir. "
                    "Set ESP32_ROOM_IP in the .env file."
                ),
            )
        except SmartRoomTimeoutError:
            log.warning("Smart room timeout")
            return ToolResult(success=False, message="I can't reach the room controller right now.")
        except SmartRoomUnavailableError:
            log.warning("Smart room unavailable")
            return ToolResult(success=False, message="I can't reach the room controller right now.")
        except SmartRoomResponseError as exc:
            log.warning("Smart room bad response: %s", exc)
            return ToolResult(
                success=False,
                message="The room controller sent an unexpected response, sir.",
            )
        except SmartRoomError as exc:
            log.exception("Unexpected smart room error: %s", exc)
            return ToolResult(success=False, message="I can't reach the room controller right now.")

        return ToolResult(success=True, message=_status_message(status), data=status.to_dict())


# ----------------------------------------------------------------------
# AI-tool adapters - one per LangChain-facing tool
# ----------------------------------------------------------------------
def _parse_control_payload(payload: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract ``(device, action)`` from a string the LLM passed in.

    The LangChain agent sometimes calls the tool with a natural
    language string (``"turn the light on please"``) and sometimes
    with a JSON document. Both are supported for robustness.
    """
    text = (payload or "").strip()
    if not text:
        return None, None

    # 1) JSON object?
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            device = data.get("device")
            action = data.get("action")
            if device and action:
                return str(device), str(action)

    # 2) Free-form string - scan for the first device + action.
    lowered = text.lower().replace('"', "").replace("'", "")
    device = next((tok for tok in ALLOWED_DEVICES if tok in lowered), None)
    action = next(
        (tok for tok in ALLOWED_ACTIONS if re.search(rf"\b{re.escape(tok)}\b", lowered)),
        None,
    )
    return device, action


class _ControlRoomDeviceTool(BaseTool):
    """LangChain-facing tool for ``control_room_device(device, action)``."""

    name = "control_room_device"
    description = (
        "Send a command to the smart-room controller (an ESP32 on the LAN). "
        "The only supported device is 'light'. The allowed actions are "
        "'on', 'off', and 'toggle'. Input is a JSON object such as "
        '``{"device": "light", "action": "on"}`` or a natural-language phrase '
        "like 'turn on the light'."
    )

    def __init__(self, room: SmartRoomTool) -> None:
        self._room = room

    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:  # noqa: D401
        # The deterministic router does not call this tool directly - the
        # LLM does. Returning ``False`` keeps the router from short-circuiting
        # structured invocations.
        return False

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        device, action = _parse_control_payload(command)
        if not device or not action:
            return ToolResult(
                success=False,
                message=(
                    "I need both a device and an action. Try device='light' with "
                    "action 'on', 'off', or 'toggle'."
                ),
            )
        return self._room.control_room_device(device=device, action=action)


class _GetRoomStatusTool(BaseTool):
    """LangChain-facing tool for ``get_room_status()``."""

    name = "get_room_status"
    description = (
        "Read the current state of the smart-room controller. Use this when the "
        "user asks whether a room device is on or off, or wants the current room "
        "status. Input is ignored."
    )

    def __init__(self, room: SmartRoomTool) -> None:
        self._room = room

    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:  # noqa: D401
        return False

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        return self._room.get_room_status()


# ----------------------------------------------------------------------
# Registration helper
# ----------------------------------------------------------------------
def register_smart_room_tools(router) -> List[BaseTool]:
    """Register the smart-room tools with *router* and return them.

    Three tools are registered:

    * :class:`SmartRoomTool` - handles the deterministic fast-path used
      by the router for obvious room commands.
    * :class:`_ControlRoomDeviceTool` - exposes ``control_room_device`` to
      the LangChain agent so the LLM can call it by name.
    * :class:`_GetRoomStatusTool` - exposes ``get_room_status`` to the
      LangChain agent.

    Only the deterministic tool is registered with high-priority
    keywords; the LLM-facing tools are added with a low priority so
    they do not steal unrelated commands.
    """
    room = SmartRoomTool()
    control = _ControlRoomDeviceTool(room)
    status = _GetRoomStatusTool(room)

    # Deterministic fast-path tool - high priority keywords.
    router.register(
        room,
        keywords=(
            "room light",
            "turn on my room",
            "turn off my room",
            "toggle my room",
            "room status",
            "is my room light",
            "is the room light",
        ),
        priority=80,
    )

    # LangChain-facing tools - no keywords (the LLM routes to them by
    # name through its tool-use machinery). The router still needs them
    # in the registry so :meth:`langchain_tools` enumerates them.
    router.register(control, keywords=(), priority=10)
    router.register(status, keywords=(), priority=10)

    return [room, control, status]


__all__ = ["SmartRoomTool", "register_smart_room_tools"]
