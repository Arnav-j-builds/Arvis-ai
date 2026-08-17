"""
smart_room.status
~~~~~~~~~~~~~~~~~

Typed view of the controller's ``/status`` payload. We keep this small
because the firmware is the only thing that produces the JSON and any
new fields can be added without breaking the rest of the assistant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class RoomStatus:
    """The structured view arvis keeps about the room controller.

    Attributes
    ----------
    controller_id:
        Friendly name (the value of ``ESP32_ROOM_ID``).
    light:
        ``"on"`` or ``"off"`` - what the firmware reported for the light.
    wifi:
        ``True`` if the firmware reports the Wi-Fi station as connected.
    ip:
        The IP the firmware believes it has. Useful when the controller
        picks its address via DHCP.
    """

    controller_id: str
    light: str
    wifi: bool
    ip: str

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, controller_id: str) -> "RoomStatus":
        """Build a :class:`RoomStatus` from a decoded JSON object.

        The function accepts whatever keys the firmware chooses to
        advertise. Required keys (``device``, ``light``, ``wifi``) are
        strict; ``ip`` defaults to an empty string when the firmware
        omits it.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("Status payload is not a JSON object.")

        light_raw = payload.get("light", "off")
        light = str(light_raw).strip().lower() or "off"
        if light not in {"on", "off"}:
            raise ValueError(f"Unexpected light state {light_raw!r} from controller.")

        wifi_raw = payload.get("wifi", False)
        if isinstance(wifi_raw, bool):
            wifi = wifi_raw
        elif isinstance(wifi_raw, (int, float)):
            wifi = bool(wifi_raw)
        elif isinstance(wifi_raw, str):
            wifi = wifi_raw.strip().lower() in {"1", "true", "yes", "on", "connected"}
        else:
            raise ValueError("Unexpected wifi state from controller.")

        ip = str(payload.get("ip", "") or "").strip()

        return cls(
            controller_id=str(payload.get("device", controller_id) or controller_id),
            light=light,
            wifi=wifi,
            ip=ip,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain dict that mirrors the JSON arvis exposes."""
        return asdict(self)


__all__ = ["RoomStatus"]
