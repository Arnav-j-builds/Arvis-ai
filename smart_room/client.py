"""
smart_room.client
~~~~~~~~~~~~~~~~~

The only module in the package that ever opens a socket to the ESP32.
Every other piece of smart-room code goes through :class:`ESP32RoomClient`.

Design notes
------------

* **Configuration is injected.** Callers pass a :class:`SmartRoomSettings`
  (or use :meth:`ESP32RoomClient.from_config` to read it from
  :mod:`core.config`). The class never reads the environment directly -
  this is what makes the client trivial to test.
* **No shell, no URL injection.** The base URL is built once from the
  immutable IP and port. The HTTP client can only hit ``/status`` and
  ``/command``; the device / action sent in the body is validated by
  :mod:`smart_room.devices` *before* it ever reaches this client.
* **Bounded timeouts.** Both endpoints honour ``settings.timeout``. The
  client distinguishes "host down" from "host slow" via the underlying
  ``requests`` exception types.
* **Typed errors.** The client maps the noisy ``requests`` exception
  hierarchy to the small set of :mod:`smart_room.exceptions` classes so
  callers do not have to import anything from ``requests``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import requests

from core.logger import get_logger
from smart_room.devices import ALLOWED_ACTIONS, ALLOWED_DEVICES, validate_command
from smart_room.exceptions import (
    SmartRoomInvalidCommandError,
    SmartRoomNotConfiguredError,
    SmartRoomResponseError,
    SmartRoomTimeoutError,
    SmartRoomUnavailableError,
)
from smart_room.status import RoomStatus

log = get_logger(__name__)


@dataclass(frozen=True)
class SmartRoomSettings:
    """Immutable connection settings for the room controller.

    Attributes
    ----------
    ip:
        The ESP32's IPv4 address on the LAN. Must be set; the client
        raises :class:`SmartRoomNotConfiguredError` otherwise.
    port:
        TCP port for the HTTP server. Defaults to ``80``.
    timeout:
        Per-request timeout in seconds. Defaults to ``5``.
    controller_id:
        Friendly name that arvis uses when it speaks about the room.
    """

    ip: Optional[str]
    port: int = 80
    timeout: int = 5
    controller_id: str = "arvis-room-controller"

    @property
    def base_url(self) -> str:
        """Return the ``http://ip:port`` base URL, with no trailing slash."""
        if not self.ip:
            raise SmartRoomNotConfiguredError(
                "ESP32_ROOM_IP is not set. Add it to your .env file to enable the room controller."
            )
        return f"http://{self.ip}:{self.port}"

    @classmethod
    def from_config(cls) -> "SmartRoomSettings":
        """Build settings from :func:`core.config.get_config`.

        Equivalent to :meth:`ESP32RoomClient.from_config` but returns the
        settings dataclass so callers can inspect the values before
        constructing a client.
        """
        # Local import keeps ``smart_room`` importable even if ``core``
        # is unavailable (e.g. during isolated unit tests).
        from core.config import get_config

        cfg = get_config()
        return cls(
            ip=cfg.esp32_room_ip,
            port=cfg.esp32_room_port,
            timeout=cfg.esp32_room_timeout,
            controller_id=cfg.esp32_room_id,
        )


class ESP32RoomClient:
    """Thin HTTP client for the room controller.

    Use the static :meth:`from_config` helper to build an instance that
    picks its settings up from the environment, or instantiate directly
    when testing.

    >>> client = ESP32RoomClient(SmartRoomSettings(ip="192.168.1.50"))
    >>> client.send_command("light", "on")
    {'success': True, 'device': 'light', 'state': 'on'}
    """

    #: Endpoints exposed by the firmware. Centralising them prevents
    #: typos and makes the URL surface auditable.
    STATUS_PATH = "/status"
    COMMAND_PATH = "/command"

    def __init__(
        self,
        settings: SmartRoomSettings,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._settings = settings
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, *, session: Optional[requests.Session] = None) -> "ESP32RoomClient":
        """Build an :class:`ESP32RoomClient` from the active :class:`~core.config.Config`.

        Pulls ``esp32_room_ip``, ``esp32_room_port``, ``esp32_room_timeout``
        and ``esp32_room_id`` from the environment. The local import
        keeps :mod:`smart_room` importable even when :mod:`core.config`
        is unavailable (handy for isolated unit tests).
        """
        from core.config import get_config

        cfg = get_config()
        return cls(
            SmartRoomSettings(
                ip=cfg.esp32_room_ip,
                port=cfg.esp32_room_port,
                timeout=cfg.esp32_room_timeout,
                controller_id=cfg.esp32_room_id,
            ),
            session=session,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        """Return ``True`` when the IP has been set."""
        return bool(self._settings.ip)

    def check(self) -> bool:
        """Lightweight reachability probe.

        Performs a ``GET /status`` and returns ``True`` if the controller
        answers with a 2xx status code. Network failures are swallowed
        and reported as ``False`` so the caller can use the return value
        as a yes/no health check.
        """
        try:
            self.get_status()
            return True
        except SmartRoomError as exc:
            log.info("Room controller health check failed: %s", exc)
            return False

    def send_command(self, device: str, action: str) -> dict:
        """Send a validated command to ``/command`` and return the body.

        Raises
        ------
        SmartRoomInvalidCommandError
            If *device* or *action* is not in the whitelist. This
            validation happens **before** the network call.
        SmartRoomUnavailableError
            The controller is not reachable (connection refused, DNS
            failure, etc.).
        SmartRoomTimeoutError
            The controller did not answer in time.
        SmartRoomResponseError
            The controller answered but the payload was malformed or
            the status code indicated an error.
        """
        device_n, action_n = validate_command(device, action)

        if not self.is_configured():
            raise SmartRoomNotConfiguredError(
                "ESP32_ROOM_IP is not set. Add it to your .env file to enable the room controller."
            )

        url = f"{self._settings.base_url}{self.COMMAND_PATH}"
        payload = {"device": device_n, "action": action_n}
        log.debug("POST %s payload=%s", url, payload)

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self._settings.timeout,
            )
        except requests.Timeout as exc:
            raise SmartRoomTimeoutError(
                f"The room controller did not respond within {self._settings.timeout}s."
            ) from exc
        except requests.ConnectionError as exc:
            raise SmartRoomUnavailableError(
                "I cannot reach the room controller right now."
            ) from exc
        except requests.RequestException as exc:
            raise SmartRoomUnavailableError(
                f"The room controller request failed: {exc}"
            ) from exc

        if response.status_code >= 500:
            raise SmartRoomResponseError(
                f"The room controller replied with HTTP {response.status_code}."
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise SmartRoomResponseError(
                "The room controller returned a non-JSON response."
            ) from exc

        if response.status_code >= 400:
            message = body.get("error") if isinstance(body, Mapping) else None
            raise SmartRoomResponseError(
                message or f"The room controller rejected the command (HTTP {response.status_code})."
            )

        if not isinstance(body, Mapping):
            raise SmartRoomResponseError(
                "The room controller returned an unexpected response shape."
            )

        # Surface controller-level errors as an explicit failure.
        if body.get("success") is False:
            message = body.get("error") or body.get("message") or "Unknown controller error."
            raise SmartRoomResponseError(str(message))

        return dict(body)

    def get_status(self) -> RoomStatus:
        """Fetch the controller's status payload and return a typed view."""
        if not self.is_configured():
            raise SmartRoomNotConfiguredError(
                "ESP32_ROOM_IP is not set. Add it to your .env file to enable the room controller."
            )

        url = f"{self._settings.base_url}{self.STATUS_PATH}"
        log.debug("GET %s", url)

        try:
            response = self._session.get(url, timeout=self._settings.timeout)
        except requests.Timeout as exc:
            raise SmartRoomTimeoutError(
                f"The room controller did not respond within {self._settings.timeout}s."
            ) from exc
        except requests.ConnectionError as exc:
            raise SmartRoomUnavailableError(
                "I cannot reach the room controller right now."
            ) from exc
        except requests.RequestException as exc:
            raise SmartRoomUnavailableError(
                f"The room controller request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise SmartRoomResponseError(
                f"The room controller returned HTTP {response.status_code} for /status."
            )

        try:
            body: Any = response.json()
        except ValueError as exc:
            raise SmartRoomResponseError(
                "The room controller returned a non-JSON response."
            ) from exc

        if not isinstance(body, Mapping):
            raise SmartRoomResponseError(
                "The room controller returned an unexpected /status response shape."
            )

        try:
            return RoomStatus.from_payload(
                body,
                controller_id=self._settings.controller_id,
            )
        except ValueError as exc:
            raise SmartRoomResponseError(str(exc)) from exc


__all__ = ["ESP32RoomClient", "SmartRoomSettings"]
