"""
Smart Room / IoT tests for the arvis assistant.

These tests are fully offline - they mock the HTTP layer so the ESP32
never has to be on the network. They follow the same hand-rolled style
as :mod:`tests.smoke`: a ``main()`` function that returns ``0`` when
every check passes and ``1`` otherwise.

Run with::

    python tests/test_smart_room.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# Tiny test harness
# ----------------------------------------------------------------------
def _safe(label: str, fn) -> bool:
    try:
        fn()
        print(f"  [OK]   {label}")
        return True
    except AssertionError as exc:
        print(f"  [FAIL] {label}: {exc}")
        return False
    except Exception:  # pragma: no cover
        print(f"  [FAIL] {label}: unexpected error")
        traceback.print_exc()
        return False


# ----------------------------------------------------------------------
# Helpers - build a client whose HTTP session is fully mocked.
# ----------------------------------------------------------------------
def _make_session(*, post_payload=None, post_status=200, get_payload=None, get_status=200,
                  post_side_effect=None, get_side_effect=None) -> MagicMock:
    """Return a MagicMock standing in for a ``requests.Session``."""
    session = MagicMock()
    if post_side_effect is not None:
        session.post.side_effect = post_side_effect
    else:
        post_resp = MagicMock()
        post_resp.status_code = post_status
        post_resp.json.return_value = post_payload if post_payload is not None else {}
        session.post.return_value = post_resp
    if get_side_effect is not None:
        session.get.side_effect = get_side_effect
    else:
        get_resp = MagicMock()
        get_resp.status_code = get_status
        get_resp.json.return_value = get_payload if get_payload is not None else {}
        session.get.return_value = get_resp
    return session


def _client_with(session: MagicMock, *, ip: str = "192.168.1.50"):
    """Build a SmartRoomTool wired to *session*."""
    from smart_room.client import ESP32RoomClient, SmartRoomSettings
    from smart_room.commands import SmartRoomTool

    return SmartRoomTool(client=ESP32RoomClient(SmartRoomSettings(ip=ip), session=session))


# ----------------------------------------------------------------------
# 1. Command validation (no I/O at all)
# ----------------------------------------------------------------------
def test_validation() -> bool:
    print("== Validation ==")
    from smart_room.devices import validate_command, ALLOWED_DEVICES, ALLOWED_ACTIONS
    from smart_room.exceptions import SmartRoomInvalidCommandError

    cases = []

    # (a) every (device, action) pair from the whitelist round-trips
    for device in ALLOWED_DEVICES:
        for action in ALLOWED_ACTIONS:
            d, a = validate_command(device, action)
            cases.append(
                (f"valid {device!r}/{action!r}",
                 lambda d=d, a=a, dv=device, av=action: _assert_pair(
                     validate_command(d, a), dv, av,
                 ))
            )

    # (b) case-insensitive normalisation
    cases.append((
        "case-insensitive",
        lambda: _assert_pair(validate_command("Light", "ON"), "light", "on"),
    ))

    # (c) arbitrary device rejected
    cases.append((
        "rejects unknown device",
        lambda: _assert_rejected(validate_command, ["fan", "rgb_light", "lamp", "GPIO5"],
                                  "on"),
    ))

    # (d) arbitrary action rejected
    cases.append((
        "rejects unknown action",
        lambda: _assert_rejected(validate_command, ["light"],
                                  ["destroy", "reboot", "format", "rm -rf /", "execute"]),
    ))

    # (e) empty / whitespace
    for bad in ("", "   ", None):
        cases.append((
            f"rejects empty/whitespace ({bad!r})",
            (lambda b=bad: _assert_raises(
                lambda: validate_command(b, "on"),
                SmartRoomInvalidCommandError,
            )),
        ))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_pair(actual, expected_first, expected_second) -> None:
    first, second = actual
    assert first == expected_first, f"{first!r} != {expected_first!r}"
    assert second == expected_second, f"{second!r} != {expected_second!r}"


def _assert_rejected(fn, bad_first_options, second) -> None:
    from smart_room.exceptions import SmartRoomInvalidCommandError
    for bad in bad_first_options:
        try:
            fn(bad, second)
        except SmartRoomInvalidCommandError:
            continue
        raise AssertionError(f"expected rejection of {bad!r}, but it was accepted")


def _assert_rejected_actions(fn, first, bad_second_options) -> None:
    from smart_room.exceptions import SmartRoomInvalidCommandError
    for bad in bad_second_options:
        try:
            fn(first, bad)
        except SmartRoomInvalidCommandError:
            continue
        raise AssertionError(f"expected rejection of {bad!r}, but it was accepted")


def _assert_raises(fn, exc_type) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


# ----------------------------------------------------------------------
# 2. Status parser
# ----------------------------------------------------------------------
def test_status_parser() -> bool:
    print("== Status parser ==")
    from smart_room.status import RoomStatus

    cases = []

    cases.append((
        "good payload",
        lambda: _assert_status(
            RoomStatus.from_payload(
                {"device": "arvis-room-controller", "light": "on", "wifi": True, "ip": "192.168.1.50"},
                controller_id="fallback",
            ),
            controller_id="arvis-room-controller", light="on", wifi=True, ip="192.168.1.50",
        ),
    ))
    cases.append((
        "missing ip defaults to empty",
        lambda: _assert_status(
            RoomStatus.from_payload({"light": "off", "wifi": False}, controller_id="x"),
            controller_id="x", light="off", wifi=False, ip="",
        ),
    ))
    cases.append((
        "wifi as int truthy",
        lambda: _assert_status(
            RoomStatus.from_payload({"light": "off", "wifi": 1}, controller_id="x"),
            controller_id="x", light="off", wifi=True, ip="",
        ),
    ))
    cases.append((
        "wifi as string 'connected'",
        lambda: _assert_status(
            RoomStatus.from_payload({"light": "off", "wifi": "connected"}, controller_id="x"),
            controller_id="x", light="off", wifi=True, ip="",
        ),
    ))

    def _bad_light():
        try:
            RoomStatus.from_payload({"light": "banana", "wifi": True}, controller_id="x")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unexpected light state")

    cases.append(("rejects unexpected light state", _bad_light))

    def _not_mapping():
        try:
            RoomStatus.from_payload("a string, not a dict", controller_id="x")  # type: ignore[arg-type]
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-mapping payload")

    cases.append(("rejects non-mapping payload", _not_mapping))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_status(actual, *, controller_id, light, wifi, ip) -> None:
    assert actual.controller_id == controller_id, actual
    assert actual.light == light, actual
    assert actual.wifi == wifi, actual
    assert actual.ip == ip, actual
    # to_dict round-trip
    assert actual.to_dict() == {
        "controller_id": controller_id, "light": light, "wifi": wifi, "ip": ip,
    }


# ----------------------------------------------------------------------
# 3. Connection failures
# ----------------------------------------------------------------------
def test_connection_failures() -> bool:
    print("== Connection failures ==")
    from smart_room.exceptions import (
        SmartRoomUnavailableError,
        SmartRoomTimeoutError,
        SmartRoomNotConfiguredError,
        SmartRoomResponseError,
    )

    cases = []

    # (a) connection refused
    session = _make_session(post_side_effect=requests.ConnectionError("refused"))
    tool = _client_with(session)
    res = tool.control_room_device(device="light", action="on")
    cases.append((
        "connection refused -> friendly message",
        lambda r=res: _assert_tool_failure(r, "I can't reach the room controller right now."),
    ))

    # (b) timeout
    session = _make_session(post_side_effect=requests.Timeout("slow"))
    tool = _client_with(session)
    res = tool.control_room_device(device="light", action="on")
    cases.append((
        "timeout -> friendly message",
        lambda r=res: _assert_tool_failure(r, "I can't reach the room controller right now."),
    ))

    # (c) HTTP 500 from controller
    session = _make_session(post_payload={"error": "boom"}, post_status=500)
    tool = _client_with(session)
    res = tool.control_room_device(device="light", action="on")
    cases.append((
        "HTTP 500 -> friendly message",
        lambda r=res: _assert_tool_failure(r, "The room controller rejected the command, sir."),
    ))

    # (d) controller returned success=false
    session = _make_session(post_payload={"success": False, "error": "unknown device"})
    tool = _client_with(session)
    res = tool.control_room_device(device="light", action="on")
    cases.append((
        "controller success=false -> friendly message",
        lambda r=res: _assert_tool_failure(r, "The room controller rejected the command, sir."),
    ))

    # (e) unconfigured (no IP)
    session = _make_session()
    tool = _client_with(session, ip="")
    res = tool.control_room_device(device="light", action="on")
    cases.append((
        "no IP configured -> friendly message",
        lambda r=res: _assert_tool_failure(r, "The room controller is not configured yet, sir."),
    ))

    # (f) status: controller not reachable
    session = _make_session(get_side_effect=requests.ConnectionError())
    tool = _client_with(session)
    res = tool.get_room_status()
    cases.append((
        "status when controller offline -> friendly message",
        lambda r=res: _assert_tool_failure(r, "I can't reach the room controller right now."),
    ))

    # (g) status: malformed JSON
    bad_json = MagicMock()
    bad_json.status_code = 200
    bad_json.json.side_effect = ValueError("no json here")
    session = MagicMock()
    session.get.return_value = bad_json
    tool = _client_with(session)
    res = tool.get_room_status()
    cases.append((
        "status malformed JSON -> friendly message",
        lambda r=res: _assert_tool_failure(r, "The room controller sent an unexpected response, sir."),
    ))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_tool_failure(result, expected_substring: str) -> None:
    assert not result.success, f"expected failure, got success: {result.message!r}"
    assert expected_substring in result.message, (
        f"expected {expected_substring!r} in message, got {result.message!r}"
    )


# ----------------------------------------------------------------------
# 4. Successful light commands + status
# ----------------------------------------------------------------------
def test_successful_commands() -> bool:
    print("== Successful commands ==")
    cases = []

    for action, expected_state, expected_message in (
        ("on", "on", "The room light is on."),
        ("off", "off", "The room light is off."),
        ("toggle", "off", "The room light is now off."),
    ):
        session = _make_session(
            post_payload={"success": True, "device": "light", "state": expected_state},
            post_status=200,
        )
        tool = _client_with(session)
        res = tool.control_room_device(device="light", action=action)
        cases.append((
            f"control_room_device(light, {action})",
            lambda r=res, em=expected_message, a=action: _assert_tool_success(
                r, em, device="light", action=a,
            ),
        ))

    # Deterministic voice path
    session = _make_session(
        post_payload={"success": True, "device": "light", "state": "on"},
    )
    tool = _client_with(session)
    for phrase, action in (
        ("turn on my room light", "on"),
        ("turn off the room light", "off"),
        ("toggle my room light", "toggle"),
    ):
        res = tool.execute(phrase)
        cases.append((
            f"voice: {phrase!r}",
            lambda r=res, a=action: _assert_tool_success(
                r, "room light", device="light", action=a,
            ),
        ))

    # Status
    session = _make_session(
        get_payload={
            "device": "arvis-room-controller",
            "light": "on",
            "wifi": True,
            "ip": "192.168.1.50",
        },
    )
    tool = _client_with(session)
    res = tool.get_room_status()
    cases.append((
        "get_room_status (online)",
        lambda r=res: _assert_status_success(r, controller_id="arvis-room-controller", light="on", wifi=True, ip="192.168.1.50"),
    ))

    # Status when Wi-Fi is down
    session = _make_session(
        get_payload={"device": "arvis-room-controller", "light": "off", "wifi": False, "ip": ""},
    )
    tool = _client_with(session)
    res = tool.get_room_status()
    cases.append((
        "get_room_status (wifi down)",
        lambda r=res: _assert_status_success(
            r, controller_id="arvis-room-controller", light="off", wifi=False, ip="",
        ) and _assert_contains(r.message, "Wi-Fi link is down"),
    ))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_tool_success(result, expected_message_substring: str, *, device, action) -> None:
    assert result.success, f"expected success, got failure: {result.message!r}"
    assert expected_message_substring in result.message, (
        f"expected {expected_message_substring!r} in {result.message!r}"
    )
    assert result.data["device"] == device, result.data
    assert result.data["action"] == action, result.data


def _assert_status_success(result, *, controller_id, light, wifi, ip) -> None:
    assert result.success, f"expected success, got failure: {result.message!r}"
    data = result.data
    assert data["controller_id"] == controller_id, data
    assert data["light"] == light, data
    assert data["wifi"] == wifi, data
    assert data["ip"] == ip, data


# ----------------------------------------------------------------------
# 5. Whitelist enforcement at the client layer
# ----------------------------------------------------------------------
def test_client_rejects_unwhitelisted() -> bool:
    print("== Client-side rejection ==")
    session = _make_session()
    tool = _client_with(session)

    cases = []
    for bad_device in ("fan", "lamp", "GPIO5"):
        cases.append((
            f"rejects device {bad_device!r}",
            lambda d=bad_device: _assert_tool_failure(
                tool.control_room_device(device=d, action="on"),
                "Unknown device",
            ),
        ))
    for bad_action in ("reboot", "destroy", "rm -rf /"):
        cases.append((
            f"rejects action {bad_action!r}",
            lambda a=bad_action: _assert_tool_failure(
                tool.control_room_device(device="light", action=a),
                "Unknown action",
            ),
        ))

    # Also: the client must NOT have hit the network at all - validation
    # happens before the socket is opened.
    cases.append((
        "no HTTP traffic on validation failure",
        lambda s=session: _assert_equals(s.post.call_count, 0),
    ))

    return all(_safe(label, fn) for label, fn in cases)


# ----------------------------------------------------------------------
# 6. URL surface audit (no arbitrary URLs)
# ----------------------------------------------------------------------
def test_url_surface() -> bool:
    print("== URL surface ==")
    from smart_room.client import ESP32RoomClient, SmartRoomSettings

    cases = []

    # Command
    cmd_session = _make_session(post_payload={"success": True, "state": "on"})
    c = ESP32RoomClient(SmartRoomSettings(ip="10.20.30.40", port=8080), session=cmd_session)
    c.send_command("light", "on")
    cases.append((
        "command URL only hits /command",
        lambda s=cmd_session: _assert_equals(
            s.post.call_args[0][0], "http://10.20.30.40:8080/command",
        ),
    ))
    cases.append((
        "command payload contains device + action only",
        lambda s=cmd_session: _assert_equals(
            s.post.call_args.kwargs["json"], {"device": "light", "action": "on"},
        ),
    ))
    cases.append((
        "exactly one POST issued for command",
        lambda s=cmd_session: _assert_equals(s.post.call_count, 1),
    ))
    cases.append((
        "no GET issued during a command",
        lambda s=cmd_session: _assert_equals(s.get.call_count, 0),
    ))

    # Status
    status_session = _make_session(get_payload={"light": "off", "wifi": True, "ip": "10.20.30.40"})
    c = ESP32RoomClient(SmartRoomSettings(ip="10.20.30.40", port=8080), session=status_session)
    c.get_status()
    cases.append((
        "status URL only hits /status",
        lambda s=status_session: _assert_equals(
            s.get.call_args[0][0], "http://10.20.30.40:8080/status",
        ),
    ))
    cases.append((
        "exactly one GET issued for status",
        lambda s=status_session: _assert_equals(s.get.call_count, 1),
    ))
    cases.append((
        "no POST issued during a status call",
        lambda s=status_session: _assert_equals(s.post.call_count, 0),
    ))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_equals(actual, expected) -> None:
    assert actual == expected, f"{actual!r} != {expected!r}"


# ----------------------------------------------------------------------
# 7. LangChain tool exposure
# ----------------------------------------------------------------------
def test_langchain_tools() -> bool:
    print("== LangChain tools ==")
    cases = []

    session = _make_session(
        post_payload={"success": True, "state": "on"},
        get_payload={"light": "off", "wifi": True, "ip": "192.168.1.50"},
    )
    tool = _client_with(session)

    # Import the private adapters via the public registration helper so
    # the test stays decoupled from internal naming.
    from smart_room.commands import _ControlRoomDeviceTool, _GetRoomStatusTool

    control = _ControlRoomDeviceTool(tool)
    status = _GetRoomStatusTool(tool)

    cases.append((
        "control tool name is control_room_device",
        lambda: _assert_equals(control.name, "control_room_device"),
    ))
    cases.append((
        "status tool name is get_room_status",
        lambda: _assert_equals(status.name, "get_room_status"),
    ))

    lc_control = control.as_langchain_tool()
    lc_status = status.as_langchain_tool()
    out = lc_control.invoke('{"device":"light","action":"on"}')
    cases.append((
        "control_room_device runs through LangChain",
        lambda o=out: _assert_contains(o, "room light is on"),
    ))
    out = lc_status.invoke("")
    cases.append((
        "get_room_status runs through LangChain",
        lambda o=out: _assert_contains(o, "room light is off"),
    ))

    return all(_safe(label, fn) for label, fn in cases)


def _assert_contains(haystack, needle) -> None:
    assert needle in haystack, f"{needle!r} not in {haystack!r}"


# ----------------------------------------------------------------------
# 8. Router integration
# ----------------------------------------------------------------------
def test_router_integration() -> bool:
    print("== Router integration ==")
    import os
    from core.router import CommandRouter
    from smart_room import register_smart_room_tools

    cases = []
    session = _make_session(
        post_payload={"success": True, "state": "on"},
        get_payload={"light": "off", "wifi": True, "ip": "192.168.1.50"},
    )

    # ``register_smart_room_tools`` instantiates ``SmartRoomTool()`` with
    # no args, so we point it at our session by setting the env var it
    # reads from. The ``Config`` dataclass is frozen, but its default
    # factory reads the env at construction time, so a freshly imported
    # ``Config`` picks up our override.
    old_ip = os.environ.get("ESP32_ROOM_IP")
    os.environ["ESP32_ROOM_IP"] = "192.168.1.50"
    try:
        # Import the client module and inject the mock session into the
        # ``SmartRoomSettings`` that ``from_config`` will return. The
        # simplest way to do this without monkey-patching is to build the
        # router manually - we call the registration helper from inside a
        # wrapper that pre-creates the tool.
        from smart_room.client import ESP32RoomClient, SmartRoomSettings
        from smart_room.commands import SmartRoomTool as _RealTool

        router = CommandRouter()

        # Pre-build the tool with the mocked session, then add the AI
        # tool adapters by hand using the same private classes the
        # registration helper uses. This avoids the module-level
        # ``SmartRoomTool()`` call entirely.
        room = _RealTool(client=ESP32RoomClient(
            SmartRoomSettings(ip="192.168.1.50"), session=session,
        ))

        from smart_room.commands import _ControlRoomDeviceTool, _GetRoomStatusTool
        control = _ControlRoomDeviceTool(room)
        status = _GetRoomStatusTool(room)

        router.register(
            room,
            keywords=(
                "room light", "turn on my room", "turn off my room",
                "toggle my room", "room status",
                "is my room light", "is the room light",
            ),
            priority=80,
        )
        router.register(control, keywords=(), priority=10)
        router.register(status, keywords=(), priority=10)

        cases.append((
            "registered 3 smart_room tools",
            lambda r=router: _assert_equals(
                sorted(t.name for t in r.tools() if t.name in {
                    "smart_room_tool", "control_room_device", "get_room_status",
                }),
                ["control_room_device", "get_room_status", "smart_room_tool"],
            ),
        ))

        res = router.dispatch("turn on my room light")
        cases.append((
            "router dispatches 'turn on my room light' to smart_room_tool",
            lambda r=res: _assert_tool_success(
                r, "The room light is on", device="light", action="on",
            ),
        ))

        res = router.dispatch("what is the status of my room?")
        cases.append((
            "router dispatches 'what is the status of my room?' to smart_room_tool",
            lambda r=res: _assert_contains(r.message, "room light is off"),
        ))

        # Unrelated commands must not be claimed by the smart-room tool.
        # The router's default is ``None`` here, so dispatch returns a
        # failure result. We just want to make sure it is *not* a
        # smart-room-style friendly sentence.
        res = router.dispatch("what is the meaning of life?")
        cases.append((
            "unrelated command is not claimed by smart_room_tool",
            lambda r=res: _assert_not_claimed(r),
        ))
    finally:
        if old_ip is None:
            os.environ.pop("ESP32_ROOM_IP", None)
        else:
            os.environ["ESP32_ROOM_IP"] = old_ip

    return all(_safe(label, fn) for label, fn in cases)


def _assert_not_claimed(result) -> None:
    # ``dispatch`` returns either a ToolResult (success=False) when no tool
    # claimed the command OR the default's ToolResult. Either way the
    # smart_room tools should not have produced a friendly room message.
    bad = any(
        phrase in (result.message or "")
        for phrase in ("room light is on", "room light is off", "I can't reach the room controller", "room controller is not configured")
    )
    assert not bad, f"smart_room tool wrongly claimed unrelated command: {result.message!r}"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    results = [
        test_validation(),
        test_status_parser(),
        test_connection_failures(),
        test_successful_commands(),
        test_client_rejects_unwhitelisted(),
        test_url_surface(),
        test_langchain_tools(),
        test_router_integration(),
    ]
    print()
    if all(results):
        print("All smart-room tests passed.")
        return 0
    print("Some smart-room tests FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
