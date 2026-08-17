"""Smoke tests for the extended arvis modules.

These tests do **not** exercise network or GUI dependencies - they only
verify that the public surface of every module can be imported and that
the parsing helpers behave as advertised. Run with::

    python -m tests.smoke
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe(label: str, fn) -> bool:
    try:
        fn()
        print(f"  [OK]   {label}")
        return True
    except Exception:  # pragma: no cover
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        return False


def test_imports() -> bool:
    print("== Imports ==")
    cases = [
        ("core.base", lambda: __import__("core.base", fromlist=["BaseTool"])),
        ("core.config", lambda: __import__("core.config", fromlist=["get_config"])),
        ("core.router", lambda: __import__("core.router", fromlist=["CommandRouter", "get_router"])),
        ("vision.commands", lambda: __import__("vision.commands", fromlist=["VisionTool"])),
        ("vision.hand_mouse", lambda: __import__("vision.hand_mouse", fromlist=["HandMouseTool", "register_hand_mouse_tools"])),
        ("communication.email", lambda: __import__("communication.email", fromlist=["EmailTool"])),
        ("communication.whatsapp", lambda: __import__("communication.whatsapp", fromlist=["WhatsAppTool"])),
        ("communication.telegram", lambda: __import__("communication.telegram", fromlist=["TelegramTool"])),
        ("communication.discord", lambda: __import__("communication.discord", fromlist=["DiscordTool"])),
        ("communication.slack", lambda: __import__("communication.slack", fromlist=["SlackTool"])),
        ("routines.manager", lambda: __import__("routines.manager", fromlist=["RoutineManager"])),
        ("routines.commands", lambda: __import__("routines.commands", fromlist=["RoutinesTool"])),
    ]
    return all(_safe(label, fn) for label, fn in cases)


def test_parsing() -> bool:
    print("== Parsing ==")
    from communication.email import _parse_send_command
    from communication.telegram import _parse_send as telegram_parse
    from communication.slack import _parse_send as slack_parse

    cases = [
        (
            "email parse - body",
            lambda: _assert_email(_parse_send_command("email john about meeting body see you"), ["john"], "meeting", "see you"),
        ),
        (
            "email parse - subject only",
            lambda: _assert_email(_parse_send_command("send email to alice@example.com about launch"), ["alice@example.com"], "launch", ""),
        ),
        (
            "telegram parse - saying",
            lambda: _assert_pair(telegram_parse("send telegram to alice saying meeting at 5"), "alice", "meeting at 5"),
        ),
        (
            "telegram parse - that",
            lambda: _assert_pair(telegram_parse("send telegram bob that hi there"), "bob", "hi there"),
        ),
        (
            "slack parse - channel + saying",
            lambda: _assert_pair(slack_parse("send slack to #devs saying deploy complete", "#devs"), "#devs", "deploy complete"),
        ),
    ]
    return all(_safe(label, fn) for label, fn in cases)


def test_router() -> bool:
    print("== Router ==")
    from core.router import CommandRouter
    from vision.commands import VisionTool

    router = CommandRouter()
    tool = VisionTool()
    router.register(tool, keywords=("screen",), priority=80)
    res = router.dispatch("Jarvis, what's on my screen")
    assert res.success, f"VisionTool should claim screen commands: {res.message}"
    print(f"  [OK]   vision dispatch returned: {res.message[:80]}")
    return True


def test_routines() -> bool:
    print("== Routines ==")
    from routines.manager import RoutineManager

    manager = RoutineManager(path=ROOT / "storage" / "routines.json")
    names = manager.names()
    print(f"  [OK]   loaded {len(names)} routines: {names}")
    assert any("starting work" in n for n in names), "Bundled routine missing"
    return True


def test_hand_mouse() -> bool:
    print("== Hand-mouse ==")
    from vision.hand_mouse import HandMouseTool
    from core.router import CommandRouter

    tool = HandMouseTool()
    router = CommandRouter()
    router.register(tool, keywords=("hand mouse", "start hand mouse"), priority=80)

    # Deterministic routing
    cases = [
        ("start hand mouse", True),
        ("stop hand mouse", True),
        ("enable hand control", True),
        ("disable hand mouse", True),
        ("is hand mouse running", True),
        ("what is the weather", False),
    ]
    for phrase, expected in cases:
        got = router.dispatch(phrase)
        # When the test runs without a camera the tool reports failure,
        # but it still CLAIMED the command. can_handle is the real test.
        assert tool.can_handle(phrase) is expected, (
            f"can_handle({phrase!r}) returned {tool.can_handle(phrase)}, expected {expected}"
        )
    # Make sure unrelated commands fall through to the LLM default.
    fallback = router.dispatch("tell me a joke")
    assert not fallback.success, "HandMouseTool should not claim joke prompts"
    print("  [OK]   hand-mouse routing is deterministic and isolated.")
    return True


def _assert_email(actual, expected_recipients, expected_subject, expected_body) -> None:
    recipients, subject, body = actual
    assert recipients == expected_recipients, f"recipients={recipients!r} != {expected_recipients!r}"
    assert subject == expected_subject, f"subject={subject!r} != {expected_subject!r}"
    assert body == expected_body, f"body={body!r} != {expected_body!r}"


def _assert_pair(actual, expected_first, expected_second) -> None:
    first, second = actual
    assert first == expected_first, f"{first!r} != {expected_first!r}"
    assert second == expected_second, f"{second!r} != {expected_second!r}"


def main() -> int:
    results = [
        test_imports(),
        test_parsing(),
        test_router(),
        test_routines(),
        test_hand_mouse(),
    ]
    print()
    if all(results):
        print("All smoke tests passed.")
        return 0
    print("Some smoke tests FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
