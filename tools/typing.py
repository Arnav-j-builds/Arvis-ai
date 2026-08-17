"""
tools.typing
~~~~~~~~~~~~

A :class:`BaseTool` that types text into whatever application is currently
focused. The user can say things like:

* "type hello world"
* "type this: hello, how are you?"
* "write the quick brown fox"
* "type this into notepad: ..."
* "paste from clipboard"
* "press enter" / "press tab" / "press escape"

The tool uses the Windows clipboard + ``Ctrl+V`` rather than
``pyautogui.write`` for two reasons:

1. ``pyautogui.write`` sends characters one at a time and chokes on
   Unicode (em-dashes, accented letters, emoji) - arvis should be able
   to type *anything*.
2. Pasting is dramatically faster - a 5,000-character paragraph is one
   keystroke rather than 5,000.

Special key handling (``enter``, ``tab``, ``escape``, ...) goes through
``pyautogui.press`` so it works the same on every focused window.

The module never raises: every error becomes a friendly :class:`ToolResult`.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency probing
# ---------------------------------------------------------------------------
def _try_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # pragma: no cover - depends on env
        log.debug("Optional import %s unavailable: %s", name, exc)
        return None


pyautogui = _try_import("pyautogui")
pyperclip = _try_import("pyperclip")

HAS_PYAUTOGUI = pyautogui is not None
HAS_PYPERCLIP = pyperclip is not None

if HAS_PYAUTOGUI:  # pragma: no cover - depends on env
    try:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------
_TYPE_PREFIXES = (
    "type ",
    "type this ",
    "type this: ",
    "type the following: ",
    "type the following ",
    "type this text: ",
    "type this text ",
    "write ",
    "write this ",
    "write this: ",
    "write the following ",
    "write the following: ",
    "type out ",
    "type out: ",
    "enter ",
    "enter this ",
    "enter this: ",
    "input ",
    "input this ",
    "input the following: ",
    "input the following ",
)

_PASTE_PHRASES = (
    "paste from clipboard",
    "paste clipboard",
    "paste the clipboard",
    "paste it",
    "paste now",
    "just paste",
    "press control v",
    "press ctrl v",
    "press control+v",
    "press ctrl+v",
)

_PRESS_KEY_PHRASES = (
    "press enter",
    "hit enter",
    "press return",
    "press tab",
    "hit tab",
    "press escape",
    "hit escape",
    "press esc",
    "press backspace",
    "press delete",
    "press space",
    "press spacebar",
    "press up",
    "press down",
    "press left",
    "press right",
    "press home",
    "press end",
    "press page up",
    "press page down",
    "press shift enter",
    "press ctrl enter",
    "press alt enter",
    "press control enter",
    "press alt tab",
    "press windows",
    "press super",
)

_CLEAR_PHRASES = (
    "clear text",
    "clear the text",
    "erase all",
    "erase everything",
    "select all",
)


# Map friendly key names -> pyautogui key names. ``pyautogui`` accepts the
# same names (``enter``, ``tab``, ``escape``, ``backspace``, ``delete``,
# ``space``, ...) so this table is mostly an explicit allow-list.
_KEY_ALIASES = {
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "escape": "escape",
    "esc": "escape",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "space": "space",
    "spacebar": "space",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "home": "home",
    "end": "end",
    "pageup": "pageup",
    "page_up": "pageup",
    "pagedown": "pagedown",
    "page_down": "pagedown",
}


def _contains_any(text: str, needles: tuple) -> bool:
    lowered = (text or "").lower()
    return any(needle in lowered for needle in needles)


# ---------------------------------------------------------------------------
# Type-text helpers
# ---------------------------------------------------------------------------
# Strip a leading quote from the user input so "type \"hello\"" -> "hello".
_LEADING_QUOTES = ('"', "“", "‘", "'")


def _strip_prefix(command: str) -> str:
    """Strip a known "type ..." prefix from *command*."""
    text = (command or "").strip()
    lower = text.lower()
    # Match the longest prefix first so "type this text: " wins over "type ".
    for prefix in sorted(_TYPE_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _strip_quotes(text: str) -> str:
    """Strip a single pair of wrapping quotes if present."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _LEADING_QUOTES:
        return text[1:-1]
    return text


def _split_segments(command: str) -> List[str]:
    """Split a phrase like ``"press enter then type hello"`` into the
    individual commands the tool understands. Each segment is then
    matched against :data:`_TYPE_PREFIXES` / :data:`_PASTE_PHRASES` /
    :data:`_PRESS_KEY_PHRASES` / :data:`_CLEAR_PHRASES`.

    This lets the user chain actions in a single utterance, e.g.
    ``"press ctrl a then type hello world"``.
    """
    # Split on common conjunctions.
    raw = re.split(r"\b(?:then|and then|after that|afterwards|, and)\b", command, flags=re.IGNORECASE)
    return [seg.strip() for seg in raw if seg.strip()]


def _paste_text(text: str) -> Tuple[bool, str]:
    """Put *text* on the clipboard, then send Ctrl+V to the focused window."""
    if not HAS_PYPERCLIP:
        return False, (
            "I cannot type that, sir - the ``pyperclip`` library is not "
            "installed. Run ``pip install pyperclip`` to enable typing."
        )
    if not HAS_PYAUTOGUI:
        return False, (
            "I cannot type that, sir - the ``pyautogui`` library is not "
            "installed. Run ``pip install pyautogui`` to enable typing."
        )
    try:
        pyperclip.copy(text)
    except Exception as exc:
        return False, f"I could not put the text on the clipboard, sir: {exc}"
    # Tiny pause so the focused app notices the clipboard change before
    # we send the keystroke.
    time.sleep(0.05)
    try:
        # ``pyautogui.hotkey`` sends the chord atomically.
        pyautogui.hotkey("ctrl", "v")
    except Exception as exc:
        return False, f"I could not send Ctrl+V to the focused window, sir: {exc}"
    return True, ""


def _press_key(name: str) -> Tuple[bool, str]:
    """Send a single named key (or hotkey chord) to the focused window."""
    if not HAS_PYAUTOGUI:
        return False, (
            "I cannot press keys, sir - the ``pyautogui`` library is not "
            "installed. Run ``pip install pyautogui`` to enable typing."
        )
    keys = [k.strip() for k in name.lower().split() if k.strip()]
    keys = [_KEY_ALIASES.get(k, k) for k in keys]
    try:
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
    except Exception as exc:
        return False, f"I could not press {name!r}, sir: {exc}"
    return True, ""


def _select_all() -> Tuple[bool, str]:
    if not HAS_PYAUTOGUI:
        return False, "pyautogui is not installed."
    try:
        pyautogui.hotkey("ctrl", "a")
    except Exception as exc:
        return False, f"I could not select all, sir: {exc}"
    return True, ""


def _clear_text() -> Tuple[bool, str]:
    """Select all and press Delete - works in notepad, browsers, IDEs."""
    ok1, msg1 = _select_all()
    if not ok1:
        return False, msg1
    ok2, msg2 = _press_key("delete")
    if not ok2:
        return False, msg2
    return True, ""


def _paste_clipboard() -> Tuple[bool, str]:
    """Send Ctrl+V without touching the clipboard contents."""
    if not HAS_PYAUTOGUI:
        return False, "pyautogui is not installed."
    try:
        pyautogui.hotkey("ctrl", "v")
    except Exception as exc:
        return False, f"I could not send Ctrl+V, sir: {exc}"
    return True, ""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------
class TypingTool(BaseTool):
    """Voice entry point for typing text into the focused window."""

    name = "typing_tool"
    description = (
        "Type text into the currently focused window. Use 'type hello "
        "world' to write a string, 'type this: <text>' to be explicit "
        "about what to type, 'press enter' / 'press tab' / 'press "
        "escape' to send a key, 'paste from clipboard' to send Ctrl+V "
        "with the existing clipboard contents, and 'clear text' to "
        "select-all and delete. The tool uses the clipboard + Ctrl+V "
        "so it handles Unicode and is fast for long paragraphs."
    )

    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        if _contains_any(text, _TYPE_PREFIXES):
            return True
        if _contains_any(text, _PASTE_PHRASES):
            return True
        if _contains_any(text, _PRESS_KEY_PHRASES):
            return True
        if _contains_any(text, _CLEAR_PHRASES):
            return True
        return False

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        raw = (command or "").strip()
        try:
            segments = _split_segments(raw)
            if not segments:
                return ToolResult(
                    success=False,
                    message="I did not catch anything to type, sir.",
                )

            # Walk the segments left-to-right so the user can chain actions
            # like "press ctrl a then type hello world".
            for seg in segments:
                seg_low = seg.lower()

                if _contains_any(seg_low, _CLEAR_PHRASES):
                    ok, msg = _clear_text()
                    if not ok:
                        return ToolResult(success=False, message=msg)
                    continue

                if _contains_any(seg_low, _PASTE_PHRASES):
                    ok, msg = _paste_clipboard()
                    if not ok:
                        return ToolResult(success=False, message=msg)
                    continue

                if _contains_any(seg_low, _PRESS_KEY_PHRASES):
                    # Extract the key name(s) from the segment.
                    name = seg_low
                    for phrase in _PRESS_KEY_PHRASES:
                        if phrase in name:
                            name = name.replace(phrase, "").strip()
                            break
                    # Default: if user said "press enter" we want to
                    # press "enter". Map bare phrases to their canonical
                    # key so we don't lose the verb.
                    if not name:
                        for phrase in _PRESS_KEY_PHRASES:
                            if phrase in seg_low:
                                # ``press enter`` -> "enter"
                                name = phrase.replace("press ", "").replace("hit ", "")
                                break
                    ok, msg = _press_key(name or "enter")
                    if not ok:
                        return ToolResult(success=False, message=msg)
                    continue

                if _contains_any(seg_low, _TYPE_PREFIXES):
                    payload = _strip_quotes(_strip_prefix(seg))
                    if not payload:
                        return ToolResult(
                            success=False,
                            message="What should I type, sir?",
                        )
                    ok, msg = _paste_text(payload)
                    if not ok:
                        return ToolResult(success=False, message=msg)
                    continue

                # Fallback - the user said "type" or "write" without a
                # recognised prefix but the segment looks like text. Be
                # lenient: just paste it.
                if seg.strip():
                    ok, msg = _paste_text(seg)
                    if not ok:
                        return ToolResult(success=False, message=msg)
                    continue

            return ToolResult(
                success=True,
                message=f"Typed {len(segments)} segment(s), sir.",
                data={"segments": len(segments)},
            )
        except Exception as exc:
            log.exception("TypingTool failed: %s", exc)
            return ToolResult(success=False, message=f"Typing failed: {exc}")

    # ------------------------------------------------------------------
    # LangChain glue
    # ------------------------------------------------------------------
    def as_langchain_tool(self):
        """Convert this tool to a LangChain tool.

        Exposes the tool with a strict schema so the LLM can call it
        with ``{"text": "hello world"}`` rather than a free-form string.
        """
        from langchain.tools import Tool

        def _run(payload: str) -> str:
            # ``payload`` may be a JSON string from a tool-calling agent.
            text = payload
            try:
                import json

                parsed = json.loads(payload)
                if isinstance(parsed, dict) and "text" in parsed:
                    text = str(parsed["text"])
                elif isinstance(parsed, str):
                    text = parsed
            except Exception:
                # Not JSON - treat the payload as the raw command.
                pass
            result = self.safe_execute(f"type {text}" if text else "")
            return result.message

        return Tool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
        )


def register_typing_tool(router) -> list:
    """Register the typing tool with *router* and return the new tools."""
    tool = TypingTool()
    router.register(
        tool,
        keywords=(
            "type ",
            "type this",
            "write ",
            "write this",
            "enter this",
            "input ",
            "press enter",
            "press tab",
            "press escape",
            "press esc",
            "press backspace",
            "press delete",
            "press space",
            "press up",
            "press down",
            "press left",
            "press right",
            "press home",
            "press end",
            "press page up",
            "press page down",
            "press shift enter",
            "press ctrl enter",
            "press alt enter",
            "press alt tab",
            "paste from clipboard",
            "paste clipboard",
            "clear text",
            "select all",
        ),
        priority=85,
    )
    return [tool]


__all__ = [
    "TypingTool",
    "register_typing_tool",
    "HAS_PYAUTOGUI",
    "HAS_PYPERCLIP",
]
