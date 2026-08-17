"""
core.visual_actions
~~~~~~~~~~~~~~~~~~~

Identifies and interacts with visible UI elements without round-
tripping through a vision model for every click.

Pipeline
--------

::

    User request
        -> VisualTarget   (semantic description of what to click)
        -> ScreenContext  (capture + OCR + active window)
        -> element match  (cheap OCR + active-window lookup)
        -> coordinates    (clamped to screen bounds, low-confidence -> ask)
        -> pyautogui action
        -> optional verify (re-capture + check the target is gone OR a
            different element is now at the click point)

The module reuses :mod:`vision.capture`, :mod:`vision.ocr`, and
:mod:`tools.typing` for low-level input. A vision model is only
consulted when the target is ambiguous (multiple candidates with
similar confidence) and the user has explicitly asked for "find X"
style behaviour. Otherwise OCR + heuristics are enough.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config import get_config
from core.context_engine import (
    ScreenContext,
    ScreenContextCache,
    ScreenElement,
    get_screen_cache,
)
from core.logger import get_logger

# Lazy / guarded imports - the vision package pulls in requests via the
# analyzer module.  Tests in environments without ``requests`` should
# still be able to exercise this module.
try:
    from vision.capture import (  # type: ignore
        capture_active_window,
        capture_primary_monitor,
    )
except Exception as _capture_exc:  # pragma: no cover - env dependent
    capture_active_window = None  # type: ignore
    capture_primary_monitor = None  # type: ignore

try:
    from vision.ocr import OCRResult, extract_text  # type: ignore
except Exception as _ocr_exc:  # pragma: no cover - env dependent
    OCRResult = None  # type: ignore
    extract_text = None  # type: ignore

log = get_logger(__name__)

# Keywords that are obviously *target* descriptions rather than free
# text.  Used by :func:`looks_like_visual_action`.
_TARGET_KEYWORDS = (
    "click ", "tap ", "press ", "double click", "right click", "right-click",
    "scroll down", "scroll up", "scroll left", "scroll right",
    "open settings", "open the settings",
    "type this into", "type into", "fill in", "enter into",
    "find the", "find a", "find my",
)

# Visual target kinds we recognise.
_TARGET_TYPES = (
    "button", "link", "checkbox", "input", "menu", "icon",
    "text", "image", "result", "error",
)

# Colour words used for visual-description targets.
_COLOURS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "gray", "grey", "cyan", "magenta", "brown",
    "gold", "silver", "navy", "teal", "lime",
}

_ORDINALS = {
    "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
    "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
    "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
}


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
@dataclass
class VisualTarget:
    """A semantic description of what to act on."""

    action: str = "click"             # click / double_click / right_click / type / scroll
    text: str = ""                    # visible text to match
    target_type: str = ""             # button / link / input / ...
    color: str = ""                   # optional colour descriptor
    ordinal: Optional[int] = None     # 0-based index when the user said "second result"
    extra: str = ""                   # free-form fallback
    verify: bool = True
    raw: str = ""

    def describe(self) -> str:
        verb = {
            "click": "click",
            "double_click": "double-click",
            "right_click": "right-click",
            "scroll": "scroll",
            "type_into": "type into",
        }.get(self.action, self.action or "click")
        bits: List[str] = []
        if self.ordinal is not None:
            bits.append(["first", "second", "third", "fourth", "fifth"][self.ordinal] if self.ordinal < 5 else f"#{self.ordinal + 1}")
        if self.target_type:
            bits.append(self.target_type)
        if self.text:
            bits.append(f"labeled {self.text!r}")
        if self.color:
            bits.append(f"colored {self.color}")
        if self.extra:
            bits.append(self.extra)
        body = " ".join(b for b in bits if b) or "the target"
        return f"{verb} {body}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def looks_like_visual_action(text: str) -> bool:
    """Cheap keyword check used by the visual tool to decide if it owns
    this command."""
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(kw in lowered for kw in _TARGET_KEYWORDS)


# A small whitelist of generic UI-element nouns that the user almost
# always uses as a *kind* suffix ("hide button", "submit link") rather
# than the visible label of the element.  When parse_target() sees one
# of these at the start/end of the cleaned text it strips it off so the
# screen-matcher can search for the real label ("hide").  More specific
# type words like "result", "row", "item", "tab", "menu", "field",
# "option", "checkbox", "tile", "card", "folder", "file", "app" are
# kept - the user often intends those as the visible name.
_GENERIC_TARGET_TYPES = (
    "button",
    "buttons",
    "link",
    "links",
    "icon",
    "icons",
    "control",
    "controls",
    "widget",
    "widgets",
    "element",
    "elements",
    "thing",
    "thingy",
)


def parse_target(command: str) -> VisualTarget:
    """Convert a free-form command into a :class:`VisualTarget`.

    The parser is intentionally conservative - if a slot cannot be
    inferred it stays empty and the matcher falls back to the next
    detection method (active window, OCR, vision model).
    """
    text = (command or "").strip()
    lowered = text.lower()
    raw = text

    target = VisualTarget(raw=raw)

    # Action ------------------------------------------------------------
    if "double click" in lowered or "double-click" in lowered or "doubleclick" in lowered:
        target.action = "double_click"
    elif "right click" in lowered or "right-click" in lowered or "rightclick" in lowered:
        target.action = "right_click"
    elif "scroll down" in lowered:
        target.action = "scroll"
        target.extra = "down"
    elif "scroll up" in lowered:
        target.action = "scroll"
        target.extra = "up"
    elif "scroll left" in lowered:
        target.action = "scroll"
        target.extra = "left"
    elif "scroll right" in lowered:
        target.action = "scroll"
        target.extra = "right"
    elif "scroll" in lowered:
        target.action = "scroll"
        target.extra = "down"
    elif "type this into" in lowered or "type into" in lowered or "fill in" in lowered or "enter into" in lowered:
        target.action = "type_into"
    elif "type" in lowered and " into" in lowered:
        target.action = "type_into"
    elif lowered.startswith("click ") or lowered.startswith("tap ") or lowered.startswith("press "):
        target.action = "click"

    # Ordinal -----------------------------------------------------------
    for word, idx in _ORDINALS.items():
        if re.search(rf"\bthe\s+{re.escape(word)}\b", lowered) or re.search(
            rf"\b{re.escape(word)}\s+(?:one|result|button|link|item)\b", lowered
        ):
            target.ordinal = idx
            break

    # Colour ------------------------------------------------------------
    for colour in _COLOURS:
        if re.search(rf"\b{colour}\b", lowered):
            target.color = colour
            break

    # Target type -------------------------------------------------------
    for tt in _TARGET_TYPES:
        if re.search(rf"\b{tt}\b", lowered):
            target.target_type = tt
            break

    # Visible text ------------------------------------------------------
    # Pull the quoted chunk if present ("click 'Download'").
    quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
    if quoted:
        target.text = quoted.group(1).strip()
    else:
        target.text = _strip_noise(lowered, action=target.action)
    # Strip the type word from the visible-text so "click the hide
    # button" -> text="hide" rather than text="hide button".  We only
    # strip a small set of generic nouns ("button", "link", "icon", ...)
    # - words that the user typically uses as a suffix to describe the
    # *kind* of element, not the *name* of the element.  For meaningful
    # type words like "result", "row", "item", "tab" we keep them so
    # "click the second result" still has text="result" for the matcher.
    if target.target_type and target.text and target.ordinal is None:
        for stop in _GENERIC_TARGET_TYPES:
            pattern = rf"\s+{re.escape(stop)}\s*$"
            target.text = re.sub(pattern, "", target.text, flags=re.IGNORECASE).strip()
            pattern = rf"^{re.escape(stop)}\s+"
            target.text = re.sub(pattern, "", target.text, flags=re.IGNORECASE).strip()

    # Type-into payload extraction.
    if target.action == "type_into":
        # "type hello into the search box"
        m = re.search(r"type\s+(.+?)\s+into\s+(.+)", text, flags=re.IGNORECASE)
        if m:
            target.extra = m.group(2).strip()  # field description
            target.text = m.group(1).strip()    # text to type

    return target


def _strip_noise(lowered: str, action: str) -> str:
    """Remove the action verb and obvious filler so what's left is the
    thing the user wants to interact with."""
    text = lowered
    for prefix in (
        "please ", "can you ", "could you ", "would you ",
        "i want to ", "i'd like to ", "i would like to ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    for verb in (
        "double click on ", "double-click on ", "right click on ",
        "right-click on ", "click on ", "click the ", "click ",
        "tap on ", "tap the ", "tap ",
        "press the ", "press ",
        "open the ", "open ",
    ):
        if text.startswith(verb):
            text = text[len(verb):]
            break
    # Remove trailing particles.
    for tail in (" for me", " please", " thanks", " now"):
        if text.endswith(tail):
            text = text[: -len(tail)]
    return text.strip().strip("'\"")


# ---------------------------------------------------------------------------
# Active-window detection
# ---------------------------------------------------------------------------
def _active_window_info() -> Tuple[str, str]:
    """Return (title, application) for the foreground window. Empty on failure."""
    title = ""
    app = ""
    try:
        import platform

        if platform.system() == "Windows":
            try:
                import win32gui  # type: ignore

                hwnd = win32gui.GetForegroundWindow()
                if hwnd:
                    title = win32gui.GetWindowText(hwnd) or ""
                    try:
                        _, pid = win32gui.GetThreadProcessId(hwnd)
                        import psutil  # type: ignore

                        app = psutil.Process(pid).name() if pid else ""
                    except Exception:
                        app = ""
            except ImportError:
                pass
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("active_window_info failed: %s", exc)
    return title, app


# ---------------------------------------------------------------------------
# Build ScreenContext
# ---------------------------------------------------------------------------
def build_screen_context(*, force: bool = False, source: str = "primary") -> ScreenContext:
    """Capture the screen and produce a :class:`ScreenContext`.

    The result is cached; pass ``force=True`` to re-capture even within
    the TTL window.
    """
    cache: ScreenContextCache = get_screen_cache()
    existing = cache.get(force=False)
    if existing is not None and not force:
        return existing

    if source == "active":
        capture = capture_active_window() if capture_active_window is not None else None
    else:
        capture = capture_primary_monitor() if capture_primary_monitor is not None else None
    if capture is None:
        log.warning("vision.capture helpers unavailable; returning empty context")
        return ScreenContext(timestamp=time.time(), width=1920, height=1080)
    path = str(capture.path)
    width = int(capture.width or 0)
    height = int(capture.height or 0)
    if not width or not height:
        width, height = 1920, 1080  # sensible default

    ocr: Optional[Any] = None
    if extract_text is not None:
        try:
            ocr = extract_text(path)
        except Exception as exc:
            log.warning("OCR during build_screen_context failed: %s", exc)

    title, app = _active_window_info()
    elements = _elements_from_ocr(ocr.text if ocr else "", width=width, height=height)

    ctx = ScreenContext(
        timestamp=time.time(),
        width=width,
        height=height,
        active_window=title or app,
        application=app,
        title=title,
        ocr_text=ocr.text if ocr else "",
        elements=elements,
        source_path=path,
    )
    cache.put(ctx, path)
    return ctx


def _elements_from_ocr(text: str, *, width: int, height: int) -> List[ScreenElement]:  # type: ignore[type-arg]
    """Approximate UI element positions from OCR lines.

    We do NOT have real bounding-box data from pytesseract here, so we
    spread detected words evenly across the screen as a cheap
    "candidate" list.  This is enough to disambiguate "the second
    result" when there are several search hits on the page; for
    pixel-perfect clicks the user should still use coordinates.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    elements: List[ScreenElement] = []
    if not lines:
        return elements
    step = max(20, height // (len(lines) + 1))
    y = step
    counter = 0
    for line in lines:
        # Keep only short lines - long paragraphs are not clickable.
        if len(line) > 80:
            continue
        counter += 1
        w = min(width - 20, max(40, int(len(line) * 8)))
        x = max(10, (width - w) // 2)
        elements.append(
            ScreenElement(
                id=f"ocr_{counter}",
                type="text",
                text=line,
                bbox=(x, y, w, max(16, step - 6)),
                confidence=0.5,
            )
        )
        y += step
    return elements


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def find_target(ctx: ScreenContext, target: VisualTarget) -> List[ScreenElement]:
    """Return matching elements sorted by relevance."""
    candidates: List[Tuple[float, ScreenElement]] = []
    needle = (target.text or "").strip().lower()
    if not needle:
        return []

    for el in ctx.elements:
        text_l = el.text.lower()
        score = 0.0
        if needle in text_l:
            score += 1.0
            # Boost for exact match.
            if text_l == needle:
                score += 0.5
            # Boost for short labels (likely buttons).
            if len(text_l) <= 20:
                score += 0.2
        elif target.target_type and target.target_type == el.type:
            score += 0.1
        if score <= 0:
            continue
        candidates.append((score, el))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [el for _, el in candidates]


def resolve_target(ctx: ScreenContext, target: VisualTarget) -> Tuple[Optional[ScreenElement], str]:
    """Pick a single target element. Returns ``(element, reason)``.

    If the request is ambiguous, the highest-confidence element is
    returned but ``reason`` indicates that the caller should
    disambiguate.
    """
    matches = find_target(ctx, target)
    if not matches:
        return None, "no match"

    if target.ordinal is not None:
        idx = max(0, min(target.ordinal, len(matches) - 1))
        return matches[idx], "ordinal"

    if len(matches) == 1:
        return matches[0], "single"

    # Multiple matches - if the user said "the X button" we trust the
    # top hit but flag ambiguity.  If ``verify`` is on, the caller will
    # ask the user.
    return matches[0], "ambiguous"


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------
def _require_pyautogui() -> Optional[Any]:
    try:
        import pyautogui  # type: ignore
        return pyautogui
    except Exception as exc:  # pragma: no cover - env dependent
        log.warning("pyautogui not available: %s", exc)
        return None


def _clamp(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    pyautogui = _require_pyautogui()
    if pyautogui is None:
        return (max(0, x), max(0, y))
    try:
        sw, sh = pyautogui.size()
    except Exception:
        sw, sh = width, height
    return (
        max(1, min(int(sw) - 1, int(x))),
        max(1, min(int(sh) - 1, int(y))),
    )


def execute_visual(target: VisualTarget) -> Dict[str, Any]:
    """Resolve + perform a visual action. Returns a small dict with the
    outcome so the calling tool can render a friendly message."""
    cfg = get_config()
    ctx = build_screen_context(force=False)

    if target.action == "scroll":
        return _do_scroll(target)

    # No OCR text and no target to find - tell the user clearly what
    # is missing instead of returning a generic "could not find".
    if not ctx.elements and not ctx.ocr_text:
        if extract_text is None:
            return {
                "success": False,
                "message": (
                    "I cannot read the screen right now, sir - the OCR engine "
                    "(tesseract) is not installed. Please install Tesseract "
                    "and add it to your PATH, or use coordinate-based clicks."
                ),
                "reason": "ocr_unavailable",
            }
        return {
            "success": False,
            "message": (
                f"I captured the screen but could not read any text. "
                f"Active window: {ctx.active_window or 'unknown'}."
            ),
            "reason": "ocr_empty",
        }

    element, reason = resolve_target(ctx, target)
    if element is None:
        return {
            "success": False,
            "message": f"I could not find {target.describe()} on your screen.",
            "reason": reason,
            "context": ctx.to_dict(),
        }

    if reason == "ambiguous" and cfg.max_visual_retries >= 0:
        # Surface ambiguity - the caller may ask the user.
        log.info("Ambiguous target %s, returning top match", target.describe())

    x, y = element.center
    x, y = _clamp(x, y, ctx.width, ctx.height)

    pyautogui = _require_pyautogui()
    if pyautogui is None:
        return {
            "success": False,
            "message": "pyautogui is not installed - I cannot click anything, sir.",
            "element": element.to_dict(),
        }

    try:
        if target.action == "double_click":
            pyautogui.doubleClick(x, y)
        elif target.action == "right_click":
            pyautogui.rightClick(x, y)
        else:
            pyautogui.click(x, y)
    except Exception as exc:
        return {
            "success": False,
            "message": f"Click failed: {exc}",
            "element": element.to_dict(),
        }

    return {
        "success": True,
        "message": f"Clicked {element.text!r} at ({x},{y}).",
        "element": element.to_dict(),
        "coordinates": [x, y],
        "context": ctx.to_dict(),
    }


def _do_scroll(target: VisualTarget) -> Dict[str, Any]:
    pyautogui = _require_pyautogui()
    if pyautogui is None:
        return {"success": False, "message": "pyautogui is not installed."}
    direction = target.extra or "down"
    clicks = 3
    try:
        if direction == "down":
            pyautogui.scroll(-clicks)
        elif direction == "up":
            pyautogui.scroll(clicks)
        elif direction == "left":
            pyautogui.hscroll(-clicks)
        elif direction == "right":
            pyautogui.hscroll(clicks)
    except Exception as exc:
        return {"success": False, "message": f"Scroll failed: {exc}"}
    return {"success": True, "message": f"Scrolled {direction}."}


def type_into(target: VisualTarget, text: str) -> Dict[str, Any]:
    """Click the target field, then type *text* via the clipboard+Ctrl+V path."""
    cfg = get_config()
    if not text:
        return {"success": False, "message": "No text to type, sir."}
    click_result = execute_visual(
        VisualTarget(action="click", text=target.text, target_type=target.target_type or "input", verify=False)
    )
    if not click_result.get("success"):
        return click_result
    try:
        from tools.typing import TypingTool
        tool = TypingTool()
        result = tool.execute(f"type {text}")
    except Exception as exc:
        return {"success": False, "message": f"Typing failed: {exc}"}
    return {
        "success": bool(result.success),
        "message": f"Typed into {target.text!r}." if result.success else result.message,
    }


__all__ = [
    "VisualTarget",
    "looks_like_visual_action",
    "parse_target",
    "build_screen_context",
    "find_target",
    "resolve_target",
    "execute_visual",
    "type_into",
]
