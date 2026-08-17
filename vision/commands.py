"""
vision.commands
~~~~~~~~~~~~~~~

The :class:`VisionTool` and the :func:`register_vision_tools` helper that
wires the vision subsystem into :mod:`core.router`.

Voice patterns the tool recognises
---------------------------------

``"Jarvis, what is on my screen?"``
   -> Capture primary monitor and analyse.
``"Jarvis, read this page."``
   -> Same as above, but emphasise OCR.
``"Explain this error."``
   -> Capture screen, then analyse with a debugging-flavoured prompt.
``"What am I looking at?"``
   -> Capture active window.
``"Read the selected text."``
   -> Capture the screen and only return the OCR portion.
``"Describe the image."``
   -> Capture a webcam frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger
from vision.analyzer import AnalysisResult, analyze_image
from vision.capture import (
    capture_active_window,
    capture_full_screen,
    capture_primary_monitor,
    capture_region,
)
from vision.ocr import extract_text
from vision.webcam import capture_webcam

log = get_logger(__name__)


_TRIGGERS_SCREEN = (
    "what is on my screen",
    "on my screen",
    "what's on my screen",
    "read this page",
    "read the screen",
    "explain this error",
    "explain the error",
    "what is on screen",
    "what's on screen",
)

_TRIGGERS_ACTIVE = (
    "what am i looking at",
    "this window",
    "active window",
)

_TRIGGERS_WEBCAM = (
    "describe the image",
    "describe my image",
    "what does the camera see",
    "webcam image",
    "camera image",
    "take a photo",
    "snap a photo",
)

_TRIGGERS_OCR = (
    "read the selected text",
    "read selected text",
    "extract the text",
    "just the text",
)

_TRIGGERS_REGION = (
    "capture region",
    "selected region",
    "capture selection",
    "screenshot this area",
)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _build_debug_prompt(user_command: str) -> str:
    return (
        "You are helping the user understand an error or technical issue shown on "
        "their screen. Be precise, list the relevant stack trace lines if any, and "
        "offer a probable cause and next step.\n\n"
        f"User said: {user_command}"
    )


class VisionTool(BaseTool):
    """Tool that owns every vision-related voice command."""

    name = "vision_tool"
    description = (
        "Inspect the user's screen, the active window, the webcam, or a region "
        "of the screen. Returns a natural-language description suitable for "
        "text-to-speech. Input is the user's raw voice command."
    )

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        triggers = _TRIGGERS_SCREEN + _TRIGGERS_ACTIVE + _TRIGGERS_WEBCAM + _TRIGGERS_OCR + _TRIGGERS_REGION
        return any(trigger in text for trigger in triggers) or "screenshot" in text or "screen" in text

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()

        try:
            # ---- Region ---------------------------------------------------
            if _contains_any(lowered, _TRIGGERS_REGION):
                return self._handle_region(text, context)

            # ---- Webcam ---------------------------------------------------
            if _contains_any(lowered, _TRIGGERS_WEBCAM):
                return self._handle_webcam(text)

            # ---- OCR-only -------------------------------------------------
            if _contains_any(lowered, _TRIGGERS_OCR):
                return self._handle_screen(text, ocr_only=True)

            # ---- Active window -------------------------------------------
            if _contains_any(lowered, _TRIGGERS_ACTIVE):
                return self._handle_active_window(text)

            # ---- Full / primary screen -----------------------------------
            return self._handle_screen(text, ocr_only=False)

        except Exception as exc:
            log.exception("Vision tool failed: %s", exc)
            return ToolResult(success=False, message=f"I could not complete that vision task: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_screen(self, command: str, ocr_only: bool) -> ToolResult:
        capture = capture_primary_monitor()
        return self._analyse(capture.path, command, ocr_only=ocr_only)

    def _handle_active_window(self, command: str) -> ToolResult:
        capture = capture_active_window()
        return self._analyse(capture.path, command)

    def _handle_webcam(self, command: str) -> ToolResult:
        capture = capture_webcam()
        return self._analyse(capture.path, command)

    def _handle_region(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        cfg = get_config()
        region = (context or {}).get("region") if context else None
        if (
            region
            and isinstance(region, dict)
            and all(k in region for k in ("left", "top", "width", "height"))
        ):
            try:
                capture = capture_region(
                    left=int(region["left"]),
                    top=int(region["top"]),
                    width=int(region["width"]),
                    height=int(region["height"]),
                )
            except Exception as exc:
                return ToolResult(success=False, message=f"Region capture failed: {exc}")
            return self._analyse(capture.path, command)

        # Without coordinates we cannot truly crop, so we capture the full
        # screen and surface only the OCR text. Future overlays should
        # supply real coordinates via context["region"].
        capture = capture_full_screen()
        ocr = extract_text(capture.path, languages=cfg.vision_ocr_languages)
        if ocr.text:
            return ToolResult(
                success=True,
                message=(
                    f"I do not yet know which region to crop, sir, so I read the "
                    f"entire screen. Here is the text:\n\n{ocr.text}"
                ),
                data={"path": str(capture.path), "ocr": ocr.text},
            )
        return ToolResult(
            success=True,
            message=f"I captured the screen to {capture.path} but could not detect any region to crop.",
            data={"path": str(capture.path)},
        )

    def _analyse(self, image_path: Path, command: str, ocr_only: bool = False) -> ToolResult:
        if ocr_only:
            cfg = get_config()
            ocr = extract_text(image_path, languages=cfg.vision_ocr_languages)
            if ocr.text:
                return ToolResult(success=True, message=ocr.text, data={"path": str(image_path), "ocr_engine": ocr.engine})
            return ToolResult(success=False, message="I could not read any text in the image, sir.")

        prompt = _build_debug_prompt(command) if "error" in command.lower() else (
            "Describe what is on the screen in clear, natural language. Mention any "
            "error messages, headings, or notable UI elements."
        )
        result: AnalysisResult = analyze_image(image_path, prompt=prompt)
        return ToolResult(
            success=True,
            message=result.description,
            data={
                "path": str(image_path),
                "model": result.model,
                "used_vision_model": result.used_vision_model,
                "ocr_text": result.ocr.text if result.ocr else "",
            },
        )


# ----------------------------------------------------------------------
# Registration helper
# ----------------------------------------------------------------------
def register_vision_tools(router) -> List[BaseTool]:
    """Register the vision tool with *router* and return the new tools."""
    tool = VisionTool()
    router.register(
        tool,
        keywords=(
            "screen",
            "screenshot",
            "webcam",
            "describe the image",
            "what is on my screen",
            "what am i looking at",
            "read this page",
            "read the selected text",
            "explain this error",
        ),
        priority=80,
    )
    return [tool]


__all__ = ["VisionTool", "register_vision_tools"]
