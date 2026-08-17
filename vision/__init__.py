"""
Vision package for the arvis assistant.

Capabilities:

* :mod:`vision.capture`  - grab the entire screen, the active window, or a
  user-defined region via ``mss``.
* :mod:`vision.webcam`   - capture a single frame from the local webcam via
  ``opencv-python`` (with a ``PIL.ImageGrab`` fallback).
* :mod:`vision.ocr`      - extract text from any image using Tesseract or
  EasyOCR (whichever is configured and installed).
* :mod:`vision.analyzer` - ask a vision-capable Ollama model about an
  image, falling back to OCR + captioning.
* :mod:`vision.commands` - the :class:`~core.base.BaseTool` that exposes
  every vision capability through the command router.
* :mod:`vision.hand_mouse` - hand-gesture mouse control via the webcam
  (MediaPipe when available, opencv contour fallback otherwise).
* :mod:`vision.eye_mouse` - eye-gaze mouse control via the webcam
  (MediaPipe ``FaceLandmarker`` + iris landmarks).

The module depends only on ``core`` and third-party libraries. The existing
screenshot / OCR tools in ``tools/`` are kept untouched; vision is the new,
preferred entry point.
"""

from vision.commands import VisionTool, register_vision_tools
from vision.hand_mouse import (
    HandMouseTool,
    register_hand_mouse_tools,
    HAS_CV2,
    HAS_MEDIAPIPE,
    HAS_PYAUTOGUI,
)
from vision.eye_mouse import (
    EyeMouseTool,
    register_eye_mouse_tools,
)

__all__ = [
    "VisionTool",
    "register_vision_tools",
    "HandMouseTool",
    "register_hand_mouse_tools",
    "EyeMouseTool",
    "register_eye_mouse_tools",
    "HAS_CV2",
    "HAS_MEDIAPIPE",
    "HAS_PYAUTOGUI",
]
