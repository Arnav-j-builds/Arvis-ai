"""
vision.webcam
~~~~~~~~~~~~~

Capture a single image from the local webcam. We try ``opencv-python`` first
because it is the most reliable cross-platform option, then fall back to
``PIL.ImageGrab`` (Windows only) and finally to a clear error message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.config import get_config
from core.logger import get_logger
from vision.capture import CaptureResult

log = get_logger(__name__)


def capture_webcam(
    destination: Optional[Path] = None,
    device_index: Optional[int] = None,
    warmup_seconds: float = 0.5,
) -> CaptureResult:
    """Capture a single frame from the webcam.

    Parameters
    ----------
    destination:
        Where to save the PNG. Defaults to ``storage/screenshots/webcam_*.png``.
    device_index:
        Camera index. ``None`` uses ``JARVIS_WEBCAM_INDEX`` from config.
    warmup_seconds:
        Sleep this long after opening the camera to let auto-exposure
        stabilise before grabbing the frame.
    """
    cfg = get_config()
    if not cfg.vision_use_webcam:
        raise RuntimeError("Webcam capture is disabled (JARVIS_USE_WEBCAM=False).")

    index = int(device_index if device_index is not None else cfg.vision_webcam_index)
    target = destination or (cfg.screenshots_dir / "webcam_latest.png")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Try OpenCV first
    try:
        import cv2  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open webcam index {index}")

        if warmup_seconds > 0:
            import time

            time.sleep(warmup_seconds)

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Webcam returned an empty frame.")

        cv2.imwrite(str(target), frame)
        height, width = frame.shape[:2]
        log.info("Webcam frame captured via OpenCV -> %s", target)
        return CaptureResult(path=target, width=width, height=height, monitor_index=None, source="webcam")

    except ImportError:
        log.info("opencv-python not available; trying PIL fallback.")
    except Exception as exc:
        log.warning("OpenCV webcam capture failed: %s", exc)

    # Fallback: PIL.ImageGrab (Windows only)
    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]

        image = ImageGrab.grab()
        image.save(target)
        log.info("Webcam fallback via PIL.ImageGrab -> %s", target)
        return CaptureResult(path=target, width=image.width, height=image.height, monitor_index=None, source="webcam")
    except Exception as exc:
        raise RuntimeError(
            "Webcam capture failed. Install opencv-python (`pip install opencv-python`) "
            f"or rely on the legacy screenshot tool. Underlying error: {exc}"
        ) from exc


__all__ = ["capture_webcam"]
