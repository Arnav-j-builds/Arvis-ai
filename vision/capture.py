"""
vision.capture
~~~~~~~~~~~~~~

Wraps ``mss`` (cross-platform screen capture) into dataclass-friendly
helpers used by :mod:`vision.analyzer` and :mod:`vision.commands`.

Public API
----------

* :func:`capture_full_screen`   - the entire virtual screen.
* :func:`capture_primary_monitor` - just the primary monitor.
* :func:`capture_active_window` - the foreground window (Windows/macOS).
* :func:`capture_region`        - ``(left, top, width, height)`` rectangle.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class CaptureResult:
    """The outcome of a screen capture."""

    path: Path
    width: int
    height: int
    monitor_index: Optional[int]
    source: str  # "screen" / "active_window" / "region" / "webcam"


def _write_png(image_bytes, size: Tuple[int, int], destination: Path) -> Path:
    """Persist *image_bytes* to *destination* using :mod:`mss.tools`."""
    import mss.tools  # local import keeps the import cost off the hot path

    destination.parent.mkdir(parents=True, exist_ok=True)
    mss.tools.to_png(image_bytes, size, output=str(destination))
    return destination


def _timestamped_filename(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def capture_full_screen(destination: Optional[Path] = None) -> CaptureResult:
    """Capture every connected monitor (the virtual screen)."""
    import mss

    cfg = get_config()
    target = destination or (cfg.screenshots_dir / _timestamped_filename("screen"))
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # virtual screen
        shot = sct.grab(monitor)
        path = _write_png(shot.rgb, shot.size, target)
    log.info("Captured full screen -> %s", path)
    return CaptureResult(path=path, width=shot.size[0], height=shot.size[1], monitor_index=0, source="screen")


def capture_primary_monitor(destination: Optional[Path] = None) -> CaptureResult:
    """Capture the primary monitor only."""
    import mss

    cfg = get_config()
    target = destination or (cfg.screenshots_dir / _timestamped_filename("primary"))
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        path = _write_png(shot.rgb, shot.size, target)
    log.info("Captured primary monitor -> %s", path)
    return CaptureResult(path=path, width=shot.size[0], height=shot.size[1], monitor_index=1, source="screen")


def capture_region(
    left: int,
    top: int,
    width: int,
    height: int,
    destination: Optional[Path] = None,
) -> CaptureResult:
    """Capture the rectangle described by ``(left, top, width, height)``.

    Negative or oversized coordinates are clipped by ``mss`` itself.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    import mss

    cfg = get_config()
    target = destination or (cfg.screenshots_dir / _timestamped_filename("region"))
    region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
    with mss.mss() as sct:
        shot = sct.grab(region)
        path = _write_png(shot.rgb, shot.size, target)
    log.info("Captured region %s -> %s", region, path)
    return CaptureResult(path=path, width=shot.size[0], height=shot.size[1], monitor_index=None, source="region")


def capture_active_window(destination: Optional[Path] = None) -> CaptureResult:
    """Capture the foreground window.

    Implemented for:

    * Windows via :mod:`pywin32` (``GetForegroundWindow`` + ``GetWindowRect``)
    * macOS via :mod:`Quartz` (``CGWindowListCopyWindowInfo``)
    * Other platforms fall back to the primary monitor with a warning.
    """
    cfg = get_config()
    target = destination or (cfg.screenshots_dir / _timestamped_filename("active_window"))

    system = platform.system()
    if system == "Windows":
        try:
            import win32gui  # type: ignore[import-not-found]
            import win32ui  # type: ignore[import-not-found]
            import win32con  # type: ignore[import-not-found]
            from PIL import Image

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                log.warning("No foreground window detected, falling back to primary monitor")
                return capture_primary_monitor(target)

            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = max(1, right - left)
            height = max(1, bottom - top)

            w_dc = win32gui.GetWindowDC(hwnd)
            src_dc = win32ui.CreateDCFromHandle(w_dc)
            mem_dc = src_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(src_dc, width, height)
            mem_dc.SelectObject(bitmap)
            # PW_RENDERFULLCONTENT (0x00000002) is required on Windows 10+ to
            # capture hardware-accelerated (DWM) windows.
            # pywin32 BitBlt signature: BitBlt(destPos, size, dc, srcPos, rop)
            # destPos and size are separate (x, y) and (w, h) tuples.
            mem_dc.BitBlt((0, 0), (width, height), src_dc, (0, 0), win32con.SRCCOPY)

            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
                bmpstr,
                "raw",
                "BGRX",
                0,
                1,
            )

            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)

            # Release GDI handles - each DeleteObject only takes a single handle.
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
            try:
                mem_dc.DeleteDC()
            except Exception:
                pass
            try:
                src_dc.DeleteDC()
            except Exception:
                pass
            try:
                win32gui.ReleaseDC(hwnd, w_dc)
            except Exception:
                pass

            return CaptureResult(
                path=target,
                width=width,
                height=height,
                monitor_index=None,
                source="active_window",
            )
        except ImportError:
            log.warning("pywin32 not installed; falling back to primary monitor")
            return capture_primary_monitor(target)

    if system == "Darwin":
        try:
            from Quartz import (  # type: ignore[import-not-found]
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
            )
            import mss

            info_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            if not info_list:
                log.warning("No active window reported by Quartz; using primary monitor")
                return capture_primary_monitor(target)

            bounds = info_list[0].get("kCGWindowBounds", {})
            if not bounds:
                return capture_primary_monitor(target)
            return capture_region(
                left=int(bounds.get("X", 0)),
                top=int(bounds.get("Y", 0)),
                width=int(bounds.get("Width", 0)),
                height=int(bounds.get("Height", 0)),
                destination=target,
            )
        except ImportError:
            log.warning("pyobjc/Quartz not installed; falling back to primary monitor")
            return capture_primary_monitor(target)

    log.warning("Active-window capture not implemented on %s; using primary monitor", system)
    return capture_primary_monitor(target)


__all__ = [
    "CaptureResult",
    "capture_full_screen",
    "capture_primary_monitor",
    "capture_active_window",
    "capture_region",
]


def _module_loaded() -> None:
    """Quietly check that the platform is supported - logs a warning only."""
    if platform.system() not in {"Windows", "Linux", "Darwin"}:
        log.warning("vision.capture: unrecognised platform %s", platform.system())


if __name__ == "__main__":  # pragma: no cover - manual debugging
    _module_loaded()
    out = capture_primary_monitor()
    print(f"Saved {out.path} ({out.width}x{out.height})")
    sys.exit(0)
