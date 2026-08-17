"""
vision.hand_mouse
~~~~~~~~~~~~~~~~~

A :class:`BaseTool` that lets the user control the mouse with hand gestures
captured from the local webcam.

Architecture
------------
The tool exposes three intents:

* "start hand mouse"  - launch a background control loop that opens the
  webcam, tracks the user's hand, and converts gestures into real mouse
  events (movement, click, right-click, double-click, scroll).
* "stop hand mouse"   - stop the loop cleanly and release the camera.
* "hand mouse status" - report the current state (running, backend, etc.).

The control loop runs in a daemon thread. It is **opt-in** - nothing
starts until the user explicitly says the trigger phrase. The tool never
crashes the rest of the assistant: every import / runtime error is
translated into a friendly :class:`ToolResult`.

Dependencies (all optional at import time)
------------------------------------------
* ``opencv-python``  - webcam capture + drawing helpers. Required.
* ``mediapipe``      - hand tracking. Supports BOTH the modern 1.0+
  ``mediapipe.tasks.vision.HandLandmarker`` API and the legacy
  ``mediapipe.solutions.hands`` API. When missing the tool falls back
  to a simple skin-colour contour tracker so the feature still works
  on a stock install.
* ``pyautogui``      - actual mouse event injection. When missing the
  tool reports the gesture but does not move the real cursor.

The MediaPipe Tasks API needs a small ``.task`` model file. We download
it on first use into the project's ``storage/`` directory and reuse it
on every subsequent start. No network access is required after the
first launch.
"""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.logger import get_logger
from core.config import get_config

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


cv2 = _try_import("cv2")
mediapipe = _try_import("mediapipe")
pyautogui = _try_import("pyautogui")

HAS_CV2 = cv2 is not None
HAS_PYAUTOGUI = pyautogui is not None

# Probe the MediaPipe API. Newer versions (>= 0.10.14 / 1.0) removed
# ``mediapipe.solutions`` and exposed a ``HandLandmarker`` task in
# ``mediapipe.tasks.python.vision``. We support both, and use a
# direct-file import as a fallback because some 1.0.x wheels ship with
# a broken ``mediapipe.tasks.__init__`` that fails on circular imports.
_HAS_LEGACY_MEDIAPIPE = (
    mediapipe is not None
    and hasattr(mediapipe, "solutions")
    and hasattr(mediapipe.solutions, "hands")
)
_HAS_MODERN_MEDIAPIPE = False
_modern_mp_hand_landmarker = None
_modern_mp_options = None
_modern_mp_image = None
_modern_mp_image_format = None
_modern_mp_running_mode = None
_modern_mp_base_options = None
if mediapipe is not None and hasattr(mediapipe, "tasks"):
    # 1. Try the conventional nested import.
    try:
        from mediapipe.tasks.python.vision import (
            HandLandmarker as _HandLandmarker,
            HandLandmarkerOptions as _HandLandmarkerOptions,
            RunningMode as _RunningMode,
        )
        from mediapipe.tasks.python.vision.core import image as _image_mod
        from mediapipe.tasks.python.core import base_options as _bo_mod
        _modern_mp_hand_landmarker = _HandLandmarker
        _modern_mp_options = _HandLandmarkerOptions
        _modern_mp_image = _image_mod.Image
        _modern_mp_image_format = _image_mod.ImageFormat
        _modern_mp_running_mode = _RunningMode
        _modern_mp_base_options = _bo_mod.BaseOptions
        _HAS_MODERN_MEDIAPIPE = True
    except Exception as exc:
        log.debug("Conventional MediaPipe import failed: %s", exc)
        # 2. Fall back to direct file-path imports. Some 1.0.x wheels
        #    ship a broken ``mediapipe.tasks.__init__`` that prevents
        #    the conventional nested import from succeeding even though
        #    every individual module file is intact.
        try:
            import importlib.util as _ilu
            import pathlib as _pl

            def _load(name: str, relpath: str):
                path = _pl.Path(mediapipe.__file__).parent / relpath
                spec = _ilu.spec_from_file_location(name, str(path))
                if spec is None or spec.loader is None:
                    return None
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            _vision_mod = _load(
                "mp_vision_direct",
                "tasks/python/vision/hand_landmarker.py",
            )
            _image_mod = _load(
                "mp_image_direct",
                "tasks/python/vision/core/image.py",
            )
            _bo_mod = _load(
                "mp_base_options_direct",
                "tasks/python/core/base_options.py",
            )
            if _vision_mod and _image_mod and _bo_mod:
                _modern_mp_hand_landmarker = _vision_mod.HandLandmarker
                _modern_mp_options = _vision_mod.HandLandmarkerOptions
                _modern_mp_running_mode = _vision_mod.RunningMode
                _modern_mp_image = _image_mod.Image
                _modern_mp_image_format = _image_mod.ImageFormat
                _modern_mp_base_options = _bo_mod.BaseOptions
                _HAS_MODERN_MEDIAPIPE = True
        except Exception as exc2:  # pragma: no cover - defensive
            log.debug("Direct MediaPipe import also failed: %s", exc2)

HAS_MEDIAPIPE = _HAS_LEGACY_MEDIAPIPE or _HAS_MODERN_MEDIAPIPE

# Disable pyautogui's fail-safe (moving to corner aborts) - we WANT to be
# able to use the corners of the screen as part of the mapping.
if HAS_PYAUTOGUI:  # pragma: no cover - depends on env
    try:
        pyautogui.FAILSAFE = False
        # Small pause makes the cursor less jittery on slow hardware.
        pyautogui.PAUSE = 0.0
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------
_START_PHRASES = (
    "start hand mouse",
    "start the hand mouse",
    "start hand tracking",
    "enable hand mouse",
    "enable hand control",
    "control my mouse with my hand",
    "control the mouse with my hand",
    "control mouse with hand",
    "hand mouse on",
    "hand tracking on",
    "begin hand tracking",
    "begin hand mouse",
)

_STOP_PHRASES = (
    "stop hand mouse",
    "stop the hand mouse",
    "stop hand tracking",
    "disable hand mouse",
    "disable hand control",
    "hand mouse off",
    "hand tracking off",
    "release the mouse",
    "let go of the mouse",
)

_STATUS_PHRASES = (
    "is hand mouse running",
    "is hand mouse on",
    "hand mouse status",
    "hand tracking status",
)


def _contains_any(text: str, needles: tuple) -> bool:
    lowered = (text or "").lower()
    return any(needle in lowered for needle in needles)


# ---------------------------------------------------------------------------
# Model download for the modern MediaPipe API
# ---------------------------------------------------------------------------
_HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_TASK_FILENAME = "hand_landmarker.task"


def _ensure_hand_model() -> Optional[str]:
    """Download the HandLandmarker ``.task`` file if it is not cached.

    The file is stored in ``storage/`` so subsequent launches do not need
    a network connection. Returns the absolute path to the model file or
    ``None`` when the download failed.
    """
    try:
        cfg = get_config()
        target = cfg.storage_dir / _HAND_TASK_FILENAME
        if target.exists() and target.stat().st_size > 100_000:
            return str(target)
        cfg.storage_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading MediaPipe hand_landmarker.task model...")
        # Use a short timeout so a missing internet connection does not
        # block the start of the control loop for long.
        req = urllib.request.Request(_HAND_TASK_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        target.write_bytes(data)
        log.info("Downloaded hand_landmarker.task (%d bytes).", len(data))
        return str(target)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.warning("Could not download hand_landmarker.task: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unexpected error downloading hand_landmarker.task: %s", exc)
        return None


# ---------------------------------------------------------------------------
# The control loop - lives in a background thread, talks to the OS through
# pyautogui, reads frames from opencv, finds the hand with mediapipe.
# ---------------------------------------------------------------------------
class _HandMouseController:
    """Background controller that maps hand gestures to mouse events.

    The class is intentionally a plain Python object (not a BaseTool) so
    that :class:`HandMouseTool` can own it and route commands to it. The
    controller never raises - all errors are logged and surfaced through
    the ``last_error`` attribute.
    """

    # Gesture tunables - the defaults work well on a 720p webcam.
    SMOOTHING = 6            # higher = smoother but laggier
    FINE_SMOOTHING = 14      # smoothing when the hand is barely moving
    FINE_PIXELS = 6          # px-of-target-change below which we treat the hand as "still"
    CLICK_HOLD = 0.08        # seconds the pinch must hold before clicking
    CLICK_DROPOUT = 0.10     # short dropouts don't reset the click timer
    DOUBLE_CLICK_GAP = 0.45  # seconds between two clicks = double-click
    FRAME_WIDTH = 640        # resize target for performance
    FRAME_HEIGHT = 480
    MARGIN = 100             # deadzone in pixels around the frame edge
    SCROLL_SENSITIVITY = 25  # pixels of vertical movement = 1 scroll tick
    ZOOM_SENSITIVITY = 220   # px-spread change = 1 ctrl+scroll tick
    DRAG_HOLD = 0.45         # seconds of sustained pinch = drag begins
    MOVE_COOLDOWN = 0.0      # seconds between pyautogui.moveTo calls
    CAMERA_PROBE_INDEXES = (0, 1, 2)  # try these in order
    FIST_HOLD = 0.10         # seconds a closed fist must hold to fire a click
    FIST_RELEASE = 0.20      # seconds the hand must be open before the next fist click is allowed
    PINCH_HYSTERESIS = 0.012 # fraction of frame width to add when releasing pinch
    FIST_HYSTERESIS = 0.015  # fraction of frame height added when releasing a fist
    # Two-hand pinch zoom: when both hands are pinching
    # (thumb+index) and the user pulls them apart / together, send
    # Ctrl+wheel to zoom the foreground app. ``ZOOM_HOLD_SEC`` is the
    # time both pinches must be held before zooming starts so an
    # accidental mid-click pinch does not fire a zoom.
    ZOOM_HOLD_SEC = 0.12
    ZOOM_DEADZONE_PX = 4      # px of distance change ignored per frame
    ZOOM_COOLDOWN = 0.05      # seconds between Ctrl+scroll events

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._click_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.running: bool = False
        self.started_at: float = 0.0
        self.frames_seen: int = 0
        self.last_action: str = "idle"
        self.last_action_at: float = 0.0
        self.last_error: str = ""
        self.backend: str = "none"  # "mediapipe-modern" | "mediapipe-legacy" | "opencv" | "none"
        self._last_move_at: float = 0.0
        # (active, started_at, cooldown_until) - state for closed-fist
        # click detection. Persisted across frames via _fist_state.
        self._fist_state: Tuple[bool, float, float] = (False, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> Tuple[bool, str]:
        """Start the control loop. Returns ``(ok, message)``."""
        if self.running:
            return True, "Hand mouse is already running, sir."
        if not HAS_CV2:
            return False, (
                "I cannot start the hand mouse, sir. OpenCV is not installed. "
                "Install opencv-python to use hand tracking."
            )
        if not HAS_PYAUTOGUI:
            log.warning("pyautogui missing - cursor will not move.")
        # Reset transient state so a previous run cannot poison the new one.
        self._stop.clear()
        with self._state_lock:
            self.running = False
            self.started_at = 0.0
            self.frames_seen = 0
            self.last_action = "idle"
            self.last_error = ""
            self.backend = "none"
            self._fist_state = (False, 0.0, 0.0)
        self._thread = threading.Thread(
            target=self._run, name="hand-mouse", daemon=True
        )
        self._thread.start()
        # Camera initialisation on Windows can take 2-3 seconds on a cold
        # start. Poll for up to 6 seconds: only fail when the loop itself
        # has reported a *terminal* error (camera probe failure, runtime
        # crash). Backend-selection warnings (e.g. "modern MediaPipe init
        # failed, falling back to legacy") log as warnings and never set
        # ``last_error``, so they don't trigger a false-negative here.
        deadline = time.time() + 6.0
        while time.time() < deadline:
            time.sleep(0.1)
            if self.running:
                return True, "Hand mouse is now active, sir. Show your hand to the camera."
            if self.last_error and "failed to start" in self.last_error:
                break
        # Final check: if the thread set ``running`` between the last poll
        # and now, treat that as a success. Otherwise the loop really
        # failed to start.
        if self.running:
            return True, "Hand mouse is now active, sir. Show your hand to the camera."
        err = self.last_error or "the camera failed to open."
        # Last-ditch effort: did the thread actually finish cleanly? If it
        # did, the controller was running and we just polled too early.
        if self._thread is not None and not self._thread.is_alive() and self.frames_seen == 0:
            return False, f"Hand mouse did not start: {err}"
        return True, (
            "Hand mouse is starting up, sir. Give it a moment and show your "
            "hand to the camera."
        )

    def stop(self) -> Tuple[bool, str]:
        if not self.running:
            return True, "Hand mouse is already off, sir."
        self._stop.set()
        # Give the thread up to ~1.5s to clean up.
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.5)
        self._thread = None
        self.running = False
        return True, "Hand mouse is now off, sir."

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "backend": self.backend,
            "frames_seen": self.frames_seen,
            "last_action": self.last_action,
            "last_action_at": self.last_action_at,
            "last_error": self.last_error,
            "started_at": self.started_at,
        }

    # ------------------------------------------------------------------
    # The loop itself
    # ------------------------------------------------------------------
    def _set_state(self, **kwargs) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _run(self) -> None:
        cap = None
        landmarker = None
        legacy_hands = None
        try:
            cap = self._open_camera()
            if cap is None:
                # ``_open_camera`` already populated ``last_error`` with a
                # user-friendly message; mark it fatal so the start()
                # poll loop returns it as a failure rather than as a
                # success-after-timeout.
                self._set_state(last_error=(
                    "failed to start: " + (self.last_error or "no camera available")
                ))
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

            # The camera is up. We can mark the controller as running
            # *before* backend selection, so the start() poll loop sees
            # success even if a MediaPipe download is slow.
            self._set_state(
                running=True,
                started_at=time.time(),
                last_error="",
            )

            if _HAS_MODERN_MEDIAPIPE:
                model_path = _ensure_hand_model()
                if model_path is not None:
                    try:
                        landmarker = _modern_mp_hand_landmarker.create_from_options(
                            _modern_mp_options(
                                base_options=_modern_mp_base_options(
                                    model_asset_path=model_path
                                ),
                                running_mode=_modern_mp_running_mode.VIDEO,
                                # Track up to two hands so the user can
                                # drive the two-hand pinch-to-zoom
                                # gesture. The single-hand code path is
                                # unchanged - it just uses ``hands[0]``.
                                num_hands=2,
                                min_hand_detection_confidence=0.5,
                                min_hand_presence_confidence=0.5,
                                min_tracking_confidence=0.5,
                            )
                        )
                        self._set_state(backend="mediapipe-modern")
                    except Exception as exc:
                        log.warning("Modern MediaPipe init failed: %s", exc)
                        landmarker = None

            if landmarker is None and _HAS_LEGACY_MEDIAPIPE:
                try:
                    legacy_hands = mediapipe.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5,
                    )
                    self._set_state(backend="mediapipe-legacy")
                except Exception as exc:
                    log.warning("Legacy MediaPipe init failed: %s", exc)
                    legacy_hands = None

            if landmarker is None and legacy_hands is None:
                self._set_state(backend="opencv")

            log.info("Hand mouse loop running (backend=%s).", self.backend)

            # --- Per-frame state -----------------------------------------
            smooth_x: Optional[float] = None
            smooth_y: Optional[float] = None
            # Display cursor stays at (display_x, display_y) while the
            # user is idle so the cursor does not jitter when the hand
            # is still.
            display_x: Optional[float] = None
            display_y: Optional[float] = None
            last_target_x: Optional[float] = None
            last_target_y: Optional[float] = None

            pinch_active: bool = False       # index+thumb currently pinched
            pinch_started: float = 0.0       # first frame the pinch was held
            last_pinch_frame: float = 0.0    # last frame the pinch was held
            dragging: bool = False
            drag_started: bool = False

            last_click_at: float = 0.0
            last_click_was_left: bool = True
            prev_index_y: Optional[float] = None
            prev_two_finger_spread: Optional[float] = None
            video_ts_ms: int = 0

            # Two-hand zoom state. ``zoom_active`` is True while both
            # hands are pinching and the user is moving them apart /
            # together; ``zoom_started`` is the wall-clock time both
            # pinches were first observed together; ``zoom_baseline`` is
            # the inter-index distance at that moment (pixels); ``prev_
            # zoom_distance`` is the last-frame distance used to compute
            # per-frame deltas. ``zoom_last_at`` rate-limits the Ctrl+
            # scroll events.
            zoom_active: bool = False
            zoom_started: float = 0.0
            zoom_baseline: float = 0.0
            prev_zoom_distance: float = 0.0
            zoom_last_at: float = 0.0

            screen_w, screen_h = self._screen_size()

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                self._set_state(frames_seen=self.frames_seen + 1)

                # Mirror so the user sees themselves like a mirror (and so
                # x-axis mappings feel natural).
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                hands, hand_count = self._detect(
                    frame, landmarker, legacy_hands, video_ts_ms
                )
                video_ts_ms += 33  # ~30 fps

                if hands is None:
                    # No hand in view - release any in-flight drag, reset
                    # the smoothing buffers, and let the cursor stay put.
                    if dragging and HAS_PYAUTOGUI:
                        try:
                            pyautogui.mouseUp()
                        except Exception:
                            pass
                    smooth_x, smooth_y = None, None
                    display_x, display_y = None, None
                    last_target_x, last_target_y = None, None
                    pinch_active = False
                    dragging = False
                    drag_started = False
                    prev_index_y = None
                    prev_two_finger_spread = None
                    zoom_active = False
                    zoom_baseline = 0.0
                    prev_zoom_distance = 0.0
                    self._fist_state = (False, 0.0, 0.0)
                    continue

                # The single-hand code path always reads hands[0]; the
                # two-hand zoom block below reads hands[1] when both are
                # visible.
                landmarks, handedness = hands[0]

                # --- Two-hand zoom detection (runs before single-hand
                # pinch handling so the single-hand click logic can
                # still fire when only one hand is visible). ---
                # Each hand is "pinching" when its thumb and index
                # fingertips are within PINCH pixels of each other,
                # normalised by frame width.
                def _hand_is_pinching(coords) -> bool:
                    try:
                        t = coords[4]
                        i = coords[8]
                        return math.hypot(t[0] - i[0], t[1] - i[1]) < 0.06 * w
                    except Exception:
                        return False

                if hand_count >= 2:
                    h1, _ = hands[0]
                    h2, _ = hands[1]
                    both_pinching = _hand_is_pinching(h1) and _hand_is_pinching(h2)
                    now_zoom = time.time()
                    if both_pinching:
                        # Distance between the two index fingertips,
                        # measured in pixels. This is what the user is
                        # actively "stretching" or "compressing".
                        dist = math.hypot(
                            h1[8][0] - h2[8][0], h1[8][1] - h2[8][1]
                        )
                        if not zoom_active:
                            zoom_active = True
                            zoom_started = now_zoom
                            zoom_baseline = dist
                            prev_zoom_distance = dist
                        else:
                            held = now_zoom - zoom_started
                            delta = dist - prev_zoom_distance
                            prev_zoom_distance = dist
                            if (
                                held >= self.ZOOM_HOLD_SEC
                                and abs(delta) >= self.ZOOM_DEADZONE_PX
                                and (now_zoom - zoom_last_at) >= self.ZOOM_COOLDOWN
                            ):
                                # Convert pixel-delta to scroll ticks.
                                # ``ZOOM_SENSITIVITY`` already exists
                                # for single-hand zoom, reuse it.
                                ticks = int(delta / self.ZOOM_SENSITIVITY * w)
                                if ticks != 0:
                                    # Positive delta = hands pulled
                                    # apart = zoom in (Ctrl+scroll up).
                                    # Negative = hands pulled together
                                    # = zoom out (Ctrl+scroll down).
                                    self._zoom_mouse(ticks)
                                    self._set_state(
                                        last_action=(
                                            "zoom_in" if ticks > 0
                                            else "zoom_out"
                                        ),
                                        last_action_at=now_zoom,
                                    )
                                    zoom_last_at = now_zoom
                    else:
                        # The user let go of at least one hand - reset
                        # the baseline so the next two-hand pinch
                        # starts fresh.
                        if zoom_active:
                            zoom_active = False
                            zoom_baseline = 0.0
                            prev_zoom_distance = 0.0
                else:
                    # Only one hand visible - drop the zoom state so a
                    # stray detection from a previous frame cannot fire
                    # a stale zoom.
                    if zoom_active:
                        zoom_active = False
                        zoom_baseline = 0.0
                        prev_zoom_distance = 0.0

                # Landmark indices (mediapipe hands convention).
                thumb = landmarks[4]
                index = landmarks[8]
                middle = landmarks[12]
                ring = landmarks[16]
                pinky = landmarks[20]

                # --- 1. Smooth cursor movement. --------------------------
                ix, iy = int(index[0]), int(index[1])
                mx = self.MARGIN
                my = self.MARGIN
                inner_w = max(1, w - 2 * mx)
                inner_h = max(1, h - 2 * my)
                ix_c = max(0, min(inner_w, ix - mx))
                iy_c = max(0, min(inner_h, iy - my))
                target_sx = ix_c / inner_w * screen_w
                target_sy = iy_c / inner_h * screen_h

                now = time.time()
                if display_x is None:
                    # First frame with a hand - snap the cursor to the
                    # hand position so the user does not see a long
                    # slide-in animation from (0, 0).
                    display_x, display_y = target_sx, target_sy
                    smooth_x = target_sx
                    smooth_y = target_sy
                else:
                    # Adaptive smoothing: when the target is barely
                    # moving, slow the EMA dramatically so the cursor
                    # settles into a still position instead of dancing
                    # around the pixel.
                    if last_target_x is not None:
                        ddx = abs(target_sx - last_target_x)
                        ddy = abs(target_sy - last_target_y)
                        still = ddx < self.FINE_PIXELS and ddy < self.FINE_PIXELS
                    else:
                        still = False
                    alpha = (
                        1.0 / self.FINE_SMOOTHING
                        if still
                        else 1.0 / self.SMOOTHING
                    )
                    smooth_x += (target_sx - smooth_x) * alpha
                    smooth_y += (target_sy - smooth_y) * alpha

                    # Display cursor only moves when the smoothed value
                    # has actually changed by a pixel. This eliminates
                    # sub-pixel jitter and the resulting pyautogui spam.
                    if (
                        abs(smooth_x - display_x) >= 1.0
                        or abs(smooth_y - display_y) >= 1.0
                    ):
                        display_x = smooth_x
                        display_y = smooth_y
                        self._move_mouse(display_x, display_y)
                        self._set_state(
                            last_action="move", last_action_at=now
                        )
                last_target_x, last_target_y = target_sx, target_sy

                # --- 2. Pinch detection (dropout-tolerant + hysteresis). --
                # We never reset ``pinch_started`` while the gesture is
                # held, even across single-frame dropouts - the click
                # timer keeps running. ``PINCH_HYSTERESIS`` widens the
                # release threshold so the gesture does not toggle at
                # the boundary.
                dx_p = thumb[0] - index[0]
                dy_p = thumb[1] - index[1]
                pinch_dist = math.hypot(dx_p, dy_p)
                pinch_on = pinch_dist < 0.05 * w
                pinch_off = pinch_dist > (0.05 + self.PINCH_HYSTERESIS) * w
                if pinch_on and not pinch_active:
                    pinch_active = True
                    pinch_started = now
                    last_pinch_frame = now
                elif pinch_on and pinch_active:
                    last_pinch_frame = now
                elif pinch_off and pinch_active:
                    # Hand is genuinely open again - cancel any in-flight
                    # drag and reset.
                    if dragging and HAS_PYAUTOGUI:
                        try:
                            pyautogui.mouseUp()
                        except Exception:
                            pass
                    dragging = False
                    drag_started = False
                    pinch_active = False
                    pinch_started = 0.0

                # Decide whether the pinch is "still being held" for the
                # purpose of firing clicks / starting a drag. The
                # CLICK_DROPOUT window absorbs up to 100ms of noise so a
                # single bad frame does not cancel an in-progress click.
                pinch_held = (
                    pinch_active
                    and (now - last_pinch_frame) <= self.CLICK_DROPOUT
                )
                if pinch_held:
                    held_for = now - pinch_started
                    if held_for >= self.DRAG_HOLD and not drag_started:
                        # The pinch has been held long enough that the
                        # user is almost certainly dragging something.
                        # Begin a drag so the eventual release drops the
                        # item where they let go.
                        drag_started = True
                        if HAS_PYAUTOGUI:
                            try:
                                pyautogui.mouseDown()
                            except Exception as exc:
                                log.debug("mouseDown failed: %s", exc)

                if pinch_held and not drag_started:
                    held_for = now - pinch_started
                    if held_for >= self.CLICK_HOLD:
                        # Two-finger variant: thumb also close to the
                        # middle fingertip -> right click.
                        mid_dist = math.hypot(
                            thumb[0] - middle[0], thumb[1] - middle[1]
                        )
                        is_two_finger = mid_dist < 0.05 * w * 1.3
                        with self._click_lock:
                            if is_two_finger:
                                self._click_mouse(right=True)
                                self._set_state(
                                    last_action="right-click",
                                    last_action_at=now,
                                )
                                last_click_was_left = False
                            else:
                                gap = now - last_click_at
                                if (
                                    last_click_was_left
                                    and gap < self.DOUBLE_CLICK_GAP
                                ):
                                    self._click_mouse(double=True)
                                    self._set_state(
                                        last_action="double-click",
                                        last_action_at=now,
                                    )
                                    last_click_at = 0.0
                                else:
                                    self._click_mouse(left=True)
                                    self._set_state(
                                        last_action="left-click",
                                        last_action_at=now,
                                    )
                                    last_click_was_left = True
                                    last_click_at = now
                        # Consume this pinch so a held gesture does not
                        # fire repeatedly.
                        pinch_started = now - 1000

                # --- 3. Drag-scroll. While dragging, vertical motion of
                # the index finger scrolls the underlying window. This
                # makes "drag a window" / "drag in a 3D viewport" / "drag
                # a slider" feel native.
                if drag_started and pinch_held and prev_index_y is not None:
                    dy2 = prev_index_y - index[1]
                    if abs(dy2) > 3:
                        ticks = int(dy2 / self.SCROLL_SENSITIVITY)
                        if ticks != 0:
                            self._scroll_mouse(ticks)
                            self._set_state(
                                last_action="drag-scroll",
                                last_action_at=now,
                            )
                prev_index_y = index[1] if pinch_held else None

                # --- 4. Closed fist = click. Dropout-tolerant + hysteresis.
                fingers_up = self._fingers_up(landmarks, h)
                index_up, middle_up, ring_up, pinky_up = fingers_up
                # Use hysteresis: it's easier to *enter* a fist than to
                # stay detected as one. A 0.015*frame-height padding on
                # release keeps the gesture from flickering.
                fist_thresh = 0.02 * h
                fist_release = fist_thresh + self.FIST_HYSTERESIS * h
                # Treat each finger as curled when its tip is below its
                # PIP by `fist_thresh` (on) or `fist_release` (off).
                def _finger_curled(tip_idx: int, pip_idx: int, threshold: float) -> bool:
                    return (landmarks[tip_idx][1] - landmarks[pip_idx][1]) > threshold
                index_curled = _finger_curled(8, 6, fist_thresh if not index_up else fist_release)
                middle_curled = _finger_curled(12, 10, fist_thresh if not middle_up else fist_release)
                ring_curled = _finger_curled(16, 14, fist_thresh if not ring_up else fist_release)
                pinky_curled = _finger_curled(20, 18, fist_thresh if not pinky_up else fist_release)
                is_fist = index_curled and middle_curled and ring_curled and pinky_curled
                # We need to refresh ``fingers_up`` so the scroll block
                # below sees the same values.
                index_up = not index_curled
                middle_up = not middle_curled
                ring_up = not ring_curled
                pinky_up = not pinky_curled

                with self._state_lock:
                    fist_state = self._fist_state
                fist_active, fist_started_at, fist_cooldown_until = fist_state
                if is_fist:
                    if not fist_active:
                        fist_active = True
                        fist_started_at = now
                    elif (
                        now >= fist_cooldown_until
                        and (now - fist_started_at) >= self.FIST_HOLD
                    ):
                        mid_dist = math.hypot(
                            thumb[0] - middle[0], thumb[1] - middle[1]
                        )
                        is_two_finger = mid_dist < 0.05 * w * 1.3
                        with self._click_lock:
                            if is_two_finger:
                                self._click_mouse(right=True)
                                self._set_state(
                                    last_action="fist-right-click",
                                    last_action_at=now,
                                )
                                last_click_was_left = False
                            else:
                                gap = now - last_click_at
                                if (
                                    last_click_was_left
                                    and gap < self.DOUBLE_CLICK_GAP
                                ):
                                    self._click_mouse(double=True)
                                    self._set_state(
                                        last_action="fist-double-click",
                                        last_action_at=now,
                                    )
                                    last_click_at = 0.0
                                else:
                                    self._click_mouse(left=True)
                                    self._set_state(
                                        last_action="fist-click",
                                        last_action_at=now,
                                    )
                                    last_click_was_left = True
                                    last_click_at = now
                        fist_cooldown_until = now + 0.35
                else:
                    if (
                        fist_active
                        and (now - fist_started_at) >= self.FIST_RELEASE
                    ):
                        fist_active = False
                        fist_started_at = 0.0
                self._fist_state = (
                    fist_active, fist_started_at, fist_cooldown_until,
                )

                # --- 5. Two-finger gestures: zoom + scroll. ---------------
                # Holding index + middle up (no pinch, no drag) is the
                # "trackpad" gesture. Horizontal spread change zooms
                # (Ctrl+scroll); vertical movement scrolls. Both run
                # from the same frame so the response is immediate.
                if (
                    index_up and middle_up and not ring_up and not pinky_up
                    and not pinch_held and not drag_started
                ):
                    spread = math.hypot(
                        index[0] - middle[0], index[1] - middle[1]
                    )
                    if prev_two_finger_spread is not None:
                        dspread = spread - prev_two_finger_spread
                        if abs(dspread) > 3:
                            ticks = int(dspread / self.ZOOM_SENSITIVITY)
                            if ticks != 0:
                                # Positive spread = zoom in. ctrl+scroll
                                # up is positive, so the sign is right.
                                self._zoom_mouse(ticks)
                                self._set_state(
                                    last_action=(
                                        "zoom-in" if ticks > 0 else "zoom-out"
                                    ),
                                    last_action_at=now,
                                )
                    prev_two_finger_spread = spread
                    if prev_index_y is not None:
                        dy2 = prev_index_y - index[1]
                        if abs(dy2) > 5:
                            ticks = int(dy2 / self.SCROLL_SENSITIVITY)
                            if ticks != 0:
                                self._scroll_mouse(ticks)
                                self._set_state(
                                    last_action=(
                                        "scroll" if ticks > 0 else "scroll-down"
                                    ),
                                    last_action_at=now,
                                )
                    prev_index_y = index[1]
                else:
                    prev_two_finger_spread = None
                    prev_index_y = None

        except Exception as exc:
            log.exception("Hand mouse loop crashed: %s", exc)
            self._set_state(last_error=f"failed to start: {exc}")
        finally:
            # If we ended mid-drag, release the mouse button.
            if dragging and HAS_PYAUTOGUI:
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass
            if legacy_hands is not None:
                try:
                    legacy_hands.close()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._set_state(running=False)
            log.info("Hand mouse loop stopped.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _open_camera(self):
        """Try the configured webcam indexes and return an open capture."""
        for idx in self.CAMERA_PROBE_INDEXES:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Confirm the camera actually yields frames; some
                    # indexes open successfully but return empty buffers.
                    ok, _ = cap.read()
                    if ok:
                        log.info("Opened webcam at index %d.", idx)
                        return cap
                cap.release()
            except Exception as exc:
                log.debug("Camera index %d probe failed: %s", idx, exc)
        self._set_state(last_error=(
            "could not open any webcam (tried indexes "
            f"{list(self.CAMERA_PROBE_INDEXES)}). "
            "Close other apps using the camera and try again."
        ))
        return None

    def _screen_size(self) -> Tuple[int, int]:
        if HAS_PYAUTOGUI:  # pragma: no cover - depends on env
            try:
                size = pyautogui.size()
                return int(size.width), int(size.height)
            except Exception:
                pass
        return 1920, 1080

    def _move_mouse(self, x: float, y: float) -> None:
        if not HAS_PYAUTOGUI:  # pragma: no cover
            return
        now = time.time()
        if now - self._last_move_at < self.MOVE_COOLDOWN:
            return
        self._last_move_at = now
        try:
            pyautogui.moveTo(int(x), int(y), duration=0)
        except Exception as exc:
            log.debug("pyautogui.moveTo failed: %s", exc)

    def _click_mouse(self, *, left: bool = False, right: bool = False, double: bool = False) -> None:
        if not HAS_PYAUTOGUI:  # pragma: no cover
            return
        try:
            if double:
                pyautogui.doubleClick()
            elif right:
                pyautogui.rightClick()
            elif left:
                pyautogui.click()
        except Exception as exc:
            log.debug("pyautogui click failed: %s", exc)

    def _scroll_mouse(self, ticks: int) -> None:
        if not HAS_PYAUTOGUI:  # pragma: no cover
            return
        try:
            pyautogui.scroll(int(ticks))
        except Exception as exc:
            log.debug("pyautogui.scroll failed: %s", exc)

    def _zoom_mouse(self, ticks: int) -> None:
        """Send a Ctrl+wheel event so the foreground app zooms.

        On Windows and most browsers / editors, ``Ctrl+scroll`` is the
        universal zoom shortcut. We use :func:`pyautogui.hscroll` /
        ``scroll`` with the ``ctrl`` modifier held down. Falls back to
        plain scroll if the hotkey API is missing.
        """
        if not HAS_PYAUTOGUI:  # pragma: no cover
            return
        try:
            # ``pyautogui`` does not expose a direct "scroll with
            # modifier" API, but ``hotkey`` + ``scroll`` works on all
            # supported platforms. We keyDown ``ctrl`` once, do all the
            # scroll work, then keyUp - the key stays held for a few
            # frames but no scroll event in between can leak through
            # because we use the explicit scroll() call.
            pyautogui.keyDown("ctrl")
            try:
                pyautogui.scroll(int(ticks))
            finally:
                pyautogui.keyUp("ctrl")
        except Exception as exc:
            log.debug("pyautogui zoom failed: %s", exc)

    def _detect(self, frame, landmarker, legacy_hands, video_ts_ms):
        """Dispatch to whichever backend is configured.

        Returns ``(hands, count)`` where ``hands`` is a list of
        ``(coords, handedness)`` tuples in image-pixel coordinates. The
        caller indexes ``hands[0]`` for the single-hand code path and
        ``hands[0] / hands[1]`` for the two-hand zoom gesture. Returns
        ``(None, 0)`` when no hand is in view.
        """
        if landmarker is not None:
            return self._detect_modern(frame, landmarker, video_ts_ms)
        if legacy_hands is not None:
            return self._detect_legacy(frame, legacy_hands)
        return self._detect_opencv(frame)

    def _detect_modern(self, frame, landmarker, video_ts_ms):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = _modern_mp_image(
                image_format=_modern_mp_image_format.SRGB, data=rgb
            )
            result = landmarker.detect_for_video(mp_image, video_ts_ms)
        except Exception as exc:
            log.debug("Modern MediaPipe detect failed: %s", exc)
            return None, 0
        if not result.hand_landmarks:
            return None, 0
        h, w = frame.shape[:2]
        hands = []
        for idx, hand in enumerate(result.hand_landmarks):
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            handedness = None
            if result.handedness and idx < len(result.handedness):
                try:
                    handedness = result.handedness[idx][0].category_name
                except Exception:
                    handedness = None
            hands.append((coords, handedness))
        return hands, len(hands)

    def _detect_legacy(self, frame, hands):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
        except Exception as exc:
            log.debug("Legacy MediaPipe detect failed: %s", exc)
            return None, 0
        if not results.multi_hand_landmarks:
            return None, 0
        h, w = frame.shape[:2]
        out = []
        for idx, hand in enumerate(results.multi_hand_landmarks):
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
            label = None
            if results.multi_handedness and idx < len(results.multi_handedness):
                try:
                    label = results.multi_handedness[idx].classification[0].label
                except Exception:
                    label = None
            out.append((coords, label))
        return out, len(out)

    def _detect_opencv(self, frame):
        """Fallback tracker: skin-colour contour + convex-hull fingertip.

        The mediapipe backend is far more accurate, but this gets the
        feature working on a stock install. The contour-based "fingertip"
        detection is intentionally simple - move the cursor with the top
        of the largest contour, click with a quick pinch of the contour
        bounding box (width collapses).
        """
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Wide skin range so the fallback works across complexions
            # without per-user calibration.
            lower = (0, 30, 60)
            upper = (25, 200, 255)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 3000:
                return None, None
            # Convex-hull defects give rough finger-tip positions.
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is None or len(hull) < 3:
                return None, None
            defects = cv2.convexityDefects(contour, hull)
            tips: list = []
            if defects is not None:
                for d in defects:
                    s, e, f, depth = d[0]
                    if depth / 256.0 > 12:
                        tip = tuple(contour[e][0])
                        if tip not in tips:
                            tips.append(tip)
            if not tips:
                # Fall back to the topmost point of the contour so the
                # cursor still moves when no defects are detected.
                topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                tips = [topmost]
            # Use the topmost detected fingertip as the index finger.
            tips.sort(key=lambda p: p[1])
            index = tips[0]
            # Approximate the thumb as the leftmost fingertip when there
            # is more than one, else use the contour bounding box.
            if len(tips) > 1:
                thumb = min(tips, key=lambda p: p[0])
            else:
                x, y, w_, _ = cv2.boundingRect(contour)
                thumb = (x, y)
            # Pad to 21 landmark-like entries so the gesture math in the
            # main loop keeps working.
            pseudo: list = [(0, 0)] * 21
            pseudo[4] = (int(thumb[0]), int(thumb[1]))
            pseudo[8] = (int(index[0]), int(index[1]))
            # Use the contour centroid for the remaining landmarks so
            # the "fingers up" check always returns a sensible answer.
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = index
            for idx in (12, 16, 20):
                pseudo[idx] = (cx, cy)
            # Two-hand zoom requires both hands, which the OpenCV
            # contour fallback cannot reliably detect. We return a
            # single-hand tuple so the existing single-hand code path
            # keeps working; the zoom block is simply a no-op when
            # hand_count < 2.
            return [(pseudo, "Right")], 1
        except Exception as exc:
            log.debug("OpenCV fallback tracker failed: %s", exc)
            return None, 0

    def _fingers_up(self, landmarks, frame_h: int) -> Tuple[bool, bool, bool, bool]:
        """Return (index, middle, ring, pinky) extended booleans.

        A finger is considered "up" if its tip is meaningfully above the
        PIP joint. We compare y coordinates - in image coordinates smaller
        y = higher on screen.
        """
        try:
            tips = (8, 12, 16, 20)
            pips = (6, 10, 14, 18)
            out = []
            for tip, pip in zip(tips, pips):
                out.append(landmarks[tip][1] < landmarks[pip][1] - 0.02 * frame_h)
            return tuple(out)  # type: ignore[return-value]
        except Exception:
            return False, False, False, False


# ---------------------------------------------------------------------------
# Public tool wrapper
# ---------------------------------------------------------------------------
class HandMouseTool(BaseTool):
    """Voice entry point for the hand-mouse control loop."""

    name = "hand_mouse_tool"
    description = (
        "Control the mouse using hand gestures captured from the webcam. "
        "Use 'start hand mouse' to enable, 'stop hand mouse' to disable. "
        "Gestures (single hand): index finger moves the cursor (with "
        "adaptive smoothing - the cursor settles when your hand is "
        "still), pinch = left click, pinch with index + middle = right "
        "click, sustained pinch (held > 0.45s) = drag-and-drop, closed "
        "fist (all fingers curled) = left click, closed fist with thumb "
        "over middle = right click, two fingers up = scroll (vertical) "
        "+ zoom (Ctrl+scroll, by spread). "
        "Two-hand gesture: pinch with BOTH hands' index+thumb, then "
        "pull them apart / together to zoom the foreground app "
        "(Ctrl+wheel: apart = zoom in, together = zoom out). The "
        "two-hand gesture needs both pinches to be held for "
        "~0.12s before zooming starts so an accidental mid-click "
        "pinch does not fire a zoom. All gestures use hysteresis and "
        "dropout-tolerance so a single noisy frame does not cancel an "
        "in-progress action."
    )

    def __init__(self) -> None:
        self._controller = _HandMouseController()

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if not text:
            return False
        if _contains_any(text, _START_PHRASES):
            return True
        if _contains_any(text, _STOP_PHRASES):
            return True
        if _contains_any(text, _STATUS_PHRASES):
            return True
        if "hand mouse" in text or "hand tracking" in text:
            return True
        return False

    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()
        try:
            if _contains_any(lowered, _STOP_PHRASES):
                ok, msg = self._controller.stop()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            if _contains_any(lowered, _STATUS_PHRASES):
                status = self._controller.status()
                if status["running"]:
                    return ToolResult(
                        success=True,
                        message=(
                            f"Hand mouse is running, sir. "
                            f"Backend: {status['backend']}. "
                            f"Frames seen: {status['frames_seen']}."
                        ),
                        data=status,
                    )
                return ToolResult(
                    success=True,
                    message="Hand mouse is currently off, sir.",
                    data=status,
                )

            if _contains_any(lowered, _START_PHRASES):
                ok, msg = self._controller.start()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            # Bare "hand mouse" - toggle.
            if "hand mouse" in lowered or "hand tracking" in lowered:
                if self._controller.running:
                    ok, msg = self._controller.stop()
                else:
                    ok, msg = self._controller.start()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            return ToolResult(
                success=False,
                message="I did not understand that hand-mouse command, sir.",
            )
        except Exception as exc:
            log.exception("HandMouseTool failed: %s", exc)
            return ToolResult(success=False, message=f"Hand mouse error: {exc}")

    # ------------------------------------------------------------------
    # Public helpers used by the web API
    # ------------------------------------------------------------------
    def start(self) -> ToolResult:
        ok, msg = self._controller.start()
        return ToolResult(success=ok, message=msg, data=self._controller.status())

    def stop(self) -> ToolResult:
        ok, msg = self._controller.stop()
        return ToolResult(success=ok, message=msg, data=self._controller.status())

    def status(self) -> Dict[str, Any]:
        return self._controller.status()


# ----------------------------------------------------------------------
# Registration helper
# ----------------------------------------------------------------------
def register_hand_mouse_tools(router) -> list:
    """Register the hand-mouse tool with *router* and return the new tools.

    The deterministic router runs the tool *before* the LLM, so phrases
    like "start hand mouse" never reach the agent executor. The same tool
    instance is also exposed to the web API so the dashboard can drive
    it directly.
    """
    tool = HandMouseTool()
    router.register(
        tool,
        keywords=(
            "hand mouse",
            "hand tracking",
            "control my mouse with my hand",
            "control the mouse with my hand",
            "start hand mouse",
            "stop hand mouse",
        ),
        priority=80,
    )
    return [tool]


__all__ = [
    "HandMouseTool",
    "register_hand_mouse_tools",
    "HAS_CV2",
    "HAS_MEDIAPIPE",
    "HAS_PYAUTOGUI",
]
