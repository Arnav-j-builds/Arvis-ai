"""
vision.eye_mouse
~~~~~~~~~~~~~~~~

A :class:`BaseTool` that lets the user control the mouse with eye gaze
captured from the local webcam.

Architecture
------------
The tool exposes four intents:

* "start eye mouse"   - launch a background control loop that opens the
  webcam, locates the user's iris with the MediaPipe iris model, and
  converts the gaze direction into real mouse events.
* "stop eye mouse"    - stop the loop cleanly and release the camera.
* "eye mouse status"  - report the current state (running, backend, etc.).
* "calibrate eye mouse" - reset calibration. The controller self-calibrates
  on first use (recording the current gaze as "center"); saying
  "calibrate eye mouse" re-records the center so the user can move their
  head or change camera position.

Gesture model
-------------
* Looking around moves the cursor. The eye-direction is mapped to screen
  coordinates using a small "active area" in the camera frame, exactly
  like the ``Control_pc_using_Hand-Gesture`` project (``MouseController``)
  but driven by iris landmarks rather than the index fingertip.
* Blinking (eyes closed for >=0.25 s) triggers a left click. The blink
  must release for >=0.2 s before the next click can fire.
* Two quick blinks (within 0.45 s) trigger a double-click.
* Both eyes closed for >=0.4 s performs a right-click (useful when you
  only want to drag the cursor without firing a click).
* Looking far up then back down scrolls up; far down then back up
  scrolls down.

The control loop runs in a daemon thread. It is opt-in - nothing starts
until the user explicitly says the trigger phrase. The tool never crashes
the rest of the assistant: every import / runtime error is translated
into a friendly :class:`ToolResult`.

Dependencies (all optional at import time)
------------------------------------------
* ``opencv-python`` - webcam capture + drawing helpers. Required.
* ``mediapipe``     - iris tracking. Supports the modern 1.0+
  ``mediapipe.tasks.vision.FaceLandmarker`` API (we use the iris model
  embedded in the face landmarker; the iris-only model is not available
  standalone from Google's CDN). When missing the tool reports a clear
  install error.
* ``pyautogui``     - actual mouse event injection. When missing the
  tool reports the gaze but does not move the real cursor.

The MediaPipe Tasks API needs a ``.task`` model file. We download it on
first use into the project's ``storage/`` directory and reuse it on every
subsequent start. No network access is required after the first launch.
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
from core.config import get_config
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


cv2 = _try_import("cv2")
mediapipe = _try_import("mediapipe")
pyautogui = _try_import("pyautogui")

HAS_CV2 = cv2 is not None
HAS_PYAUTOGUI = pyautogui is not None

# Probe the modern MediaPipe API (same dance as hand_mouse.py).
_HAS_MODERN_MEDIAPIPE = False
_modern_mp_face_landmarker = None
_modern_mp_options = None
_modern_mp_image = None
_modern_mp_image_format = None
_modern_mp_running_mode = None
_modern_mp_base_options = None
if mediapipe is not None and hasattr(mediapipe, "tasks"):
    try:
        from mediapipe.tasks.python.vision import (
            FaceLandmarker as _FaceLandmarker,
            FaceLandmarkerOptions as _FaceLandmarkerOptions,
            RunningMode as _RunningMode,
        )
        from mediapipe.tasks.python.vision.core import image as _image_mod
        from mediapipe.tasks.python.core import base_options as _bo_mod

        _modern_mp_face_landmarker = _FaceLandmarker
        _modern_mp_options = _FaceLandmarkerOptions
        _modern_mp_image = _image_mod.Image
        _modern_mp_image_format = _image_mod.ImageFormat
        _modern_mp_running_mode = _RunningMode
        _modern_mp_base_options = _bo_mod.BaseOptions
        _HAS_MODERN_MEDIAPIPE = True
    except Exception as exc:
        log.debug("Conventional MediaPipe import failed: %s", exc)
        # Fallback: direct file-path imports for broken 1.0.x wheels.
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
                "tasks/python/vision/face_landmarker.py",
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
                _modern_mp_face_landmarker = _vision_mod.FaceLandmarker
                _modern_mp_options = _vision_mod.FaceLandmarkerOptions
                _modern_mp_running_mode = _vision_mod.RunningMode
                _modern_mp_image = _image_mod.Image
                _modern_mp_image_format = _image_mod.ImageFormat
                _modern_mp_base_options = _bo_mod.BaseOptions
                _HAS_MODERN_MEDIAPIPE = True
        except Exception as exc2:  # pragma: no cover - defensive
            log.debug("Direct MediaPipe import also failed: %s", exc2)

HAS_MEDIAPIPE = _HAS_MODERN_MEDIAPIPE

# Legacy ``mediapipe.solutions.face_mesh`` - available on the common
# 0.10.x line, no .task file needed, iris landmarks built-in (468..477).
# We probe it as a fallback so the vision mouse works on wheels that do
# not expose the new ``FaceLandmarker`` Tasks API.
_HAS_LEGACY_FACE_MESH = (
    mediapipe is not None
    and hasattr(mediapipe, "solutions")
    and hasattr(mediapipe.solutions, "face_mesh")
)
_legacy_face_mesh = None
_legacy_face_mesh_module = None
if _HAS_LEGACY_FACE_MESH:
    try:
        _legacy_face_mesh_module = mediapipe.solutions.face_mesh
        log.debug("Legacy MediaPipe face_mesh is available.")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("Could not import legacy face_mesh: %s", exc)
        _HAS_LEGACY_FACE_MESH = False

if HAS_PYAUTOGUI:  # pragma: no cover - depends on env
    try:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Phrase matching
#
# The user-facing phrasing is "vision mouse" / "vision tracking" - this
# stops Google Speech from mis-hearing it as "I tracking". The old "eye"
# phrases are kept as fallbacks so anyone who already wired them up
# keeps working.
# ---------------------------------------------------------------------------
_START_PHRASES = (
    "start vision mouse",
    "start the vision mouse",
    "start vision tracking",
    "enable vision mouse",
    "enable vision tracking",
    "control my mouse with my vision",
    "control the mouse with my vision",
    "control mouse with vision",
    "vision mouse on",
    "vision tracking on",
    "begin vision tracking",
    "begin vision mouse",
    # Eye-based fallbacks.
    "start eye mouse",
    "start the eye mouse",
    "start eye tracking",
    "enable eye mouse",
    "enable eye tracking",
    "control my mouse with my eye",
    "control the mouse with my eye",
    "control mouse with my eye",
    "eye mouse on",
    "eye tracking on",
    "begin eye tracking",
    "begin eye mouse",
)

_STOP_PHRASES = (
    "stop vision mouse",
    "stop the vision mouse",
    "stop vision tracking",
    "disable vision mouse",
    "disable vision tracking",
    "vision mouse off",
    "vision tracking off",
    # Eye-based fallbacks.
    "stop eye mouse",
    "stop the eye mouse",
    "stop eye tracking",
    "disable eye mouse",
    "disable eye tracking",
    "eye mouse off",
    "eye tracking off",
)

_STATUS_PHRASES = (
    "is vision mouse running",
    "is vision mouse on",
    "vision mouse status",
    "vision tracking status",
    # Eye-based fallbacks.
    "is eye mouse running",
    "is eye mouse on",
    "eye mouse status",
    "eye tracking status",
)

_CALIBRATE_PHRASES = (
    "calibrate vision mouse",
    "calibrate vision tracking",
    "reset vision mouse",
    "reset vision tracking",
    "recalibrate vision mouse",
    # Eye-based fallbacks.
    "calibrate eye mouse",
    "calibrate eye tracking",
    "reset eye mouse",
    "reset eye tracking",
    "recalibrate eye mouse",
)


def _contains_any(text: str, needles: tuple) -> bool:
    lowered = (text or "").lower()
    return any(needle in lowered for needle in needles)


# ---------------------------------------------------------------------------
# Model download for the modern MediaPipe API
# ---------------------------------------------------------------------------
_FACE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_TASK_FILENAME = "face_landmarker.task"


def _ensure_face_model() -> Optional[str]:
    """Download the FaceLandmarker ``.task`` file if it is not cached.

    The file is stored in ``storage/`` so subsequent launches do not need
    a network connection. Returns the absolute path to the model file or
    ``None`` when the download failed.
    """
    try:
        cfg = get_config()
        target = cfg.storage_dir / _FACE_TASK_FILENAME
        if target.exists() and target.stat().st_size > 100_000:
            return str(target)
        cfg.storage_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading MediaPipe face_landmarker.task model...")
        req = urllib.request.Request(_FACE_TASK_URL, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        target.write_bytes(data)
        log.info("Downloaded face_landmarker.task (%d bytes).", len(data))
        return str(target)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        log.warning("Could not download face_landmarker.task: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unexpected error downloading face_landmarker.task: %s", exc)
        return None


# ---------------------------------------------------------------------------
# The control loop
# ---------------------------------------------------------------------------
class _EyeMouseController:
    """Background controller that maps iris gaze to mouse events.

    MediaPipe's face landmarker returns 478 face landmarks. Indices
    468..472 are the left iris (5 points) and 473..477 are the right
    iris. We average the iris center across both eyes and use the ratio
    between the iris and the eye-bounding box to estimate gaze
    direction.
    """

    SMOOTHING = 7            # higher = smoother but laggier
    FINE_SMOOTHING = 16
    FINE_PIXELS = 6
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    MARGIN = 120             # deadzone in pixels around the frame edge
    # Gaze gain: the iris-in-eye-socket ratio is in [0, 1] and the
    # "comfortable eye-rotation range" is roughly the middle 30-40%
    # of that. ``GAZE_GAIN`` amplifies it so a comfortable look to
    # the corner of the screen reaches the corner of the monitor
    # (rather than only the centre). Bump this if the cursor feels
    # stuck in the middle of the screen; lower it if it shoots past
    # the edge.
    GAZE_GAIN = 4.0
    BLINK_HOLD = 0.25        # seconds eyes closed before a click fires
    BLINK_RELEASE = 0.20     # seconds eyes open between clicks
    DOUBLE_BLINK_GAP = 0.45  # seconds between two blinks = double-click
    BOTH_EYES_HOLD = 0.40    # seconds both eyes closed = right click
    SCROLL_THRESHOLD = 0.15  # gaze-offset to trigger scroll
    SCROLL_COOLDOWN = 0.6    # seconds between scroll events
    CAMERA_PROBE_INDEXES = (0, 1, 2)

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self.running: bool = False
        self.started_at: float = 0.0
        self.frames_seen: int = 0
        self.last_action: str = "idle"
        self.last_action_at: float = 0.0
        self.last_error: str = ""
        self.backend: str = "none"
        # Calibration: the gaze ratio at startup is treated as the
        # center of the screen. We average the first few frames' gaze
        # so the user does not have to hold their head perfectly still
        # while calibration is captured. ``recalibrate`` re-records this
        # baseline.
        self._center_x: float = 0.5
        self._center_y: float = 0.5
        self._calibrated: bool = False
        # Calibration sample buffer (gaze ratios from the first ~10
        # frames after start / recalibrate, used to compute the
        # average baseline).
        self._calib_samples: list = []
        self._calib_target: int = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> Tuple[bool, str]:
        if self.running:
            return True, "Vision mouse is already running, sir."
        if not HAS_CV2:
            return False, (
                "I cannot start the vision mouse, sir. OpenCV is not installed. "
                "Install opencv-python to use vision tracking."
            )
        if not _HAS_MODERN_MEDIAPIPE and not _HAS_LEGACY_FACE_MESH:
            return False, (
                "I cannot start the vision mouse, sir. The MediaPipe library "
                "is not installed (or is too old to expose the FaceLandmarker "
                "task or the legacy face_mesh module). Install "
                "``mediapipe>=0.10.14`` to use vision tracking."
            )
        if not HAS_PYAUTOGUI:
            log.warning("pyautogui missing - cursor will not move.")
        self._stop.clear()
        with self._state_lock:
            self.running = False
            self.started_at = 0.0
            self.frames_seen = 0
            self.last_action = "idle"
            self.last_error = ""
            self.backend = "none"
            self._calibrated = False
            self._center_x = 0.5
            self._center_y = 0.5
            self._calib_samples = []
        self._thread = threading.Thread(
            target=self._run, name="vision-mouse", daemon=True
        )
        self._thread.start()
        # Poll for up to ~8 seconds; the first run downloads the model.
        deadline = time.time() + 8.0
        while time.time() < deadline:
            time.sleep(0.1)
            if self.running:
                return True, (
                    "Vision mouse is now active, sir. Look around to move the cursor "
                    "and blink to click."
                )
            if self.last_error and "failed to start" in self.last_error:
                break
        if self.running:
            return True, (
                "Vision mouse is now active, sir. Look around to move the cursor "
                "and blink to click."
            )
        err = self.last_error or "the camera failed to open."
        if self._thread is not None and not self._thread.is_alive() and self.frames_seen == 0:
            return False, f"Vision mouse did not start: {err}"
        return True, (
            "Vision mouse is starting up, sir. Give it a moment and look at the screen."
        )

    def stop(self) -> Tuple[bool, str]:
        if not self.running:
            return True, "Vision mouse is already off, sir."
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.5)
        self._thread = None
        self.running = False
        return True, "Vision mouse is now off, sir."

    def recalibrate(self) -> Tuple[bool, str]:
        if not self.running:
            return False, (
                "Vision mouse is not running, sir - I cannot recalibrate. "
                "Start the vision mouse first."
            )
        with self._state_lock:
            self._calibrated = False  # reset sample buffer next frame
            self._calib_samples = []
        return True, (
            "Recalibrating - please look at the screen centre, sir. "
            "I will average the next second of gaze samples."
        )

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "backend": self.backend,
            "frames_seen": self.frames_seen,
            "last_action": self.last_action,
            "last_action_at": self.last_action_at,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "calibrated": self._calibrated,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_state(self, **kwargs) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def _screen_size(self) -> Tuple[int, int]:
        if HAS_PYAUTOGUI:
            try:
                w, h = pyautogui.size()
                return int(w), int(h)
            except Exception:
                pass
        # Fallback - 1080p on a 100% DPI Windows desktop.
        return 1920, 1080

    def _move_mouse(self, x: float, y: float) -> None:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.moveTo(x, y)
            except Exception:
                pass

    def _click(self) -> None:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.click()
            except Exception:
                pass

    def _double_click(self) -> None:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.doubleClick()
            except Exception:
                pass

    def _right_click(self) -> None:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.rightClick()
            except Exception:
                pass

    def _scroll(self, clicks: int) -> None:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.scroll(clicks)
            except Exception:
                pass

    def _open_camera(self) -> Optional[Any]:
        if not HAS_CV2:
            return None
        last_exc: Optional[Exception] = None
        for idx in self.CAMERA_PROBE_INDEXES:
            cap = cv2.VideoCapture(idx)
            if cap is None or not cap.isOpened():
                last_exc = Exception(f"camera index {idx} did not open")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue
            # Confirm we can actually pull a frame.
            ok, _ = cap.read()
            if not ok:
                try:
                    cap.release()
                except Exception:
                    pass
                last_exc = Exception(f"camera index {idx} returned no frame")
                continue
            return cap
        self._set_state(last_error=(
            f"could not open any webcam (tried indexes "
            f"{self.CAMERA_PROBE_INDEXES}): {last_exc}"
        ))
        return None

    # ------------------------------------------------------------------
    # Iris / blink math
    # ------------------------------------------------------------------
    # MediaPipe face-mesh landmark indices for the eye sockets.
    # These are the canonical indices from the FaceMesh topology.
    # Left eye corners:  33 (inner) and 133 (outer).
    # Right eye corners: 362 (inner) and 263 (outer).
    # Left iris:  468..472 (5 points around the iris circle).
    # Right iris: 473..477 (same).
    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (362, 263)
    LEFT_IRIS = (468, 469, 470, 471, 472)
    RIGHT_IRIS = (473, 474, 475, 476, 477)
    # All iris indices - used by ``_iris_position_in_eye``.
    ALL_IRIS = LEFT_IRIS + RIGHT_IRIS

    @staticmethod
    def _iris_centers_by_eye(landmarks) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return the iris centroid for each eye in normalized 0..1 coords.

        Returns ``(left_iris, right_iris)`` or ``None`` when the model
        did not return iris landmarks.
        """
        try:
            def _centroid(indices):
                xs = [landmarks[i].x for i in indices if i < len(landmarks)]
                ys = [landmarks[i].y for i in indices if i < len(landmarks)]
                if not xs:
                    return None
                return (sum(xs) / len(xs), sum(ys) / len(ys))

            left = _centroid(_EyeMouseController.LEFT_IRIS)
            right = _centroid(_EyeMouseController.RIGHT_IRIS)
            if left is None or right is None:
                return None
            return (left, right)
        except Exception:
            return None

    @staticmethod
    def _eye_bounding_box(landmarks, indices) -> Optional[Tuple[float, float, float, float]]:
        """Return ``(min_x, min_y, max_x, max_y)`` for the given eye corners.

        The eye "socket" is approximated by its inner/outer corners -
        a small box - because MediaPipe does not expose a full eyelid
        contour that survives a blink.
        """
        try:
            xs = [landmarks[i].x for i in indices if i < len(landmarks)]
            ys = [landmarks[i].y for i in indices if i < len(landmarks)]
            if not xs:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            return None

    @staticmethod
    def _gaze_ratio(landmarks) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return ``(left, right)`` gaze ratios for both eyes.

        Each ratio is ``(x, y)`` in [0, 1] x [0, 1] describing where the
        iris sits **inside the eye's corner-to-corner box**. ``(0, 0)``
        means the iris is on the inner-corner side of the eye socket
        (looking left), ``(1, 1)`` means outer-corner + bottom (looking
        right-and-down), and ``(0.5, 0.5)`` means straight ahead.

        This is the key fix for "tracking the head instead of the eyes":
        head motion shifts the whole face equally, but the iris stays
        at the same ratio inside the eye socket, so the gaze value is
        invariant to head translation.

        Returns ``None`` when iris landmarks are missing.
        """
        try:
            iris_centers = _EyeMouseController._iris_centers_by_eye(landmarks)
            if iris_centers is None:
                return None
            left_iris, right_iris = iris_centers

            def _ratio(iris_pt, corner_indices):
                box = _EyeMouseController._eye_bounding_box(landmarks, corner_indices)
                if box is None:
                    return None
                min_x, min_y, max_x, max_y = box
                w = max_x - min_x
                h = max_y - min_y
                if w <= 0 or h <= 0:
                    return None
                rx = (iris_pt[0] - min_x) / w
                ry = (iris_pt[1] - min_y) / h
                # Clamp to [0, 1] so a tiny iris offset outside the
                # corner box does not generate a huge gaze value.
                return (
                    max(0.0, min(1.0, rx)),
                    max(0.0, min(1.0, ry)),
                )

            left_ratio = _ratio(left_iris, _EyeMouseController.LEFT_EYE_CORNERS)
            right_ratio = _ratio(right_iris, _EyeMouseController.RIGHT_EYE_CORNERS)
            if left_ratio is None or right_ratio is None:
                return None
            return (left_ratio, right_ratio)
        except Exception:
            return None

    @staticmethod
    def _eye_aspect(landmarks, indices: Tuple[int, ...]) -> float:
        """Estimate eye-openness from a set of eye landmarks.

        ``indices`` should be 6 points describing an eye (two corners
        plus top and bottom on two axes). We compute (vertical span) /
        (horizontal span); lower ratio = more closed.
        """
        try:
            pts = [(landmarks[i].x, landmarks[i].y) for i in indices if i < len(landmarks)]
            if len(pts) < 6:
                return 1.0
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            v = max(ys) - min(ys)
            h = max(xs) - min(xs)
            if h <= 0:
                return 1.0
            return float(v / h)
        except Exception:
            return 1.0

    def _blink_state(
        self,
        landmarks,
        left_blink_started: Optional[float],
        right_blink_started: Optional[float],
        last_click_at: float,
        last_click_was_left: bool,
        now: float,
    ):
        """Detect left-blink, right-blink (double), and both-eyes-closed.

        Returns a tuple of updated blink state and a list of pending
        actions (``"click"``, ``"doubleclick"``, ``"rightclick"``).
        """
        actions: list = []
        # Standard MediaPipe face mesh eye indices.
        LEFT = (33, 160, 158, 133, 153, 144)
        RIGHT = (362, 385, 387, 263, 373, 380)
        left_open = self._eye_aspect(landmarks, LEFT) > 0.18
        right_open = self._eye_aspect(landmarks, RIGHT) > 0.18

        # Single-eye blink -> click (with double-click detection).
        single_closed = (not left_open) ^ (not right_open)
        both_closed = (not left_open) and (not right_open)

        # Both eyes closed long enough -> right-click.
        if both_closed:
            if right_blink_started is None:
                right_blink_started = now
            elif (now - right_blink_started) >= self.BOTH_EYES_HOLD and (
                now - last_click_at
            ) >= self.BLINK_RELEASE:
                actions.append("rightclick")
                last_click_at = now
                right_blink_started = now + 1.0  # suppress further right-clicks
        else:
            right_blink_started = None

        # Single-eye blink long enough -> click.
        if single_closed and not both_closed:
            if left_blink_started is None:
                left_blink_started = now
            elif (now - left_blink_started) >= self.BLINK_HOLD and (
                now - last_click_at
            ) >= self.BLINK_RELEASE:
                if last_click_was_left and (now - last_click_at) <= self.DOUBLE_BLINK_GAP:
                    actions.append("doubleclick")
                    last_click_at = now
                    left_blink_started = now + 1.0
                    last_click_was_left = False
                else:
                    actions.append("click")
                    last_click_at = now
                    left_blink_started = now + 1.0
                    last_click_was_left = True
        else:
            left_blink_started = None

        return (
            left_blink_started,
            right_blink_started,
            last_click_at,
            last_click_was_left,
            actions,
        )

    # ------------------------------------------------------------------
    # The loop itself
    # ------------------------------------------------------------------
    def _run(self) -> None:
        cap = None
        landmarker = None
        try:
            cap = self._open_camera()
            if cap is None:
                self._set_state(last_error=(
                    "failed to start: " + (self.last_error or "no camera available")
                ))
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

            self._set_state(running=True, started_at=time.time(), last_error="")

            model_path = _ensure_face_model()
            if model_path is not None and _modern_mp_face_landmarker is not None:
                try:
                    landmarker = _modern_mp_face_landmarker.create_from_options(
                        _modern_mp_options(
                            base_options=_modern_mp_base_options(
                                model_asset_path=model_path
                            ),
                            running_mode=_modern_mp_running_mode.VIDEO,
                            num_faces=1,
                            output_face_blendshapes=False,
                            output_facial_transformation_matrixes=False,
                            min_face_detection_confidence=0.5,
                            min_face_presence_confidence=0.5,
                            min_tracking_confidence=0.5,
                        )
                    )
                    self._set_state(backend="mediapipe-face")
                except Exception as exc:
                    log.warning("FaceLandmarker init failed: %s", exc)
                    landmarker = None

            if landmarker is None and _HAS_LEGACY_FACE_MESH:
                try:
                    landmarker = _legacy_face_mesh_module.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,  # enables iris (468..477)
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._set_state(backend="mediapipe-face-mesh")
                except Exception as exc:
                    log.warning("Legacy FaceMesh init failed: %s", exc)
                    landmarker = None

            if landmarker is None:
                self._set_state(backend="none", last_error=(
                    "failed to start: could not initialize any MediaPipe "
                    "face tracker (FaceLandmarker task and legacy "
                    "face_mesh both unavailable)."
                ))
                return

            log.info("Eye mouse loop running (backend=%s).", self.backend)

            smooth_x: Optional[float] = None
            smooth_y: Optional[float] = None
            display_x: Optional[float] = None
            display_y: Optional[float] = None
            last_target_x: Optional[float] = None
            last_target_y: Optional[float] = None

            left_blink_started: Optional[float] = None
            right_blink_started: Optional[float] = None
            last_click_at: float = 0.0
            last_click_was_left: bool = True

            last_scroll_at: float = 0.0
            video_ts_ms: int = 0
            screen_w, screen_h = self._screen_size()
            using_modern_api = self.backend == "mediapipe-face"

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                self._set_state(frames_seen=self.frames_seen + 1)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                landmarks = None
                if using_modern_api and _modern_mp_image is not None:
                    mp_image = _modern_mp_image(
                        image_format=_modern_mp_image_format.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    try:
                        result = landmarker.detect_for_video(mp_image, video_ts_ms)
                    except Exception as exc:
                        log.debug("FaceLandmarker.detect_for_video failed: %s", exc)
                        video_ts_ms += 33
                        continue
                    video_ts_ms += 33
                    if result and result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                else:
                    # Legacy FaceMesh path. ``process`` returns either a
                    # result with ``multi_face_landmarks`` or None.
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = landmarker.process(rgb)
                    except Exception as exc:
                        log.debug("FaceMesh.process failed: %s", exc)
                        continue
                    if result and getattr(result, "multi_face_landmarks", None):
                        landmarks = result.multi_face_landmarks[0].landmark

                if landmarks is None:
                    # No face in view - drop blink / scroll state.
                    left_blink_started = None
                    right_blink_started = None
                    continue
                gaze = self._gaze_ratio(landmarks)
                if gaze is None:
                    continue
                left_gaze, right_gaze = gaze
                # Average the two eyes so one slightly squinty eye does
                # not bias the cursor. The gaze value is invariant to
                # head translation - it only changes when the iris
                # actually moves inside the eye socket.
                gx = (left_gaze[0] + right_gaze[0]) / 2.0
                gy = (left_gaze[1] + right_gaze[1]) / 2.0

                now = time.time()

                # Auto-calibrate: average the first ``_calib_target``
                # frames' gaze so a tiny jitter at startup does not
                # anchor the centre to a stray sample.
                if not self._calibrated:
                    with self._state_lock:
                        self._calib_samples.append((gx, gy))
                        if len(self._calib_samples) >= self._calib_target:
                            avg_x = sum(s[0] for s in self._calib_samples) / len(self._calib_samples)
                            avg_y = sum(s[1] for s in self._calib_samples) / len(self._calib_samples)
                            self._center_x = avg_x
                            self._center_y = avg_y
                            self._calibrated = True
                            self._calib_samples = []
                            log.info(
                                "Vision mouse calibrated at gaze=(%.3f, %.3f) "
                                "(averaged %d frames)",
                                avg_x, avg_y, self._calib_target,
                            )
                    # Skip cursor mapping until calibration is done -
                    # the first frame would otherwise snap the cursor
                    # to a random position.
                    continue

                # --- 1. Cursor mapping -----------------------------------
                # Translate gaze offset from calibration centre into
                # screen pixels. The gaze ratio lives in [0, 1] - we
                # multiply by the full screen size. ``GAIN`` amplifies
                # the small eye-rotation range so looking at the
                # corners of the camera frame reaches the corners of
                # the monitor without requiring extreme eye angles.
                dx = (gx - self._center_x) * self.GAZE_GAIN
                dy = (gy - self._center_y) * self.GAZE_GAIN
                target_sx = (screen_w / 2) + dx * screen_w
                target_sy = (screen_h / 2) + dy * screen_h
                target_sx = max(0.0, min(screen_w - 1, target_sx))
                target_sy = max(0.0, min(screen_h - 1, target_sy))

                if display_x is None:
                    display_x, display_y = target_sx, target_sy
                    smooth_x, smooth_y = target_sx, target_sy
                else:
                    if last_target_x is not None:
                        still = (
                            abs(target_sx - last_target_x) < self.FINE_PIXELS
                            and abs(target_sy - last_target_y) < self.FINE_PIXELS
                        )
                    else:
                        still = False
                    alpha = (
                        1.0 / self.FINE_SMOOTHING
                        if still
                        else 1.0 / self.SMOOTHING
                    )
                    smooth_x += (target_sx - smooth_x) * alpha
                    smooth_y += (target_sy - smooth_y) * alpha
                    if (
                        abs(smooth_x - display_x) >= 1.0
                        or abs(smooth_y - display_y) >= 1.0
                    ):
                        display_x, display_y = smooth_x, smooth_y
                        self._move_mouse(display_x, display_y)
                        self._set_state(last_action="move", last_action_at=now)
                last_target_x, last_target_y = target_sx, target_sy

                # --- 2. Blink detection ---------------------------------
                (
                    left_blink_started,
                    right_blink_started,
                    last_click_at,
                    last_click_was_left,
                    actions,
                ) = self._blink_state(
                    landmarks,
                    left_blink_started,
                    right_blink_started,
                    last_click_at,
                    last_click_was_left,
                    now,
                )
                for action in actions:
                    if action == "click":
                        self._click()
                        self._set_state(last_action="click", last_action_at=now)
                    elif action == "doubleclick":
                        self._double_click()
                        self._set_state(last_action="doubleclick", last_action_at=now)
                    elif action == "rightclick":
                        self._right_click()
                        self._set_state(last_action="rightclick", last_action_at=now)

                # --- 3. Scroll via large vertical gaze shifts -----------
                if (now - last_scroll_at) >= self.SCROLL_COOLDOWN:
                    # Use the same gaze-offset-from-centre the cursor
                    # uses, so scroll direction matches where the
                    # user is looking.
                    vy = (gy - self._center_y) * self.GAZE_GAIN
                    if vy < -self.SCROLL_THRESHOLD:
                        self._scroll(3)
                        self._set_state(last_action="scroll_up", last_action_at=now)
                        last_scroll_at = now
                    elif vy > self.SCROLL_THRESHOLD:
                        self._scroll(-3)
                        self._set_state(last_action="scroll_down", last_action_at=now)
                        last_scroll_at = now
        except Exception as exc:
            log.exception("Vision mouse loop crashed: %s", exc)
            self._set_state(last_error=f"loop crashed: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._set_state(running=False)


# ---------------------------------------------------------------------------
# Public tool wrapper
# ---------------------------------------------------------------------------
class EyeMouseTool(BaseTool):
    """Voice entry point for the eye-mouse control loop."""

    name = "vision_mouse_tool"
    description = (
        "Control the mouse using eye / iris gaze captured from the webcam. "
        "User-facing commands use the phrase 'vision mouse' / 'vision "
        "tracking' so speech recognition reliably hears them. Use 'start "
        "vision mouse' to enable, 'stop vision mouse' to disable, "
        "'calibrate vision mouse' to reset the centre, and 'vision mouse "
        "status' to check the current state. The controller tracks the "
        "iris position *inside the eye socket* (corner-to-corner ratio) "
        "so head motion does NOT move the cursor - only the eyes do. "
        "Gestures: look around to move the cursor (with adaptive "
        "smoothing - the cursor settles when your eyes are still), "
        "single-eye blink = left click, two quick blinks = double-click, "
        "both eyes closed (~0.4s) = right-click, look far up/down = "
        "scroll. The controller self-calibrates by averaging the first "
        "~10 frames' gaze so the user just needs to face the camera."
    )

    def __init__(self) -> None:
        self._controller = _EyeMouseController()

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
        if _contains_any(text, _CALIBRATE_PHRASES):
            return True
        if (
            "vision mouse" in text
            or "vision tracking" in text
            or "eye mouse" in text
            or "eye tracking" in text
        ):
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
                            f"Vision mouse is running, sir. "
                            f"Backend: {status['backend']}. "
                            f"Frames seen: {status['frames_seen']}. "
                            f"Calibrated: {status['calibrated']}."
                        ),
                        data=status,
                    )
                return ToolResult(
                    success=True,
                    message="Vision mouse is currently off, sir.",
                    data=status,
                )

            if _contains_any(lowered, _CALIBRATE_PHRASES):
                ok, msg = self._controller.recalibrate()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            if _contains_any(lowered, _START_PHRASES):
                ok, msg = self._controller.start()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            # Bare "vision mouse" / "eye mouse" - toggle.
            if (
                "vision mouse" in lowered
                or "vision tracking" in lowered
                or "eye mouse" in lowered
                or "eye tracking" in lowered
            ):
                if self._controller.running:
                    ok, msg = self._controller.stop()
                else:
                    ok, msg = self._controller.start()
                return ToolResult(success=ok, message=msg, data=self._controller.status())

            return ToolResult(
                success=False,
                message="I did not understand that vision-mouse command, sir.",
            )
        except Exception as exc:
            log.exception("VisionMouseTool failed: %s", exc)
            return ToolResult(success=False, message=f"Vision mouse error: {exc}")

    # ------------------------------------------------------------------
    # Public helpers (mirrors HandMouseTool for the web API).
    # ------------------------------------------------------------------
    def start(self) -> ToolResult:
        ok, msg = self._controller.start()
        return ToolResult(success=ok, message=msg, data=self._controller.status())

    def stop(self) -> ToolResult:
        ok, msg = self._controller.stop()
        return ToolResult(success=ok, message=msg, data=self._controller.status())

    def recalibrate(self) -> ToolResult:
        ok, msg = self._controller.recalibrate()
        return ToolResult(success=ok, message=msg, data=self._controller.status())

    def status(self) -> Dict[str, Any]:
        return self._controller.status()


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
def register_eye_mouse_tools(router) -> list:
    """Register the eye-mouse tool with *router* and return the new tools.

    The deterministic router runs the tool *before* the LLM, so phrases
    like "start eye mouse" never reach the agent executor.
    """
    tool = EyeMouseTool()
    router.register(
        tool,
        keywords=(
            "vision mouse",
            "vision tracking",
            "start vision mouse",
            "stop vision mouse",
            "calibrate vision mouse",
            # Eye-based fallbacks.
            "eye mouse",
            "eye tracking",
            "control my mouse with my eye",
            "control the mouse with my eye",
            "start eye mouse",
            "stop eye mouse",
            "calibrate eye mouse",
        ),
        priority=80,
    )
    return [tool]


__all__ = [
    "EyeMouseTool",
    "register_eye_mouse_tools",
    "HAS_CV2",
    "HAS_MEDIAPIPE",
    "HAS_PYAUTOGUI",
]
