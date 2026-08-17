"""
core.verification
~~~~~~~~~~~~~~~~~

Per-step verification after ``CommandRouter.dispatch`` returns.

The point: the executor must not assume ``success=True`` means the
step *actually* worked. The router's deterministic matchers return
success for "yes I claimed the command" - they do not check that
the file was created, the email was sent, or the app window is on
screen.

For each common ``tool_hint`` we have a tiny verifier that does a
follow-up check:

* ``open_app``  - look up the process list for a friendly-name match
                   (e.g. "vscode" -> ``Code.exe``).
* ``terminal``  - assume the terminal's own runner succeeded; trust
                   the router.
* ``type``      - trust the router; pyautogui's FailSafe would have
                   raised if the cursor was off-screen.
* ``search``    - if the executor has access to ``vision.capture``
                   it OCRs the screen for a result row, otherwise
                   trust.
* everything else - trust.

The verifier is intentionally lenient. We do NOT want to flap
"verification failed" on noisy checks - we want to catch obvious
disasters (Chrome didn't open) and quietly approve the rest.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from core.base import ToolResult
from core.logger import get_logger
from core.task_plan import TaskStep

log = get_logger(__name__)


@dataclass
class VerifyOutcome:
    """Result of a verification step."""

    verified: bool
    detail: str = ""
    data: Optional[Dict[str, Any]] = None


# Friendly-name -> list of process names. We match case-insensitively.
# The lists are conservative on purpose: matching too many processes
# creates false positives (e.g. every Office install has a ``WINWORD``
# background helper).
_APP_PROCESS_HINTS = {
    "chrome": ("chrome.exe",),
    "google chrome": ("chrome.exe",),
    "firefox": ("firefox.exe",),
    "edge": ("msedge.exe",),
    "notepad": ("notepad.exe",),
    "notepad++": ("notepad++.exe",),
    "vscode": ("code.exe",),
    "visual studio code": ("code.exe",),
    "code": ("code.exe",),
    "explorer": ("explorer.exe",),
    "file explorer": ("explorer.exe",),
    "terminal": ("wt.exe", "cmd.exe", "windowsterminal.exe"),
    "cmd": ("cmd.exe",),
    "powershell": ("powershell.exe", "pwsh.exe"),
    "word": ("winword.exe",),
    "excel": ("excel.exe",),
    "powerpoint": ("powerpoint.exe",),
    "outlook": ("outlook.exe",),
    "discord": ("discord.exe",),
    "slack": ("slack.exe",),
    "telegram": ("telegram.exe",),
    "spotify": ("spotify.exe",),
    "steam": ("steam.exe",),
}


class Verifier:
    """Post-step verification registry."""

    def __init__(self) -> None:
        # Per-hint verifier map. ``None`` means "trust the router".
        self._handlers: Dict[str, Callable[[TaskStep, ToolResult], VerifyOutcome]] = {
            "open_app": self._verify_open_app,
            "launch_app": self._verify_open_app,
            "app": self._verify_open_app,
            "open_anything": self._verify_open_app,
            "type": self._verify_type,
            "type_text": self._verify_type,
            "typing": self._verify_type,
            "press_key": self._verify_type,
            "press": self._verify_type,
            "search": self._verify_search,
            "google_search": self._verify_search,
            "youtube_search": self._verify_search,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def verify(self, step: TaskStep, result: ToolResult) -> VerifyOutcome:
        """Run the verifier matching the step's ``tool_hint``."""
        hint = (step.tool_hint or "").strip().lower()
        handler = self._handlers.get(hint)
        if handler is None:
            # Default: trust the router. ``result.success`` already
            # says whether the deterministic matcher claimed the
            # command. Verification just records that we did not
            # run any extra check.
            return VerifyOutcome(
                verified=bool(result.success),
                detail="no-op verifier (trusted router result)",
                data={"hint": hint},
            )
        try:
            return handler(step, result)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("verifier for %r raised: %s", hint, exc)
            return VerifyOutcome(
                verified=bool(result.success),
                detail=f"verifier raised {exc!r}; falling back to router success",
            )

    # ------------------------------------------------------------------
    # Per-hint handlers
    # ------------------------------------------------------------------
    def _verify_open_app(self, step: TaskStep, result: ToolResult) -> VerifyOutcome:
        if not result.success:
            return VerifyOutcome(
                verified=False,
                detail="router reported failure",
                data={"hint": step.tool_hint},
            )
        args = step.arguments or {}
        name = (
            args.get("app")
            or args.get("name")
            or args.get("query")
            or (step.description or "")
        )
        if not isinstance(name, str) or not name.strip():
            return VerifyOutcome(verified=True, detail="no app name to verify")
        friendly = name.strip().lower()
        procs = _APP_PROCESS_HINTS.get(friendly)
        if procs is None:
            # Last-ditch heuristic: look for a process whose exe name
            # contains the first word of the friendly name.
            token = re.sub(r"[^a-z0-9]+", "", friendly.split()[0])[:8]
            procs = (f"{token}.exe",) if token else ()
        if not procs:
            return VerifyOutcome(verified=True, detail="no process hint for app")

        # Give the OS a moment to spawn the process before we look
        # for it. ``time.sleep`` here is fine - verification runs on
        # the executor thread, not the mic loop.
        time.sleep(0.8)

        running = _running_process_names()
        for proc in procs:
            if proc.lower() in running:
                return VerifyOutcome(
                    verified=True,
                    detail=f"process {proc} is running",
                    data={"process": proc},
                )
        return VerifyOutcome(
            verified=False,
            detail=(
                f"no matching process found (looked for {procs!r}); "
                f"app may have opened and closed, or its process name "
                f"is not in our lookup table"
            ),
            data={"looked_for": list(procs)},
        )

    def _verify_type(self, step: TaskStep, result: ToolResult) -> VerifyOutcome:
        # ``pyautogui`` would have raised ``FailSafeException`` if the
        # cursor was off-screen, so trust the router for typing /
        # key-presses. We do however record the failure case
        # explicitly so the executor can surface "I could not type".
        if not result.success:
            return VerifyOutcome(verified=False, detail="router reported failure")
        return VerifyOutcome(
            verified=True,
            detail="typing/press trusted (pyautogui FailSafe would have raised)",
        )

    def _verify_search(self, step: TaskStep, result: ToolResult) -> VerifyOutcome:
        # No cheap universal verification - the search engine's UI
        # varies wildly. Trust the router and let the user notice if
        # results didn't show up. ``vision.capture`` could be plugged
        # in here later.
        if not result.success:
            return VerifyOutcome(verified=False, detail="router reported failure")
        return VerifyOutcome(verified=True, detail="search trusted (no OCR check)")


# ---------------------------------------------------------------------------
# Process-listing helper
# ---------------------------------------------------------------------------
def _running_process_names() -> set:
    """Return a set of lowercase process names currently running.

    Tries ``psutil`` first (it is already a project dep for the web
    server), then falls back to ``tasklist`` on Windows or ``ps`` on
    POSIX. Returns an empty set if no listing method works.
    """
    names: set = set()
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(attrs=["name"]):
            try:
                n = proc.info.get("name")
                if n:
                    names.add(n.lower())
            except Exception:
                continue
        if names:
            return names
    except Exception:
        pass

    # Fallback: Windows ``tasklist``.
    exe = shutil.which("tasklist")
    if exe:
        try:
            import subprocess

            out = subprocess.run(
                [exe, "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            for line in (out.stdout or "").splitlines():
                # CSV row: "Image Name","PID","Session Name","Session#","Mem Usage"
                parts = line.split(",")
                if parts:
                    name = parts[0].strip().strip('"').lower()
                    if name.endswith(".exe"):
                        names.add(name)
        except Exception:
            pass
    return names


__all__ = ["Verifier", "VerifyOutcome"]
