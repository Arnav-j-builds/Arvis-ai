"""
core.autostart
~~~~~~~~~~~~~~

Windows autostart helper, modeled on the approach used by the
``Brahma-Echo`` project (``start_brahma.vbs`` + an ``HKCU\\...\\Run``
registry value pointing at ``pythonw.exe main.py --startup``).

The pattern lets the user double-click a single ``.vbs`` shortcut to
toggle the assistant on or off at Windows boot, without ever showing
a console window on the user's desktop.

API
---

* :func:`is_enabled`   - ``True`` when the registry value exists.
* :func:`enable`       - add the registry value.
* :func:`disable`      - remove the registry value.
* :func:`toggle`       - flip the current state, return the new state.
* :func:`run_value`    - the command string we will write to the registry.
* :func:`launcher_vbs` - the path to the ``.vbs`` we expect to find in the
                         project root (used to power the autostart flow).

The module silently no-ops on non-Windows platforms so it can be
imported safely from cross-platform code (CI, tests, Linux dev box).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.config import project_root

log = get_logger(__name__)

# Same registry path Brahma-Echo writes to.
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "ArvisAssistant"

# Tag passed to ``main.py`` / ``app.py`` when launched at Windows boot.
# Brahma-Echo uses ``--startup`` for the same purpose; we follow suit.
STARTUP_FLAG = "--startup"


def _project_root() -> Path:
    return project_root()


def _quote_cmd_arg(value: str) -> str:
    """Quote a single argument for the Windows ``Run`` registry value.

    The Windows Run key expects a single command line string, with each
    argument wrapped in double quotes. Paths are quoted as-is so spaces
    in the project root (e.g. ``F:\\Program Files\\arvis``) work.
    """
    value = str(value)
    if '"' in value:
        # Should never happen for our paths, but defend anyway.
        value = value.replace('"', '\\"')
    return f'"{value}"'


def run_value() -> str:
    """Return the command string written to the Run registry value.

    Points at ``start_arvis.vbs`` so Windows boot opens a visible
    console window, ``cd``'s into the project folder, and runs
    ``app.py``. We deliberately use the VBS launcher (not a silent
    ``pythonw.exe`` invocation) so the user can see the exact
    project folder Python is running from and any startup errors.
    """
    return _quote_cmd_arg(str(launcher_vbs()))


def launcher_vbs() -> Path:
    """Path to the ``.vbs`` helper that lives next to ``app.py``."""
    return _project_root() / "start_arvis.vbs"


def is_enabled() -> bool:
    """Return ``True`` when the registry value currently points at us."""
    if os.name != "nt":
        return False
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_READ
        ) as key:
            try:
                value, _ = winreg.QueryValueEx(key, _VALUE_NAME)
            except FileNotFoundError:
                return False
            # Treat as enabled if the value points at *our* project.
            return str(_project_root()) in str(value)
    except Exception as exc:  # pragma: no cover - depends on OS
        log.debug("is_enabled failed: %s", exc)
        return False


def enable() -> bool:
    """Write the registry value. Returns ``True`` on success."""
    if os.name != "nt":
        log.info("Autostart is a no-op on this platform.")
        return False
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(
                key, _VALUE_NAME, 0, winreg.REG_SZ, run_value()
            )
        log.info("Autostart enabled (%s -> %s)", _VALUE_NAME, run_value())
        return True
    except Exception as exc:  # pragma: no cover - depends on OS
        log.error("Failed to enable autostart: %s", exc)
        return False


def disable() -> bool:
    """Remove the registry value. Returns ``True`` on success."""
    if os.name != "nt":
        return False
    try:
        import winreg  # type: ignore[import-not-found]

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            try:
                winreg.DeleteValue(key, _VALUE_NAME)
            except FileNotFoundError:
                return True
        log.info("Autostart disabled.")
        return True
    except Exception as exc:  # pragma: no cover - depends on OS
        log.error("Failed to disable autostart: %s", exc)
        return False


def toggle() -> bool:
    """Flip the autostart state, return the new state (``True``=enabled)."""
    if is_enabled():
        disable()
        return False
    enable()
    return is_enabled()


def launched_from_startup() -> bool:
    """``True`` when the current process was launched by the autostart
    entry point (i.e. with ``--startup`` on the command line)."""
    return any(str(arg).strip().lower() == STARTUP_FLAG for arg in sys.argv[1:])


__all__ = [
    "is_enabled",
    "enable",
    "disable",
    "toggle",
    "run_value",
    "launcher_vbs",
    "launched_from_startup",
    "STARTUP_FLAG",
]
