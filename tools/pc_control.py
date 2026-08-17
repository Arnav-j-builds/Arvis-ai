"""
tools.pc_control
~~~~~~~~~~~~~~~

Cross-platform PC control helpers.

Implemented actions
-------------------

* ``volume up | down | mute | unmute | set <0-100>``
* ``brightness set <0-100>``     (Linux + Windows best-effort)
* ``lock``                        (lock the workstation)
* ``shutdown [confirm]``          (asks the OS to shut down)
* ``reboot [confirm]``            (asks the OS to reboot)
* ``media play | pause | next | prev``

The platform-specific code is wrapped in helpers that swallow ``ImportError``
and ``OSError`` so the tool never crashes the assistant - it simply returns a
"not supported on this OS" message.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from core.base import BaseTool, ToolResult
from core.logger import get_logger

log = get_logger(__name__)

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
IS_LIN = sys.platform.startswith("linux")


def _windows_volume(level: Optional[int] = None, action: Optional[str] = None) -> Tuple[bool, str]:
    """Best-effort volume control on Windows.

    Uses ``pycaw`` if available, otherwise falls back to a PowerShell script
    that drives ``SendKeys`` / ``Wscript.Shell``.
    """
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
        from ctypes import cast, POINTER

        devices = AudioUtilities.GetAllSessions()
        if not devices:
            return False, "No audio devices found."
        # Use the default endpoint - usually index 0 / device 0
        speaker = AudioUtilities.GetSpeakers()
        interface = speaker.Activate(IAudioEndpointVolume._iid_, 0, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        if action == "mute":
            volume.SetMute(1, None)
            return True, "Muted."
        if action == "unmute":
            volume.SetMute(0, None)
            return True, "Unmuted."
        if level is not None:
            volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level / 100.0)), None)
            return True, f"Volume set to {level}%."
        # Step up/down handled by caller via relative nudge
        return False, "Volume action required."
    except Exception as exc:
        return _windows_volume_fallback(level, action, exc)


def _windows_volume_fallback(level: Optional[int], action: Optional[str], exc: Exception) -> Tuple[bool, str]:
    """PowerShell-based volume fallback for Windows when pycaw is missing."""
    if not IS_WIN:
        return False, f"pycaw unavailable ({exc}); not on Windows."
    if action in ("mute", "unmute"):
        target = 1 if action == "mute" else 0
        ps = (
            f"$obj = New-Object -ComObject WScript.Shell;"
            f"$obj.SendKeys([char]173);"  # VK_VOLUME_MUTE toggle - one press
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=5, check=False)
            return True, f"{action.capitalize()}d (via SendKeys)."
        except Exception as e:
            return False, f"Volume control failed: {e}"
    if level is not None:
        ps = (
            "$obj = New-Object -ComObject WScript.Shell;"
            "1..50 | ForEach-Object { $obj.SendKeys([char]174) };"  # volume down 50 steps
            f"1..{int(level // 2)} | ForEach-Object {{ $obj.SendKeys([char]175) }}"  # volume up
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=8, check=False)
            return True, f"Volume nudged toward {level}% (via SendKeys)."
        except Exception as e:
            return False, f"Volume control failed: {e}"
    return False, "No volume action specified."


def _nudge_volume(direction: str) -> Tuple[bool, str]:
    """Nudge master volume up or down by ~10% via WinAPI."""
    if IS_WIN:
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from ctypes import cast, POINTER
            speaker = AudioUtilities.GetSpeakers()
            interface = speaker.Activate(IAudioEndpointVolume._iid_, 0, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            cur = volume.GetMasterVolumeLevelScalar()
            new = max(0.0, min(1.0, cur + (0.1 if direction == "up" else -0.1)))
            volume.SetMasterVolumeLevelScalar(new, None)
            return True, f"Volume {direction} ({int(new * 100)}%)."
        except Exception as exc:
            log.debug("pycaw nudge failed: %s", exc)
    # Linux fallback: pactl
    if IS_LIN and shutil.which("pactl"):
        delta = "+5%" if direction == "up" else "-5%"
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", delta], check=False)
        return True, f"Volume {direction} via pactl."
    # macOS fallback: osascript
    if IS_MAC:
        key = (72 if direction == "up" else 74)  # UI keys: F12=72 volume-down, F11=73? actually F10/F12
        # Use osascript to nudge by 10%
        subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) + {10 if direction == 'up' else -10})"], check=False)
        return True, f"Volume {direction} via osascript."
    return False, f"Volume nudge not supported on {sys.platform}."


def _set_brightness(level: int) -> Tuple[bool, str]:
    """Best-effort brightness control."""
    level = max(0, min(100, level))
    if IS_WIN:
        # Use wmic to query then powershell to set via WMI
        try:
            ps = (
                "$monitors = Get-WmiObject -Namespace root\\wmi -Class WmiMonitorBrightness;"
                "foreach ($m in $monitors) { $m.WmiSetBrightness(1, $args[0]) }"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps, str(level)],
                timeout=5, check=False,
            )
            return True, f"Brightness set to {level}% (best-effort)."
        except Exception as exc:
            return False, f"Brightness control failed: {exc}"
    if IS_LIN and shutil.which("brightnessctl"):
        try:
            # Discover max brightness and scale
            max_b = int(subprocess.check_output(["brightnessctl", "max"]).strip())
            target = int(max_b * level / 100)
            subprocess.run(["brightnessctl", "set", str(target)], check=True)
            return True, f"Brightness set to {level}%."
        except Exception as exc:
            return False, f"brightnessctl failed: {exc}"
    if IS_LIN and shutil.which("xbacklight"):
        try:
            subprocess.run(["xbacklight", "-set", str(level)], check=True)
            return True, f"Brightness set to {level}%."
        except Exception as exc:
            return False, f"xbacklight failed: {exc}"
    return False, "Brightness control is not supported on this OS."


def _lock_workstation() -> Tuple[bool, str]:
    try:
        if IS_WIN:
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
            return True, "Workstation locked."
        if IS_MAC:
            subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"], check=False)
            return True, "Screen locked."
        if IS_LIN and shutil.which("loginctl"):
            subprocess.run(["loginctl", "lock-session"], check=False)
            return True, "Session locked."
        if IS_LIN and shutil.which("xdg-screensaver"):
            subprocess.run(["xdg-screensaver", "lock"], check=False)
            return True, "Screensaver locked."
    except Exception as exc:
        return False, f"Lock failed: {exc}"
    return False, "Lock not supported on this OS."


def _shutdown_or_reboot(action: str, confirm: bool) -> Tuple[bool, str]:
    """Schedule a shutdown/reboot. Requires ``confirm=True`` for safety."""
    if not confirm:
        return False, f"Refusing to {action} without explicit confirmation."

    try:
        if IS_WIN:
            args = ["shutdown", "/s" if action == "shutdown" else "/r", "/t", "30"]
            subprocess.run(args, check=False)
            return True, f"{action.title()} scheduled in 30 seconds. Cancel with: shutdown /a"
        if IS_MAC:
            target = "-halt" if action == "shutdown" else "-restart"
            subprocess.run(["sudo", "shutdown", target, "+1"], check=False)
            return True, f"{action.title()} scheduled in 1 minute."
        if IS_LIN and shutil.which("shutdown"):
            arg = "-h" if action == "shutdown" else "-r"
            subprocess.run(["sudo", "shutdown", arg, "+1"], check=False)
            return True, f"{action.title()} scheduled in 1 minute."
    except Exception as exc:
        return False, f"{action} failed: {exc}"
    return False, f"{action} is not supported on this OS."


def _media_key(action: str) -> Tuple[bool, str]:
    """Send a media key."""
    keys = {
        "play":   ("VK_MEDIA_PLAY_PAUSE", 0xB3),
        "pause":  ("VK_MEDIA_PLAY_PAUSE", 0xB3),
        "next":   ("VK_MEDIA_NEXT_TRACK", 0xB0),
        "prev":   ("VK_MEDIA_PREV_TRACK", 0xB1),
        "stop":   ("VK_MEDIA_STOP", 0xB2),
    }
    if action not in keys:
        return False, f"Unknown media action '{action}'."

    if IS_WIN:
        try:
            import ctypes
            vk = keys[action][1]
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP = 2
            return True, f"Media {action} sent."
        except Exception as exc:
            return False, f"Media key failed: {exc}"

    if IS_MAC:
        # Use osascript to send media keys via System Events.
        key_map = {
            "play":  "PLAY",
            "pause": "PLAY",
            "next":  "NEXT",
            "prev":  "PREVIOUS",
        }
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to key code {{{{ key code of (key "{key_map[action]}" of process "SystemUIServer") }}}}'],
                check=False, timeout=4,
            )
            return True, f"Media {action} sent."
        except Exception as exc:
            return False, f"Media key failed: {exc}"

    if IS_LIN:
        if shutil.which("playerctl"):
            cmd = {
                "play":  "play-pause",
                "pause": "pause",
                "next":  "next",
                "prev":  "previous",
                "stop":  "stop",
            }[action]
            subprocess.run(["playerctl", cmd], check=False)
            return True, f"Media {action} via playerctl."
        return False, "Media control needs playerctl on Linux."
    return False, "Media keys not supported on this OS."


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class PcControlTool(BaseTool):
    name = "pc_control_tool"
    description = (
        "Control the user's PC: change volume, mute, brightness, lock the screen, "
        "play/pause media, next/previous track, or shutdown / reboot (with "
        "explicit confirmation). Examples: 'turn up the volume', 'mute', "
        "'set brightness to 60', 'lock my pc', 'play next track'."
    )

    _RE_VOLUME = re.compile(r"(volume|sound)\s*(up|down|mute|unmute|set)?", re.IGNORECASE)
    _RE_SET_VOLUME = re.compile(r"set\s+(?:the\s+)?(?:volume|sound)\s+to\s+(\d{1,3})", re.IGNORECASE)
    _RE_BRIGHTNESS = re.compile(r"(?:set\s+)?brightness\s*(?:to\s+)?(\d{1,3})?", re.IGNORECASE)
    _RE_MEDIA = re.compile(r"(play|pause|next|previous|prev|stop)\s*(?:track|song|media)?", re.IGNORECASE)
    _RE_LOCK = re.compile(r"\block(?:\s+(?:the\s+)?(?:pc|computer|workstation|screen))?\b", re.IGNORECASE)
    _RE_SHUTDOWN = re.compile(r"\b(shut\s*down|power\s*off)\b", re.IGNORECASE)
    _RE_REBOOT = re.compile(r"\b(reboot|restart)\b", re.IGNORECASE)

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        return any(
            re.search(p, text)
            for p in (
                r"\bvolume\b", r"\bmute\b", r"\bunmute\b", r"\bbrightness\b",
                r"\block\b", r"\bshutdown\b", r"\breboot\b", r"\brestart\b",
                r"\bplay\s+next\b", r"\bnext\s+track\b", r"\bpause\s+music\b",
                r"\bnext\s+song\b",
            )
        )

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        text = (command or "").lower().strip()

        # 1. shutdown / reboot (require explicit confirmation)
        if self._RE_SHUTDOWN.search(text):
            confirm = bool(context and context.get("confirm")) or "confirm" in text
            ok, msg = _shutdown_or_reboot("shutdown", confirm)
            return ToolResult(success=ok, message=msg)
        if self._RE_REBOOT.search(text):
            confirm = bool(context and context.get("confirm")) or "confirm" in text
            ok, msg = _shutdown_or_reboot("reboot", confirm)
            return ToolResult(success=ok, message=msg)

        # 2. lock
        if self._RE_LOCK.search(text):
            ok, msg = _lock_workstation()
            return ToolResult(success=ok, message=msg)

        # 3. brightness
        m = self._RE_BRIGHTNESS.search(text)
        if m and m.group(1):
            ok, msg = _set_brightness(int(m.group(1)))
            return ToolResult(success=ok, message=msg)

        # 4. media keys
        m = self._RE_MEDIA.search(text)
        if m:
            verb = m.group(1).lower()
            if verb == "previous":
                verb = "prev"
            ok, msg = _media_key(verb)
            return ToolResult(success=ok, message=msg)

        # 5. volume set absolute
        m = self._RE_SET_VOLUME.search(text)
        if m:
            ok, msg = _windows_volume(level=int(m.group(1))) if IS_WIN else (False, "Set-volume only implemented on Windows.")
            return ToolResult(success=ok, message=msg)

        # 6. volume relative / mute / unmute
        if "mute" in text:
            ok, msg = _windows_volume(action="mute") if IS_WIN else (False, "Mute only implemented on Windows.")
            return ToolResult(success=ok, message=msg)
        if "unmute" in text:
            ok, msg = _windows_volume(action="unmute") if IS_WIN else (False, "Unmute only implemented on Windows.")
            return ToolResult(success=ok, message=msg)

        if "volume" in text or "sound" in text or "louder" in text or "quieter" in text:
            direction = "down" if any(w in text for w in ("down", "quieter", "lower")) else "up"
            ok, msg = _nudge_volume(direction)
            return ToolResult(success=ok, message=msg)

        return ToolResult(success=False, message="I could not parse a PC control command there.")


def register_pc_control_tool(router) -> List[BaseTool]:
    tool = PcControlTool()
    router.register(
        tool,
        keywords=(
            "volume", "mute", "unmute", "louder", "quieter", "lower the",
            "brightness", "lock ", "lock my", "shutdown", "reboot", "restart",
            "play next", "next track", "previous track", "prev track",
            "pause music", "pause the", "next song",
        ),
        priority=70,
    )
    return [tool]


__all__ = ["PcControlTool", "register_pc_control_tool"]