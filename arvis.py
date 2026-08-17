"""
Arvis — a modern, futuristic voice assistant.

Built on the same engine as `main.py` (Ollama + speech_recognition + pyttsx3),
but with two big additions:

  1. A customtkinter-powered futuristic UI overlay. The interface is a
     cinematic AI command center — animated central core/orb, HUD rings,
     live audio visualisation, command console, system monitor, side
     status panels, boot sequence, and a holographic command input.
  2. A "modes" system — pre-baked shortcuts such as
        "i'm working"  → open VS Code, YouTube, Figma, Terminal
        "i'm studying" → play study music on YouTube, open Notion
     plus a UI for adding / editing / deleting your own custom modes
     that chain together any number of the same actions the assistant
     already supports.

Usage:
    python arvis.py            # launches the GUI + voice loop
    python arvis.py --no-gui   # headless mode (no window)

Dependencies (already required by main.py):
    customtkinter, speechrecognition, pyttsx3, langchain-ollama,
    langchain-core, langchain, langchain-community, langchain-openai,
    python-dotenv, pyaudio, plus everything in `tools/`.

The optional `psutil` package enables richer system stats (CPU %, RAM %,
disk %, network throughput, battery). When it's missing the monitor
silently degrades to placeholder values instead of crashing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from typing import Callable

import customtkinter as ctk
import pyttsx3
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama

# Re-use the existing tools so we don't duplicate code.
from tools.time import get_time
from tools.OCR import read_text_from_latest_image
from tools.arp_scan import arp_scan_terminal
from tools.duckduckgo import duckduckgo_search_tool
from tools.matrix import matrix_mode
from tools.screenshot import take_screenshot
from tools.opener import open_anything

# Optional dependency for the system monitor. Imported defensively below.
try:  # pragma: no cover - depends on optional deps
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIC_INDEX = None
TRIGGER_WORD = "arvis"
CONVERSATION_TIMEOUT = 30  # seconds before exiting conversation mode

# Persistent storage for user-defined modes.
CONFIG_DIR = Path(os.getenv("ARVIS_CONFIG_DIR", Path.home() / ".arvis"))
CONFIG_FILE = CONFIG_DIR / "modes.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("arvis")


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------
# A mode is: {
#   "name":   "Working",          # display name
#   "phrases": ["i'm working",    # spoken triggers (any match fires it)
#               "lets work"],
#   "actions": [                  # list of actions to execute in order
#       {"type": "app",   "target": "vs code"},
#       {"type": "url",   "target": "https://www.youtube.com"},
#       ...
#   ],
#   "builtin": True,              # protected from accidental deletion
# }

BUILTIN_MODES: list[dict] = [
    {
        "name": "Working",
        "phrases": ["i'm working", "i am working", "lets work", "let's work", "start working"],
        "actions": [
            {"type": "app", "target": "vs code"},
            {"type": "url", "target": "https://www.youtube.com"},
            {"type": "url", "target": "https://www.figma.com"},
            {"type": "app", "target": "terminal"},
        ],
        "builtin": True,
    },
    {
        "name": "Studying",
        "phrases": ["i'm studying", "i am studying", "lets study", "let's study", "start studying"],
        "actions": [
            {"type": "youtube_search", "target": "study music lofi focus"},
            {"type": "url", "target": "https://www.notion.so"},
        ],
        "builtin": True,
    },
]


# ---------------------------------------------------------------------------
# Mode manager
# ---------------------------------------------------------------------------

class ModeManager:
    """Loads built-in + custom modes from disk and persists changes."""

    def __init__(self, path: Path = CONFIG_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._modes: list[dict] = []
        self.reload()

    # -- persistence --------------------------------------------------------

    def reload(self) -> None:
        """Read modes.json (if present), merge with built-ins."""
        saved: list[dict] = []
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(saved, list):
                    saved = []
            except Exception as exc:
                log.warning("Could not read %s: %s — using built-ins only.", self.path, exc)
                saved = []

        # Built-ins always live in the list (in a fixed order).
        by_name = {m["name"]: dict(m) for m in BUILTIN_MODES}
        for m in saved:
            if m.get("builtin"):
                continue  # never overwrite built-ins from disk
            by_name[m["name"]] = m
        self._modes = list(by_name.values())
        self.save()

    def save(self) -> None:
        """Persist only non-built-in modes to disk."""
        custom = [m for m in self._modes if not m.get("builtin")]
        try:
            self.path.write_text(json.dumps(custom, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not save %s: %s", self.path, exc)

    # -- public API ---------------------------------------------------------

    @property
    def modes(self) -> list[dict]:
        return list(self._modes)

    def find_by_phrase(self, text: str) -> dict | None:
        t = text.lower().strip()
        for mode in self._modes:
            for phrase in mode.get("phrases", []):
                if phrase.lower() in t:
                    return mode
        return None

    def add(self, name: str, phrases: list[str], actions: list[dict]) -> dict:
        if any(m["name"].lower() == name.lower() for m in self._modes):
            raise ValueError(f"A mode named '{name}' already exists.")
        mode = {"name": name, "phrases": phrases, "actions": actions, "builtin": False}
        self._modes.append(mode)
        self.save()
        return mode

    def update(self, name: str, phrases: list[str], actions: list[dict]) -> dict:
        for m in self._modes:
            if m["name"].lower() == name.lower():
                if m.get("builtin"):
                    raise ValueError("Built-in modes cannot be edited.")
                m["phrases"] = phrases
                m["actions"] = actions
                self.save()
                return m
        raise KeyError(name)

    def remove(self, name: str) -> None:
        for m in self._modes:
            if m["name"].lower() == name.lower():
                if m.get("builtin"):
                    raise ValueError("Built-in modes cannot be deleted.")
                self._modes.remove(m)
                self.save()
                return
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Action executor — runs the steps of a mode
# ---------------------------------------------------------------------------

def _open_app_quietly(target: str) -> str:
    """Best-effort launch using the same logic as tools.opener."""
    cmd = target.lower()
    APP_SHORTCUTS = {
        "vs code": ("Code.exe", "all"),
        "vscode": ("Code.exe", "all"),
        "visual studio code": ("Code.exe", "all"),
        "terminal": ("cmd.exe", "win"),
        "cmd": ("cmd.exe", "win"),
        "powershell": ("powershell.exe", "win"),
        "notepad": ("notepad.exe", "win"),
        "calculator": ("calc.exe", "win"),
        "calc": ("calc.exe", "win"),
        "chrome": ("chrome", "all"),
        "edge": ("msedge", "all"),
        "firefox": ("firefox", "all"),
        "figma": ("figma", "all"),
        "notion": ("notion", "all"),
        "spotify": ("spotify", "all"),
    }
    entry = APP_SHORTCUTS.get(cmd)
    command_str = entry[0] if entry else target
    try:
        subprocess.Popen(["cmd", "/c", "start", "", command_str], shell=False)
        return f"launched {target}"
    except Exception as exc:
        return f"failed to launch {target}: {exc}"


def _open_url_quietly(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    try:
        webbrowser.open(url)
        return f"opened {url}"
    except Exception as exc:
        return f"failed to open {url}: {exc}"


def _youtube_search_quietly(query: str) -> str:
    from urllib.parse import quote_plus
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    return _open_url_quietly(url)


def _google_search_quietly(query: str) -> str:
    from urllib.parse import quote_plus
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    return _open_url_quietly(url)


def run_action(action: dict) -> str:
    """Execute one action dict. Returns a short status string."""
    atype = action.get("type", "").lower()
    target = action.get("target", "").strip()
    if not target:
        return "skipped empty action"
    if atype == "app":
        return _open_app_quietly(target)
    if atype == "url":
        return _open_url_quietly(target)
    if atype == "youtube_search":
        return _youtube_search_quietly(target)
    if atype == "google_search":
        return _google_search_quietly(target)
    return f"unknown action type: {atype}"


# ---------------------------------------------------------------------------
# LangChain agent (identical to main.py)
# ---------------------------------------------------------------------------

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=MIC_INDEX)

llm = ChatOllama(model="minimax-m3:cloud", reasoning=False)

tools = [
    get_time,
    arp_scan_terminal,
    read_text_from_latest_image,
    duckduckgo_search_tool,
    matrix_mode,
    take_screenshot,
    open_anything,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are arvis, an intelligent, conversational AI assistant. "
            "Your goal is to be helpful, friendly, and informative. You can "
            "respond in natural, human-like language and use tools when "
            "needed. Keep responses conversational and concise.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# ---------------------------------------------------------------------------
# TTS — same logic as main.py but with a per-call engine to avoid crashes
# when called from threads.
# ---------------------------------------------------------------------------

_tts_lock = threading.Lock()


def speak_text(text: str) -> None:
    def _go() -> None:
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                for voice in engine.getProperty("voices"):
                    if "jamie" in voice.name.lower():
                        engine.setProperty("voice", voice.id)
                        break
                engine.setProperty("rate", 180)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                time.sleep(0.2)
            except Exception as exc:
                log.error("TTS failed: %s", exc)

    threading.Thread(target=_go, daemon=True).start()


# ===========================================================================
#  FUTURISTIC UI
# ===========================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

class Palette:
    """Single source of truth for all UI colors.

    Theme: 3D Holographic Glass (skeuomorphic).
    Deep cool indigos with frosted-glass panels, soft white highlights,
    brushed-metal accents, and a refined glow that hints at depth
    without the high-saturation neon of the Tron look.
    """

    # ── Backgrounds ────────────────────────────────────────────────────
    BG_VOID = "#06081a"           # window root — deep midnight indigo
    BG_DEEP = "#0b1130"           # mid-depth — used inside dialogs
    BG_PANEL = "#0f1740"          # primary panel surface
    BG_GLASS = "#1a2455"          # frosted-glass inset (alpha simulated)
    BG_GLASS_LIGHT = "#243070"    # hovered glass — brightens on interaction
    BG_GLASS_HIGHLIGHT = "#2e3c85"  # bright specular on glass

    # ── Borders / chrome ────────────────────────────────────────────────
    BORDER_DIM = "#243069"        # resting border (cool steel)
    BORDER_HOT = "#7d96e0"        # focused / active border (warm reflection)
    BORDER_METAL = "#3a4a8c"      # brushed-metal divider
    BORDER_METAL_LIGHT = "#6478b8"

    # ── Accents — softened, with a hint of warm reflection ──────────────
    NEON_CYAN = "#7fdfff"         # primary accent — sky cyan
    NEON_AQUA = "#9eecff"         # pale aqua highlight
    NEON_PINK = "#ff7fc8"         # accent (warm magenta)
    NEON_GREEN = "#a6f0c2"        # success / online (mint)
    NEON_AMBER = "#ffd28a"        # warning / speaking (warm gold)
    NEON_PURPLE = "#b59cff"       # executing (lavender)
    NEON_RED = "#ff8a9c"          # error (coral)

    # ── Specular / reflection ───────────────────────────────────────────
    SPEC_HIGHLIGHT = "#ffffff"    # pure white specular
    SPEC_SOFT = "#dde6ff"         # soft white-blue highlight
    SPEC_DIM = "#5a6db0"          # dim reflection on metal

    # ── Text ────────────────────────────────────────────────────────────
    TEXT_DIM = "#a3b1d8"          # secondary text
    TEXT_BRIGHT = "#f1f5ff"       # primary text (almost white)
    TEXT_GHOST = "#5d6da3"        # labels / captions
    TEXT_FAINT = "#2c3766"        # very dim labels


# ---------------------------------------------------------------------------
# Utility widgets
# ---------------------------------------------------------------------------

class HudCorners(tk.Canvas):
    """A 24px-tall canvas overlay that draws sci-fi corner brackets.

    Sits transparently above any frame; we draw the four bracket arms in
    cyan with metallic end-caps. The canvas itself is otherwise empty,
    so the panel underneath is unaffected.
    """

    def __init__(self, master, color: str = Palette.NEON_CYAN, **kw):
        kw.setdefault("bg", Palette.BG_PANEL)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("height", 22)
        super().__init__(master, **kw)
        self._color = color
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self) -> None:
        self.delete("c")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 6 or h < 6:
            return
        L = 16
        T = 2
        # Soft outer glow + crisp bracket — gives a slight 3D bevel.
        for off, col in ((2, self._color), (0, Palette.SPEC_HIGHLIGHT)):
            # top-left
            self.create_line(2 - off, 2, 2 + L + off, 2, fill=col, width=T, tags="c")
            self.create_line(2, 2 - off, 2, 2 + L + off, fill=col, width=T, tags="c")
            # top-right
            self.create_line(w - 2 + off, 2, w - 2 - L - off, 2, fill=col, width=T, tags="c")
            self.create_line(w - 2, 2 - off, w - 2, 2 + L + off, fill=col, width=T, tags="c")
            # bottom-left
            self.create_line(2 - off, h - 2, 2 + L + off, h - 2, fill=col, width=T, tags="c")
            self.create_line(2, h - 2 + off, 2, h - 2 - L - off, fill=col, width=T, tags="c")
            # bottom-right
            self.create_line(w - 2 + off, h - 2, w - 2 - L - off, h - 2, fill=col, width=T, tags="c")
            self.create_line(w - 2, h - 2 + off, w - 2, h - 2 - L - off, fill=col, width=T, tags="c")
            break  # only draw the glow + highlight pair once via offsets


class GlassPanel(tk.Canvas):
    """A reusable frosted-glass surface painted as a Tk canvas behind a
    transparent CTkFrame.

    Renders three layers:
      1. A vertical gradient body (deep → glass).
      2. A thin specular highlight along the top edge (faux rim-light).
      3. A 1px outer bevel (lighter top/left, darker bottom/right) to
         fake depth without a real blur.

    The accompanying CTkFrame that lives on top must use a transparent
    ``fg_color`` so the canvas below shows through.
    """

    def __init__(self, master, accent: str = Palette.NEON_CYAN,
                 top_highlight: bool = True, **kw):
        super().__init__(master, highlightthickness=0,
                         bg=Palette.BG_PANEL, **kw)
        self._accent = accent
        self._top_highlight = top_highlight
        self.bind("<Configure>", lambda e: self._paint())

    def _paint(self) -> None:
        self.delete("g")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        if w < 4 or h < 4:
            return
        # Vertical gradient — 24 stripes of decreasing alpha feel.
        steps = 28
        for i in range(steps):
            t = i / max(steps - 1, 1)
            # Interpolate between BG_PANEL (top) and BG_GLASS (bottom).
            r1, g1, b1 = 0x0f, 0x17, 0x40
            r2, g2, b2 = 0x1a, 0x24, 0x55
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            y0 = int(i * h / steps)
            y1 = int((i + 1) * h / steps) + 1
            self.create_rectangle(
                0, y0, w, y1, fill=f"#{r:02x}{g:02x}{b:02x}",
                outline="", tags="g",
            )
        # Top specular rim — a thin lighter band.
        if self._top_highlight:
            self.create_rectangle(
                1, 1, w - 1, 2, fill=Palette.BORDER_METAL_LIGHT, outline="",
                tags="g",
            )
            self.create_rectangle(
                1, 3, w - 1, 4, fill=Palette.BG_GLASS_HIGHLIGHT, outline="",
                tags="g",
            )
        # Bottom shadow rim.
        self.create_rectangle(
            1, h - 2, w - 1, h - 1, fill="#0a0e22", outline="", tags="g",
        )
        # Left/right bevel edges.
        self.create_rectangle(
            1, 1, 2, h - 1, fill=Palette.BORDER_METAL_LIGHT, outline="",
            tags="g",
        )
        self.create_rectangle(
            w - 2, 1, w - 1, h - 1, fill="#0a0e22", outline="", tags="g",
        )


class HudPanel(ctk.CTkFrame):
    """A reusable glass-style panel with corner brackets.

    The panel is a transparent CTkFrame sitting on top of a ``GlassPanel``
    canvas that paints the actual gradient + bevel.
    """

    def __init__(self, master, title: str | None = None,
                 accent: str = Palette.NEON_CYAN, **kw):
        super().__init__(
            master, fg_color="transparent", corner_radius=14,
            border_width=0, **kw,
        )
        self._accent = accent
        self._glass = GlassPanel(self, accent=accent)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        # 1px cool-steel border drawn last so it sits on top of the bevel.
        self._border = tk.Frame(self, bg=Palette.BORDER_DIM, highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        # Inner body container — children add here for proper layering.
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Corner bracket overlay sits above everything else.
        self._corners = HudCorners(self, color=accent, height=22)
        self._corners.place(x=0, y=0, relwidth=1, height=22)
        self._title = title

    def set_title(self, text: str, accent: str | None = None) -> None:
        self._title = text
        if accent:
            self._accent = accent
            self._corners._color = accent
            self._corners._draw()


# ---------------------------------------------------------------------------
# Central AI core + HUD rings + audio visualiser
# ---------------------------------------------------------------------------

class AICoreView(tk.Canvas):
    """A 3D-feeling glass orb canvas.

    Renders:
      • A frosted-glass sphere with multiple stacked layers:
          – back rim (soft glow halo, low alpha)
          – glass body (deep cool tone with radial gradient feel)
          – outer rim shadow (bottom-right darker)
          – inner sphere (smaller, accent color, gives glass depth)
          – specular highlight (off-center upper-left)
          – pinpoint hot-spot
          – equatorial refraction line (horizontal arc across the orb)
          – three orbiting wireframe rings (perspective-tilted ellipses)
      • 60 tick marks around the orb
      • 64 radial audio bars beyond the rings
      • Drifting particles
      • A faint floor reflection (ellipse below the orb)
      • Subtle vertical reflection sweep

    State-driven palette + pulse keeps the orb reactive while still
    reading as a single 3D object rather than a collection of 2D parts.
    """

    # Per-state palette (used by the orb body + glow + hot-spot).
    _PAL = {
        "idle":      {"core": "#7fdfff", "rim": "#3a4a8c", "glow": "#0e1a48",
                      "hot": "#ffffff", "ring": "#a3b1d8"},
        "listening": {"core": "#a6f0c2", "rim": "#2a6b4a", "glow": "#0a3a26",
                      "hot": "#f1fff1", "ring": "#9eecff"},
        "thinking":  {"core": "#ff7fc8", "rim": "#7a3a78", "glow": "#3a1640",
                      "hot": "#fff0fa", "ring": "#b59cff"},
        "speaking":  {"core": "#ffd28a", "rim": "#7a5a2a", "glow": "#3a2a08",
                      "hot": "#fff8e0", "ring": "#9eecff"},
        "executing": {"core": "#b59cff", "rim": "#5a3a8c", "glow": "#241a4a",
                      "hot": "#f3eaff", "ring": "#7fdfff"},
        "error":     {"core": "#ff8a9c", "rim": "#7a2a3a", "glow": "#3a0a16",
                      "hot": "#ffe6ec", "ring": "#ff8a9c"},
    }

    # Pulse amplitude per state.
    _PULSE = {
        "idle": 0.03, "listening": 0.07, "thinking": 0.06,
        "speaking": 0.10, "executing": 0.09, "error": 0.14,
    }

    def __init__(self, master, **kw):
        super().__init__(master, bg=Palette.BG_PANEL, highlightthickness=0, **kw)
        self._state = "idle"
        self._phase = 0.0
        self._items: dict[str, int | list[int]] = {}
        self._particles: list[dict] = []
        self._eq_bars: list[tuple[int, float]] = []
        self._ring_labels: list[int] = []
        self._geom: dict[str, float] = {}
        self._audio_amp = 0.0  # smoothed amplitude 0..1, set by UI
        self._loop_started = False
        self.bind("<Configure>", lambda e: self._redraw())

    # -- public state API ---------------------------------------------------

    def set_state(self, state: str) -> None:
        if state not in self._PAL:
            state = "idle"
        if state == self._state:
            return
        self._state = state
        self._apply_palette()

    def state(self) -> str:
        return self._state

    def set_audio_amplitude(self, amp: float) -> None:
        """0..1 audio amplitude. Drives orb pulse + visualiser bars."""
        self._audio_amp = max(0.0, min(1.0, amp))

    # -- helpers ------------------------------------------------------------

    def _hex_to_rgb(self, hex_str: str) -> tuple[int, int, int]:
        s = hex_str.lstrip("#")
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

    def _rgb(self, r: int, g: int, b: int) -> str:
        return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"

    def _lerp(self, a: str, b: str, t: float) -> str:
        ra, ga, ba = self._hex_to_rgb(a)
        rb, gb, bb = self._hex_to_rgb(b)
        return self._rgb(int(ra + (rb - ra) * t),
                         int(ga + (gb - ga) * t),
                         int(ba + (bb - ba) * t))

    # -- initial layout -----------------------------------------------------

    def _redraw(self) -> None:
        self.delete("all")
        self._items.clear()
        self._eq_bars.clear()
        self._ring_labels.clear()

        w = max(self.winfo_width(), 320)
        h = max(self.winfo_height(), 360)
        cx, cy = w / 2, h / 2 - 6  # nudge up so reflection + bars fit below
        self._geom = {"cx": cx, "cy": cy, "w": w, "h": h}

        pal = self._PAL[self._state]

        # ── 1. Soft outer halo (very large, low intensity) ─────────────
        for i, r in enumerate((195, 175, 155)):
            self._items[f"halo_{i}"] = self.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=pal["glow"], outline="",
            )

        # ── 2. Floor reflection (ellipse below the orb, mirrored) ──────
        # Drawn first so the orb body covers most of it.
        self._items["floor"] = self.create_oval(
            cx - 78, cy + 95, cx + 78, cy + 130,
            fill=pal["glow"], outline="",
        )

        # ── 3. Three orbiting rings (perspective-tilted ellipses) ──────
        # Each is drawn as a thin oval outline so the tilt reads as 3D.
        for i, (key, ry_factor, dash, color) in enumerate([
            ("ring_outer", 0.34, (1, 5),  pal["ring"]),
            ("ring_mid",   0.46, (4, 3),  pal["core"]),
            ("ring_inner", 0.62, (2, 4),  pal["ring"]),
        ]):
            rx = 162 - i * 14
            ry = max(int(rx * ry_factor), 8)
            self._items[key] = self.create_oval(
                cx - rx, cy - ry, cx + rx, cy + ry,
                outline=color, width=1, dash=dash,
            )

        # ── 4. Tick marks around the orb's equator ─────────────────────
        tick_items: list[int] = []
        tick_count = 60
        r_in = 138
        r_out = 146
        for i in range(tick_count):
            a = (i / tick_count) * math.tau
            x0 = cx + r_in * math.cos(a)
            y0 = cy + r_in * math.sin(a)
            x1 = cx + r_out * math.cos(a)
            y1 = cy + r_out * math.sin(a)
            tick_items.append(self.create_line(
                x0, y0, x1, y1, fill=pal["ring"], width=1,
            ))
        self._items["ticks"] = tick_items

        # ── 5. Glass orb body — gradient sphere faked with stacked discs ─
        # Outer dark rim (slightly bigger than core — gives the edge).
        self._items["rim"] = self.create_oval(
            cx - 96, cy - 96, cx + 96, cy + 96,
            fill=pal["rim"], outline="",
        )
        # Glass body — mid-tone, slightly smaller.
        self._items["body"] = self.create_oval(
            cx - 88, cy - 88, cx + 88, cy + 88,
            fill=self._lerp(pal["rim"], pal["core"], 0.18), outline="",
        )
        # Inner accent sphere — gives glass depth.
        self._items["core"] = self.create_oval(
            cx - 64, cy - 64, cx + 64, cy + 64,
            fill=self._lerp(pal["rim"], pal["core"], 0.55), outline="",
        )
        # Inner-most glow.
        self._items["inner"] = self.create_oval(
            cx - 44, cy - 44, cx + 44, cy + 44,
            fill=pal["core"], outline="",
        )
        # Hot spot.
        self._items["hot"] = self.create_oval(
            cx - 12, cy - 12, cx + 12, cy + 12,
            fill=pal["hot"], outline="",
        )

        # ── 6. Specular highlight — upper-left of the orb ──────────────
        # A soft white ellipse + a smaller brighter ellipse.
        self._items["spec"] = self.create_oval(
            cx - 70, cy - 80, cx - 10, cy - 30,
            fill=Palette.SPEC_SOFT, outline="",
        )
        self._items["spec_hot"] = self.create_oval(
            cx - 58, cy - 70, cx - 28, cy - 45,
            fill=Palette.SPEC_HIGHLIGHT, outline="",
        )
        # Small bright dot near the top — "fresnel" reflection.
        self._items["spec_dot"] = self.create_oval(
            cx - 22, cy - 80, cx - 14, cy - 72,
            fill=Palette.SPEC_HIGHLIGHT, outline="",
        )

        # ── 7. Bottom-right rim shadow — fakes depth shading ───────────
        self._items["shadow"] = self.create_arc(
            cx - 88, cy - 88, cx + 88, cy + 88,
            start=290, extent=120, style=tk.PIESLICE,
            fill="#0a1130", outline="",
        )

        # ── 8. Equatorial refraction line — a thin highlight across ────
        self._items["equator"] = self.create_arc(
            cx - 88, cy - 6, cx + 88, cy + 6,
            start=200, extent=140, style=tk.ARC,
            outline=Palette.SPEC_SOFT, width=1,
        )
        self._items["equator2"] = self.create_arc(
            cx - 70, cy - 3, cx + 70, cy + 3,
            start=210, extent=120, style=tk.ARC,
            outline=Palette.SPEC_HIGHLIGHT, width=1,
        )

        # ── 9. Spiral arc inside the orb ────────────────────────────────
        self._items["spiral"] = self.create_arc(
            cx - 36, cy - 36, cx + 36, cy + 36,
            start=20, extent=110, style=tk.ARC,
            outline=pal["hot"], width=1,
        )

        # ── 10. HUD ring labels around the orb ─────────────────────────
        self._draw_ring_labels(cx, cy, pal["ring"])

        # ── 11. Radial audio bars beyond the rings ─────────────────────
        bar_count = 64
        bar_r0 = 156
        for i in range(bar_count):
            a = (i / bar_count) * math.tau
            x0 = cx + bar_r0 * math.cos(a)
            y0 = cy + bar_r0 * math.sin(a)
            x1 = x0 + 7 * math.cos(a)
            y1 = y0 + 7 * math.sin(a)
            bar = self.create_line(x0, y0, x1, y1, fill=pal["ring"],
                                   width=2, tags="eq")
            self._eq_bars.append((bar, a))

        # ── 12. Reset + spawn particles ────────────────────────────────
        for p in self._particles:
            try:
                self.delete(p["id"])
            except Exception:
                pass
        self._particles = []
        for _ in range(34):
            self._spawn_particle()

        # ── 13. Reflection sweep (a thin diagonal highlight across) ────
        self._items["sweep"] = self.create_rectangle(
            0, 0, w, 1, fill=Palette.BG_GLASS_HIGHLIGHT, outline="",
        )

        # ── 14. Caption text below the orb ─────────────────────────────
        self._items["caption"] = self.create_text(
            cx, cy + 180,
            text="",
            fill=pal["core"],
            font=("Segoe UI", 11, "bold"),
        )

        if not self._loop_started:
            self._loop_started = True
            self._animate()

    def _draw_ring_labels(self, cx: float, cy: float, color: str) -> None:
        """Drop a few short technical labels around the orb."""
        labels = [
            ("CORE-01", 0.0, 165),
            ("0xA7FF", 0.7, 165),
            ("SYNC OK", 1.6, 165),
            ("LAT 04ms", 2.5, 165),
            ("MEM 64%", 3.6, 165),
            ("v3.07", 4.7, 165),
            ("NEURAL LINK", 5.4, 165),
        ]
        for text, ang, r in labels:
            a = ang
            x = cx + r * math.cos(a)
            y = cy + r * math.sin(a)
            t = self.create_text(
                x, y, text=text, fill=Palette.TEXT_GHOST,
                font=("Consolas", 8), anchor="center",
            )
            self._ring_labels.append(t)

    def _spawn_particle(self) -> None:
        g = self._geom
        if not g:
            return
        angle = random.uniform(0, math.tau)
        radius = random.uniform(70, 120)
        x = g["cx"] + radius * math.cos(angle)
        y = g["cy"] + radius * math.sin(angle)
        size = random.uniform(0.8, 2.2)
        color = random.choice([
            Palette.NEON_CYAN, Palette.NEON_PINK,
            Palette.NEON_GREEN, Palette.NEON_AMBER,
            Palette.NEON_AQUA, Palette.SPEC_HIGHLIGHT,
        ])
        item = self.create_oval(
            x - size, y - size, x + size, y + size,
            fill=color, outline="",
        )
        self._particles.append({
            "id": item,
            "angle": angle,
            "radius": radius,
            "speed": random.uniform(0.15, 0.6) * (1 if random.random() > 0.5 else -1),
            "twinkle": random.uniform(0, math.tau),
        })

    # -- animation loop -----------------------------------------------------

    def _animate(self) -> None:
        try:
            self._phase += 0.06
            g = self._geom
            if not g:
                self.after(33, self._animate)
                return
            cx, cy = g["cx"], g["cy"]
            w, h = g["w"], g["h"]
            state = self._state
            pal = self._PAL[state]

            base_pulse = self._PULSE[state]
            pulse = 1.0 + base_pulse * math.sin(self._phase * 1.6)
            pulse += 0.10 * self._audio_amp

            # ── Orb body pulse (rim + body + core + inner scale together)
            body_scale = 88 * pulse
            rim_r = 96 * pulse
            core_r = 64 * pulse
            inner_r = 44 * pulse
            self.coords(self._items["rim"],
                        cx - rim_r, cy - rim_r, cx + rim_r, cy + rim_r)
            self.coords(self._items["body"],
                        cx - body_scale, cy - body_scale,
                        cx + body_scale, cy + body_scale)
            self.coords(self._items["core"],
                        cx - core_r, cy - core_r, cx + core_r, cy + core_r)
            self.coords(self._items["inner"],
                        cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r)
            self.coords(self._items["hot"],
                        cx - 12, cy - 12, cx + 12, cy + 12)

            # ── Halo pulse
            for i, base_r in enumerate((155, 175, 195)):
                r = int(base_r * (1 + 0.02 * math.sin(self._phase * 1.4 + i)))
                self.coords(self._items[f"halo_{i}"],
                            cx - r, cy - r, cx + r, cy + r)

            # ── Specular highlights wobble slightly with the orb ──────
            wob = 0
            if state == "speaking":
                wob = int(3 * math.sin(self._phase * 7))
            self.move(self._items["spec"], wob, 0)
            self.move(self._items["spec_hot"], wob, 0)
            self.move(self._items["spec_dot"], wob, 0)
            # Drift the spec up/down a hair.
            self.move(self._items["spec"], 0, int(0.5 * math.sin(self._phase * 0.6)))
            self.move(self._items["spec_hot"], 0, int(0.5 * math.sin(self._phase * 0.6)))
            # Keep the spec inside the orb bounds — clamp by recentering
            # every so often so it doesn't drift off.
            x0, y0, x1, y1 = self.coords(self._items["spec"])
            if x0 < cx - 96 or x1 > cx + 96:
                target_x = cx - 70
                target_y = cy - 80
                self.coords(self._items["spec"],
                            target_x, target_y,
                            target_x + 60, target_y + 50)
                self.coords(self._items["spec_hot"],
                            target_x + 12, target_y + 10,
                            target_x + 42, target_y + 35)
                self.coords(self._items["spec_dot"],
                            cx - 22, cy - 80, cx - 14, cy - 72)

            # ── Spiral inside the core ────────────────────────────────
            spiral_start = (self._phase * 22) % 360
            extent = 90 + 25 * math.sin(self._phase * 1.2)
            self.itemconfigure(self._items["spiral"],
                               start=spiral_start, extent=extent)

            # ── Orbiting rings — parallax scaling ─────────────────────
            for key, factor in (("ring_outer", 1.0), ("ring_inner", -1.0)):
                k = self._items[key]
                scale = 1.0 + 0.012 * math.sin(self._phase * 2 * factor)
                x0, y0, x1, y1 = self.coords(k)
                bw = (x1 - x0) * scale
                bh = (y1 - y0) * scale
                self.coords(k, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            # Mid ring spins the other way around the long axis.
            k = self._items["ring_mid"]
            scale_x = 1.0 + 0.018 * math.sin(self._phase * 1.3)
            scale_y = 1.0 + 0.05 * math.sin(self._phase * 2.1)
            x0, y0, x1, y1 = self.coords(k)
            self.coords(k, cx - (x1 - x0) * scale_x / 2,
                           cy - (y1 - y0) * scale_y / 2,
                           cx + (x1 - x0) * scale_x / 2,
                           cy + (y1 - y0) * scale_y / 2)

            # ── Equator highlight pulse ──────────────────────────────
            eq_amp = 1.0 + 0.18 * math.sin(self._phase * 2)
            self.itemconfigure(self._items["equator"],
                               outline=self._lerp(Palette.SPEC_SOFT,
                                                  Palette.SPEC_HIGHLIGHT,
                                                  0.5 + 0.4 * math.sin(self._phase * 3)))

            # ── Tick ring rotation ────────────────────────────────────
            ticks = self._items["ticks"]
            tick_count = len(ticks)
            for i, item in enumerate(ticks):
                a = ((i / tick_count) + self._phase * 0.025) * math.tau
                r_in, r_out = 138, 146
                x0 = cx + r_in * math.cos(a)
                y0 = cy + r_in * math.sin(a)
                x1 = cx + r_out * math.cos(a)
                y1 = cy + r_out * math.sin(a)
                self.coords(item, x0, y0, x1, y1)

            # ── Audio bars ────────────────────────────────────────────
            eq_scale = {
                "idle": 6, "listening": 14, "thinking": 10,
                "speaking": 22, "executing": 16, "error": 18,
            }[state]
            bar_count = len(self._eq_bars)
            for i, (bar, a) in enumerate(self._eq_bars):
                amp = eq_scale + 6 * math.sin(self._phase * 1.5 + i * 0.4)
                amp += 18 * self._audio_amp * (
                    0.6 + 0.4 * math.sin(self._phase * 4 + i * 0.3)
                )
                r0 = 156
                r1 = 156 + amp
                x0 = cx + r0 * math.cos(a)
                y0 = cy + r0 * math.sin(a)
                x1 = cx + r1 * math.cos(a)
                y1 = cy + r1 * math.sin(a)
                self.coords(bar, x0, y0, x1, y1)
                if state == "speaking":
                    self.itemconfigure(
                        bar,
                        fill=(Palette.NEON_AMBER
                              if i % 2 else Palette.NEON_CYAN),
                    )
                elif state == "error":
                    self.itemconfigure(bar, fill=Palette.NEON_RED)
                else:
                    self.itemconfigure(bar, fill=pal["ring"])

            # ── Particles ─────────────────────────────────────────────
            new_particles: list[dict] = []
            for p in self._particles:
                p["angle"] += 0.010 * p["speed"]
                r = p["radius"] + 5 * math.sin(self._phase + p["twinkle"])
                x = cx + r * math.cos(p["angle"])
                y = cy + r * math.sin(p["angle"])
                p["twinkle"] += 0.1
                size = 0.8 + 1.3 * (0.5 + 0.5 * math.sin(p["twinkle"]))
                self.coords(p["id"], x - size, y - size, x + size, y + size)
                if r > 165 or r < 55:
                    self.delete(p["id"])
                    self._spawn_particle()
                else:
                    new_particles.append(p)
            self._particles = new_particles

            # ── Reflection sweep across the whole canvas ──────────────
            sweep_y = (self._phase * 50) % (h + 80) - 40
            self.coords(self._items["sweep"], 0, sweep_y, w, sweep_y + 1)
            self.itemconfigure(
                self._items["sweep"],
                fill=(Palette.NEON_AQUA
                      if state != "error" else Palette.NEON_RED),
            )

            # ── Caption ───────────────────────────────────────────────
            captions = {
                "idle":      "● IDLE",
                "listening": "MIC ACTIVE",
                "thinking":  "PROCESSING",
                "speaking":  "VOICE OUTPUT",
                "executing": "EXECUTING",
                "error":     "ERROR",
            }
            self.itemconfigure(
                self._items["caption"],
                text=captions[state],
                fill=pal["core"],
            )
        except Exception:
            pass
        self.after(33, self._animate)

    def _apply_palette(self) -> None:
        pal = self._PAL[self._state]
        try:
            self.itemconfigure(self._items["core"],
                               fill=self._lerp(pal["rim"], pal["core"], 0.55))
            self.itemconfigure(self._items["inner"], fill=pal["core"])
            self.itemconfigure(self._items["rim"], fill=pal["rim"])
            self.itemconfigure(self._items["body"],
                               fill=self._lerp(pal["rim"], pal["core"], 0.18))
            self.itemconfigure(self._items["ring_outer"], outline=pal["ring"])
            self.itemconfigure(self._items["ring_mid"], outline=pal["core"])
            self.itemconfigure(self._items["ring_inner"], outline=pal["ring"])
            self.itemconfigure(self._items["spiral"], outline=pal["hot"])
            for i, paint in enumerate((pal["glow"], pal["glow"], pal["glow"])):
                self.itemconfigure(self._items[f"halo_{i}"], fill=paint)
            for bar, _ in self._eq_bars:
                self.itemconfigure(bar, fill=pal["ring"])
            for t in self._ring_labels:
                self.itemconfigure(t, fill=Palette.TEXT_GHOST)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# System monitor widget — CPU/RAM/disk/network/battery/temperature bars.
# ---------------------------------------------------------------------------

class SystemMonitor(ctk.CTkFrame):
    """A small panel of animated usage bars + a sparkline."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent",
                         corner_radius=12, border_width=0, **kw)
        # Glass background.
        self._glass = GlassPanel(self)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        # Steel border.
        self._border = tk.Frame(self, bg=Palette.BORDER_METAL,
                                highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        # Corner brackets.
        self._corners = HudCorners(self, color=Palette.NEON_CYAN,
                                   bg=Palette.BG_PANEL, height=22)
        self._corners.place(x=0, y=0, relwidth=1, height=22)
        self._net_prev = None
        self._cpu_hist: list[float] = [0] * 30
        self._ram_hist: list[float] = [0] * 30
        self._last_update = 0.0
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(
            self, text="◤ SYSTEM TELEMETRY",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        )
        title.grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")

        # Sparkline (small canvas, 200x36).
        self._spark = tk.Canvas(
            self, bg=Palette.BG_GLASS, highlightthickness=0, height=42,
        )
        self._spark.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")
        self._spark.bind("<Configure>", lambda e: self._draw_spark())

        # CPU/RAM rows.
        self._rows_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._rows_frame.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")
        self._rows_frame.grid_columnconfigure(0, weight=1)
        self._rows: dict[str, dict] = {}
        for i, key in enumerate(("CPU", "RAM", "DISK")):
            row = ctk.CTkFrame(self._rows_frame, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(1, weight=1)

            lbl = ctk.CTkLabel(
                row, text=key, width=46, anchor="w",
                font=("Consolas", 10, "bold"),
                text_color=Palette.TEXT_DIM,
            )
            lbl.grid(row=0, column=0, padx=(0, 8))

            track = tk.Canvas(row, bg=Palette.BG_DEEP, highlightthickness=0,
                              height=8)
            track.grid(row=0, column=1, sticky="ew")

            value = ctk.CTkLabel(
                row, text="--%", width=58, anchor="e",
                font=("Consolas", 10, "bold"),
                text_color=Palette.NEON_CYAN,
            )
            value.grid(row=0, column=2, padx=(8, 0))

            self._rows[key] = {"track": track, "value": value}

        # Network / battery row.
        self._net_lbl = ctk.CTkLabel(
            self, text="NET ↓ -- KB/s   ↑ -- KB/s",
            font=("Consolas", 10), text_color=Palette.TEXT_DIM,
            anchor="w",
        )
        self._net_lbl.grid(row=3, column=0, padx=12, pady=(2, 2), sticky="ew")

        self._bat_lbl = ctk.CTkLabel(
            self, text="BAT --   TEMP --°C",
            font=("Consolas", 10), text_color=Palette.TEXT_DIM,
            anchor="w",
        )
        self._bat_lbl.grid(row=4, column=0, padx=12, pady=(0, 10), sticky="ew")

        self.after(200, self._tick)

    def _draw_bars(self) -> None:
        # Update the three horizontal tracks.
        for key, pct in (
            ("CPU", self._cpu_hist[-1] if self._cpu_hist else 0),
            ("RAM", self._ram_hist[-1] if self._ram_hist else 0),
            ("DISK", self._rows["DISK"].get("pct", 0)),
        ):
            data = self._rows[key]
            track = data["track"]
            track.delete("bar")
            track.delete("frame")
            w = max(track.winfo_width(), 1)
            h = max(track.winfo_height(), 1)
            track.create_rectangle(
                0, 0, w, h, fill=Palette.BG_DEEP, outline="",
                tags="frame",
            )
            track.create_rectangle(
                0, 0, int(w * pct / 100), h,
                fill=Palette.NEON_CYAN, outline="",
                tags="bar",
            )
            data["value"].configure(text=f"{int(pct):>3d}%")

    def _draw_spark(self) -> None:
        c = self._spark
        c.delete("line")
        w = max(c.winfo_width(), 1)
        h = max(c.winfo_height(), 1)
        if not self._cpu_hist:
            return
        n = len(self._cpu_hist)
        step = max(w / (n - 1), 1)
        pts = []
        for i, v in enumerate(self._cpu_hist):
            x = i * step
            y = h - (v / 100.0) * (h - 4) - 2
            pts.extend([x, y])
        c.create_line(*pts, fill=Palette.NEON_CYAN, width=1, tags="line")
        # Faint secondary line for RAM.
        pts2 = []
        for i, v in enumerate(self._ram_hist):
            x = i * step
            y = h - (v / 100.0) * (h - 4) - 2
            pts2.extend([x, y])
        c.create_line(*pts2, fill=Palette.NEON_PINK, width=1, tags="line")

    def _tick(self) -> None:
        now = time.time()
        if now - self._last_update > 1.0:
            self._sample()
            self._last_update = now
            self._draw_bars()
            self._draw_spark()
        self.after(250, self._tick)

    def _sample(self) -> None:
        cpu = 0.0
        ram = 0.0
        disk = 0.0
        net_down = 0.0
        net_up = 0.0
        battery = "--"
        temp = "--"

        if _HAS_PSUTIL:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent
                net = psutil.net_io_counters()
                cur = (net.bytes_recv, net.bytes_sent)
                if self._net_prev:
                    dt = max(time.time() - self._last_update, 0.001)
                    net_down = (cur[0] - self._net_prev[0]) / dt / 1024.0
                    net_up = (cur[1] - self._net_prev[1]) / dt / 1024.0
                self._net_prev = cur

                batt = psutil.sensors_battery()
                if batt is not None:
                    plug = " ⚡" if batt.power_plugged else ""
                    battery = f"{int(batt.percent)}%{plug}"

                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for entries in temps.values():
                            for entry in entries:
                                if entry.current is not None:
                                    temp = f"{int(entry.current)}"
                                    break
                            if temp != "--":
                                break
            except Exception:
                pass

        self._cpu_hist.append(cpu)
        self._ram_hist.append(ram)
        if len(self._cpu_hist) > 30:
            self._cpu_hist.pop(0)
        if len(self._ram_hist) > 30:
            self._ram_hist.pop(0)
        self._rows["DISK"]["pct"] = disk

        self._net_lbl.configure(
            text=f"NET ↓ {net_down:5.1f} KB/s   ↑ {net_up:5.1f} KB/s",
        )
        self._bat_lbl.configure(
            text=f"BAT {battery}   TEMP {temp}°C",
        )


# ---------------------------------------------------------------------------
# Command console — animated voice intent / execution pipeline.
# ---------------------------------------------------------------------------

class CommandConsole(ctk.CTkFrame):
    """A panel that visualises the lifecycle of a voice command:
    VOICE INPUT → INTENT DETECTED → EXECUTING → SUCCESS / FAIL."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent",
                         corner_radius=12, border_width=0, **kw)
        self._glass = GlassPanel(self)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        self._border = tk.Frame(self, bg=Palette.BORDER_METAL,
                                highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        self._corners = HudCorners(self, color=Palette.NEON_PINK,
                                   bg=Palette.BG_PANEL, height=22)
        self._corners.place(x=0, y=0, relwidth=1, height=22)
        self._stages: list[dict] = []
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self, text="◤ COMMAND EXECUTION",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")

        for i, (tag, default_text) in enumerate([
            ("VOICE INPUT",      "—"),
            ("INTENT DETECTED",  "—"),
            ("EXECUTING",        "—"),
            ("RESULT",           "—"),
        ], start=1):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.grid(row=i, column=0, padx=12, pady=3, sticky="ew")
            row.grid_columnconfigure(1, weight=1)

            dot = tk.Canvas(row, bg=Palette.BG_GLASS, highlightthickness=0,
                            width=12, height=12)
            dot.grid(row=0, column=0, padx=(0, 8))
            dot.create_oval(2, 2, 10, 10, fill=Palette.BORDER_DIM,
                            outline="", tags="d")

            tag_lbl = ctk.CTkLabel(
                row, text=tag, font=("Consolas", 10, "bold"),
                text_color=Palette.TEXT_DIM, width=140, anchor="w",
            )
            tag_lbl.grid(row=0, column=1, padx=(0, 8))

            val_lbl = ctk.CTkLabel(
                row, text=default_text,
                font=("Consolas", 11),
                text_color=Palette.TEXT_BRIGHT, anchor="w",
            )
            val_lbl.grid(row=0, column=2, sticky="ew")

            self._stages.append({
                "tag": tag_lbl, "val": val_lbl, "dot": dot,
                "state": "idle",  # idle / active / done / fail
            })

    def reset(self) -> None:
        for stage in self._stages:
            stage["state"] = "idle"
            stage["tag"].configure(text_color=Palette.TEXT_DIM)
            stage["val"].configure(text="—", text_color=Palette.TEXT_BRIGHT)
            stage["dot"].itemconfigure("d", fill=Palette.BORDER_DIM)

    def set_stage(self, index: int, value: str, state: str = "active") -> None:
        if not (0 <= index < len(self._stages)):
            return
        stage = self._stages[index]
        stage["val"].configure(text=value)
        stage["state"] = state
        if state == "active":
            stage["tag"].configure(text_color=Palette.NEON_CYAN)
            stage["val"].configure(text_color=Palette.TEXT_BRIGHT)
            stage["dot"].itemconfigure("d", fill=Palette.NEON_CYAN)
        elif state == "done":
            stage["tag"].configure(text_color=Palette.NEON_GREEN)
            stage["val"].configure(text_color=Palette.TEXT_BRIGHT)
            stage["dot"].itemconfigure("d", fill=Palette.NEON_GREEN)
        elif state == "fail":
            stage["tag"].configure(text_color=Palette.NEON_RED)
            stage["val"].configure(text_color=Palette.NEON_RED)
            stage["dot"].itemconfigure("d", fill=Palette.NEON_RED)
        else:
            stage["tag"].configure(text_color=Palette.TEXT_DIM)
            stage["val"].configure(text_color=Palette.TEXT_BRIGHT)
            stage["dot"].itemconfigure("d", fill=Palette.BORDER_DIM)

    def animate(self, values: list[tuple[int, str, str]],
                on_done: Callable[[], None] | None = None,
                delay: float = 0.35) -> None:
        """Animate through the given stage updates.

        ``values`` is a list of (index, text, state) tuples. Each one is
        applied after ``delay`` seconds; once the last one is applied
        ``on_done`` is invoked (if given)."""
        def _step(i: int) -> None:
            if i >= len(values):
                if on_done:
                    on_done()
                return
            idx, text, state = values[i]
            self.set_stage(idx, text, state)
            self.after(int(delay * 1000), lambda: _step(i + 1))
        _step(0)


# ---------------------------------------------------------------------------
# Thinking pipeline — visualises the Ollama "thinking…" stage.
# ---------------------------------------------------------------------------

class ThinkingPipeline(ctk.CTkFrame):
    """Shows a 5-step pipeline (ANALYZING → UNDERSTANDING → SELECTING TOOL
    → EXECUTING ACTION → GENERATING RESPONSE) with a moving highlight."""

    STEPS = (
        "ANALYZING REQUEST",
        "UNDERSTANDING INTENT",
        "SELECTING TOOL",
        "EXECUTING ACTION",
        "GENERATING RESPONSE",
    )

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent",
                         corner_radius=12, border_width=0, **kw)
        self._glass = GlassPanel(self)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        self._border = tk.Frame(self, bg=Palette.BORDER_METAL,
                                highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        self._corners = HudCorners(self, color=Palette.NEON_AMBER,
                                   bg=Palette.BG_PANEL, height=22)
        self._corners.place(x=0, y=0, relwidth=1, height=22)
        self._active = -1
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self, text="◤ AI PROCESSING PIPELINE",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")

        self._rows: list[ctk.CTkLabel] = []
        for i, step in enumerate(self.STEPS, start=1):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.grid(row=i, column=0, padx=12, pady=2, sticky="ew")
            row.grid_columnconfigure(1, weight=1)

            indicator = ctk.CTkLabel(
                row, text="◇", width=18, anchor="w",
                font=("Consolas", 12, "bold"),
                text_color=Palette.TEXT_GHOST,
            )
            indicator.grid(row=0, column=0, padx=(0, 8))

            label = ctk.CTkLabel(
                row, text=step, font=("Consolas", 11),
                text_color=Palette.TEXT_DIM, anchor="w",
            )
            label.grid(row=0, column=1, sticky="ew")

            self._rows.append((indicator, label))

    def reset(self) -> None:
        self._active = -1
        for ind, lbl in self._rows:
            ind.configure(text="◇", text_color=Palette.TEXT_GHOST)
            lbl.configure(text_color=Palette.TEXT_DIM)

    def set_active(self, index: int) -> None:
        """Mark step ``index`` as currently running and earlier steps as
        completed. Index 0..len(STEPS)-1; pass -1 to clear."""
        if index == self._active:
            return
        self._active = index
        for i, (ind, lbl) in enumerate(self._rows):
            if i < index:
                ind.configure(text="◆", text_color=Palette.NEON_GREEN)
                lbl.configure(text_color=Palette.TEXT_DIM)
            elif i == index:
                ind.configure(text="◈", text_color=Palette.NEON_CYAN)
                lbl.configure(text_color=Palette.TEXT_BRIGHT)
            else:
                ind.configure(text="◇", text_color=Palette.TEXT_GHOST)
                lbl.configure(text_color=Palette.TEXT_DIM)


# ---------------------------------------------------------------------------
# System status list (left side panel)
# ---------------------------------------------------------------------------

class SystemStatusPanel(ctk.CTkFrame):
    """A small column of status rows: AI CORE, VOICE ENGINE, OLLAMA,
    MEMORY, TTS, MIC."""

    ITEMS = (
        ("AI CORE",       "ONLINE"),
        ("VOICE ENGINE",  "ONLINE"),
        ("OLLAMA",        "CONNECTED"),
        ("MEMORY",        "ACTIVE"),
        ("TTS",           "ONLINE"),
        ("MIC",           "STANDBY"),
    )

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent",
                         corner_radius=14, border_width=0, **kw)
        self._glass = GlassPanel(self)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        self._border = tk.Frame(self, bg=Palette.BORDER_METAL,
                                highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        self._corners = HudCorners(self, color=Palette.NEON_GREEN,
                                   bg=Palette.BG_PANEL, height=22)
        self._corners.place(x=0, y=0, relwidth=1, height=22)
        self._rows: dict[str, dict] = {}
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self, text="◤ SYSTEM STATUS",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")

        for i, (name, default) in enumerate(self.ITEMS, start=1):
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.grid(row=i, column=0, padx=14, pady=4, sticky="ew")
            row.grid_columnconfigure(1, weight=1)

            dot = tk.Canvas(row, bg=Palette.BG_PANEL, highlightthickness=0,
                            width=10, height=10)
            dot.grid(row=0, column=0, padx=(0, 10))
            dot.create_oval(2, 2, 8, 8, fill=Palette.NEON_GREEN, outline="",
                            tags="d")

            name_lbl = ctk.CTkLabel(
                row, text=name, font=("Consolas", 10, "bold"),
                text_color=Palette.TEXT_DIM, anchor="w",
            )
            name_lbl.grid(row=0, column=1, sticky="ew")

            val_lbl = ctk.CTkLabel(
                row, text=default, font=("Consolas", 10, "bold"),
                text_color=Palette.NEON_GREEN, anchor="e",
            )
            val_lbl.grid(row=0, column=2, padx=(8, 0))

            self._rows[name] = {"dot": dot, "val": val_lbl,
                                "name": name_lbl}

    def set_status(self, name: str, value: str, color: str | None = None) -> None:
        if name not in self._rows:
            return
        row = self._rows[name]
        row["val"].configure(text=value)
        c = color or Palette.NEON_GREEN
        row["val"].configure(text_color=c)
        row["dot"].itemconfigure("d", fill=c)


# ---------------------------------------------------------------------------
# Recent commands — animated list on the right side.
# ---------------------------------------------------------------------------

class RecentCommands(ctk.CTkScrollableFrame):
    """Scrollable list of the last few commands. Each item slides in."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent",
                         corner_radius=10, border_width=0, **kw)
        self._glass = GlassPanel(self)
        self._glass.place(x=0, y=0, relwidth=1, relheight=1)
        self._border = tk.Frame(self, bg=Palette.BORDER_METAL,
                                highlightthickness=0)
        self._border.place(x=0, y=0, relwidth=1, relheight=1)
        self._items: list[ctk.CTkFrame] = []
        self._max_items = 8

    def add(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        frame = ctk.CTkFrame(self, fg_color=Palette.BG_GLASS,
                             corner_radius=8, border_width=1,
                             border_color=Palette.BORDER_METAL)
        frame.grid(row=len(self._items), column=0, padx=4, pady=3,
                   sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ts_lbl = ctk.CTkLabel(
            frame, text=ts, width=64,
            font=("Consolas", 9, "bold"),
            text_color=Palette.NEON_CYAN, anchor="w",
        )
        ts_lbl.grid(row=0, column=0, padx=(8, 6), pady=6)

        txt_lbl = ctk.CTkLabel(
            frame, text=text,
            font=("Consolas", 10),
            text_color=Palette.TEXT_BRIGHT, anchor="w",
            wraplength=240, justify="left",
        )
        txt_lbl.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="ew")

        self._items.append(frame)
        # Trim old items.
        while len(self._items) > self._max_items:
            old = self._items.pop(0)
            old.destroy()
            for i, it in enumerate(self._items):
                it.grid(row=i, column=0)


# ---------------------------------------------------------------------------
# Boot sequence overlay
# ---------------------------------------------------------------------------

class BootSequence(ctk.CTkFrame):
    """A full-window overlay that plays a 2-3 second cinematic boot and
    fades itself out when finished."""

    STEPS = (
        ("INITIALIZING ARVIS CORE", 320),
        ("LOADING VOICE ENGINE",     260),
        ("LOADING AI ENGINE",        260),
        ("CONNECTING OLLAMA",        260),
        ("LOADING MEMORY",           220),
        ("INITIALIZING SYSTEM CTRL", 240),
    )

    def __init__(self, master, on_done: Callable[[], None], **kw):
        super().__init__(master, fg_color=Palette.BG_VOID, **kw)
        self._on_done = on_done
        self._step_index = 0
        self._build()
        self.after(120, self._step)

    def _build(self) -> None:
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Soft radial glow behind the title — a low-saturation halo.
        self._glow = tk.Canvas(self, bg=Palette.BG_VOID, highlightthickness=0)
        self._glow.place(x=0, y=0, relwidth=1, relheight=1)
        self._glow.bind("<Configure>", lambda e: self._draw_glow())

        center = ctk.CTkFrame(self, fg_color="transparent")
        center.grid(row=0, column=0, sticky="nsew")
        center.grid_columnconfigure(0, weight=1)

        # Outer glass card for the boot UI.
        boot_card = ctk.CTkFrame(center, fg_color="transparent",
                                 corner_radius=16, border_width=0)
        boot_card.grid(row=0, column=0, padx=120, pady=40, sticky="nsew")
        bg = GlassPanel(boot_card)
        bg.place(x=0, y=0, relwidth=1, relheight=1)
        bd = tk.Frame(boot_card, bg=Palette.BORDER_METAL,
                      highlightthickness=0)
        bd.place(x=0, y=0, relwidth=1, relheight=1)
        co = HudCorners(boot_card, color=Palette.NEON_CYAN,
                        bg=Palette.BG_PANEL, height=22)
        co.place(x=0, y=0, relwidth=1, height=22)
        inner = ctk.CTkFrame(boot_card, fg_color="transparent")
        inner.place(relx=0, rely=0, relwidth=1, relheight=1)
        inner.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            inner, text="◢ ARVIS",
            font=("Segoe UI", 44, "bold"),
            text_color=Palette.NEON_CYAN,
        )
        title.grid(row=0, column=0, pady=(28, 8))

        sub = ctk.CTkLabel(
            inner, text="ADAPTIVE VOICE INTERFACE · v3",
            font=("Consolas", 12),
            text_color=Palette.TEXT_DIM,
        )
        sub.grid(row=1, column=0, pady=(0, 28))

        # Progress bar with glass + metallic feel.
        self._bar_outer = ctk.CTkFrame(
            inner, fg_color=Palette.BG_GLASS, height=12, corner_radius=6,
            border_width=1, border_color=Palette.BORDER_METAL,
        )
        self._bar_outer.grid(row=2, column=0, padx=80, pady=6, sticky="ew")

        self._bar_fill = ctk.CTkFrame(
            self._bar_outer, fg_color=Palette.NEON_CYAN,
            corner_radius=4, width=0, height=10,
        )
        self._bar_fill.place(x=1, y=1, relheight=1)

        # Status text.
        self._status = ctk.CTkLabel(
            inner, text="INITIALIZING ARVIS CORE",
            font=("Consolas", 11, "bold"),
            text_color=Palette.NEON_CYAN,
        )
        self._status.grid(row=3, column=0, pady=(16, 0))

        # Step list.
        self._step_labels: list[ctk.CTkLabel] = []
        for i, (text, _) in enumerate(self.STEPS, start=4):
            lbl = ctk.CTkLabel(
                inner, text=f"○ {text}",
                font=("Consolas", 10),
                text_color=Palette.TEXT_DIM,
            )
            lbl.grid(row=i, column=0, pady=1)
            self._step_labels.append(lbl)

        self._total_steps = len(self.STEPS) + 1

    def _draw_glow(self) -> None:
        c = self._glow
        c.delete("g")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        cx, cy = w / 2, h / 2
        # Three faint concentric rings behind the boot card.
        for i, r in enumerate((280, 220, 160)):
            shade = ["#0e1a48", "#0a1240", "#080e2c"][i]
            c.create_oval(cx - r, cy - r, cx + r, cy + r,
                          fill=shade, outline="", tags="g")
        # Soft cyan halo behind the title.
        c.create_oval(cx - 120, cy - 90, cx + 120, cy + 90,
                      fill="#0a1a3a", outline="", tags="g")

    def _step(self) -> None:
        if self._step_index < len(self.STEPS):
            text, dur = self.STEPS[self._step_index]
            self._status.configure(text=text)
            for i in range(self._step_index):
                self._step_labels[i].configure(
                    text=self._step_labels[i]._text.replace("○", "✓"),
                    text_color=Palette.NEON_GREEN,
                )
            self._step_labels[self._step_index].configure(
                text=self._step_labels[self._step_index]._text.replace("○", "◈"),
                text_color=Palette.NEON_CYAN,
            )
            self._bar_fill.configure(
                width=int(self._bar_outer.winfo_width()
                          * (self._step_index + 1) / self._total_steps),
            )
            self._step_index += 1
            self.after(dur, self._step)
        else:
            self._finish()

    def _finish(self) -> None:
        # Final "ALL SYSTEMS ONLINE" line.
        self._status.configure(text="ALL SYSTEMS ONLINE", text_color=Palette.NEON_GREEN)
        for lbl in self._step_labels:
            lbl.configure(
                text=lbl._text.replace("◈", "✓").replace("○", "✓"),
                text_color=Palette.NEON_GREEN,
            )
        self._bar_fill.configure(width=self._bar_outer.winfo_width())
        self._bar_fill.configure(fg_color=Palette.NEON_GREEN)
        self.after(450, self._tear_down)

    def _tear_down(self) -> None:
        try:
            self.destroy()
        finally:
            self._on_done()


# ---------------------------------------------------------------------------
# Main futuristic UI
# ---------------------------------------------------------------------------

class ArvisUI(ctk.CTk):
    """The main futuristic command-center window."""

    # Per-state visual config for the command console.
    _CONSOLE_PROFILES = {
        "open_chrome": (
            [(0, "OPEN CHROME",                   "active"),
             (1, "APPLICATION.LAUNCH",            "active"),
             (2, "CHROME.EXE",                    "active"),
             (3, "APPLICATION READY",             "done")],
            0.32,
        ),
    }

    def __init__(self, on_command: Callable[[str], None]):
        super().__init__()

        self.title("ARVIS — Adaptive Voice Interface")
        self.geometry("1480x880")
        self.minsize(1200, 740)
        self.configure(fg_color=Palette.BG_VOID)

        self._always_on_top = False
        self.attributes("-topmost", False)

        self._on_command = on_command
        self._on_user_speaks: Callable[[str], None] | None = None
        self._on_ai_responds: Callable[[str], None] | None = None
        self._on_status: Callable[[str], None] | None = None
        self._on_mode_fired: Callable[[str], None] | None = None

        # State machine.
        self._orb_state = "idle"

        # Mode-save handler (set by controller later).
        self._mode_save_handler: Callable | None = None
        self._all_modes: list[dict] = []

        self._mode_chip_widgets: list[ctk.CTkFrame] = []

        # Pending set_modes call (deferred until layout exists).
        self._pending_modes: tuple[list[dict], Callable, Callable] | None = None

        # Defer layout until the window is on-screen.
        self.after(40, self._build_layout)
        self.after(60, self._animate_clock)
        self.after(80, self._start_scanlines)
        self.after(100, self._start_boot)

    # -- top-level layout ---------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top HUD strip ----------------------------------------------------
        self._build_top_hud()

        # Body (left/center/right grid) ------------------------------------
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=14, pady=(2, 4))
        body.grid_columnconfigure(0, weight=0, minsize=240)   # left
        body.grid_columnconfigure(1, weight=1)                # center
        body.grid_columnconfigure(2, weight=0, minsize=320)   # right
        body.grid_rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_center_panel(body)
        self._build_right_panel(body)

        # Bottom command input --------------------------------------------
        self._build_bottom_input()

        # Layout is now real — drain any deferred set_modes call and let
        # future calls render immediately.
        self._layout_ready = True
        if self._pending_modes is not None:
            modes, on_run, on_edit = self._pending_modes
            self._pending_modes = None
            self._render_modes(modes, on_run, on_edit)

    # -- top HUD ------------------------------------------------------------

    def _build_top_hud(self) -> None:
        # Container with a glass background.
        wrap = ctk.CTkFrame(self, fg_color="transparent",
                            corner_radius=12, height=58)
        wrap.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 4))
        wrap.grid_propagate(False)
        wrap.grid_columnconfigure(1, weight=1)
        # Layer 1: glass canvas background.
        bg = GlassPanel(wrap)
        bg.place(x=0, y=0, relwidth=1, relheight=1)
        # Layer 2: 1px metallic border.
        border = tk.Frame(wrap, bg=Palette.BORDER_METAL, highlightthickness=0)
        border.place(x=0, y=0, relwidth=1, relheight=1)
        # Layer 3: actual content (transparent so glass shows).
        hud = ctk.CTkFrame(wrap, fg_color="transparent")
        hud.place(relx=0, rely=0, relwidth=1, relheight=1)
        hud.grid_columnconfigure(1, weight=1)
        hud.grid_rowconfigure(0, weight=1)

        # Logo.
        self.logo = ctk.CTkLabel(
            hud, text="◢ ARVIS",
            font=("Segoe UI", 22, "bold"),
            text_color=Palette.NEON_CYAN,
        )
        self.logo.grid(row=0, column=0, padx=(22, 14), pady=10, sticky="w")

        # Tagline.
        self.tagline = ctk.CTkLabel(
            hud, text="// adaptive voice interface · v3.07",
            font=("Consolas", 11),
            text_color=Palette.TEXT_DIM,
        )
        self.tagline.grid(row=0, column=1, padx=4, pady=10, sticky="w")

        # Right-aligned segments with metallic bevel.
        right = ctk.CTkFrame(hud, fg_color="transparent")
        right.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        self.hud_segments: dict[str, ctk.CTkLabel] = {}
        for col, (key, label, color) in enumerate([
            ("system",  "SYSTEM",  Palette.NEON_GREEN),
            ("ollama",  "OLLAMA",  Palette.NEON_CYAN),
            ("mic",     "MIC",     Palette.NEON_GREEN),
            ("tts",     "TTS",     Palette.NEON_AMBER),
            ("net",     "NET",     Palette.NEON_AQUA),
        ]):
            # Each segment is a small beveled "pill" with rim + face.
            seg_holder = tk.Frame(right, bg=Palette.BG_PANEL,
                                  highlightthickness=0)
            seg_holder.grid(row=0, column=col, padx=4)
            seg = ctk.CTkLabel(
                seg_holder, text=f" {label} · ONLINE ",
                font=("Consolas", 10, "bold"),
                text_color=Palette.TEXT_BRIGHT,
                fg_color=Palette.BG_GLASS,
                corner_radius=4,
            )
            seg.pack(padx=1, pady=1)
            self.hud_segments[key] = seg

        # Clock.
        self.clock_label = ctk.CTkLabel(
            right, text="00:00:00",
            font=("Consolas", 13, "bold"),
            text_color=Palette.NEON_CYAN,
        )
        self.clock_label.grid(row=0, column=99, padx=(12, 6))

    # -- left side (status panel + modes) -----------------------------------

    def _build_left_panel(self, master) -> None:
        wrap = ctk.CTkFrame(master, fg_color="transparent")
        wrap.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_rowconfigure(2, weight=1)

        # Status panel.
        self.status_panel = SystemStatusPanel(wrap)
        self.status_panel.grid(row=0, column=0, pady=(0, 6), sticky="ew")

        # System monitor.
        self.system_monitor = SystemMonitor(wrap)
        self.system_monitor.grid(row=1, column=0, pady=(0, 6), sticky="ew")

        # Modes panel — glass card.
        self.modes_panel = ctk.CTkFrame(
            wrap, fg_color="transparent", corner_radius=14,
            border_width=0,
        )
        self.modes_panel.grid(row=2, column=0, sticky="nsew")
        mp_bg = GlassPanel(self.modes_panel)
        mp_bg.place(x=0, y=0, relwidth=1, relheight=1)
        mp_b = tk.Frame(self.modes_panel, bg=Palette.BORDER_METAL,
                        highlightthickness=0)
        mp_b.place(x=0, y=0, relwidth=1, relheight=1)
        mp_c = HudCorners(self.modes_panel, color=Palette.NEON_PINK,
                          bg=Palette.BG_PANEL, height=22)
        mp_c.place(x=0, y=0, relwidth=1, height=22)
        self.modes_panel.grid_columnconfigure(0, weight=1)
        self.modes_panel.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(self.modes_panel, fg_color="transparent")
        head.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head, text="◤ MODES",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            head, text="click RUN or speak a phrase",
            font=("Consolas", 9),
            text_color=Palette.TEXT_GHOST, anchor="e",
        ).grid(row=0, column=1, sticky="e")

        self.modes_frame = ctk.CTkScrollableFrame(
            self.modes_panel, fg_color="transparent",
        )
        self.modes_frame.grid(row=2, column=0, padx=8, pady=(4, 8),
                              sticky="nsew")
        self.modes_frame.grid_columnconfigure((0, 1), weight=1)

        # Mode action buttons.
        actions = ctk.CTkFrame(self.modes_panel, fg_color="transparent")
        actions.grid(row=3, column=0, padx=8, pady=(0, 10), sticky="ew")
        actions.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            actions, text="✚ NEW MODE", height=34,
            fg_color=Palette.BG_GLASS, hover_color=Palette.BG_GLASS_LIGHT,
            text_color=Palette.NEON_GREEN,
            border_width=1, border_color=Palette.NEON_GREEN,
            font=("Consolas", 11, "bold"),
            command=self._open_add_mode_dialog,
        ).grid(row=0, column=0, padx=4, sticky="ew")

        ctk.CTkButton(
            actions, text="⚙ MANAGE", height=34,
            fg_color=Palette.BG_GLASS, hover_color=Palette.BG_GLASS_LIGHT,
            text_color=Palette.NEON_PINK,
            border_width=1, border_color=Palette.NEON_PINK,
            font=("Consolas", 11, "bold"),
            command=self._open_manage_dialog,
        ).grid(row=0, column=1, padx=4, sticky="ew")

    # -- center (AI core) ---------------------------------------------------

    def _build_center_panel(self, master) -> None:
        center = ctk.CTkFrame(master, fg_color="transparent",
                              corner_radius=14, border_width=0)
        center.grid(row=0, column=1, padx=6, sticky="nsew")
        cbg = GlassPanel(center)
        cbg.place(x=0, y=0, relwidth=1, relheight=1)
        cb = tk.Frame(center, bg=Palette.BORDER_METAL,
                      highlightthickness=0)
        cb.place(x=0, y=0, relwidth=1, relheight=1)
        cc = HudCorners(center, color=Palette.NEON_CYAN,
                        bg=Palette.BG_PANEL, height=22)
        cc.place(x=0, y=0, relwidth=1, height=22)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)

        # Inner content holder so the orb canvas sits over the glass.
        center_inner = ctk.CTkFrame(center, fg_color="transparent")
        center_inner.place(x=1, y=1, relwidth=1, relheight=1)
        center_inner.grid_columnconfigure(0, weight=1)
        center_inner.grid_rowconfigure(1, weight=1)
        # Stash reference so we can still grid the orb/header correctly.
        center._inner = center_inner  # type: ignore[attr]

        # Header.
        head = ctk.CTkFrame(center._inner, fg_color="transparent")
        head.grid(row=0, column=0, padx=18, pady=(12, 4), sticky="ew")
        head.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            head, text="◤ AI CORE",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.state_badge = ctk.CTkLabel(
            head, text="  IDLE  ",
            font=("Consolas", 11, "bold"),
            text_color=Palette.NEON_CYAN,
            fg_color=Palette.BG_GLASS, corner_radius=4,
        )
        self.state_badge.grid(row=0, column=1, sticky="e")

        # Orb.
        self.orb_canvas = AICoreView(center._inner)
        self.orb_canvas.grid(row=1, column=0, padx=12, pady=(4, 12),
                             sticky="nsew")

        # Bottom panels (thinking pipeline + command console).
        bottom = ctk.CTkFrame(center._inner, fg_color="transparent")
        bottom.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        bottom.grid_columnconfigure((0, 1), weight=1)

        self.thinking_pipeline = ThinkingPipeline(bottom)
        self.thinking_pipeline.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.command_console = CommandConsole(bottom)
        self.command_console.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    # -- right (chat + recent commands) -------------------------------------

    def _build_right_panel(self, master) -> None:
        wrap = ctk.CTkFrame(master, fg_color="transparent")
        wrap.grid(row=0, column=2, padx=(6, 0), sticky="nsew")
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_rowconfigure(1, weight=1)

        # Dialogue — glass card.
        dialogue = ctk.CTkFrame(
            wrap, fg_color="transparent", corner_radius=14,
            border_width=0,
        )
        dialogue.grid(row=0, column=0, pady=(0, 6), sticky="ew")
        dbg = GlassPanel(dialogue)
        dbg.place(x=0, y=0, relwidth=1, relheight=1)
        dbo = tk.Frame(dialogue, bg=Palette.BORDER_METAL,
                       highlightthickness=0)
        dbo.place(x=0, y=0, relwidth=1, relheight=1)
        dco = HudCorners(dialogue, color=Palette.NEON_AQUA,
                         bg=Palette.BG_PANEL, height=22)
        dco.place(x=0, y=0, relwidth=1, height=22)
        dialogue._inner = ctk.CTkFrame(dialogue, fg_color="transparent")  # type: ignore[attr]
        dialogue._inner.place(x=1, y=1, relwidth=1, relheight=1)  # type: ignore[attr]
        dialogue._inner.grid_columnconfigure(0, weight=1)  # type: ignore[attr]

        head = ctk.CTkFrame(dialogue._inner, fg_color="transparent")  # type: ignore[attr]
        head.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head, text="◤ DIALOGUE",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.dialogue_state = ctk.CTkLabel(
            head, text="—",
            font=("Consolas", 10, "bold"),
            text_color=Palette.NEON_CYAN, anchor="e",
        )
        self.dialogue_state.grid(row=0, column=1, sticky="e")

        self.chat = ctk.CTkTextbox(
            dialogue._inner, fg_color=Palette.BG_GLASS, corner_radius=10,  # type: ignore[attr]
            border_width=1, border_color=Palette.BORDER_METAL,
            text_color=Palette.TEXT_BRIGHT,
            font=("Segoe UI", 12), wrap="word", height=220,
        )
        self.chat.grid(row=1, column=0, padx=10, pady=(2, 12), sticky="ew")
        self.chat.configure(state="disabled")
        self._append_chat("arvis", "Online. Say my name to wake me up.")

        # Recent commands — glass card.
        recent_wrap = ctk.CTkFrame(
            wrap, fg_color="transparent", corner_radius=14,
            border_width=0,
        )
        recent_wrap.grid(row=1, column=0, sticky="nsew")
        rbg = GlassPanel(recent_wrap)
        rbg.place(x=0, y=0, relwidth=1, relheight=1)
        rbo = tk.Frame(recent_wrap, bg=Palette.BORDER_METAL,
                       highlightthickness=0)
        rbo.place(x=0, y=0, relwidth=1, relheight=1)
        rco = HudCorners(recent_wrap, color=Palette.NEON_GREEN,
                         bg=Palette.BG_PANEL, height=22)
        rco.place(x=0, y=0, relwidth=1, height=22)
        recent_wrap._inner = ctk.CTkFrame(recent_wrap, fg_color="transparent")  # type: ignore[attr]
        recent_wrap._inner.place(x=1, y=1, relwidth=1, relheight=1)  # type: ignore[attr]
        recent_wrap._inner.grid_columnconfigure(0, weight=1)  # type: ignore[attr]
        recent_wrap._inner.grid_rowconfigure(1, weight=1)  # type: ignore[attr]

        head2 = ctk.CTkFrame(recent_wrap._inner, fg_color="transparent")  # type: ignore[attr]
        head2.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")
        head2.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head2, text="◤ RECENT COMMANDS",
            font=("Consolas", 10, "bold"),
            text_color=Palette.TEXT_GHOST, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.recent = RecentCommands(recent_wrap._inner, height=180)  # type: ignore[attr]
        self.recent.grid(row=1, column=0, padx=10, pady=(2, 12), sticky="nsew")

    # -- bottom input -------------------------------------------------------

    def _build_bottom_input(self) -> None:
        # Glass-backed command bar.
        wrap = ctk.CTkFrame(
            self, fg_color="transparent", corner_radius=14,
            border_width=0, height=72,
        )
        wrap.grid(row=2, column=0, sticky="ew", padx=14, pady=(2, 12))
        wrap.grid_columnconfigure(1, weight=1)
        wrap.grid_propagate(False)
        bbg = GlassPanel(wrap)
        bbg.place(x=0, y=0, relwidth=1, relheight=1)
        bbo = tk.Frame(wrap, bg=Palette.BORDER_METAL,
                       highlightthickness=0)
        bbo.place(x=0, y=0, relwidth=1, relheight=1)
        bco = HudCorners(wrap, color=Palette.NEON_CYAN,
                         bg=Palette.BG_PANEL, height=22)
        bco.place(x=0, y=0, relwidth=1, height=22)
        bar = ctk.CTkFrame(wrap, fg_color="transparent")
        bar.place(relx=0, rely=0, relwidth=1, relheight=1)
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_rowconfigure(0, weight=1)

        # Mic indicator (visual only — actual capture is on the voice thread).
        mic_btn = ctk.CTkButton(
            bar, text="◉", width=48, height=48,
            fg_color=Palette.BG_GLASS, hover_color=Palette.BG_GLASS_LIGHT,
            text_color=Palette.NEON_CYAN,
            border_width=1, border_color=Palette.NEON_CYAN,
            font=("Segoe UI", 18, "bold"),
            command=self._toggle_mic_visual,
        )
        mic_btn.grid(row=0, column=0, padx=(12, 8), pady=12)
        self.mic_btn = mic_btn

        # Inline mini-waveform.
        self.input_wave = tk.Canvas(
            bar, bg=Palette.BG_GLASS, highlightthickness=0, height=24,
        )
        self.input_wave.grid(row=0, column=1, padx=(0, 8), pady=0, sticky="ew")
        self._wave_phase = 0.0
        self._wave_amp = 0.1
        self._draw_input_wave()

        # Entry.
        self.entry = ctk.CTkEntry(
            bar, height=48,
            placeholder_text="› speak or type a command …",
            font=("Consolas", 13),
            fg_color=Palette.BG_GLASS,
            border_color=Palette.BORDER_DIM,
            border_width=1,
            text_color=Palette.TEXT_BRIGHT,
            placeholder_text_color=Palette.TEXT_GHOST,
        )
        self.entry.grid(row=0, column=2, padx=(8, 8), pady=12, sticky="ew")
        self.entry.bind("<Return>", self._on_submit)
        self.entry.bind("<FocusIn>",
                        lambda e: self.entry.configure(
                            border_color=Palette.NEON_CYAN))
        self.entry.bind("<FocusOut>",
                        lambda e: self.entry.configure(
                            border_color=Palette.BORDER_DIM))

        # Send button.
        self.send_btn = ctk.CTkButton(
            bar, text="➤", width=58, height=48,
            fg_color=Palette.NEON_CYAN, hover_color="#33f7ff",
            text_color="#001018", font=("Segoe UI", 18, "bold"),
            command=self._on_submit,
        )
        self.send_btn.grid(row=0, column=3, padx=(4, 12), pady=12)

        # Pin button.
        pin_btn = ctk.CTkButton(
            bar, text="◇", width=36, height=48,
            fg_color="transparent", hover_color=Palette.BG_GLASS_LIGHT,
            text_color=Palette.TEXT_DIM,
            border_width=0,
            font=("Segoe UI", 14),
            command=self._toggle_pin,
        )
        pin_btn.grid(row=0, column=4, padx=(0, 8), pady=12)
        self.pin_btn = pin_btn

        # Status footer.
        self._input_status = ctk.CTkLabel(
            self, text="● ONLINE   ·   ollama · minimax-m3:cloud · streaming",
            font=("Consolas", 10),
            text_color=Palette.TEXT_GHOST,
        )
        self._input_status.grid(row=3, column=0, sticky="ew", padx=18,
                                pady=(0, 8))

    # -- boot sequence ------------------------------------------------------

    def _start_boot(self) -> None:
        def _done():
            self.deiconify()
            self.lift()
            self.focus_force()
            self.set_listening(False)

        BootSequence(self, on_done=_done)

    # -- scanlines ----------------------------------------------------------

    def _start_scanlines(self) -> None:
        # Subtle horizontal scanlines drawn directly on a low-priority canvas
        # placed behind everything. Tk canvas place uses bg to draw, but
        # using a frame with a striped pattern is more reliable. We instead
        # drop in a single low-priority overlay canvas.
        self._scan = tk.Canvas(self, bg=Palette.BG_VOID, highlightthickness=0)
        self._scan.place(x=0, y=0, relwidth=1, relheight=1)
        try:
            self.tk.call("lower", self._scan._w)
        except Exception:
            pass
        self.after(80, self._redraw_scanlines)
        self.bind("<Configure>", lambda e: self._redraw_scanlines())

    def _redraw_scanlines(self) -> None:
        c = self._scan
        c.delete("scan")
        w = max(self.winfo_width(), 100)
        h = max(self.winfo_height(), 100)
        # Subtler scanlines spaced every 5px, very dim indigo.
        for y in range(0, h, 5):
            c.create_line(0, y, w, y, fill="#0a1030", tags="scan")
        # Add faint horizontal highlight stripes every 60px for the
        # holographic feel.
        for y in range(60, h, 60):
            c.create_line(0, y, w, y, fill="#1a2455", tags="scan")
        try:
            self.tk.call("lower", self._scan._w)
        except Exception:
            pass

    # -- clock --------------------------------------------------------------

    def _animate_clock(self) -> None:
        try:
            self.clock_label.configure(text=time.strftime("%H:%M:%S"))
            hue = (math.sin(time.time() * 0.5) + 1) * 0.5
            if hue > 0.85:
                self.logo.configure(text_color=Palette.NEON_PINK)
            else:
                self.logo.configure(text_color=Palette.NEON_CYAN)
        except Exception:
            pass
        self.after(500, self._animate_clock)

    # -- input waveform ----------------------------------------------------

    def _draw_input_wave(self) -> None:
        c = self.input_wave
        c.delete("wave")
        w = max(c.winfo_width(), 60)
        h = max(c.winfo_height(), 12)
        mid = h / 2
        # Always-on low-amplitude wave; mic_btn click boosts amplitude.
        amp = 2 + self._wave_amp * (h / 2 - 3)
        n = 60
        pts = []
        for i in range(n):
            x = i * (w / (n - 1))
            y = mid + amp * math.sin(self._wave_phase + i * 0.35)
            pts.extend([x, y])
        c.create_line(*pts, fill=Palette.NEON_CYAN, width=1, smooth=True,
                      tags="wave")
        self._wave_phase += 0.18
        # Decay amplitude back to baseline.
        self._wave_amp = max(0.1, self._wave_amp * 0.95)
        self.after(40, self._draw_input_wave)

    # -- chat & log helpers (kept for backwards compatibility) --------------

    def _append_chat(self, who: str, msg: str) -> None:
        self.chat.configure(state="normal")
        if who == "arvis":
            self.chat.insert("end", "ARVIS  ", "arvis_name")
            self.chat.insert("end", msg + "\n\n", "arvis_msg")
        elif who == "you":
            self.chat.insert("end", "YOU    ", "you_name")
            self.chat.insert("end", msg + "\n\n", "you_msg")
        else:
            self.chat.insert("end", msg + "\n")
        self.chat.tag_config("arvis_name", foreground=Palette.NEON_CYAN)
        self.chat.tag_config("arvis_msg", foreground=Palette.TEXT_BRIGHT)
        self.chat.tag_config("you_name", foreground=Palette.NEON_PINK)
        self.chat.tag_config("you_msg", foreground=Palette.TEXT_BRIGHT)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _append_log(self, msg: str) -> None:
        # Logs are no longer a separate widget — fold them into the
        # recent-commands panel as a faint ghost entry.
        if msg.startswith("listening for") or msg.startswith("could not"):
            return
        if "wake word" in msg:
            return
        # Only surface interesting lines.
        self.recent.add(f"// {msg}")

    # -- modes rendering (kept mostly identical to old UI) ------------------

    def set_modes(self, modes: list[dict], on_run: Callable[[dict], None],
                  on_edit: Callable[[dict], None]) -> None:
        # Layout may not exist yet (controller constructs before our
        # _build_layout runs). Queue the call and let _build_layout drain
        # the queue.
        if not getattr(self, "_layout_ready", False):
            self._pending_modes = (modes, on_run, on_edit)
            return
        self._render_modes(modes, on_run, on_edit)

    def _render_modes(self, modes: list[dict], on_run: Callable,
                      on_edit: Callable) -> None:
        for w in self._mode_chip_widgets:
            w.destroy()
        self._mode_chip_widgets.clear()

        cols = 2
        for i, mode in enumerate(modes):
            row, col = divmod(i, cols)
            chip = ctk.CTkFrame(
                self.modes_frame, fg_color=Palette.BG_GLASS, corner_radius=10,
                border_width=1, border_color=Palette.BORDER_METAL,
            )
            chip.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            self._mode_chip_widgets.append(chip)
            chip.grid_columnconfigure(0, weight=1)

            head = ctk.CTkFrame(chip, fg_color="transparent")
            head.grid(row=0, column=0, padx=10, pady=(8, 0), sticky="ew")
            head.grid_columnconfigure(1, weight=1)

            is_builtin = mode.get("builtin")
            accent = Palette.NEON_PINK if is_builtin else Palette.NEON_GREEN

            idx_lbl = ctk.CTkLabel(
                head, text=f"0{i + 1}",
                font=("Consolas", 10, "bold"),
                text_color=Palette.TEXT_GHOST,
            )
            idx_lbl.grid(row=0, column=0, sticky="w")

            name_lbl = ctk.CTkLabel(
                head, text=mode["name"].upper(),
                font=("Consolas", 13, "bold"),
                text_color=accent,
            )
            name_lbl.grid(row=0, column=1, padx=(6, 0), sticky="w")

            if not is_builtin:
                edit_btn = ctk.CTkButton(
                    head, text="✎", width=24, height=20,
                    fg_color="transparent", hover_color=Palette.BG_GLASS_LIGHT,
                    text_color=Palette.TEXT_DIM,
                    command=lambda m=mode: on_edit(m),
                )
                edit_btn.grid(row=0, column=2, padx=(6, 0))

            phrase = ", ".join(f'"{p}"' for p in mode.get("phrases", [])[:2])
            if len(mode.get("phrases", [])) > 2:
                phrase += ", …"
            ctk.CTkLabel(
                chip, text=phrase,
                font=("Consolas", 10),
                text_color=Palette.TEXT_BRIGHT, wraplength=180, justify="left",
            ).grid(row=1, column=0, padx=10, pady=(2, 4), sticky="w")

            actions = ", ".join(
                a.get("target", "?") for a in mode.get("actions", [])
            ) or "(no actions)"
            ctk.CTkLabel(
                chip, text=actions,
                font=("Consolas", 9),
                text_color=Palette.TEXT_DIM, wraplength=180, justify="left",
            ).grid(row=2, column=0, padx=10, pady=(0, 6), sticky="w")

            run_btn = ctk.CTkButton(
                chip, text="▶ EXECUTE", height=28,
                fg_color=accent, hover_color="#ffffff",
                text_color="#001018",
                font=("Consolas", 11, "bold"),
                command=lambda m=mode: on_run(m),
            )
            run_btn.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="ew")

    # -- state transitions (called by controller / voice loop) -------------

    def _apply_orb_palette(self) -> None:
        self.orb_canvas.set_state(self._orb_state)

    def set_speaking(self, on: bool) -> None:
        new_state = "speaking" if on else "idle"
        if new_state == self._orb_state:
            return
        self._orb_state = new_state
        self._apply_orb_palette()
        self._update_state_badge()

    def set_thinking(self, on: bool) -> None:
        new_state = "thinking" if on else "idle"
        if new_state == self._orb_state:
            return
        if self._orb_state == "speaking":
            return
        self._orb_state = new_state
        self._apply_orb_palette()
        self._update_state_badge()

    def set_executing(self, on: bool) -> None:
        new_state = "executing" if on else "idle"
        if new_state == self._orb_state:
            return
        self._orb_state = new_state
        self._apply_orb_palette()
        self._update_state_badge()

    def set_error(self, on: bool) -> None:
        new_state = "error" if on else "idle"
        if new_state == self._orb_state:
            return
        self._orb_state = new_state
        self._apply_orb_palette()
        self._update_state_badge()

    def set_listening(self, on: bool) -> None:
        new_state = "listening" if on else "idle"
        if new_state != self._orb_state:
            self._orb_state = new_state
            self._apply_orb_palette()
        self._update_state_badge()

        self.mic_btn.configure(
            text="◉" if on else "◌",
            text_color=Palette.NEON_CYAN if on else Palette.TEXT_DIM,
            border_color=Palette.NEON_CYAN if on else Palette.BORDER_DIM,
        )

        seg = self.hud_segments.get("mic")
        if seg:
            seg.configure(
                text=f"  MIC  ·  {'LIVE' if on else 'STANDBY'}  ",
                text_color=Palette.NEON_GREEN if on else Palette.TEXT_DIM,
            )

        # Boost the input-wave amplitude when actively listening.
        if on:
            self._wave_amp = 0.9

    def _update_state_badge(self) -> None:
        labels = {
            "idle":      ("  IDLE  ",      Palette.NEON_CYAN),
            "listening": ("  LISTENING  ", Palette.NEON_GREEN),
            "thinking":  ("  PROCESSING  ", Palette.NEON_PINK),
            "speaking":  ("  SPEAKING  ",  Palette.NEON_AMBER),
            "executing": ("  EXECUTING  ", Palette.NEON_PURPLE),
            "error":     ("  ERROR  ",     Palette.NEON_RED),
        }
        text, color = labels.get(self._orb_state, ("  IDLE  ", Palette.NEON_CYAN))
        try:
            self.state_badge.configure(text=text, text_color=color)
        except Exception:
            pass
        try:
            self.dialogue_state.configure(
                text=text.strip(),
                text_color=color,
            )
        except Exception:
            pass

        # HUD strips.
        for key, seg in self.hud_segments.items():
            if key in ("system", "ollama", "tts", "net"):
                continue  # leave connectivity indicators alone
            if key == "mic":
                active = self._orb_state == "listening"
                seg.configure(
                    text=f"  MIC  ·  {'LIVE' if active else 'STANDBY'}  ",
                    text_color=Palette.NEON_GREEN if active else Palette.TEXT_DIM,
                )

    # -- intent visualisation helpers --------------------------------------

    def show_intent(self, voice_text: str, intent: str,
                    target: str, result: str, success: bool) -> None:
        """Animate the command console with the standard lifecycle."""
        self.command_console.reset()
        self.set_executing(True)
        states = ["active", "active", "active", "done" if success else "fail"]
        self.command_console.animate([
            (0, voice_text.upper(),    states[0]),
            (1, intent,                states[1]),
            (2, target,                states[2]),
            (3, result,                states[3]),
        ], delay=0.32, on_done=lambda: self.set_executing(False))

    # -- thinking pipeline --------------------------------------------------

    def show_thinking(self, on: bool) -> None:
        if on:
            self.thinking_pipeline.reset()
            self.thinking_pipeline.set_active(0)
            # Animate the active step moving forward.
            self._thinking_active = True
            def _tick(step: int = 0) -> None:
                if not self._thinking_active:
                    return
                if step >= len(ThinkingPipeline.STEPS):
                    return
                self.thinking_pipeline.set_active(step)
                self.after(380, lambda: _tick(step + 1))
            self.after(20, lambda: _tick(0))
        else:
            self._thinking_active = False
            self.thinking_pipeline.reset()

    # -- audio amplitude (used by the orb) ---------------------------------

    def pulse_audio(self, amp: float) -> None:
        """Forward an audio amplitude (0..1) to the orb visualiser."""
        self.orb_canvas.set_audio_amplitude(amp)

    # -- input handlers -----------------------------------------------------

    def _on_submit(self, _event=None) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._append_chat("you", text)
        self.recent.add(text)
        self._on_command(text)

    def _toggle_mic_visual(self) -> None:
        # Visual-only — actual listening is owned by the voice thread.
        self.set_listening(not getattr(self, "_is_listening", False))

    def _toggle_pin(self) -> None:
        self._always_on_top = not self._always_on_top
        self.attributes("-topmost", self._always_on_top)
        self.pin_btn.configure(
            text="◆" if self._always_on_top else "◇",
            text_color=Palette.NEON_CYAN if self._always_on_top else Palette.TEXT_DIM,
        )

    # -- mode dialogs -------------------------------------------------------

    def _open_add_mode_dialog(self) -> None:
        ModeEditor(self, on_save=lambda name, phrases, actions:
                   self._on_save_mode(None, name, phrases, actions))

    def _open_manage_dialog(self) -> None:
        ManageModesDialog(self)

    def _on_save_mode(self, existing: dict | None,
                      name: str, phrases: list[str], actions: list[dict]) -> None:
        if self._mode_save_handler:
            self._mode_save_handler(existing, name, phrases, actions)


class ModeEditor(ctk.CTkToplevel):
    """A pop-up for creating or editing a mode."""

    def __init__(self, master: ArvisUI, on_save: Callable):
        super().__init__(master)
        self.title("Mode editor")
        self.geometry("520x620")
        self.configure(fg_color=Palette.BG_DEEP)
        self.grab_set()

        self._on_save = on_save

        ctk.CTkLabel(
            self, text="Mode name",
            font=("Segoe UI", 12, "bold"),
            text_color=Palette.TEXT_DIM,
        ).pack(anchor="w", padx=20, pady=(20, 4))
        self.name_entry = ctk.CTkEntry(
            self, height=38, placeholder_text="e.g. Workout",
            fg_color=Palette.BG_GLASS, border_color=Palette.BORDER_DIM,
        )
        self.name_entry.pack(fill="x", padx=20)

        ctk.CTkLabel(
            self, text="Trigger phrases (comma-separated)",
            font=("Segoe UI", 12, "bold"),
            text_color=Palette.TEXT_DIM,
        ).pack(anchor="w", padx=20, pady=(16, 4))
        self.phrases_entry = ctk.CTkEntry(
            self, height=38,
            placeholder_text='"im working", "lets grind", "deep work mode"',
            fg_color=Palette.BG_GLASS, border_color=Palette.BORDER_DIM,
        )
        self.phrases_entry.pack(fill="x", padx=20)

        ctk.CTkLabel(
            self,
            text="Actions (one per line) — format: TYPE | TARGET\n"
                 "  type = app | url | youtube_search | google_search",
            font=("Segoe UI", 12, "bold"),
            text_color=Palette.TEXT_DIM,
            justify="left",
        ).pack(anchor="w", padx=20, pady=(16, 4))

        self.actions_box = ctk.CTkTextbox(
            self, fg_color=Palette.BG_GLASS, border_color=Palette.BORDER_DIM,
            border_width=1, corner_radius=10, height=240,
        )
        self.actions_box.pack(fill="both", padx=20, expand=True)
        self.actions_box.insert(
            "1.0",
            "app | vs code\n"
            "url | https://www.youtube.com\n"
            "youtube_search | study music\n",
        )

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=20)
        ctk.CTkButton(
            btn_row, text="CANCEL", height=38,
            fg_color="#101a32", hover_color="#172246",
            text_color=Palette.TEXT_DIM,
            border_width=1, border_color=Palette.BORDER_DIM,
            command=self.destroy,
        ).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="SAVE MODE", height=38,
            fg_color=Palette.NEON_CYAN, hover_color="#33f7ff",
            text_color="#001018", font=("Segoe UI", 12, "bold"),
            command=self._save,
        ).pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _save(self) -> None:
        name = self.name_entry.get().strip()
        phrases_raw = self.phrases_entry.get().strip()
        actions_raw = self.actions_box.get("1.0", "end").strip()

        if not name:
            return
        phrases = [p.strip().strip('"').strip("'")
                   for p in phrases_raw.split(",") if p.strip()]
        actions = []
        for line in actions_raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                atype, target = line.split("|", 1)
                actions.append({"type": atype.strip().lower(),
                                "target": target.strip()})
        if not phrases or not actions:
            return

        self._on_save(name, phrases, actions)
        self.destroy()


class ManageModesDialog(ctk.CTkToplevel):
    """List, edit, and delete custom modes."""

    def __init__(self, master: ArvisUI):
        super().__init__(master)
        self.title("Manage modes")
        self.geometry("560x520")
        self.configure(fg_color=Palette.BG_DEEP)
        self.grab_set()

        self._master_ui = master
        self._list = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._list.pack(fill="both", expand=True, padx=14, pady=14)
        self._refresh()

    def _refresh(self) -> None:
        for w in self._list.winfo_children():
            w.destroy()
        for mode in getattr(self._master_ui, "_all_modes", []):
            row = ctk.CTkFrame(self._list, fg_color=Palette.BG_GLASS,
                               corner_radius=10,
                               border_width=1, border_color=Palette.BORDER_DIM)
            row.pack(fill="x", padx=4, pady=4)
            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                row, text=mode["name"],
                font=("Segoe UI", 12, "bold"),
                text_color=(Palette.NEON_PINK
                            if mode.get("builtin")
                            else Palette.NEON_GREEN),
            ).grid(row=0, column=0, padx=14, pady=10, sticky="w")

            if not mode.get("builtin"):
                ctk.CTkButton(
                    row, text="DELETE", height=30, width=90,
                    fg_color="#3a0a1a", hover_color="#5a0a26",
                    text_color=Palette.NEON_PINK,
                    border_width=1, border_color=Palette.NEON_PINK,
                    command=lambda m=mode: self._delete(m),
                ).grid(row=0, column=1, padx=8, pady=8)


# ---------------------------------------------------------------------------
# Controller — wires UI events to voice loop and agent
# ---------------------------------------------------------------------------

class ArvisController:
    def __init__(self, ui: ArvisUI | None):
        self.ui = ui
        self.modes = ModeManager()

        self.event_q: queue.Queue = queue.Queue()
        self.conversation_mode = False
        self.last_interaction_time = 0.0
        self.running = True

        if ui is not None:
            ui._mode_save_handler = self._save_mode_from_dialog
            ui._on_command = self._handle_text_command
            ui.set_modes(self.modes.modes, self._run_mode, self._edit_mode_from_chip)
            ui._all_modes = self.modes.modes

    # -- helpers ------------------------------------------------------------

    def _refresh_modes_ui(self) -> None:
        if self.ui is None:
            return
        self.ui._all_modes = self.modes.modes
        self.ui.set_modes(self.modes.modes, self._run_mode, self._edit_mode_from_chip)

    def _emit(self, event: str, *args) -> None:
        """Send an event to the UI thread."""
        if self.ui is None:
            return
        def _go():
            fn = getattr(self.ui, event, None)
            if fn:
                try:
                    fn(*args)
                except Exception as exc:
                    log.debug("UI event %s failed: %s", event, exc)
        self.ui.after(0, _go)

    # -- intent heuristics -------------------------------------------------

    @staticmethod
    def _classify_intent(text: str) -> tuple[str, str]:
        """Return (intent_name, target) for simple voice intents we want
        to visualise in the command console. Falls back to a generic
        ('LLM.INFER', text) tuple when nothing matches."""
        t = text.lower().strip()
        # Mode fire.
        for mode in ModeManager().modes:
            for phrase in mode.get("phrases", []):
                if phrase.lower() in t:
                    return ("MODE.FIRE", mode["name"].upper())
        # App open.
        for kw in ("open ", "launch ", "start ", "go to "):
            if t.startswith(kw):
                target = t[len(kw):].strip()
                return ("APPLICATION.LAUNCH", target.upper())
        # Search.
        for kw in ("search for ", "search ", "look up "):
            if t.startswith(kw):
                target = t[len(kw):].strip()
                return ("SEARCH.WEB", target.upper())
        # Time.
        for kw in ("what time", "current time", "tell me the time"):
            if kw in t:
                return ("TIME.QUERY", "LOCAL CLOCK")
        # Screenshots.
        if "screenshot" in t or "capture the screen" in t:
            return ("SCREENSHOT.CAPTURE", "PRIMARY")
        # Network scan.
        if "arp" in t or "scan the network" in t or "scan network" in t:
            return ("NETWORK.SCAN", "LOCAL SUBNET")
        return ("LLM.INFER", text.upper())

    # -- mode execution -----------------------------------------------------

    def _run_mode(self, mode: dict) -> None:
        name = mode["name"]
        self._emit("_append_chat", "arvis", f"Switching to {name} mode.")
        self._emit("show_intent",
                   f"fire {name} mode",
                   "MODE.FIRE", name.upper(),
                   f"EXECUTING {len(mode.get('actions', []))} ACTIONS",
                   True)
        for action in mode.get("actions", []):
            result = run_action(action)
            self._emit("_append_log",
                       f"{action.get('type')} {action.get('target')} → {result}")

    def _edit_mode_from_chip(self, mode: dict) -> None:
        if self.ui is None:
            return
        editor = ModeEditor(self.ui, on_save=lambda name, phrases, actions:
                            self._save_mode_from_dialog(mode, name, phrases, actions))
        editor.name_entry.insert(0, mode["name"])
        editor.phrases_entry.insert(0, ", ".join(mode.get("phrases", [])))
        editor.actions_box.delete("1.0", "end")
        editor.actions_box.insert(
            "1.0",
            "\n".join(f"{a.get('type')} | {a.get('target')}"
                      for a in mode.get("actions", [])),
        )

    def _save_mode_from_dialog(self, existing: dict | None,
                               name: str, phrases: list[str],
                               actions: list[dict]) -> None:
        try:
            if existing is None:
                self.modes.add(name, phrases, actions)
            else:
                try:
                    self.modes.remove(existing["name"])
                except KeyError:
                    pass
                self.modes.add(name, phrases, actions)
        except Exception as exc:
            log.warning("Could not save mode: %s", exc)
            self._emit("_append_chat", "arvis", f"Couldn't save mode: {exc}")
            return
        self._refresh_modes_ui()
        self._emit("_append_chat", "arvis", f"Mode '{name}' saved.")

    # -- text command -------------------------------------------------------

    def _handle_text_command(self, text: str) -> None:
        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    # -- core processing ----------------------------------------------------

    def _process(self, text: str) -> None:
        intent, target = self._classify_intent(text)
        mode = self.modes.find_by_phrase(text)
        if mode:
            self._run_mode(mode)
            speak_text(f"{mode['name']} mode activated.")
            return

        # Visualise the intent + thinking pipeline while the LLM runs.
        self._emit("show_intent", text, intent, target, "—", False)
        self._emit("show_thinking", True)
        self._emit("set_thinking", True)

        try:
            response = executor.invoke({"input": text})
            content = response.get("output", "")
        except Exception as exc:
            log.error("Agent error: %s", exc)
            content = f"I ran into an error: {exc}"
            self._emit("set_error", True)

        self._emit("show_thinking", False)
        self._emit("set_thinking", False)
        self._emit("set_speaking", True)
        self._emit("_append_chat", "arvis", content)
        self._emit("_append_log", f"Agent → {content[:80]}…")

        # Finalise the command console as success.
        self._emit("show_intent", text, intent, target,
                   "RESPONSE GENERATED", True)

        speak_text(content)

        def _stop():
            self._emit("set_speaking", False)
        threading.Timer(max(1.5, len(content) * 0.05), _stop).start()


# ---------------------------------------------------------------------------
# Voice loop (mirrors main.py, but posts events to the UI)
# ---------------------------------------------------------------------------

def voice_loop(controller: ArvisController) -> None:
    """Background thread: listens for the wake word and dispatches commands."""

    def emit_listening(on: bool) -> None:
        if controller.ui:
            controller.ui.after(0, lambda: controller.ui.set_listening(on))

    def emit_log(msg: str) -> None:
        if controller.ui:
            controller.ui.after(0, lambda m=msg: controller.ui._append_log(m))

    def emit_chat(who: str, msg: str) -> None:
        if controller.ui:
            controller.ui.after(0, lambda w=who, m=msg: controller.ui._append_chat(w, m))

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while controller.running:
            try:
                if not controller.conversation_mode:
                    emit_listening(True)
                    audio = recognizer.listen(source, timeout=10)
                    transcript = recognizer.recognize_google(audio).lower()
                    if TRIGGER_WORD in transcript:
                        emit_chat("arvis", "Yes sir?")
                        speak_text("Yes sir?")
                        controller.conversation_mode = True
                        controller.last_interaction_time = time.time()
                else:
                    emit_listening(True)
                    audio = recognizer.listen(source, timeout=10)
                    command = recognizer.recognize_google(audio)
                    emit_chat("you", command)
                    controller._process(command)
                    controller.last_interaction_time = time.time()
            except sr.WaitTimeoutError:
                if (controller.conversation_mode
                        and time.time() - controller.last_interaction_time
                        > CONVERSATION_TIMEOUT):
                    controller.conversation_mode = False
            except sr.UnknownValueError:
                pass
            except Exception as exc:
                log.error("voice loop error: %s", exc)
                time.sleep(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    headless = "--no-gui" in sys.argv

    if headless:
        controller = ArvisController(ui=None)
        log.info("Headless mode — say '%s' then your command.", TRIGGER_WORD)
        try:
            voice_loop(controller)
        except KeyboardInterrupt:
            pass
        return

    # Build UI
    ui = ArvisUI(on_command=lambda _t: None)  # wired properly below
    controller = ArvisController(ui=ui)
    ui._on_command = controller._handle_text_command

    # Background voice listener
    threading.Thread(target=voice_loop, args=(controller,), daemon=True).start()

    try:
        ui.mainloop()
    except KeyboardInterrupt:
        controller.running = False
        ui.destroy()


if __name__ == "__main__":
    main()
