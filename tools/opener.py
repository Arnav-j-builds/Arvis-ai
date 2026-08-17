"""
Tool for opening websites, launching applications, and searching
Google / YouTube directly from a voice command.
"""

import os
import platform
import shlex
import subprocess
import webbrowser
from urllib.parse import quote_plus

from langchain.tools import tool


# Common website shortcuts. Extend this freely.
SITE_SHORTCUTS = {
    "youtube": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "chatgpt": "https://chatgpt.com",
    "wikipedia": "https://www.wikipedia.org",
    "netflix": "https://www.netflix.com",
    "whatsapp": "https://web.whatsapp.com",
    "amazon": "https://www.amazon.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "IIT-competition" : "https://iit-competition-ajcordex.lovable.app"
}

# Common desktop application shortcuts. Format: (display name, command).
# Add more entries here as needed.
APP_SHORTCUTS = {
    "notepad": ("notepad.exe", "win"),
    "calculator": ("calc.exe", "win"),
    "calc": ("calc.exe", "win"),
    "explorer": ("explorer.exe", "win"),
    "file explorer": ("explorer.exe", "win"),
    "chrome": ("chrome", "all"),
    "google chrome": ("chrome", "all"),
    "firefox": ("firefox", "all"),
    "edge": ("msedge", "all"),
    "microsoft edge": ("msedge", "all"),
    "vscode": ("Code.exe", "all"),
    "vs code": ("Code.exe", "all"),
    "visual studio code": ("Code.exe", "all"),
    "terminal": ("cmd.exe", "win"),
    "cmd": ("cmd.exe", "win"),
    "powershell": ("powershell.exe", "win"),
    "spotify": ("spotify", "all"),
    "discord": ("discord", "all"),
    "slack": ("slack", "all"),
    "word": ("winword", "win"),
    "excel": ("excel", "win"),
    "powerpoint": ("powerpnt", "win"),
    "outlook": ("outlook", "win"),
}


def _detect_intent(text: str) -> str:
    """Return one of: 'open_website', 'open_app', 'youtube_search',
    'google_search', 'play_youtube', or 'unknown'."""
    t = text.lower().strip()

    # Strip common conversational wrappers like "please ", "can you ", etc.
    for prefix in (
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "i want to ",
        "i'd like to ",
        "i would like to ",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()

    if t.startswith("play ") or t.startswith("youtube play "):
        return "play_youtube"
    if t.startswith("youtube search ") or t.startswith("search youtube ") or t.startswith("youtube "):
        return "youtube_search"
    if t.startswith("google search ") or t.startswith("search google ") or t.startswith("google "):
        return "google_search"
    if t.startswith("open app ") or t.startswith("launch app ") or t.startswith("start app "):
        return "open_app"
    if t.startswith("open ") or t.startswith("launch ") or t.startswith("go to "):
        return "open_website"
    return "unknown"


def _resolve_website(name: str) -> str | None:
    """Map a spoken site name to a URL, falling back to treating it as a
    domain or full URL."""
    cleaned = _strip_wrapper(name).strip()
    key = cleaned.strip().lower().rstrip(".")
    if not key:
        return None
    if key in SITE_SHORTCUTS:
        return SITE_SHORTCUTS[key]

    # Already a URL-ish string
    if "." in key and " " not in key:
        return key if key.startswith("http") else f"https://{key}"

    # If it looks like a single word domain guess (e.g. "github" -> "github.com")
    if key.replace(" ", "").isalpha():
        return f"https://{key.replace(' ', '')}.com"

    return None


def _strip_wrapper(name: str) -> str:
    """Remove conversational wrappers from a website/app name."""
    text = name.strip()
    lowered = text.lower()
    # Drop trailing conversational tails.
    for tail in (" for me", " please", " thanks", " now"):
        if lowered.endswith(tail):
            text = text[: -len(tail)].rstrip()
            lowered = text.lower()
    # Collapse the bare command word into a single site keyword.
    # "open chrome" -> "chrome"
    for prefix in ("please open ", "can you open ", "could you open ",
                   "would you open ", "open ", "launch ", "go to ",
                   "i want to open ", "i'd like to open "):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            lowered = text.lower()
            break
    return text


def _open_url(url: str) -> None:
    """Open a URL using the system default browser. Tries Windows-specific
    start, then falls back to the default webbrowser module."""
    try:
        if platform.system() == "Windows":
            # shell=True is the most reliable way on Windows; the URL is
            # never user-typed shell input here.
            subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
        else:
            webbrowser.open(url)
    except Exception:
        webbrowser.open(url)


def _launch_app(command: str) -> tuple[bool, str]:
    """Attempt to launch a desktop app. Returns (success, message)."""
    system = platform.system()
    try:
        if system == "Windows":
            # Use `start` so .exe lookup happens via PATH / App Paths.
            subprocess.Popen(["cmd", "/c", "start", "", command], shell=False)
        elif system == "Darwin":
            subprocess.Popen(["open", "-a", command])
        else:
            subprocess.Popen([command])
        return True, f"Launched {command}"
    except FileNotFoundError:
        return False, f"Could not find an application called '{command}'."
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Failed to launch '{command}': {exc}"


@tool("open_anything", return_direct=True)
def open_anything(command: str) -> str:
    """
    Open a website, launch a desktop application, or run a Google / YouTube
    search based on a single natural-language command.

    Supported intents (the command may begin with or omit these verbs):
      - "open <site>"      -> open a website in the default browser
      - "launch <app>"     -> launch a desktop application
      - "google <query>" / "google search <query>" -> Google search
      - "youtube <query>" / "youtube search <query>" -> YouTube search
      - "play <query>"     -> open the top YouTube result for <query>

    Examples of valid commands:
      - "open youtube"
      - "open github.com"
      - "launch chrome"
      - "google search python decorators"
      - "youtube search lofi music"
      - "play never gonna give you up"
    """
    try:
        intent = _detect_intent(command)
        text = command.strip()

        if intent == "unknown":
            cleaned = _strip_wrapper(text)
            # Best-effort guess: if it has spaces, treat as a Google search;
            # otherwise treat as a website to open.
            if " " in cleaned:
                return google_search(cleaned)
            url = _resolve_website(cleaned)
            if url:
                _open_url(url)
                return f"Opening {url} sir."
            return f"I wasn't sure what to do with '{command}', sir."

        if intent == "open_website":
            target = _strip_wrapper(text)
            url = _resolve_website(target)
            if not url:
                return f"I don't know how to open '{target}', sir."
            _open_url(url)
            return f"Opening {url} sir."

        if intent == "open_app":
            target = (
                text[len("open app "):]
                if text.lower().startswith("open app ")
                else text[len("launch app "):]
                if text.lower().startswith("launch app ")
                else text[len("start app "):]
            ).strip()
            key = target.lower()
            cmd = APP_SHORTCUTS.get(key)
            if cmd is None:
                # Try launching the raw name as a command.
                ok, msg = _launch_app(target)
                return ("Yes sir. " + msg) if ok else msg
            command_str, scope = cmd
            if scope == "win" and platform.system() != "Windows":
                return f"'{target}' is a Windows-only application, sir."
            ok, msg = _launch_app(command_str)
            return ("Yes sir. " + msg) if ok else msg

        if intent == "google_search":
            query = (
                text[len("google search "):]
                if text.lower().startswith("google search ")
                else text[len("search google "):]
                if text.lower().startswith("search google ")
                else text[len("google "):]
            ).strip()
            return google_search(query)

        if intent == "youtube_search":
            query = (
                text[len("youtube search "):]
                if text.lower().startswith("youtube search ")
                else text[len("search youtube "):]
                if text.lower().startswith("search youtube ")
                else text[len("youtube "):]
            ).strip()
            return youtube_search(query)

        if intent == "play_youtube":
            query = (
                text[len("play "):]
                if text.lower().startswith("play ")
                else text[len("youtube play "):]
            ).strip()
            return play_on_youtube(query)

        return f"I couldn't understand the command: '{command}', sir."

    except Exception as exc:
        return f"Something went wrong while handling '{command}': {exc}"


def google_search(query: str) -> str:
    """Open a Google search results page for the given query."""
    if not query.strip():
        return "I need a search query, sir."
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    _open_url(url)
    return f"Searching Google for '{query}', sir."


def youtube_search(query: str) -> str:
    """Open a YouTube search results page for the given query."""
    if not query.strip():
        return "I need a search query, sir."
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    _open_url(url)
    return f"Searching YouTube for '{query}', sir."


def play_on_youtube(query: str) -> str:
    """Open the first YouTube search result for the given query."""
    if not query.strip():
        return "I need something to play, sir."
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    _open_url(url)
    return f"Playing '{query}' on YouTube, sir."
