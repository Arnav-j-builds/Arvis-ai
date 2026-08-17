"""
core.config
~~~~~~~~~~~

Centralised configuration for the assistant. Values are read from environment
variables (and therefore from a local ``.env`` file courtesy of
``python-dotenv``) and exposed as typed attributes.

The point of this module is to **avoid global state** and to keep all the
magic strings (paths, model names, credentials) in a single place that can
be overridden in tests.

Example
-------
>>> from core.config import get_config
>>> cfg = get_config()
>>> cfg.vision_model
'llava'
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - python-dotenv is optional
    load_dotenv = None  # type: ignore

# Load .env once on import. ``override=False`` keeps real environment variables
# (e.g. those set by the OS) authoritative.
if load_dotenv is not None:
    try:
        load_dotenv(override=False)
    except Exception:  # pragma: no cover - tolerate missing dotenv
        pass

log = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return the trimmed value of an environment variable, or *default*."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid integer for %s=%r, using default %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = _env(name)
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items if items else list(default)


def project_root() -> Path:
    """Return the absolute path to the project root (the directory containing
    ``main.py``)."""
    here = Path(__file__).resolve()
    # core/ -> project root
    return here.parent.parent


@dataclass(frozen=True)
class Config:
    """Immutable configuration object."""

    # ----- General --------------------------------------------------------
    trigger_word: str = field(default_factory=lambda: (_env("JARVIS_TRIGGER_WORD", "friday") or "friday").lower())
    conversation_timeout: int = field(default_factory=lambda: _env_int("JARVIS_CONVERSATION_TIMEOUT", 30))

    # ----- Paths ----------------------------------------------------------
    project_root: Path = field(default_factory=project_root)
    storage_dir: Path = field(default_factory=lambda: project_root() / "storage")
    screenshots_dir: Path = field(default_factory=lambda: project_root() / "storage" / "screenshots")
    routines_file: Path = field(default_factory=lambda: project_root() / "storage" / "routines.json")
    env_file: Path = field(default_factory=lambda: project_root() / ".env")

    # ----- LLM ------------------------------------------------------------
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434")
    llm_model: str = field(default_factory=lambda: _env("JARVIS_LLM_MODEL", "minimax-m3:cloud") or "minimax-m3:cloud")
    vision_model: str = field(default_factory=lambda: _env("JARVIS_VISION_MODEL", "llava") or "llava")

    # ----- Vision ---------------------------------------------------------
    vision_enabled: bool = field(default_factory=lambda: _env_bool("JARVIS_VISION_ENABLED", True))
    vision_ocr_engine: str = field(default_factory=lambda: (_env("JARVIS_OCR_ENGINE", "tesseract") or "tesseract").lower())
    vision_ocr_languages: List[str] = field(default_factory=lambda: _env_list("JARVIS_OCR_LANGUAGES", ["en"]))
    vision_use_webcam: bool = field(default_factory=lambda: _env_bool("JARVIS_USE_WEBCAM", True))
    vision_webcam_index: int = field(default_factory=lambda: _env_int("JARVIS_WEBCAM_INDEX", 0))

    # ----- Email ----------------------------------------------------------
    email_imap_host: Optional[str] = field(default_factory=lambda: _env("EMAIL_IMAP_HOST"))
    email_imap_port: int = field(default_factory=lambda: _env_int("EMAIL_IMAP_PORT", 993))
    email_smtp_host: Optional[str] = field(default_factory=lambda: _env("EMAIL_SMTP_HOST"))
    email_smtp_port: int = field(default_factory=lambda: _env_int("EMAIL_SMTP_PORT", 587))
    email_username: Optional[str] = field(default_factory=lambda: _env("EMAIL_USERNAME"))
    email_password: Optional[str] = field(default_factory=lambda: _env("EMAIL_PASSWORD"))
    email_use_ssl: bool = field(default_factory=lambda: _env_bool("EMAIL_USE_SSL", True))

    # ----- Google / Gmail OAuth -------------------------------------------
    google_client_id: Optional[str] = field(default_factory=lambda: _env("GOOGLE_CLIENT_ID"))
    google_client_secret: Optional[str] = field(default_factory=lambda: _env("GOOGLE_CLIENT_SECRET"))
    google_redirect_uri: Optional[str] = field(default_factory=lambda: _env("GOOGLE_REDIRECT_URI"))
    google_scopes: List[str] = field(default_factory=lambda: _env_list(
        "GOOGLE_SCOPES",
        [
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/userinfo.email",
        ],
    ))

    # ----- WhatsApp -------------------------------------------------------
    whatsapp_default_contact: Optional[str] = field(default_factory=lambda: _env("WHATSAPP_DEFAULT_CONTACT"))
    whatsapp_web_url: str = field(default_factory=lambda: _env("WHATSAPP_WEB_URL", "https://web.whatsapp.com") or "https://web.whatsapp.com")

    # ----- Telegram -------------------------------------------------------
    telegram_bot_token: Optional[str] = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_default_chat_id: Optional[str] = field(default_factory=lambda: _env("TELEGRAM_DEFAULT_CHAT_ID"))

    # ----- Discord --------------------------------------------------------
    discord_webhook_url: Optional[str] = field(default_factory=lambda: _env("DISCORD_WEBHOOK_URL"))
    discord_bot_token: Optional[str] = field(default_factory=lambda: _env("DISCORD_BOT_TOKEN"))
    discord_default_channel_id: Optional[str] = field(default_factory=lambda: _env("DISCORD_DEFAULT_CHANNEL_ID"))

    # ----- Slack ----------------------------------------------------------
    slack_bot_token: Optional[str] = field(default_factory=lambda: _env("SLACK_BOT_TOKEN"))
    slack_default_channel: Optional[str] = field(default_factory=lambda: _env("SLACK_DEFAULT_CHANNEL"))

    # ----- Routers --------------------------------------------------------
    router_max_routine_actions: int = field(default_factory=lambda: _env_int("JARVIS_MAX_ROUTINE_ACTIONS", 25))

    # ----- Task planner / executor ----------------------------------------
    # ``planner_model`` is a smaller, faster Ollama model used to turn
    # a multi-step user request into a structured plan. If it's missing
    # the executor falls back to a single-step plan so nothing breaks.
    # When ``JARVIS_PLANNER_MODEL`` is unset we deliberately fall back to
    # ``llm_model`` rather than a hard-coded default like ``mistral`` -
    # the user's primary LLM is almost always installed locally, whereas a
    # generic planner model may not be.  This avoids spurious
    # ``model 'mistral' not found`` 404s on first run.
    planner_model: str = field(
        default_factory=lambda: (
            _env("JARVIS_PLANNER_MODEL")
            or _env("JARVIS_LLM_MODEL")
            or "minimax-m3:cloud"
        )
    )
    planner_temperature: float = field(default_factory=lambda: float(_env("JARVIS_PLANNER_TEMPERATURE", "0.2") or 0.2))
    task_max_retries: int = field(default_factory=lambda: _env_int("JARVIS_TASK_MAX_RETRIES", 2))
    task_max_replans: int = field(default_factory=lambda: _env_int("JARVIS_TASK_MAX_REPLANS", 1))
    task_max_steps: int = field(default_factory=lambda: _env_int("JARVIS_TASK_MAX_STEPS", 30))
    task_max_duration_s: float = field(default_factory=lambda: float(_env("JARVIS_TASK_MAX_DURATION_S", "120") or 120))
    # HIGH_RISK confirmations require an explicit "yes" spoken after a
    # short delay (this discourages accidental confirmations caused by
    # background noise or a too-eager STT).
    confirmation_high_risk_pause_s: float = field(default_factory=lambda: float(_env("JARVIS_CONFIRM_HIGH_RISK_PAUSE_S", "3.0") or 3.0))
    confirmation_listen_timeout_s: float = field(default_factory=lambda: float(_env("JARVIS_CONFIRM_LISTEN_TIMEOUT_S", "8.0") or 8.0))
    # ``confirm_extra_tools`` lets the user (or a future routine) flag
    # additional tools as confirmation-required even if they are not in
    # the default risk map. Comma-separated env value.
    confirm_extra_tools: List[str] = field(default_factory=lambda: _env_list("JARVIS_CONFIRM_EXTRA", []))

    # ----- Conversation ----------------------------------------------------
    # ``conversation_history`` caps the bounded turn list. ``followup_timeout_s``
    # is how long we wait for a follow-up before returning to wake-word
    # mode (kept separate from the legacy ``conversation_timeout`` which
    # is now only used as a fallback). ``conversation_max_turns`` is the
    # hard cap on turns per session - prevents runaway sessions.
    conversation_history: int = field(default_factory=lambda: _env_int("JARVIS_CONVERSATION_HISTORY", 20))
    followup_timeout_s: float = field(default_factory=lambda: float(_env("JARVIS_FOLLOWUP_TIMEOUT_S", "15") or 15))
    conversation_max_turns: int = field(default_factory=lambda: _env_int("JARVIS_CONVERSATION_MAX_TURNS", 20))

    # ----- Universal Screen / Visual / Skills / Browser ------------------
    # These tunables bound the four new capabilities so they cannot blow
    # through LLM credits, hold the screen lock, or spend forever
    # researching a single question. All values can be overridden via
    # the corresponding ``JARVIS_*`` env variable.
    screen_context_ttl_s: float = field(default_factory=lambda: float(_env("JARVIS_SCREEN_CONTEXT_TTL_S", "2.0") or 2.0))
    max_visual_retries: int = field(default_factory=lambda: _env_int("JARVIS_MAX_VISUAL_RETRIES", 1))
    max_browser_results: int = field(default_factory=lambda: _env_int("JARVIS_MAX_BROWSER_RESULTS", 5))
    max_research_sources: int = field(default_factory=lambda: _env_int("JARVIS_MAX_RESEARCH_SOURCES", 5))
    max_skill_steps: int = field(default_factory=lambda: _env_int("JARVIS_MAX_SKILL_STEPS", 50))
    max_agent_steps: int = field(default_factory=lambda: _env_int("JARVIS_MAX_AGENT_STEPS", 20))
    max_agent_tool_calls: int = field(default_factory=lambda: _env_int("JARVIS_MAX_AGENT_TOOL_CALLS", 40))

    # ----- Smart Room / IoT -----------------------------------------------
    # The ESP32 controller runs on the local LAN. ``ESP32_ROOM_IP`` is
    # intentionally *not* a hardcoded value inside the codebase - it must
    # come from the environment. Leaving it empty disables the feature
    # cleanly instead of crashing.
    smart_room_enabled: bool = field(default_factory=lambda: _env_bool("SMART_ROOM_ENABLED", True))
    esp32_room_ip: Optional[str] = field(default_factory=lambda: _env("ESP32_ROOM_IP"))
    esp32_room_port: int = field(default_factory=lambda: _env_int("ESP32_ROOM_PORT", 80))
    esp32_room_timeout: int = field(default_factory=lambda: _env_int("ESP32_ROOM_TIMEOUT", 5))
    esp32_room_id: str = field(default_factory=lambda: (_env("ESP32_ROOM_ID", "arvis-room-controller") or "arvis-room-controller"))

    # ----- Boot / Windows startup -----------------------------------------
    # Apps and websites to open automatically when arvis is launched at
    # Windows boot (i.e. when ``core.autostart.STARTUP_FLAG`` is on the
    # command line). These lists are read from ``.env`` and are comma-
    # separated, e.g. ``JARVIS_STARTUP_APPS=chrome,notepad``.
    startup_urls: List[str] = field(default_factory=lambda: _env_list("JARVIS_STARTUP_URLS", []))
    startup_apps: List[str] = field(default_factory=lambda: _env_list("JARVIS_STARTUP_APPS", []))
    # When ``True``, the hand-mouse controller is auto-started at boot so
    # arvis can drive the cursor by itself.
    startup_hand_mouse: bool = field(default_factory=lambda: _env_bool("JARVIS_STARTUP_HAND_MOUSE", False))
    # When ``True``, arvis will *also* register itself in the Windows Run
    # registry key on first launch, if it is not already registered. This
    # makes "open as soon as Windows starts" happen automatically the
    # first time the user runs the app.
    auto_register_startup: bool = field(default_factory=lambda: _env_bool("JARVIS_AUTO_REGISTER_STARTUP", True))

    def __post_init__(self) -> None:
        # Ensure storage paths exist - they are cheap to create and avoid
        # FileNotFoundError deep inside feature code.
        for path in (self.storage_dir, self.screenshots_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a singleton :class:`Config` instance."""
    cfg = Config()
    log.debug("Loaded configuration: trigger=%s, vision=%s, llm=%s", cfg.trigger_word, cfg.vision_model, cfg.llm_model)
    return cfg
