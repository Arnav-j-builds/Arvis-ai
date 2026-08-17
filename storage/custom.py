"""
storage.custom
~~~~~~~~~~~~~~

Thread-safe JSON-file storage for user-defined:

* Custom commands   - utterance → response  (and optional follow-up tool call)
* Custom modes      - named behavioural presets (colour, vibe, system prompt)
* Reminders         - schedule a one-shot notification (TTS + Socket.IO event)

Each store follows the same pattern: load-once, in-memory dict, save-on-write,
auto-backup on corrupt JSON. This keeps the web server, the desktop entry
point and any future entry point talking to the *same* truth.

Schema (all JSON, all stored under ``storage/``):

    custom_commands.json  ->  { "<name>": { trigger, response, run_tool? } }
    custom_modes.json     ->  { "<name>": { colour, vibe, prompt } }
    reminders.json        ->  { "<id>":  { text, fire_at, fired, created_at } }
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Generic JSON store
# ---------------------------------------------------------------------------
class JsonStore:
    """Tiny atomic JSON file store. Thread-safe; backs up on parse errors."""

    def __init__(self, path: Path, default: Optional[Dict[str, Any]] = None) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = default or {}
        self._load()

    # ---- I/O -------------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._write_locked()
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
                else:
                    log.warning("%s is not an object - resetting", self._path)
                    self._data = {}
                    self._write_locked()
            except json.JSONDecodeError as exc:
                backup = self._path.with_suffix(".bak")
                log.warning("%s is corrupt (%s) - backing up to %s", self._path, exc, backup)
                try:
                    backup.write_text(self._path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception:
                    pass
                self._data = {}
                self._write_locked()

    def _write_locked(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        # Note: ensure_ascii=False preserves Unicode (emoji, accents) so
        # the UI can render them directly. We still serialise with a UTF-8
        # BOM-free plain write.
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def _save(self) -> None:
        with self._lock:
            self._write_locked()

    # ---- API -------------------------------------------------------------
    def all(self) -> Dict[str, Any]:
        with self._lock:
            return {k: _deep_copy(v) for k, v in self._data.items()}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            v = self._data.get(key.lower() if isinstance(key, str) else key)
            return _deep_copy(v) if v is not None else None

    def upsert(self, key: str, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("value must be a dict")
        with self._lock:
            self._data[key.lower()] = _deep_copy(value)
            self._write_locked()

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._data.pop(key.lower(), None) is not None
            if existed:
                self._write_locked()
            return existed


def _deep_copy(value: Any) -> Any:
    """Cheap deep copy for JSON-shaped data (dicts, lists, primitives)."""
    return json.loads(json.dumps(value, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Custom commands
# ---------------------------------------------------------------------------
@dataclass
class CustomCommand:
    """A user-defined mapping: spoken phrase -> canned response / tool call."""

    name: str
    trigger: List[str] = field(default_factory=list)
    response: str = ""
    run_tool: Optional[str] = None        # optional tool name to invoke
    description: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "CustomCommand":
        return cls(
            name=name,
            trigger=[str(t).strip() for t in payload.get("trigger", []) if str(t).strip()],
            response=str(payload.get("response", "")),
            run_tool=payload.get("run_tool"),
            description=str(payload.get("description", "")),
            created_at=float(payload.get("created_at", time.time())),
        )


class CustomCommandStore(JsonStore):
    """Store of CustomCommand objects."""

    def __init__(self, path: Optional[Path] = None) -> None:
        cfg = get_config()
        super().__init__(path or cfg.storage_dir / "custom_commands.json")

    # ---- Typed wrappers --------------------------------------------------
    def list(self) -> List[CustomCommand]:
        with self._lock:
            return [CustomCommand.from_dict(k, v) for k, v in self._data.items()]

    def upsert_command(self, cmd: CustomCommand) -> None:
        if not cmd.name:
            raise ValueError("Custom command requires a name")
        self.upsert(cmd.name, cmd.to_dict())

    def matches(self, command: str) -> Optional[CustomCommand]:
        lowered = (command or "").lower()
        with self._lock:
            best: Optional[CustomCommand] = None
            best_len = 0
            for raw in self._data.values():
                cand = CustomCommand.from_dict(raw.get("name", ""), raw)
                for trig in cand.trigger:
                    if trig and trig.lower() in lowered and len(trig) > best_len:
                        best = cand
                        best_len = len(trig)
            return best


# ---------------------------------------------------------------------------
# Custom modes
# ---------------------------------------------------------------------------
@dataclass
class CustomMode:
    """A user-defined behavioural mode - orb colour + system-prompt flavour."""

    name: str
    colour: str = "#5fe4ff"
    vibe: str = ""
    prompt: str = ""
    emoji: str = "◆"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "CustomMode":
        return cls(
            name=name,
            colour=str(payload.get("colour", "#5fe4ff")),
            vibe=str(payload.get("vibe", "")),
            prompt=str(payload.get("prompt", "")),
            emoji=str(payload.get("emoji", "◆")) or "◆",
            created_at=float(payload.get("created_at", time.time())),
        )


class CustomModeStore(JsonStore):
    def __init__(self, path: Optional[Path] = None) -> None:
        cfg = get_config()
        super().__init__(path or cfg.storage_dir / "custom_modes.json")

    def list(self) -> List[CustomMode]:
        with self._lock:
            return [CustomMode.from_dict(k, v) for k, v in self._data.items()]

    def upsert_mode(self, mode: CustomMode) -> None:
        if not mode.name:
            raise ValueError("Custom mode requires a name")
        self.upsert(mode.name, mode.to_dict())


# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------
@dataclass
class Reminder:
    id: str
    text: str
    fire_at: float             # epoch seconds
    created_at: float = field(default_factory=time.time)
    fired: bool = False
    recurring: Optional[str] = None   # "daily" / "hourly" (optional future)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Reminder":
        rid = str(payload.get("id") or uuid.uuid4())
        text = str(payload.get("text", "")).strip()
        fire_at = float(payload.get("fire_at", 0.0) or 0.0)
        if fire_at <= 0:
            raise ValueError("Reminder requires a positive fire_at epoch")
        if not text:
            raise ValueError("Reminder requires non-empty text")
        return cls(
            id=rid,
            text=text,
            fire_at=fire_at,
            created_at=float(payload.get("created_at", time.time())),
            fired=bool(payload.get("fired", False)),
            recurring=payload.get("recurring"),
        )


class ReminderStore(JsonStore):
    def __init__(self, path: Optional[Path] = None) -> None:
        cfg = get_config()
        super().__init__(path or cfg.storage_dir / "reminders.json")

    def list_active(self) -> List[Reminder]:
        now = time.time()
        with self._lock:
            out: List[Reminder] = []
            for raw in self._data.values():
                rem = Reminder.from_dict(raw)
                if not rem.fired or (rem.recurring and rem.fire_at <= now):
                    out.append(rem)
            return sorted(out, key=lambda r: r.fire_at)

    def due(self, now: Optional[float] = None) -> List[Reminder]:
        now = now or time.time()
        with self._lock:
            out = []
            for raw in self._data.values():
                rem = Reminder.from_dict(raw)
                if not rem.fired and rem.fire_at <= now:
                    out.append(rem)
            return out

    def mark_fired(self, rid: str) -> None:
        with self._lock:
            for k, raw in self._data.items():
                if raw.get("id") == rid:
                    raw["fired"] = True
                    break
            self._write_locked()

    def add(self, text: str, fire_at: float, recurring: Optional[str] = None) -> Reminder:
        rem = Reminder(id=str(uuid.uuid4()), text=text.strip(), fire_at=float(fire_at), recurring=recurring)
        with self._lock:
            self._data[rem.id] = rem.to_dict()
            self._write_locked()
        return rem

    def cancel(self, rid: str) -> bool:
        with self._lock:
            existed = self._data.pop(rid, None) is not None
            if existed:
                self._write_locked()
            return existed


# ---------------------------------------------------------------------------
# Module-level singletons (so web + desktop share the same files)
# ---------------------------------------------------------------------------
_commands_singleton: Optional[CustomCommandStore] = None
_modes_singleton: Optional[CustomModeStore] = None
_reminders_singleton: Optional[ReminderStore] = None


def get_command_store() -> CustomCommandStore:
    global _commands_singleton
    if _commands_singleton is None:
        _commands_singleton = CustomCommandStore()
    return _commands_singleton


def get_mode_store() -> CustomModeStore:
    global _modes_singleton
    if _modes_singleton is None:
        _modes_singleton = CustomModeStore()
    return _modes_singleton


def get_reminder_store() -> ReminderStore:
    global _reminders_singleton
    if _reminders_singleton is None:
        _reminders_singleton = ReminderStore()
    return _reminders_singleton


__all__ = [
    "CustomCommand",
    "CustomCommandStore",
    "CustomMode",
    "CustomModeStore",
    "Reminder",
    "ReminderStore",
    "get_command_store",
    "get_mode_store",
    "get_reminder_store",
]