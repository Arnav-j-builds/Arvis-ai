"""
core.context_engine
~~~~~~~~~~~~~~~~~~~

Shared short-lived context for the four new capabilities (Screen
Context, Visual Action, Skill Builder, Browser Agent).

Three independent caches with their own TTLs:

* :class:`ScreenContextCache`  - the structured result of "look at the
  screen right now".  1-3 second TTL.
* :class:`BrowserContextCache` - the last few browser results so a
  follow-up like "open the second result" makes sense.
* :class:`SkillContext`        - records of skills just learned, run,
  or referenced.

Each cache is intentionally tiny - they are NOT a database. They exist
to avoid re-running OCR / re-capturing the screen / re-searching the
web inside a single conversation turn.

The module is deliberately import-only. There is no I/O at import time
and no global mutable state beyond a single module-level singleton
getter that is initialised lazily.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ScreenContext
# ---------------------------------------------------------------------------
@dataclass
class ScreenElement:
    """One detected UI element on the screen."""

    id: str
    type: str = "unknown"   # button / link / text / input / icon / ...
    text: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
        }

    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class ScreenContext:
    """A structured snapshot of what is currently on the screen."""

    timestamp: float
    width: int
    height: int
    active_window: str = ""
    application: str = ""
    title: str = ""
    ocr_text: str = ""
    elements: List[ScreenElement] = field(default_factory=list)
    source_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "active_window": self.active_window,
            "application": self.application,
            "title": self.title,
            "ocr_text": self.ocr_text,
            "elements": [e.to_dict() for e in self.elements],
            "source_path": self.source_path,
        }

    def find_text(self, query: str) -> List[ScreenElement]:
        """Return every element whose ``text`` contains *query* (case-insensitive)."""
        q = (query or "").strip().lower()
        if not q:
            return []
        return [el for el in self.elements if q in el.text.lower()]

    def find_first(self, query: str) -> Optional[ScreenElement]:
        matches = self.find_text(query)
        return matches[0] if matches else None


class ScreenContextCache:
    """Single-slot cache for the most recent :class:`ScreenContext`."""

    def __init__(self, ttl_s: float = 2.0) -> None:
        self._ttl = max(0.0, float(ttl_s))
        self._lock = threading.Lock()
        self._ctx: Optional[ScreenContext] = None
        self._path: str = ""

    def set_ttl(self, ttl_s: float) -> None:
        self._ttl = max(0.0, float(ttl_s))

    def put(self, ctx: ScreenContext, path: str = "") -> None:
        with self._lock:
            self._ctx = ctx
            self._path = path or ctx.source_path

    def get(self, *, force: bool = False) -> Optional[ScreenContext]:
        with self._lock:
            if self._ctx is None:
                return None
            if not force and (time.time() - self._ctx.timestamp) > self._ttl:
                return None
            return self._ctx

    def last_path(self) -> str:
        with self._lock:
            return self._path

    def invalidate(self) -> None:
        with self._lock:
            self._ctx = None
            self._path = ""


# ---------------------------------------------------------------------------
# BrowserContext
# ---------------------------------------------------------------------------
@dataclass
class BrowserHit:
    """A single search result."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class BrowserContext:
    """The recent browser state: last query, last results, recent pages."""

    timestamp: float = 0.0
    current_url: str = ""
    page_title: str = ""
    search_query: str = ""
    results: List[BrowserHit] = field(default_factory=list)
    selected_index: int = -1
    history: List[str] = field(default_factory=list)  # URLs visited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_url": self.current_url,
            "page_title": self.page_title,
            "search_query": self.search_query,
            "results": [r.to_dict() for r in self.results],
            "selected_index": self.selected_index,
            "history": list(self.history[-10:]),
        }


class BrowserContextCache:
    """Stores the most recent browser context for follow-up commands."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ctx = BrowserContext()
        self._max_results = 5
        self._max_history = 10

    def configure(self, *, max_results: int = 5, max_history: int = 10) -> None:
        with self._lock:
            self._max_results = max(1, int(max_results))
            self._max_history = max(1, int(max_history))

    def set_search(self, query: str, hits: List[BrowserHit]) -> None:
        with self._lock:
            self._ctx.timestamp = time.time()
            self._ctx.search_query = query
            self._ctx.results = list(hits[: self._max_results])
            self._ctx.selected_index = -1

    def set_page(self, url: str, title: str = "") -> None:
        with self._lock:
            self._ctx.timestamp = time.time()
            self._ctx.current_url = url
            self._ctx.page_title = title
            self._ctx.history.append(url)
            if len(self._ctx.history) > self._max_history:
                self._ctx.history = self._ctx.history[-self._max_history:]

    def select(self, index: int) -> Optional[BrowserHit]:
        with self._lock:
            if 0 <= index < len(self._ctx.results):
                self._ctx.selected_index = index
                return self._ctx.results[index]
            return None

    def resolve_ordinal(self, phrase: str) -> Optional[BrowserHit]:
        """Map "the second result" / "first one" / "result 3" to a hit."""
        with self._lock:
            if not self._ctx.results:
                return None
            phrase = (phrase or "").lower()
            words = {
                "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
                "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
            }
            for word, idx in words.items():
                if word in phrase and idx < len(self._ctx.results):
                    return self._ctx.results[idx]
            return None

    def get(self) -> BrowserContext:
        with self._lock:
            return self._ctx

    def reset(self) -> None:
        with self._lock:
            self._ctx = BrowserContext()


# ---------------------------------------------------------------------------
# SkillContext
# ---------------------------------------------------------------------------
@dataclass
class SkillRecord:
    """One learned skill - persisted to disk by :class:`SkillManager`."""

    name: str
    description: str = ""
    version: int = 1
    variables: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "variables": list(self.variables),
            "steps": list(self.steps),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SkillRecord":
        return cls(
            name=str(payload.get("name", "")).strip(),
            description=str(payload.get("description", "")),
            version=int(payload.get("version", 1) or 1),
            variables=[str(v) for v in (payload.get("variables") or [])],
            steps=list(payload.get("steps") or []),
            created_at=float(payload.get("created_at", time.time()) or time.time()),
            updated_at=float(payload.get("updated_at", time.time()) or time.time()),
        )


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_screen_cache: Optional[ScreenContextCache] = None
_browser_cache: Optional[BrowserContextCache] = None


def get_screen_cache() -> ScreenContextCache:
    global _screen_cache
    if _screen_cache is None:
        from core.config import get_config

        cfg = get_config()
        _screen_cache = ScreenContextCache(ttl_s=cfg.screen_context_ttl_s)
    return _screen_cache


def get_browser_cache() -> BrowserContextCache:
    global _browser_cache
    if _browser_cache is None:
        from core.config import get_config

        cfg = get_config()
        _browser_cache = BrowserContextCache()
        _browser_cache.configure(
            max_results=cfg.max_browser_results,
            max_history=cfg.max_browser_results,
        )
    return _browser_cache


def reset_caches() -> None:
    """Clear all cached context. Used by tests and on session end."""
    global _screen_cache, _browser_cache
    if _screen_cache is not None:
        _screen_cache.invalidate()
    if _browser_cache is not None:
        _browser_cache.reset()


__all__ = [
    "ScreenElement",
    "ScreenContext",
    "ScreenContextCache",
    "BrowserHit",
    "BrowserContext",
    "BrowserContextCache",
    "SkillRecord",
    "get_screen_cache",
    "get_browser_cache",
    "reset_caches",
]
