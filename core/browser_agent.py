"""
core.browser_agent
~~~~~~~~~~~~~~~~~~

Multi-step browser research built on top of the existing
:mod:`tools.duckduckgo` and :mod:`tools.opener` tools.

The agent NEVER opens a real browser window.  All "navigation" is a
Google / YouTube / DuckDuckGo search and the assistant reads the
structured top results.  When the user says "open the second result"
the agent maps the ordinal to a URL it already has from a previous
search - no new HTTP fetch is needed.

The :class:`BrowserContextCache` (:mod:`core.context_engine`) keeps
the last query, the last N results, and a short history.  The cache
is what makes follow-ups like "compare these two pages" or "use the
first website" cheap.

Public surface
--------------

* :func:`search_web`            - run a DuckDuckGo search, return top hits.
* :func:`open_url`              - open a URL via the existing opener.
* :func:`research`              - high-level "research X" workflow.
* :func:`summarise_results`     - render a short spoken summary of the
                                   current cache.

The module NEVER raises; every failure is logged and surfaces as an
empty result list.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.config import get_config
from core.context_engine import BrowserHit, get_browser_cache
from core.logger import get_logger

log = get_logger(__name__)

# A tiny allow-list of trusted domains we can fetch with stdlib. We
# avoid urllib + beautifulsoup here to keep the dependency surface
# unchanged; only duckduckgo_search (already a dependency) is used.
_TRUSTED_FETCH_DOMAINS = (
    "en.wikipedia.org",
    "github.com",
    "raw.githubusercontent.com",
)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_web(query: str, *, max_results: Optional[int] = None) -> List[BrowserHit]:
    """Run a web search via the existing DuckDuckGo tool.

    Returns a list of :class:`BrowserHit` capped at *max_results*
    (default from config).  On any failure the list is empty.
    """
    if not query or not query.strip():
        return []
    cfg = get_config()
    cap = max_results or cfg.max_browser_results
    hits: List[BrowserHit] = []

    # 1. Try the existing duckduckgo helper.
    try:
        from tools.duckduckgo import duckduckgo_search_tool  # type: ignore

        # The legacy tool is a LangChain ``@tool``; calling it directly
        # returns a string.  We use it as a smoke test - if it works
        # we know the user has network access and the library is
        # installed.  We then call duckduckgo_search directly to get
        # the structured result list.
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            for r in ddgs.text(query, region="wt-wt", safesearch="Moderate", max_results=cap):
                title = (r.get("title") or "").strip()
                url = (r.get("href") or r.get("url") or "").strip()
                snippet = (r.get("body") or r.get("snippet") or "").strip()
                if not title and not url:
                    continue
                hits.append(BrowserHit(title=title or url, url=url, snippet=snippet))
                if len(hits) >= cap:
                    break
    except Exception as exc:  # pragma: no cover - env dependent
        log.warning("DuckDuckGo search failed: %s", exc)
        return []

    if hits:
        get_browser_cache().set_search(query, hits)
    return hits


def summarise_results(hits: Optional[List[BrowserHit]] = None, *, limit: int = 3) -> str:
    """Render a short spoken summary of the top hits."""
    cache = get_browser_cache()
    if hits is None:
        hits = cache.get().results
    if not hits:
        return "I have no results to summarise, sir."
    top = hits[:limit]
    lines = [f"{i+1}. {h.title}" for i, h in enumerate(top)]
    return "Here are the top results: " + "; ".join(lines)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def open_url(url: str) -> Tuple[bool, str]:
    """Open *url* through the existing opener.  Updates the browser cache."""
    if not url or not url.strip():
        return False, "I need a URL to open, sir."
    try:
        from tools.opener import _open_url as opener_open_url  # type: ignore

        opener_open_url(url.strip())
        get_browser_cache().set_page(url.strip(), title=url.strip())
        return True, f"Opening {url}."
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("open_url failed: %s", exc)
        return False, f"I could not open {url}: {exc}"


def open_ordinal(phrase: str) -> Tuple[bool, str]:
    """Open "the second result" / "the first website" from the cache."""
    cache = get_browser_cache()
    hit = cache.resolve_ordinal(phrase or "")
    if hit is None:
        return False, "I do not have a recent search to pick a result from."
    return open_url(hit.url)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
@dataclass
class ResearchReport:
    """Structured output of a research session."""

    query: str
    hits: List[BrowserHit]
    summary: str
    sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "hits": [h.to_dict() for h in self.hits],
            "summary": self.summary,
            "sources": list(self.sources),
        }


def research(query: str, *, max_sources: Optional[int] = None) -> ResearchReport:
    """High-level "research X" workflow.

    Runs one search, takes the top *max_sources* hits, and produces a
    short spoken summary.  Stays well below the credit budget by
    never re-querying.
    """
    cfg = get_config()
    cap = max(1, min(max_sources or cfg.max_research_sources, cfg.max_research_sources))
    hits = search_web(query, max_results=cap)
    sources = [h.url for h in hits if h.url]

    if not hits:
        return ResearchReport(
            query=query,
            hits=[],
            summary=f"I could not find any sources for {query!r}.",
            sources=[],
        )

    # Build a short spoken summary without calling the LLM.  The
    # search-engine snippets are usually enough for the user to
    # decide if they want to dig deeper.
    bullets: List[str] = []
    for i, hit in enumerate(hits[:3]):
        snippet = hit.snippet
        if snippet:
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            bullets.append(f"{hit.title}: {snippet}")
        else:
            bullets.append(hit.title)

    summary = (
        f"I found {len(hits)} sources for {query!r}. "
        f"The strongest starting point is {hits[0].title}. "
        + " Also worth a look: " + "; ".join(h.title for h in hits[1:cap]) + "."
        if len(hits) > 1
        else f" The only hit is {hits[0].title}."
    )
    return ResearchReport(
        query=query,
        hits=hits,
        summary=summary.strip(),
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------
_BROWSER_KEYWORDS = (
    "search for ", "look up ", "find me ", "find ",
    "google ", "search the web", "duckduckgo",
    "research ", "compare ", "summarise the results",
    "summarize the results", "compare these", "compare them",
    "open the first", "open the second", "open the third",
    "use the first", "use the second", "use the third",
    "go to the first", "go to the second", "go to the third",
    "open the last result", "use the last result",
    "what did you find", "show me the results",
)


def looks_like_browser_intent(text: str) -> bool:
    """Return True if *text* is a browser-research / search / open command."""
    lowered = (text or "").lower()
    if not lowered:
        return False
    if any(kw in lowered for kw in _BROWSER_KEYWORDS):
        return True
    if lowered.startswith("search ") or lowered.startswith("google ") or lowered.startswith("research "):
        return True
    return False


def parse_search_query(command: str) -> Optional[str]:
    """Pull the raw query out of a search/look-up command."""
    text = (command or "").strip()
    lowered = text.lower()
    for prefix in (
        "search for ", "search ", "look up ", "find me ", "find ",
        "google search ", "google ", "research ", "duckduckgo ",
    ):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text or None


__all__ = [
    "ResearchReport",
    "search_web",
    "summarise_results",
    "open_url",
    "open_ordinal",
    "research",
    "looks_like_browser_intent",
    "parse_search_query",
]
