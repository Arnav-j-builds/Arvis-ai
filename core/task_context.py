"""
core.task_context
~~~~~~~~~~~~~~~~~

Mutable session-scoped memory shared by the task planner, executor,
and conversation manager.

Holds pointers to the things a follow-up sentence refers to:

* current_app         - the last ``open_app`` target
* current_folder      - the last folder the assistant worked in
* current_file        - the last file the assistant created/edited
* last_tool           - name of the last tool that ran
* last_tool_result    - the ``ToolResult`` from the last step
* last_search_results - list of strings the last search step produced
* last_intent / goal  - last user intent / mission goal

The pronoun resolver rewrites "the second result", "that folder",
"the same file", etc. against this state, so follow-ups feel natural.

The context is reset when the conversation session ends. It is NOT
persisted to disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_PRONOUN_MAP = {
    "it": ["current_file", "current_app", "last_tool_result"],
    "that": ["current_file", "current_app", "last_tool_result"],
    "this": ["current_file", "current_app", "last_tool_result"],
    "there": ["current_folder"],
    "the same file": ["current_file"],
    "the same folder": ["current_folder"],
    "the same app": ["current_app"],
    "the previous file": ["current_file"],
    "the previous folder": ["current_folder"],
    "the previous app": ["current_app"],
}


# "the first result", "the second one", "the third item", ...
# Numeric form (1st / 2nd / 3rd ...).
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:one|result|item|file|folder|link|app)\b",
    flags=re.IGNORECASE,
)
# Spelled-out ordinal form ("the second result"). Maps to its 1-based index.
_ORDINAL_WORD_RE = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(?:one|result|item|file|folder|link|app)\b",
    flags=re.IGNORECASE,
)
_ORDINAL_WORD_INDEX = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
# Bare "the result" -> the first/last
_THE_RESULT_RE = re.compile(
    r"\bthe\s+(?:first|last|next)\s+(?:result|one|item)\b",
    flags=re.IGNORECASE,
)


@dataclass
class TaskContext:
    """Session-scoped memory for follow-ups and pronoun resolution."""

    current_app: Optional[str] = None
    current_folder: Optional[str] = None
    current_file: Optional[str] = None
    last_tool: Optional[str] = None
    last_tool_result: Optional[Any] = None  # ToolResult or None
    last_search_results: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    last_goal: Optional[str] = None

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def record_tool(
        self,
        tool_name: str,
        result: Any,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update context with what the just-executed tool produced.

        ``arguments`` are inspected to lift out app names / file paths /
        folder paths so follow-ups can refer to them.
        """
        self.last_tool = tool_name
        self.last_tool_result = result
        args = arguments or {}

        # Friendly-name heuristics per tool family.
        if tool_name in {"open_app", "open_anything", "hand_mouse_tool", "vision_mouse_tool"}:
            candidate = args.get("app") or args.get("name") or args.get("query")
            if isinstance(candidate, str) and candidate:
                self.current_app = candidate

        if tool_name in {"typing_tool"}:
            text = args.get("text") or args.get("query")
            if isinstance(text, str) and text:
                # ``typing_tool`` pastes arbitrary text - track the
                # most recent text as the ``current_file`` stand-in
                # so follow-ups can refer to "the text I just typed".
                self.current_file = text

        if tool_name in {"terminal_tool", "open_anything"}:
            candidate = args.get("folder") or args.get("cwd") or args.get("path")
            if isinstance(candidate, str) and candidate:
                self.current_folder = candidate

        if isinstance(result, dict):
            data = result.get("data") if isinstance(result.get("data"), dict) else None
        else:
            data = None
        # ``ToolResult.data`` is a dict - lift any obvious keys.
        try:
            from core.base import ToolResult  # local import to avoid cycles

            if isinstance(result, ToolResult) and isinstance(result.data, dict):
                for key, target in (
                    ("file", "current_file"),
                    ("path", "current_file"),
                    ("folder", "current_folder"),
                    ("app", "current_app"),
                    ("query", "last_search_results"),
                ):
                    val = result.data.get(key)
                    if isinstance(val, str) and target in {"current_file", "current_folder", "current_app"}:
                        setattr(self, target, val)
                    elif isinstance(val, list) and target == "last_search_results":
                        self.last_search_results = [str(x) for x in val]
        except Exception:
            pass

    def reset(self) -> None:
        """Clear all session memory (called when a conversation ends)."""
        self.current_app = None
        self.current_folder = None
        self.current_file = None
        self.last_tool = None
        self.last_tool_result = None
        self.last_search_results = []
        self.last_intent = None
        self.last_goal = None

    def has_referent(self) -> bool:
        """Return True if there is at least one thing a pronoun could
        refer to.

        Used by the conversation manager to decide whether an utterance
        containing "it" / "that" / "this" should be classified as a
        follow-up vs. a brand-new command.
        """
        return any(
            getattr(self, name, None)
            for name in (
                "current_app",
                "current_folder",
                "current_file",
                "last_tool_result",
                "last_search_results",
            )
        )

    # ------------------------------------------------------------------
    # Pronoun / reference resolution
    # ------------------------------------------------------------------
    def resolve(self, text: str) -> str:
        """Rewrite *text* in place, replacing pronouns and ordinal
        references with concrete values from the context.

        Returns the rewritten string. Untouched words are returned
        verbatim so the caller can still see what was said.
        """
        if not text:
            return text
        rewritten = text

        # 1. "the first result", "the second one" - resolve against
        # the last search-result list.
        m = _ORDINAL_RE.search(rewritten)
        if m and self.last_search_results:
            idx = max(1, int(m.group(1))) - 1
            if 0 <= idx < len(self.last_search_results):
                placeholder = m.group(0)
                rewritten = rewritten.replace(placeholder, self.last_search_results[idx])

        # 1b. Spelled-out ordinal form ("the second result"). Numeric
        # form (above) handles "2nd" / "2" but spoken utterances tend
        # to use words.
        m = _ORDINAL_WORD_RE.search(rewritten)
        if m and self.last_search_results:
            idx = _ORDINAL_WORD_INDEX.get(m.group(1).lower(), 0) - 1
            if 0 <= idx < len(self.last_search_results):
                placeholder = m.group(0)
                rewritten = rewritten.replace(placeholder, self.last_search_results[idx])

        # 2. Bare "the result" / "the first one" - default to the
        # last item in the search list.
        if _THE_RESULT_RE.search(rewritten) and self.last_search_results:
            rewritten = _THE_RESULT_RE.sub(self.last_search_results[-1], rewritten)

        # 3. "it" / "that" / "this" / "there" - resolve against the
        # most relevant context slot.
        lowered = rewritten.lower()
        for pronoun, fields in _PRONOUN_MAP.items():
            if pronoun not in lowered:
                continue
            replacement: Optional[str] = None
            for fld in fields:
                value = getattr(self, fld, None)
                if value:
                    replacement = str(value)
                    break
            if replacement:
                # Match with word boundaries so we don't replace "it"
                # inside "with". Case-insensitive replace preserves
                # the original casing of the first match.
                pattern = re.compile(rf"\b{re.escape(pronoun)}\b", flags=re.IGNORECASE)
                rewritten = pattern.sub(replacement, rewritten, count=1)
                lowered = rewritten.lower()

        return rewritten

    # ------------------------------------------------------------------
    # Snapshot for LLM prompt
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict summarising the context.

        Used in the planner prompt so the LLM can see "you just
        opened Chrome" before it plans the next mission.
        """
        last_message = None
        try:
            from core.base import ToolResult

            if isinstance(self.last_tool_result, ToolResult):
                last_message = self.last_tool_result.message
        except Exception:
            last_message = str(self.last_tool_result) if self.last_tool_result else None
        return {
            "current_app": self.current_app,
            "current_folder": self.current_folder,
            "current_file": self.current_file,
            "last_tool": self.last_tool,
            "last_tool_message": last_message,
            "last_intent": self.last_intent,
            "last_goal": self.last_goal,
            "search_result_count": len(self.last_search_results),
        }


__all__ = ["TaskContext"]
