"""
tools.reminders
~~~~~~~~~~~~~~~

Wraps the persistent :mod:`storage.custom` reminder store as a LangChain-callable
:class:`core.base.BaseTool`. The tool itself does NOT spawn threads; it only
mutates the store. A separate ticker (in :mod:`web_server`) polls the store
and pushes ``reminder.fired`` Socket.IO events to the browser.

The LLM can therefore say:

    "remind me to call mom in 20 minutes"
    "list my reminders"
    "cancel the second reminder"

Natural language parsing supports two common shapes:

* "remind me to <text> in <duration>"
* "remind me to <text> at <time>"

Durations accept ``N seconds|minutes|hours``. ``at HH:MM`` / ``at 9 pm`` is
also parsed (today, or tomorrow if already past).
"""
from __future__ import annotations

import datetime as _dt
import re
import time
from typing import Any, Dict, List, Optional

from core.base import BaseTool, ToolResult
from core.logger import get_logger
from storage.custom import Reminder, get_reminder_store

log = get_logger(__name__)


_DURATION_RE = re.compile(
    r"\b(?:in\s+)?(\d+)\s+(second|sec|minute|min|hour|hr|day)s?\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_REMIND_RE = re.compile(r"remind\s+me\s+to\s+(.+?)(?:\s+(?:in|at)\b|$)", re.IGNORECASE)
_CANCEL_RE = re.compile(r"cancel(?:\s+the)?\s+(.+?)\s+reminder", re.IGNORECASE)
_LIST_RE = re.compile(r"(?:list|show|what\s+are)\s+(?:my\s+)?reminders", re.IGNORECASE)


class ReminderTool(BaseTool):
    """Set, list, or cancel reminders. Also used by the web UI for CRUD."""

    name = "reminder_tool"
    description = (
        "Schedule a reminder for the user. Examples: 'remind me to call mom in 20 minutes', "
        "'remind me to take a break at 3 pm', 'list my reminders', 'cancel reminder 2'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        lowered = (command or "").lower()
        return bool(
            _REMIND_RE.search(lowered)
            or _LIST_RE.search(lowered)
            or _CANCEL_RE.search(lowered)
            or "reminder" in lowered and any(w in lowered for w in ("set", "create", "add", "schedule"))
        )

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        store = get_reminder_store()
        lowered = (command or "").lower()

        if _LIST_RE.search(lowered):
            return self._list(store)

        m = _CANCEL_RE.search(lowered)
        if m:
            target = m.group(1).strip()
            return self._cancel_by_name(store, target)

        if lowered.startswith("cancel") or "cancel reminder" in lowered:
            # bare "cancel reminder" → cancel most recent
            return self._cancel_latest(store)

        m = _REMIND_RE.search(command or "")
        if m:
            text = m.group(1).strip()
            fire_at = _parse_when(command or "")
            if fire_at is None:
                return ToolResult(
                    success=False,
                    message=(
                        "I need a time for the reminder, sir. Try "
                        "'remind me to call mom in 20 minutes' or "
                        "'remind me to take a break at 3 pm'."
                    ),
                )
            rem = store.add(text=text, fire_at=fire_at)
            when_human = _dt.datetime.fromtimestamp(fire_at).strftime("%Y-%m-%d %H:%M")
            return ToolResult(
                success=True,
                message=f"Reminder set for {when_human}: {text}.",
                data={"reminder": rem.to_dict()},
            )

        return ToolResult(success=False, message="I could not parse that as a reminder command.")

    # ------------------------------------------------------------------
    def _list(self, store) -> ToolResult:
        active = store.list_active()
        if not active:
            return ToolResult(success=True, message="You have no active reminders, sir.")
        lines = []
        for i, rem in enumerate(active, 1):
            when = _dt.datetime.fromtimestamp(rem.fire_at).strftime("%Y-%m-%d %H:%M")
            lines.append(f"{i}. {when} — {rem.text}")
        return ToolResult(
            success=True,
            message="Here are your reminders:\n" + "\n".join(lines),
            data={"reminders": [r.to_dict() for r in active]},
        )

    def _cancel_latest(self, store) -> ToolResult:
        active = store.list_active()
        if not active:
            return ToolResult(success=True, message="There is nothing to cancel.")
        latest = active[0]
        store.cancel(latest.id)
        return ToolResult(success=True, message=f"Cancelled the reminder: {latest.text}.")

    def _cancel_by_name(self, store, name: str) -> ToolResult:
        active = store.list_active()
        name_l = name.lower().strip()
        # Try numeric index first ("cancel the 2nd reminder").
        m = re.match(r"(\d+)(?:st|nd|rd|th)?", name_l)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(active):
                rem = active[idx]
                store.cancel(rem.id)
                return ToolResult(success=True, message=f"Cancelled reminder {idx + 1}: {rem.text}.")
        # Otherwise look for a substring match on the reminder text.
        for rem in active:
            if name_l in rem.text.lower():
                store.cancel(rem.id)
                return ToolResult(success=True, message=f"Cancelled reminder: {rem.text}.")
        return ToolResult(success=False, message=f"No reminder matched '{name}'.")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_when(command: str) -> Optional[float]:
    """Extract a fire_at epoch from the command. Supports durations and HH:MM."""
    text = (command or "").strip()

    # 1. absolute time "at 3 pm" / "at 15:30"
    m = _TIME_RE.search(text)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = (m.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        now = _dt.datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += _dt.timedelta(days=1)
        return target.timestamp()

    # 2. relative duration "in 20 minutes" / "5 seconds"
    m = _DURATION_RE.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("sec"):
            return time.time() + n
        if unit.startswith("min"):
            return time.time() + n * 60
        if unit.startswith("hour") or unit.startswith("hr"):
            return time.time() + n * 3600
        if unit.startswith("day"):
            return time.time() + n * 86400

    return None


def register_reminder_tool(router) -> List[BaseTool]:
    tool = ReminderTool()
    router.register(
        tool,
        keywords=("remind", "reminder", "remind me"),
        priority=60,
    )
    return [tool]


__all__ = ["ReminderTool", "register_reminder_tool", "_parse_when"]