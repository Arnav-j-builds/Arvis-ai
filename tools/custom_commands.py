"""
tools.custom_commands
~~~~~~~~~~~~~~~~~~~~~

Allows the LLM to read and write the user's custom-command store.

Supported natural-language shapes::

    "create a custom command called <name> that says <response> when I say <trigger>"
    "list my custom commands"
    "delete the <name> custom command"
    "what does my <name> command do"

The LLM only gets the *list* and *match* actions exposed; mutation should
flow through the web UI so the user sees the change reflected immediately.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.base import BaseTool, ToolResult
from core.logger import get_logger
from storage.custom import CustomCommand, get_command_store

log = get_logger(__name__)


_CREATE_RE = re.compile(
    r"(?:create|add|make)\s+(?:a\s+)?custom\s+command\s+(?:called|named)\s+(?P<name>[\w\- ]+?)(?:\s+that\s+(?:says|replies|responds\s+with)\s+(?P<response>.+?))?(?:\s+when\s+(?:i\s+)?say\s+(?P<trigger>.+?))?$",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"(?:list|show)\s+(?:my\s+)?custom\s+commands", re.IGNORECASE)
_DELETE_RE = re.compile(r"(?:delete|remove)\s+(?:the\s+)?(?:custom\s+command\s+)?(?P<name>[\w\- ]+?)(?:\s+custom\s+command)?$", re.IGNORECASE)
_RUN_RE = re.compile(r"(?:run|execute|trigger)\s+(?:the\s+)?(?P<name>[\w\- ]+?)\s+custom\s+command$", re.IGNORECASE)


class CustomCommandTool(BaseTool):
    """Read-only LLM wrapper around the user-defined command store."""

    name = "custom_command_tool"
    description = (
        "Look up user-defined custom commands. Examples: 'list my custom "
        "commands', 'what does the joke command do', 'run the morning command'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower()
        if _LIST_RE.search(text):
            return True
        if _CREATE_RE.search(text):
            return True
        if _DELETE_RE.search(text):
            return True
        if _RUN_RE.search(text):
            return True
        return "custom command" in text

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        store = get_command_store()
        text = (command or "").strip()

        if _LIST_RE.search(text):
            cmds = store.list()
            if not cmds:
                return ToolResult(success=True, message="You have no custom commands yet, sir.")
            lines = [f"{i+1}. {c.name} — triggers: {', '.join(c.trigger) or '(none)'} — {c.response[:80]}" for i, c in enumerate(cmds)]
            return ToolResult(success=True, message="Your custom commands:\n" + "\n".join(lines),
                              data={"commands": [c.to_dict() for c in cmds]})

        m = _CREATE_RE.search(text)
        if m:
            name = (m.group("name") or "").strip()
            response = (m.group("response") or "").strip()
            trigger_phrase = (m.group("trigger") or "").strip()
            if not name or not response:
                return ToolResult(
                    success=False,
                    message=(
                        "To create a custom command I need a name and a response. "
                        "Try: 'create a custom command called joke that says why "
                        "did the chicken cross the road when I say tell me a joke'."
                    ),
                )
            triggers = [trigger_phrase] if trigger_phrase else [name]
            cmd = CustomCommand(name=name, trigger=triggers, response=response, description="created via chat")
            store.upsert_command(cmd)
            return ToolResult(
                success=True,
                message=f"Custom command '{name}' created. It will reply '{response}' when you say '{triggers[0]}'.",
                data={"command": cmd.to_dict()},
            )

        m = _DELETE_RE.search(text)
        if m:
            name = (m.group("name") or "").strip()
            ok = store.delete(name)
            return ToolResult(success=ok, message=f"Custom command '{name}' deleted." if ok else f"No command called '{name}'.")

        m = _RUN_RE.search(text)
        if m:
            name = (m.group("name") or "").strip().lower()
            for c in store.list():
                if c.name.lower() == name:
                    return ToolResult(success=True, message=c.response, data={"command": c.to_dict()})
            return ToolResult(success=False, message=f"No custom command called '{name}'.")

        return ToolResult(success=False, message="I could not parse a custom-command request there.")


def register_custom_command_tool(router) -> List[BaseTool]:
    tool = CustomCommandTool()
    router.register(tool, keywords=("custom command", "custom commands"), priority=50)
    return [tool]


__all__ = ["CustomCommandTool", "register_custom_command_tool"]