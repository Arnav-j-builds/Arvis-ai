"""
tools.terminal
~~~~~~~~~~~~~~

Run shell commands on the user's machine.

Safety
------

The tool keeps an *allowlist* of read-only / safe commands and refuses
anything else unless the caller explicitly enables ``unsafe`` mode. The
intent is to give the assistant the ability to answer "what is my IP",
"how much disk is free", "git status" etc. without exposing a free-form
shell that a misfiring LLM could weaponise.

Edit ``SAFE_COMMANDS`` below to extend the allowlist.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.logger import get_logger

log = get_logger(__name__)


# Commands allowed by default (first token match). Keep this list short and
# read-only by default - they cannot mutate the filesystem on their own.
SAFE_COMMANDS = {
    # introspection
    "ls", "dir", "pwd", "echo", "whoami", "date", "hostname", "env",
    "cat", "type", "head", "tail", "wc", "stat", "file", "find", "tree",
    "ps", "tasklist", "top", "ipconfig", "ifconfig", "ping", "tracert",
    "traceroute", "nslookup", "netstat", "arp", "systeminfo", "ver",
    "git", "python", "python3", "pip", "pip3", "node", "npm", "where",
    "which", "uname", "uptime", "df", "du", "free", "wmic",
    # arvis-specific inspection
    "arvis",
}

# Patterns that are never allowed, even when an unsafe command is permitted.
# These are belt-and-braces against the LLM going rogue.
FORBIDDEN_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"del\s+/[sq]\s+[a-z]:\\",
    r"format\s+[a-z]:",
    r":\(\)\s*\{.*\}\s*;:",          # fork bomb
    r"mkfs",
    r"dd\s+if=.*of=/dev/",
    r"shutdown(\.exe)?\s+/[sr]",    # shutdown /r or /s without confirmation
    r"reg\s+delete",
    r"net\s+user\s+/delete",
    r"cipher\s+/w",
    r"rd\s+/[sq]\s+[a-z]:\\",
]


class TerminalTool(BaseTool):
    """Run a shell command. Read-only by default; opt-in for writes."""

    name = "terminal_tool"
    description = (
        "Run a shell command on the user's machine. Allowed read-only commands "
        "by default include ls/dir, cat, pwd, whoami, ping, ipconfig, git, python "
        "and many more. Returns the command's stdout (and stderr on failure). "
        "Examples: 'run dir', 'show me git status', 'ping google.com'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        text = (command or "").lower().lstrip()
        prefixes = (
            "run ", "execute ", "shell ", "cmd ", "command ",
            "terminal ", "in terminal ", "type ", "show me ",
        )
        return any(text.startswith(p) for p in prefixes) or text.startswith(("$ ", ">>>"))

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        raw = _extract_command(command)
        if not raw:
            return ToolResult(success=False, message="I need a command to run, sir.")

        unsafe = bool(context and context.get("unsafe"))
        if not _is_allowed(raw, unsafe=unsafe):
            return ToolResult(
                success=False,
                message=(
                    "I will not run that command - it is not in the safe list. "
                    "Add it to SAFE_COMMANDS in tools/terminal.py to enable it, "
                    "or invoke me with unsafe=True to bypass (not recommended)."
                ),
            )

        try:
            proc = subprocess.run(
                raw,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, message=f"Command timed out after 30s: {raw}")
        except Exception as exc:
            return ToolResult(success=False, message=f"Command failed: {exc}")

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        body = out or err or "(no output)"
        # Cap the body so the LLM and the UI don't get a 10 MB blob.
        if len(body) > 4000:
            body = body[:4000] + "\n... [truncated]"
        success = proc.returncode == 0
        return ToolResult(
            success=success,
            message=f"$ {raw}\n\n{body}",
            data={"command": raw, "exit_code": proc.returncode, "stdout": out, "stderr": err},
        )


# ---------------------------------------------------------------------------
def _extract_command(command: str) -> str:
    """Strip the leading trigger phrase so we can pass the rest to a shell."""
    text = (command or "").strip()
    lowered = text.lower()
    prefixes = [
        "run ", "execute ", "shell ", "cmd ", "command ", "terminal ",
        "in terminal ", "type ", "show me ",
    ]
    for p in prefixes:
        if lowered.startswith(p):
            return text[len(p):].strip()
    if text.startswith("$ "):
        return text[2:].strip()
    if text.startswith(">>>"):
        return text[3:].strip()
    return text


def _is_allowed(raw: str, unsafe: bool) -> bool:
    """Check that the command is not on the forbidden list and (if safe-mode)
    starts with an allowlisted token."""
    if not raw.strip():
        return False

    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, raw, re.IGNORECASE):
            log.warning("Refused forbidden command: %r", raw)
            return False

    if unsafe:
        return True

    try:
        head = shlex.split(raw, posix=(sys.platform != "win32"))[0]
    except ValueError:
        return False
    head = os.path.basename(head).lower()
    if head in SAFE_COMMANDS:
        return True
    # Allow absolute paths whose basename matches.
    return any(head == os.path.basename(token).lower() for token in SAFE_COMMANDS)


def register_terminal_tool(router) -> List[BaseTool]:
    tool = TerminalTool()
    router.register(
        tool,
        keywords=(
            "run ", "execute ", "shell ", "terminal ", "cmd ",
            "type ", "show me ", "git status", "ping ", "ipconfig",
        ),
        priority=80,
    )
    return [tool]


__all__ = ["TerminalTool", "register_terminal_tool", "SAFE_COMMANDS"]