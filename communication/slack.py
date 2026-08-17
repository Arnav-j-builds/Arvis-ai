"""
communication.slack
~~~~~~~~~~~~~~~~~~~

Slack delivery via the official Web API.

* Uses ``chat.postMessage`` to send messages to a channel or DM.
* Credentials: ``SLACK_BOT_TOKEN`` (xoxb-...) and ``SLACK_DEFAULT_CHANNEL``.
"""

from __future__ import annotations

from typing import Optional

import requests

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


_API = "https://slack.com/api/chat.postMessage"


class SlackTool(BaseTool):
    name = "slack_tool"
    description = (
        "Send a Slack message. Examples: 'send slack to #devs saying deploy "
        "complete', 'message slack channel general hello team'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[dict] = None) -> bool:
        return "slack" in (command or "").lower()

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[dict] = None) -> ToolResult:
        cfg = get_config()
        if not cfg.slack_bot_token:
            return ToolResult(
                success=False,
                message="Slack is not configured. Set SLACK_BOT_TOKEN in your .env file.",
            )

        text = (command or "").strip()
        channel, body = _parse_send(text, default_channel=cfg.slack_default_channel)
        if not channel:
            return ToolResult(success=False, message="I need a Slack channel (or set SLACK_DEFAULT_CHANNEL).")
        if not body:
            return ToolResult(success=False, message="I need a message body for Slack, sir.")

        try:
            response = requests.post(
                _API,
                headers={"Authorization": f"Bearer {cfg.slack_bot_token}"},
                json={"channel": channel, "text": body},
                timeout=20,
            )
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(success=False, message=f"Slack send failed: {exc}")

        payload = response.json()
        if not payload.get("ok", False):
            return ToolResult(success=False, message=f"Slack rejected the message: {payload.get('error')}")

        return ToolResult(success=True, message=f"Slack message sent to {channel}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_send(command: str, default_channel: Optional[str]) -> tuple[str, str]:
    text = command.strip()
    lowered = text.lower()

    channel = default_channel or ""
    body = text

    # Strip leading verbs
    for prefix in ("send slack message to ", "send slack to ", "send slack ", "send a slack message to ", "message slack channel ", "message slack ", "slack "):
        if lowered.startswith(prefix):
            remainder = text[len(prefix):].strip()
            break
    else:
        remainder = text

    if remainder:
        parts = remainder.split(" ", 1)
        candidate = parts[0].strip()
        # If the user gave an explicit channel marker (#name, @name, name) use it.
        if candidate.startswith("#") or candidate.startswith("@") or candidate.lower() in {"channel", "dm"}:
            channel = candidate if not candidate.lower() == "channel" else (parts[1].split(" ", 1)[0] if len(parts) > 1 else default_channel)
            body = parts[1] if len(parts) > 1 else ""
        else:
            # First token is channel, remainder is body
            channel = candidate
            body = parts[1] if len(parts) > 1 else ""

    # Pull out the message body using common separators.
    # The body may legitimately start with "saying" or "that" so accept both
    # the form "saying X" and " ... saying X".
    body_lowered = body.lower()
    if " saying " in body_lowered:
        body = body[body_lowered.index(" saying ") + len(" saying "):].strip()
    elif body_lowered.startswith("saying ") or body_lowered.startswith("that "):
        body = body.split(" ", 1)[1].strip()
    elif " that " in body_lowered:
        body = body[body_lowered.index(" that ") + len(" that "):].strip()

    return channel, body.strip()


def register_slack_tool(router) -> list[BaseTool]:
    tool = SlackTool()
    router.register(tool, keywords=("slack",), priority=70)
    return [tool]


__all__ = ["SlackTool", "register_slack_tool"]
