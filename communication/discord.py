"""
communication.discord
~~~~~~~~~~~~~~~~~~~~~

Two delivery paths for Discord:

* **Webhook** (``DISCORD_WEBHOOK_URL``) - simplest, no bot account needed.
* **Bot token** (``DISCORD_BOT_TOKEN`` + ``DISCORD_DEFAULT_CHANNEL_ID``) -
  richer features, lets us list the most recent messages from a channel.

The tool exposes ``send`` and ``send webhook`` and ``read messages`` (bot
mode only).
"""

from __future__ import annotations

from typing import Optional

import requests

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


_API = "https://discord.com/api/v10"


class DiscordTool(BaseTool):
    name = "discord_tool"
    description = (
        "Send a Discord message through a webhook or bot. Examples: "
        "'send discord message to dev-team saying deploy complete', 'message my discord team'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[dict] = None) -> bool:
        lowered = (command or "").lower()
        return any(token in lowered for token in ("discord",))

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[dict] = None) -> ToolResult:
        cfg = get_config()
        text = (command or "").strip()
        lowered = text.lower()
        try:
            if "read" in lowered and ("discord" in lowered or "message" in lowered):
                return self._read_messages()
            return self._send(text)
        except Exception as exc:
            log.exception("DiscordTool failed: %s", exc)
            return ToolResult(success=False, message=f"Discord action failed: {exc}")

    # ------------------------------------------------------------------
    def _send(self, command: str) -> ToolResult:
        cfg = get_config()
        content = _extract_content(command)
        if not content:
            return ToolResult(success=False, message="I need a message body for Discord, sir.")

        # Webhook path is the simplest and needs no extra parsing.
        if cfg.discord_webhook_url:
            try:
                response = requests.post(
                    cfg.discord_webhook_url,
                    json={"content": content},
                    timeout=20,
                )
                response.raise_for_status()
                return ToolResult(success=True, message="Discord webhook delivered, sir.")
            except Exception as exc:
                log.warning("Webhook send failed: %s", exc)

        # Bot path
        if cfg.discord_bot_token and cfg.discord_default_channel_id:
            try:
                response = requests.post(
                    f"{_API}/channels/{cfg.discord_default_channel_id}/messages",
                    headers={"Authorization": f"Bot {cfg.discord_bot_token}"},
                    json={"content": content},
                    timeout=20,
                )
                response.raise_for_status()
                return ToolResult(success=True, message="Discord message sent, sir.")
            except Exception as exc:
                return ToolResult(success=False, message=f"Discord bot send failed: {exc}")

        return ToolResult(
            success=False,
            message=(
                "Discord is not configured. Set DISCORD_WEBHOOK_URL or DISCORD_BOT_TOKEN "
                "+ DISCORD_DEFAULT_CHANNEL_ID in your .env file."
            ),
        )

    def _read_messages(self, limit: int = 5) -> ToolResult:
        cfg = get_config()
        if not (cfg.discord_bot_token and cfg.discord_default_channel_id):
            return ToolResult(
                success=False,
                message="Reading Discord history requires DISCORD_BOT_TOKEN and DISCORD_DEFAULT_CHANNEL_ID.",
            )

        try:
            response = requests.get(
                f"{_API}/channels/{cfg.discord_default_channel_id}/messages",
                headers={"Authorization": f"Bot {cfg.discord_bot_token}"},
                params={"limit": limit},
                timeout=20,
            )
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(success=False, message=f"Discord history fetch failed: {exc}")

        messages = response.json()
        if not messages:
            return ToolResult(success=True, message="No recent Discord messages, sir.")

        lines = []
        for message in messages[:limit]:
            author = message.get("author", {}).get("username", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            lines.append(f"[{timestamp}] {author}: {content}")

        return ToolResult(success=True, message="\n".join(lines), data={"messages": messages})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_content(command: str) -> str:
    text = command.strip()
    lowered = text.lower()
    for prefix in (
        "send discord message saying ",
        "send discord saying ",
        "send discord ",
        "send a discord message ",
        "send message to discord ",
        "message my discord team ",
        "message the discord team ",
        "discord ",
    ):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def register_discord_tool(router) -> list[BaseTool]:
    tool = DiscordTool()
    router.register(tool, keywords=("discord",), priority=70)
    return [tool]


__all__ = ["DiscordTool", "register_discord_tool"]
