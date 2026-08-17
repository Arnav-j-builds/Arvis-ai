"""
communication.telegram
~~~~~~~~~~~~~~~~~~~~~~

Thin wrapper over the official Telegram Bot HTTP API. The tool exposes
``send`` and ``read updates`` (long polling the bot ``getUpdates`` endpoint).

Credentials
-----------

* ``TELEGRAM_BOT_TOKEN`` - the token returned by @BotFather.
* ``TELEGRAM_DEFAULT_CHAT_ID`` - chat id used when the user does not name
  one explicitly. Both values live in ``.env``.
"""

from __future__ import annotations

from typing import Optional

import requests

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramTool(BaseTool):
    name = "telegram_tool"
    description = (
        "Send a Telegram message or read recent updates from your bot. "
        "Examples: 'send telegram to @alice saying meeting at 5', 'read telegram updates'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[dict] = None) -> bool:
        lowered = (command or "").lower()
        return any(token in lowered for token in ("telegram", "tg "))

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[dict] = None) -> ToolResult:
        cfg = get_config()
        if not cfg.telegram_bot_token:
            return ToolResult(
                success=False,
                message=(
                    "Telegram is not configured. Set TELEGRAM_BOT_TOKEN (and optionally "
                    "TELEGRAM_DEFAULT_CHAT_ID) in your .env file."
                ),
            )

        text = (command or "").strip()
        lowered = text.lower()
        try:
            if lowered.startswith("send telegram") or lowered.startswith("send a telegram") or "send telegram" in lowered or "send a telegram" in lowered:
                return self._send(text)
            if "read" in lowered and ("update" in lowered or "telegram" in lowered):
                return self._read_updates()
            return ToolResult(success=False, message="I am not sure what to do with that Telegram request.")
        except Exception as exc:
            log.exception("TelegramTool failed: %s", exc)
            return ToolResult(success=False, message=f"Telegram action failed: {exc}")

    # ------------------------------------------------------------------
    def _send(self, command: str) -> ToolResult:
        cfg = get_config()
        recipient, message = _parse_send(command)
        chat_id = _resolve_chat_id(recipient) or cfg.telegram_default_chat_id
        if not chat_id:
            return ToolResult(success=False, message="I need a chat id or username to send the Telegram message.")
        if not message:
            return ToolResult(success=False, message="I need a message body, sir.")

        try:
            response = requests.post(
                _API.format(token=cfg.telegram_bot_token, method="sendMessage"),
                json={"chat_id": chat_id, "text": message},
                timeout=20,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            return ToolResult(success=False, message=f"Telegram rejected the message: {exc.response.text if exc.response else exc}")
        except Exception as exc:
            return ToolResult(success=False, message=f"Telegram send failed: {exc}")

        return ToolResult(success=True, message=f"Telegram message sent to {recipient or chat_id}.")

    def _read_updates(self, limit: int = 5) -> ToolResult:
        cfg = get_config()
        try:
            response = requests.get(
                _API.format(token=cfg.telegram_bot_token, method="getUpdates"),
                params={"limit": limit * 2},  # ask for 2x because some are bot commands
                timeout=20,
            )
            response.raise_for_status()
        except Exception as exc:
            return ToolResult(success=False, message=f"Telegram update fetch failed: {exc}")

        payload = response.json()
        updates = payload.get("result", [])
        relevant = [u for u in updates if u.get("message") or u.get("edited_message")]
        if not relevant:
            return ToolResult(success=True, message="There are no recent Telegram updates, sir.")

        lines = []
        for entry in relevant[-limit:]:
            message = entry.get("message") or entry.get("edited_message") or {}
            sender = message.get("from", {}).get("first_name", "Unknown")
            text = message.get("text", "")
            date = message.get("date", 0)
            lines.append(f"{sender}: {text} (date={date})")

        return ToolResult(success=True, message="\n".join(lines), data={"updates": relevant})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_send(command: str) -> tuple[str, str]:
    """Best-effort parse of ``"send telegram to <recipient> saying <body>"``.

    Returns ``(recipient, message)``. If no separator is found the first
    whitespace-delimited token is treated as the recipient and the rest as
    the message.
    """
    text = command.strip()
    lowered = text.lower()
    body = text
    for prefix in ("send telegram to ", "send a telegram to ", "send telegram ", "send a telegram "):
        if lowered.startswith(prefix):
            body = text[len(prefix):].strip()
            lowered = body.lower()
            break

    for sep in (" saying ", " that "):
        idx = lowered.find(sep)
        if idx != -1:
            recipient = body[:idx].strip()
            message = body[idx + len(sep):].strip()
            return recipient, message

    parts = body.split(" ", 1)
    recipient = parts[0].strip()
    message = parts[1].strip() if len(parts) > 1 else ""
    return recipient, message


def _resolve_chat_id(recipient: str) -> Optional[str]:
    """Allow usernames (``@alice``) and numeric chat ids."""
    if not recipient:
        return None
    if recipient.startswith("@"):
        # Telegram Bot API cannot message by username in some versions, but
        # the chat_id field accepts usernames when a chat has been started.
        return recipient.lstrip("@")
    if recipient.lstrip("-").isdigit():
        return recipient
    return recipient


def register_telegram_tool(router) -> list[BaseTool]:
    tool = TelegramTool()
    router.register(tool, keywords=("telegram", " tg "), priority=70)
    return [tool]


__all__ = ["TelegramTool", "register_telegram_tool"]
