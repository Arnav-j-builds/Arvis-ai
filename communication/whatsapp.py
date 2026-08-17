"""
communication.whatsapp
~~~~~~~~~~~~~~~~~~~~~~

Voice-controlled WhatsApp helpers.

The module intentionally does **not** use the unofficial ``whatsapp-web``
APIs (which break regularly and risk account bans). Instead, it talks to
the official WhatsApp Business Cloud API when credentials are configured,
and otherwise opens ``web.whatsapp.com`` so the user can send the message
manually. This is the safest design that still satisfies the spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger
from tools.opener import _open_url

log = get_logger(__name__)


_WA_SEND_URL = "https://graph.facebook.com/v20.0/{phone_id}/messages"


class WhatsAppTool(BaseTool):
    name = "whatsapp_tool"
    description = (
        "Send a WhatsApp message, open a chat, or search a contact. Examples: "
        "'send whatsapp to mom saying I'll be late', 'open whatsapp chat with john', "
        "'search whatsapp contact alice'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[dict] = None) -> bool:
        lowered = (command or "").lower()
        return any(token in lowered for token in ("whatsapp", "whats app", " wa "))

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[dict] = None) -> ToolResult:
        text = (command or "").strip()
        lowered = text.lower()

        try:
            if lowered.startswith("send whatsapp") or "send a whatsapp" in lowered or "send message to" in lowered:
                return self._send(text)
            if lowered.startswith("open whatsapp") or "open chat" in lowered or "whatsapp chat" in lowered:
                return self._open_chat(text)
            if "search" in lowered and "contact" in lowered:
                return self._search_contact(text)
            return ToolResult(success=False, message="I am not sure what to do with that WhatsApp request.")
        except Exception as exc:
            log.exception("WhatsAppTool failed: %s", exc)
            return ToolResult(success=False, message=f"WhatsApp action failed: {exc}")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------
    def _send(self, command: str) -> ToolResult:
        cfg = get_config()
        contact, message = _parse_send(command)
        if not contact:
            return ToolResult(success=False, message="I need a recipient for the WhatsApp message, sir.")

        # Preferred path: official Cloud API
        token = _env("WHATSAPP_TOKEN")
        phone_id = _env("WHATSAPP_PHONE_ID")
        recipient = _resolve_phone_number(contact)
        if token and phone_id and recipient:
            try:
                response = requests.post(
                    _WA_SEND_URL.format(phone_id=phone_id),
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={
                        "messaging_product": "whatsapp",
                        "to": recipient,
                        "type": "text",
                        "text": {"body": message or "(no message body provided)"},
                    },
                    timeout=20,
                )
                response.raise_for_status()
                return ToolResult(success=True, message=f"WhatsApp message sent to {contact}.")
            except Exception as exc:
                log.warning("WhatsApp Cloud API failed (%s) - falling back to web.", exc)

        # Fallback: open WhatsApp Web pre-filled with the text.
        url = f"https://wa.me/{recipient or quote(contact)}?text={quote(message or '')}"
        _open_url(url)
        return ToolResult(
            success=True,
            message=f"I opened WhatsApp Web so you can send the message to {contact}. Press Enter to send.",
        )

    # ------------------------------------------------------------------
    # Open chat
    # ------------------------------------------------------------------
    def _open_chat(self, command: str) -> ToolResult:
        contact = _extract_contact(command)
        recipient = _resolve_phone_number(contact) if contact else None
        url = f"https://wa.me/{recipient}" if recipient else get_config().whatsapp_web_url
        _open_url(url)
        return ToolResult(success=True, message=f"Opening WhatsApp chat for {contact or 'your default list'}.")

    # ------------------------------------------------------------------
    # Search contact (web.whatsapp.com has no public search API, so we open
    # the search interface for the user).
    # ------------------------------------------------------------------
    def _search_contact(self, command: str) -> ToolResult:
        contact = _extract_contact(command)
        url = f"{get_config().whatsapp_web_url}?search={quote(contact or '')}"
        _open_url(url)
        return ToolResult(success=True, message=f"I opened WhatsApp Web. Search for {contact}.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env(name: str) -> Optional[str]:
    import os

    value = os.getenv(name)
    return value.strip() if value else None


def _extract_contact(command: str) -> str:
    text = command.strip()
    lowered = text.lower()
    for prefix in ("send whatsapp to ", "send a whatsapp to ", "message ", "send message to ", "open chat with ", "search contact ", "search whatsapp contact "):
        if lowered.startswith(prefix):
            return text[len(prefix):].split(" saying ", 1)[0].split(" that ", 1)[0].strip()
    return ""


def _parse_send(command: str) -> tuple[str, str]:
    text = command.strip()
    contact = _extract_contact(text)
    body = ""
    lowered = text.lower()
    for sep in (" saying ", " that ", " with message ", " message "):
        idx = lowered.find(sep)
        if idx != -1 and contact:
            body = text[idx + len(sep):].strip()
            break
    if not body and contact:
        # Body may have been extracted with a leading "saying"/"that"
        # attached to the contact name (e.g. "send whatsapp to mom saying hi").
        parts = text.split(contact, 1)
        if len(parts) > 1:
            tail = parts[1].strip()
            for sep in ("saying ", "that ", "with message ", "message "):
                if tail.lower().startswith(sep):
                    body = tail[len(sep):].strip()
                    break
    return contact, body


def _resolve_phone_number(contact: str) -> Optional[str]:
    """Return the E.164 phone number for *contact*.

    The :mod:`contacts` directory (if present) may contain a JSON file with
    explicit ``"mom": "+91..."`` mappings; otherwise we assume the user
    already provided a digit-only phone number and return it unchanged.
    """
    import json
    import re

    digits = re.sub(r"\D", "", contact or "")
    if digits and len(digits) >= 7:
        return digits

    contact_file = Path(__file__).resolve().parent.parent / "storage" / "contacts.json"
    if contact_file.exists():
        try:
            data = json.loads(contact_file.read_text(encoding="utf-8"))
            value = data.get(contact.lower())
            if value:
                return re.sub(r"\D", "", str(value))
        except Exception as exc:  # pragma: no cover
            log.warning("Could not read contacts.json: %s", exc)
    return None


def register_whatsapp_tool(router) -> list[BaseTool]:
    tool = WhatsAppTool()
    router.register(tool, keywords=("whatsapp",), priority=70)
    return [tool]


__all__ = ["WhatsAppTool", "register_whatsapp_tool"]
