"""
Communication package for the arvis assistant.

Each provider lives in its own module so adding another (Microsoft Teams,
Matrix, etc.) is purely additive.

Modules
-------

* :mod:`communication.email`    - IMAP + SMTP.
* :mod:`communication.gmail`    - Gmail REST API via OAuth 2.0.
* :mod:`communication.whatsapp` - WhatsApp Cloud API + web.whatsapp.com fallback.
* :mod:`communication.telegram` - Telegram Bot HTTP API.
* :mod:`communication.discord`  - Discord webhooks + bot API.
* :mod:`communication.slack`    - Slack Web API.

Every module exposes a ``register_<provider>_tool(router)`` helper so
:mod:`core.router` can wire them up without import cycles.
"""

from communication.discord import DiscordTool, register_discord_tool
from communication.email import EmailTool, register_email_tool
from communication.gmail import GmailTool, GoogleOAuth, register_gmail_tool
from communication.slack import SlackTool, register_slack_tool
from communication.telegram import TelegramTool, register_telegram_tool
from communication.whatsapp import WhatsAppTool, register_whatsapp_tool

__all__ = [
    "DiscordTool",
    "EmailTool",
    "GmailTool",
    "GoogleOAuth",
    "SlackTool",
    "TelegramTool",
    "WhatsAppTool",
    "register_discord_tool",
    "register_email_tool",
    "register_gmail_tool",
    "register_slack_tool",
    "register_telegram_tool",
    "register_whatsapp_tool",
]
