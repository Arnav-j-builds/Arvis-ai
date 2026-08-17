"""
communication.email
~~~~~~~~~~~~~~~~~~~

IMAP + SMTP helpers used by :class:`EmailTool`.

The class never stores credentials - everything is read from
:mod:`core.config` (which itself reads ``.env`` via ``python-dotenv``). When
the credentials are missing the tool fails with a clear message instead of
crashing.

Supported actions
-----------------

* ``send <recipient(s)> <subject> <body>`` - send an email.
* ``read`` - read the latest unread message.
* ``read latest`` / ``read newest`` - read the most recent message.
* ``search <keyword>`` - search the inbox.
* ``unread`` / ``list unread`` - summarise every unread message.
"""

from __future__ import annotations

import imaplib
import smtplib
import ssl
from dataclasses import dataclass
from email import message_from_bytes
from email.header import decode_header, make_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import Iterable, List, Optional, Tuple

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class EmailMessage:
    """A decoded email message."""

    subject: str
    sender: str
    body: str
    date: str
    uid: str


def _decode_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _format_recipients(recipients: Iterable[str]) -> List[str]:
    cleaned = []
    for raw in recipients:
        name, addr = parseaddr(raw)
        cleaned.append(addr or name)
    return [r for r in cleaned if r]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class EmailTool(BaseTool):
    """Voice-driven email tool."""

    name = "email_tool"
    description = (
        "Send, read, or search email. Examples: 'send an email to alice@example.com "
        "about the meeting, body see you tomorrow', 'read my newest email', 'search "
        "inbox for invoice', 'read unread emails'."
    )

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[dict] = None) -> bool:
        lowered = (command or "").lower()
        return any(
            trigger in lowered
            for trigger in (
                "email ",
                "send an email",
                "send email",
                "inbox",
                "read my newest email",
                "read my latest email",
                "read newest email",
                "read latest email",
                "search inbox",
                "send a mail",
                "send mail",
            )
        )

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[dict] = None) -> ToolResult:
        cfg = get_config()
        if not (cfg.email_username and cfg.email_password and cfg.email_imap_host and cfg.email_smtp_host):
            return ToolResult(
                success=False,
                message=(
                    "Email credentials are missing. Please set EMAIL_USERNAME, "
                    "EMAIL_PASSWORD, EMAIL_IMAP_HOST, and EMAIL_SMTP_HOST in your .env file."
                ),
            )

        text = (command or "").strip()
        lowered = text.lower()

        try:
            if lowered.startswith("send") or "send email" in lowered or "send an email" in lowered or "send a mail" in lowered:
                return self._send(text)
            if "search" in lowered and "inbox" in lowered:
                return self._search(text)
            if "unread" in lowered:
                return self._read_unread()
            if "newest" in lowered or "latest" in lowered or "new email" in lowered:
                return self._read_latest()
            return ToolResult(success=False, message="I am not sure what to do with that email request.")
        except Exception as exc:
            log.exception("EmailTool failed: %s", exc)
            return ToolResult(success=False, message=f"Email action failed: {exc}")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------
    def _send(self, command: str) -> ToolResult:
        cfg = get_config()
        recipients, subject, body = _parse_send_command(command)
        if not recipients:
            return ToolResult(success=False, message="I need at least one recipient to send the email, sir.")
        if not subject:
            return ToolResult(success=False, message="I need a subject for the email, sir.")

        msg = MIMEMultipart()
        msg["From"] = cfg.email_username or ""
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body or "", "plain", "utf-8"))

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(cfg.email_smtp_host, cfg.email_smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(cfg.email_username, cfg.email_password)
                server.sendmail(cfg.email_username, recipients, msg.as_string())
        except smtplib.SMTPAuthenticationError:
            return ToolResult(
                success=False,
                message="Email authentication failed. For Gmail, use an App Password, not your account password.",
            )
        except Exception as exc:
            return ToolResult(success=False, message=f"I could not send the email: {exc}")

        log.info("Email sent to %s subject=%r", recipients, subject)
        return ToolResult(success=True, message=f"Email sent to {', '.join(recipients)} with subject '{subject}'.")

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------
    def _connect(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        cfg = get_config()
        if cfg.email_use_ssl:
            return imaplib.IMAP4_SSL(cfg.email_imap_host, cfg.email_imap_port)
        return imaplib.IMAP4(cfg.email_imap_host, cfg.email_imap_port)

    def _login(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        cfg = get_config()
        conn = self._connect()
        conn.login(cfg.email_username, cfg.email_password)
        conn.select("INBOX")
        return conn

    def _read_latest(self) -> ToolResult:
        with self._login() as conn:
            status, data = conn.search(None, "ALL")
            if status != "OK" or not data or not data[0]:
                return ToolResult(success=True, message="Your inbox is empty, sir.")
            ids = data[0].split()
            latest_id = ids[-1]
            message = self._fetch_message(conn, latest_id)
        if not message:
            return ToolResult(success=False, message="I could not read the latest email.")
        return ToolResult(success=True, message=self._format_message(message), data={"message": message})

    def _read_unread(self) -> ToolResult:
        with self._login() as conn:
            status, data = conn.search(None, "UNSEEN")
            if status != "OK" or not data or not data[0]:
                return ToolResult(success=True, message="You have no unread emails, sir.")
            ids = data[0].split()
            messages: List[EmailMessage] = []
            for uid in ids[-10:]:  # cap at 10 to keep responses short
                message = self._fetch_message(conn, uid)
                if message:
                    messages.append(message)
        if not messages:
            return ToolResult(success=True, message="You have no unread emails, sir.")
        formatted = "\n\n".join(self._format_message(msg) for msg in messages)
        return ToolResult(success=True, message=formatted, data={"messages": messages})

    def _search(self, command: str) -> ToolResult:
        keyword = _extract_keyword(command, after="search inbox")
        if not keyword:
            return ToolResult(success=False, message="Please tell me which keyword to search for.")
        with self._login() as conn:
            status, data = conn.search(None, "TEXT", f'"{keyword}"')
            if status != "OK" or not data or not data[0]:
                return ToolResult(success=True, message=f"No emails matched '{keyword}', sir.")
            ids = data[0].split()
            messages: List[EmailMessage] = []
            for uid in ids[-10:]:
                message = self._fetch_message(conn, uid)
                if message:
                    messages.append(message)
        if not messages:
            return ToolResult(success=True, message=f"No emails matched '{keyword}', sir.")
        formatted = "\n\n".join(self._format_message(msg) for msg in messages)
        return ToolResult(success=True, message=formatted, data={"messages": messages, "keyword": keyword})

    # ------------------------------------------------------------------
    # IMAP helpers
    # ------------------------------------------------------------------
    def _fetch_message(self, conn: imaplib.IMAP4, uid: bytes) -> Optional[EmailMessage]:
        status, payload = conn.fetch(uid, "(RFC822)")
        if status != "OK" or not payload:
            return None
        raw = None
        for part in payload:
            if isinstance(part, tuple) and part[1]:
                raw = part[1]
                break
        if raw is None:
            return None
        message = message_from_bytes(raw)
        subject = _decode_header(message.get("Subject", ""))
        sender = _decode_header(message.get("From", ""))
        date = _decode_header(message.get("Date", ""))
        body = _extract_body(message)
        return EmailMessage(subject=subject, sender=sender, body=body, date=date, uid=uid.decode("utf-8", "ignore"))

    @staticmethod
    def _format_message(msg: EmailMessage) -> str:
        snippet = msg.body if len(msg.body) <= 600 else msg.body[:600] + "..."
        return f"From: {msg.sender}\nSubject: {msg.subject}\nDate: {msg.date}\n\n{snippet}"


# ---------------------------------------------------------------------------
# Parsing helpers (shared with tests)
# ---------------------------------------------------------------------------
def _extract_keyword(command: str, after: str) -> str:
    lowered = command.lower()
    if after in lowered:
        return command[lowered.index(after) + len(after):].strip().strip("'\"")
    if "for " in lowered:
        return command[lowered.index("for ") + 4:].strip().strip("'\"")
    return ""


def _parse_send_command(command: str) -> Tuple[List[str], str, str]:
    """Best-effort parser for natural-language email commands.

    Recognises forms such as::

        "send email to alice@example.com about meeting body see you tomorrow"
        "email john at john@example.com subject update body hello"
    """
    text = command.strip()
    lowered = text.lower()

    # Remove leading verb
    for prefix in ("send an email ", "send email ", "send a mail ", "send mail ", "send ", "email "):
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            lowered = text.lower()
            break

    # Recipient: look for "to <name>" or first quoted email
    recipients: List[str] = []
    remainder = text
    if lowered.startswith("to "):
        # take everything up to the first boundary keyword
        boundary = _find_boundary(lowered[3:])
        recipient_blob = remainder[3:3 + boundary].strip()
        remainder = remainder[3 + boundary:].strip()
        recipients = _format_recipients(_split_recipients(recipient_blob))
    else:
        # No "to" prefix - treat the first whitespace token as the recipient.
        head, _, tail = remainder.partition(" ")
        if head and "@" not in head and not head.startswith("to "):
            # only split off if we do not already have recipients and the
            # head token looks like a name (no @). Caller can still supply
            # the recipient inline elsewhere; this branch only fires when
            # there is no other signal in the command.
            tokens = tail.split(" ", 1)
            # Require a boundary word in the tail to avoid eating part of
            # the subject.
            if tokens[0] and any(tok in ("about", "subject", "re:") for tok in [tokens[0].lower()]):
                remainder = tail
                recipients = _format_recipients(_split_recipients(head.strip()))

    subject = ""
    body = ""
    remainder_lowered = remainder.lower()
    for token in (" about ", " subject ", " re: "):
        if token in remainder_lowered:
            idx = remainder_lowered.find(token)
            subject_piece = remainder[:idx]
            remainder = remainder[idx + len(token):]
            remainder_lowered = remainder.lower()
            # If we never captured a recipient, treat the first word as a contact alias
            if not recipients and subject_piece:
                recipients = _format_recipients(_split_recipients(subject_piece.strip()))
            if " body " in remainder_lowered:
                body_idx = remainder_lowered.find(" body ")
                subject = remainder[:body_idx].strip()
                body = remainder[body_idx + len(" body "):].strip()
            else:
                subject = remainder.strip()
            break

    if not subject:
        # Strip leading boundary word if present ("about launch" -> "launch")
        remainder_lowered = remainder.lower()
        for token in (" about ", " subject ", " re: "):
            if remainder_lowered.startswith(token.lstrip()):
                remainder = remainder[len(token) - 1:].lstrip()
                remainder_lowered = remainder.lower()
                break
        if " body " in remainder_lowered:
            body_idx = remainder_lowered.find(" body ")
            subject = remainder[:body_idx].strip()
            body = remainder[body_idx + len(" body "):].strip()
        else:
            subject = remainder.strip()
            body = ""

    return recipients, subject, body


def _find_boundary(after_to: str) -> int:
    for token in (" about ", " subject ", " body "):
        idx = after_to.find(token)
        if idx != -1:
            return idx
    return len(after_to)


def _split_recipients(blob: str) -> List[str]:
    # Split on " and " / commas / whitespace
    cleaned = blob.replace(" and ", ",").replace("&", ",")
    return [chunk.strip() for chunk in cleaned.split(",") if chunk.strip()]


def _extract_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except LookupError:
                        return payload.decode("utf-8", errors="replace")
        # Fall back to HTML if no plain text part was found
        for part in message.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        return payload.decode(charset, errors="replace")
                    except LookupError:
                        return payload.decode("utf-8", errors="replace")
        return ""
    payload = message.get_payload(decode=True)
    if not payload:
        return ""
    charset = message.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def register_email_tool(router) -> List[BaseTool]:
    """Register :class:`EmailTool` with *router*."""
    tool = EmailTool()
    router.register(tool, keywords=("email", "inbox", "mail "), priority=70)
    return [tool]


__all__ = [
    "EmailMessage",
    "EmailTool",
    "register_email_tool",
    "_parse_send_command",
    "_extract_body",
]
