"""
communication.gmail
~~~~~~~~~~~~~~~~~~~~

Gmail integration using the Gmail REST API (gmail.googleapis.com) with full
OAuth 2.0 installed-app flow.

Why Gmail REST and not SMTP?
----------------------------

* Modern Gmail accounts block plain-password SMTP - App Passwords are awkward.
* OAuth tokens survive password resets and 2FA changes.
* We can read labels (INBOX, UNREAD, STARRED, SENT, etc.), threads, drafts.
* We get free profile / contact info from people.googleapis.com.

Configuration
-------------

Add the following to your ``.env``::

    GOOGLE_CLIENT_ID=xxxxx.apps.googleusercontent.com
    GOOGLE_CLIENT_SECRET=xxxxx
    GOOGLE_REDIRECT_URI=http://localhost:5050/api/auth/google/callback
    GMAIL_SCOPES=https://www.googleapis.com/auth/gmail.send
                    https://www.googleapis.com/auth/gmail.readonly
                    https://www.googleapis.com/auth/gmail.modify
                    https://www.googleapis.com/auth/userinfo.email

Storage
-------

OAuth tokens are persisted at ``storage/google_token.json``. Re-running the
web server picks up the existing token so the user only has to consent once.

If OAuth credentials are missing we *fall back* to the original SMTP/IMAP
:class:`EmailTool` so the assistant keeps working.
"""
from __future__ import annotations

import base64
import json
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.base import BaseTool, ToolResult
from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"
PEOPLE_BASE = "https://people.googleapis.com/v1"

DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
]


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------
@dataclass
class GoogleToken:
    access_token: str
    refresh_token: Optional[str]
    expires_at: float
    scope: str
    email: Optional[str] = None

    def is_valid(self) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GoogleToken":
        return cls(
            access_token=str(payload.get("access_token", "")),
            refresh_token=payload.get("refresh_token"),
            expires_at=float(payload.get("expires_at", 0.0)),
            scope=str(payload.get("scope", "")),
            email=payload.get("email"),
        )


class TokenStore:
    """Reads / writes the Google OAuth token JSON file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        cfg = get_config()
        self._path = path or cfg.storage_dir / "google_token.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[GoogleToken]:
        if not self._path.exists():
            return None
        try:
            return GoogleToken.from_dict(json.loads(self._path.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("google_token.json is unreadable: %s", exc)
            return None

    def save(self, token: GoogleToken) -> None:
        self._path.write_text(json.dumps(token.to_dict(), indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()


# ---------------------------------------------------------------------------
# OAuth client
# ---------------------------------------------------------------------------
class GoogleOAuth:
    """Tiny OAuth 2.0 client with local-loopback flow."""

    def __init__(self) -> None:
        cfg = get_config()
        self.client_id = cfg.google_client_id or ""
        self.client_secret = cfg.google_client_secret or ""
        self.redirect_uri = cfg.google_redirect_uri or "http://localhost:5050/api/auth/google/callback"
        self.scopes = cfg.google_scopes or DEFAULT_SCOPES
        self.store = TokenStore()

    # ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def is_authenticated(self) -> bool:
        token = self.store.load()
        return token is not None and (token.is_valid() or token.refresh_token is not None)

    def get_valid_token(self) -> Optional[GoogleToken]:
        token = self.store.load()
        if token is None:
            return None
        if token.is_valid():
            return token
        if token.refresh_token:
            refreshed = self._refresh(token)
            if refreshed:
                return refreshed
        return None

    # ------------------------------------------------------------------
    def build_auth_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
        }
        return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str) -> GoogleToken:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
        return self._token_from_response(resp.json())

    def _refresh(self, token: GoogleToken) -> Optional[GoogleToken]:
        if not token.refresh_token:
            return None
        try:
            resp = requests.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": token.refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=20,
            )
        except Exception as exc:
            log.warning("Token refresh failed: %s", exc)
            return None
        if resp.status_code != 200:
            log.warning("Token refresh HTTP %d: %s", resp.status_code, resp.text)
            return None
        data = resp.json()
        new_token = GoogleToken(
            access_token=data["access_token"],
            refresh_token=token.refresh_token,        # not rotated on refresh
            expires_at=time.time() + int(data.get("expires_in", 3600)),
            scope=data.get("scope", token.scope),
            email=token.email,
        )
        self.store.save(new_token)
        return new_token

    def _token_from_response(self, data: Dict[str, Any]) -> GoogleToken:
        token = GoogleToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=time.time() + int(data.get("expires_in", 3600)),
            scope=data.get("scope", ""),
        )
        # Optionally resolve the user's email via userinfo endpoint.
        try:
            userinfo = requests.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {token.access_token}"},
                timeout=10,
            ).json()
            token.email = userinfo.get("email")
        except Exception:
            pass
        self.store.save(token)
        return token


# ---------------------------------------------------------------------------
# Gmail client (uses an access token)
# ---------------------------------------------------------------------------
class GmailClient:
    def __init__(self, token: GoogleToken) -> None:
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token.access_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    def send(self, to: List[str], subject: str, body: str) -> Dict[str, Any]:
        msg = MIMEMultipart()
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.attach(MIMEText(body or "", "plain", "utf-8"))
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        resp = requests.post(
            f"{GMAIL_BASE}/users/me/messages/send",
            headers=self.headers,
            json={"raw": raw},
            timeout=30,
        )
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"Gmail send failed ({resp.status_code}): {resp.text}")
        return resp.json()

    # ------------------------------------------------------------------
    def list_messages(self, label: str = "INBOX", max_results: int = 10) -> List[Dict[str, Any]]:
        resp = requests.get(
            f"{GMAIL_BASE}/users/me/messages",
            headers=self.headers,
            params={"labelIds": label, "maxResults": max_results},
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gmail list failed ({resp.status_code}): {resp.text}")
        return resp.json().get("messages", [])

    def get_message(self, msg_id: str) -> Dict[str, Any]:
        resp = requests.get(
            f"{GMAIL_BASE}/users/me/messages/{msg_id}",
            headers=self.headers,
            params={"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]},
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gmail get failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        resp = requests.get(
            f"{GMAIL_BASE}/users/me/messages",
            headers=self.headers,
            params={"q": query, "maxResults": max_results},
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gmail search failed ({resp.status_code}): {resp.text}")
        return resp.json().get("messages", [])


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class GmailTool(BaseTool):
    name = "gmail_tool"
    description = (
        "Send, read or search email via the user's Gmail account using "
        "OAuth 2.0. Requires the user to sign in once via /api/auth/google."
    )

    def __init__(self) -> None:
        self.oauth = GoogleOAuth()

    # ------------------------------------------------------------------
    def can_handle(self, command: str, context: Optional[Dict[str, Any]] = None) -> bool:
        return self.oauth.is_authenticated() and EmailTool.can_handle(self, command, context)

    # ------------------------------------------------------------------
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        if not self.oauth.configured:
            return ToolResult(
                success=False,
                message=(
                    "Gmail OAuth is not configured. Set GOOGLE_CLIENT_ID and "
                    "GOOGLE_CLIENT_SECRET in your .env, then visit "
                    "/api/auth/google to sign in."
                ),
            )
        token = self.oauth.get_valid_token()
        if token is None:
            return ToolResult(
                success=False,
                message=(
                    "You are not signed in to Google yet. Open "
                    "http://localhost:5050/api/auth/google in your browser, "
                    "grant access, then try again."
                ),
            )

        client = GmailClient(token)
        text = (command or "").strip()
        lowered = text.lower()

        try:
            if "send" in lowered or "send email" in lowered or "send an email" in lowered:
                recipients, subject, body = _parse_send_command(text)
                if not recipients:
                    return ToolResult(success=False, message="I need at least one recipient, sir.")
                if not subject:
                    return ToolResult(success=False, message="I need a subject for the email, sir.")
                client.send(recipients, subject, body)
                return ToolResult(success=True, message=f"Email sent via Gmail to {', '.join(recipients)} with subject '{subject}'.")
            if "newest" in lowered or "latest" in lowered:
                msgs = client.list_messages("INBOX", max_results=1)
                if not msgs:
                    return ToolResult(success=True, message="Your inbox is empty, sir.")
                msg = client.get_message(msgs[0]["id"])
                return ToolResult(success=True, message=_format_gmail_msg(msg), data={"message": msg})
            if "unread" in lowered:
                msgs = client.list_messages("UNREAD", max_results=10)
                if not msgs:
                    return ToolResult(success=True, message="You have no unread emails, sir.")
                body = "\n\n".join(_format_gmail_msg(client.get_message(m["id"])) for m in msgs)
                return ToolResult(success=True, message=body)
            if "search" in lowered and "inbox" in lowered:
                keyword = text.lower().split("search inbox", 1)[-1].strip()
                msgs = client.search(keyword)
                if not msgs:
                    return ToolResult(success=True, message=f"No emails matched '{keyword}'.")
                body = "\n\n".join(_format_gmail_msg(client.get_message(m["id"])) for m in msgs[:10])
                return ToolResult(success=True, message=body)
            return ToolResult(success=False, message="I am not sure what to do with that Gmail request.")
        except Exception as exc:
            log.exception("Gmail tool failed: %s", exc)
            return ToolResult(success=False, message=f"Gmail action failed: {exc}")


# ---------------------------------------------------------------------------
# Tiny local OAuth callback helper (used by /api/auth/google)
# ---------------------------------------------------------------------------
def run_oauth_callback_server(port: int, on_code: callable, on_state: callable) -> HTTPServer:
    """Run a one-shot HTTP server on *port* that captures the auth code."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server convention
            url = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(url.query)
            code = params.get("code", [""])[0]
            state = params.get("state", [""])[0]
            err = params.get("error", [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if err:
                self.wfile.write(f"<h1>Sign-in failed: {err}</h1>".encode())
            else:
                self.wfile.write(b"<h1>Arvis is now signed in to Google. You can close this tab.</h1>")
            on_code(code)
            on_state(state)

        def log_message(self, format, *args):  # silence the default log
            return

    server = HTTPServer(("127.0.0.1", port), Handler)
    return server


# ---------------------------------------------------------------------------
# Helpers shared with EmailTool
# ---------------------------------------------------------------------------
def _parse_send_command(command: str) -> Tuple[List[str], str, str]:
    """Delegate to communication.email's parser for now."""
    from communication.email import _parse_send_command as _parser
    return _parser(command)


def _format_gmail_msg(msg: Dict[str, Any]) -> str:
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "(no subject)")
    sender = headers.get("from", "(unknown sender)")
    date = headers.get("date", "")
    snippet = msg.get("snippet", "")
    return f"From: {sender}\nSubject: {subject}\nDate: {date}\n\n{snippet}"


def register_gmail_tool(router) -> List[BaseTool]:
    tool = GmailTool()
    router.register(
        tool,
        keywords=("gmail", "google mail"),
        priority=65,
    )
    return [tool]


# Imported here for the can_handle fallback so GmailTool can be used even
# before the user signs in (it then just returns the "sign in" message).
from communication.email import EmailTool  # noqa: E402


__all__ = [
    "GmailTool",
    "GoogleOAuth",
    "GoogleToken",
    "GmailClient",
    "TokenStore",
    "register_gmail_tool",
    "run_oauth_callback_server",
]