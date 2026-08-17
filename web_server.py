"""
web_server.py
~~~~~~~~~~~~~

Full-stack web backend for Arvis.

* Serves the static frontend under ``web/`` (HTML / CSS / JS / Three.js orb).
* Exposes a JSON REST API for chat, history, system telemetry, modes, routines.
* Streams live events (state, status, response, telemetry tick) over Socket.IO.

The runtime does not modify ``main.py`` or ``arvis.py`` - it reuses the
existing :class:`core.router.CommandRouter` and friends. Voice input is not
captured from the browser in this build; the browser records audio via the
Web Speech API and sends the resulting transcript to ``/api/chat``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from core.base import ToolResult
from core.config import get_config
from core.confirmation import ConfirmationManager
from core.conversation_manager import ConversationManager, ConversationSnapshot
from core.logger import configure_logging, get_logger
from core.planner import TaskPlanner
from core.router import CommandRouter, get_router
from core.task_context import TaskContext
from core.task_executor import TaskExecutor, TaskStateInfo
from core.verification import Verifier
from storage.custom import (
    CustomCommand, CustomMode, Reminder,
    get_command_store, get_mode_store, get_reminder_store,
)

configure_logging()
log = get_logger("arvis.web")

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"


# ---------------------------------------------------------------------------
# In-memory conversation log + state hub
# ---------------------------------------------------------------------------
class ChatHub:
    """Holds the conversation transcript and broadcasts events."""

    def __init__(self, max_history: int = 200) -> None:
        self.history: deque = deque(maxlen=max_history)
        self.listening: bool = False
        self.thinking: bool = False
        self.last_tool: Optional[str] = None
        self._lock = threading.Lock()

    # ---------------- transcript ---------------------------------------
    def add_message(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        msg = {
            "role": role,
            "content": content,
            "ts": time.time(),
            "meta": meta or {},
        }
        with self._lock:
            self.history.append(msg)
        return msg

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.history)

    # ---------------- runtime flags ------------------------------------
    def set_state(self, *, listening: Optional[bool] = None,
                  thinking: Optional[bool] = None,
                  last_tool: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if listening is not None:
                self.listening = listening
            if thinking is not None:
                self.thinking = thinking
            if last_tool is not None:
                self.last_tool = last_tool
        return {
            "listening": self.listening,
            "thinking": self.thinking,
            "last_tool": self.last_tool,
        }


chat = ChatHub()


# ---------------------------------------------------------------------------
# System telemetry - lightweight, no extra deps beyond psutil
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False


def _system_snapshot() -> Dict[str, Any]:
    if not _HAS_PSUTIL:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "battery_percent": None,
            "uptime_seconds": int(time.time() - _BOOT_TIME),
            "platform": os.name,
            "ollama_reachable": _ping_ollama(),
        }
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    battery: Optional[Dict[str, Any]] = None
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            battery = {
                "percent": float(batt.percent),
                "plugged": bool(batt.power_plugged),
            }
    except Exception:
        battery = None
    return {
        "cpu_percent": float(cpu),
        "memory_percent": float(mem.percent),
        "memory_used_gb": round(mem.used / (1024 ** 3), 2),
        "memory_total_gb": round(mem.total / (1024 ** 3), 2),
        "battery": battery,
        "uptime_seconds": int(time.time() - _BOOT_TIME),
        "platform": os.name,
        "ollama_reachable": _ping_ollama(),
    }


_BOOT_TIME = time.time()


def _ping_ollama() -> bool:
    """Best-effort check that the local Ollama server is reachable."""
    try:
        import urllib.request

        cfg = get_config()
        req = urllib.request.Request(f"{cfg.ollama_base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _google_status() -> Dict[str, Any]:
    """Lightweight Gmail OAuth status for the frontend."""
    try:
        from communication.gmail import GoogleOAuth
        oauth = GoogleOAuth()
        token = oauth.store.load()
        return {
            "configured": oauth.configured,
            "signed_in": oauth.is_authenticated(),
            "email": (token.email if token else None),
        }
    except Exception:
        return {"configured": False, "signed_in": False, "email": None}


# ---------------------------------------------------------------------------
# Flask + SocketIO
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)
app.config["SECRET_KEY"] = "arvis-web-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ---------------------------------------------------------------------------
# Build router lazily (mirrors app.py) so missing optional deps don't crash
# ---------------------------------------------------------------------------
_router_lock = threading.Lock()
_router: Optional[CommandRouter] = None


def _build_router() -> CommandRouter:
    """Build a router with the same registrations as ``app.build_router``."""
    router = get_router()
    try:
        # Local import so the desktop UI keeps working unchanged.
        from app import build_router  # type: ignore
        router = build_router()
    except Exception as exc:
        log.warning("app.build_router unavailable, falling back to bare router: %s", exc)
    return router


def get_built_router() -> CommandRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = _build_router()
    return _router


def _get_hand_mouse_tool():
    """Return the registered HandMouseTool from the router, or None."""
    try:
        router = get_built_router()
    except Exception:
        return None
    for t in router.tools():
        if getattr(t, "name", "") == "hand_mouse_tool":
            return t
    return None


# ---------------------------------------------------------------------------
# LangChain executor - mirrors app._build_agent
# Built lazily so missing dependencies don't crash the web server.
# ---------------------------------------------------------------------------
_agent_lock = threading.Lock()
_agent: Optional[Any] = None


def _build_agent() -> Any:
    """Build a LangChain AgentExecutor for free-form chat.

    Returns ``None`` if LangChain / Ollama / dependencies are missing - the
    web server still works in that case; the default fallback just echoes.
    """
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama
    except Exception as exc:
        log.warning("LangChain deps unavailable: %s", exc)
        return None
    try:
        cfg = get_config()
        llm = ChatOllama(
            model=cfg.llm_model,
            base_url=cfg.ollama_base_url,
            reasoning=False,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are arvis, an intelligent, conversational AI assistant. "
                    "You can respond in natural, human-like language and use tools "
                    "when needed. Keep responses conversational and concise.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        router = get_built_router()
        tools = router.langchain_tools()
        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)
    except Exception as exc:
        log.warning("Agent build failed: %s", exc)
        return None


def get_agent() -> Any:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = _build_agent()
    return _agent


# ---------------------------------------------------------------------------
# Task + conversation singletons
# ---------------------------------------------------------------------------
_task_layer_lock = threading.Lock()
_task_ctx: Optional[TaskContext] = None
_task_executor: Optional[TaskExecutor] = None
_task_planner: Optional[TaskPlanner] = None
_conversation: Optional[ConversationManager] = None


def _broadcast_task_state(info: TaskStateInfo) -> None:
    """SocketIO hook for live task progress."""
    try:
        socketio.emit("task:state", info.to_dict())
    except Exception:  # pragma: no cover - defensive
        log.debug("task:state emit failed", exc_info=True)


def _broadcast_conversation_state(snapshot: ConversationSnapshot) -> None:
    """SocketIO hook for live conversation transitions."""
    try:
        socketio.emit("conversation:state", snapshot.to_dict())
    except Exception:  # pragma: no cover - defensive
        log.debug("conversation:state emit failed", exc_info=True)


def get_task_layer():
    """Lazy-init and return (ctx, executor, planner, conversation)."""
    global _task_ctx, _task_executor, _task_planner, _conversation
    with _task_layer_lock:
        if _task_ctx is None:
            _task_ctx = TaskContext()
        if _task_executor is None:
            _task_executor = TaskExecutor(
                router=get_built_router(),
                ctx=_task_ctx,
                confirmation=ConfirmationManager(),
                verifier=Verifier(),
                planner=TaskPlanner(router=get_built_router()),
                state_callback=_broadcast_task_state,
            )
        if _task_planner is None:
            _task_planner = TaskPlanner(router=get_built_router())
        if _conversation is None:
            _conversation = ConversationManager(
                ctx=_task_ctx,
                state_callback=_broadcast_conversation_state,
            )
    return _task_ctx, _task_executor, _task_planner, _conversation


def _run_agent(executor: Any, command: str) -> ToolResult:
    try:
        response = executor.invoke({"input": command})
        content = response.get("output", "") if isinstance(response, dict) else str(response)
    except Exception as exc:
        log.exception("Agent invocation failed: %s", exc)
        return ToolResult(success=False, message=f"LLM error: {exc}")
    if not content:
        return ToolResult(success=False, message="My language model returned an empty response.")
    return ToolResult(success=True, message=content)


# ---------------------------------------------------------------------------
# Routes - static + API
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> Any:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return ("Web frontend not built yet. Expected: " + str(index_path), 500)
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename: str) -> Any:
    return send_from_directory(str(WEB_DIR), filename)


@app.route("/api/health")
def health() -> Any:
    return jsonify({"status": "ok", "uptime": int(time.time() - _BOOT_TIME)})


@app.route("/api/state")
def state() -> Any:
    hand = _get_hand_mouse_tool()
    hand_status = hand.status() if hand is not None else {"registered": False}
    ctx, executor, planner, conversation = get_task_layer()
    return jsonify({
        "chat": chat.snapshot(),
        "runtime": chat.set_state(),
        "system": _system_snapshot(),
        "trigger": get_config().trigger_word,
        "tools": [t.name for t in get_built_router().tools()],
        "custom_commands": [c.to_dict() for c in get_command_store().list()],
        "custom_modes": [m.to_dict() for m in get_mode_store().list()],
        "reminders": [r.to_dict() for r in get_reminder_store().list_active()],
        "google": _google_status(),
        "hand_mouse": hand_status,
        "task": executor.state_snapshot().to_dict(),
        "conversation": conversation.snapshot().to_dict(),
    })


@app.route("/api/task/cancel", methods=["POST"])
def api_task_cancel() -> Any:
    """Cancel the currently running task, if any."""
    _, executor, _, _ = get_task_layer()
    if not executor.is_running():
        return jsonify({"ok": False, "message": "no task running"})
    executor.cancel()
    return jsonify({"ok": True})


@app.route("/api/task/state")
def api_task_state() -> Any:
    """Snapshot of the task layer only (lighter than /api/state)."""
    _, executor, _, _ = get_task_layer()
    return jsonify(executor.state_snapshot().to_dict())


@app.route("/api/system")
def system() -> Any:
    return jsonify(_system_snapshot())


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    payload = request.get_json(silent=True) or {}
    text = (payload.get("message") or "").strip()
    if not text:
        return jsonify({"success": False, "message": "Empty message"}), 400

    chat.add_message("user", text)
    socketio.emit("chat", chat.snapshot()[-1])
    socketio.emit("state", chat.set_state(thinking=True))

    router = get_built_router()
    ctx, executor, planner, conversation = get_task_layer()

    # Begin a session if this is the first message in a while - the
    # dashboard always wants to feel conversational even when the
    # user is typing chat bubbles.
    if not conversation.is_active():
        conversation.begin_session()

    kind = conversation.classify(text)
    if kind.value == "cancel":
        executor.cancel()
        executor.wait(timeout=2.0)
        conversation.end_session(reason="user cancelled")
        chat.add_message("assistant", "Cancelled.")
        socketio.emit("chat", chat.snapshot()[-1])
        socketio.emit("state", chat.set_state(thinking=False))
        return jsonify({"success": True, "message": "Cancelled.", "history": chat.snapshot()})
    if kind.value == "end_session":
        executor.cancel()
        executor.wait(timeout=2.0)
        conversation.end_session(reason="user ended")
        chat.add_message("assistant", "Goodbye, sir.")
        socketio.emit("chat", chat.snapshot()[-1])
        socketio.emit("state", chat.set_state(thinking=False))
        return jsonify({"success": True, "message": "Goodbye, sir.", "history": chat.snapshot()})

    resolved = ctx.resolve(text) if kind.value in {"followup", "question", "new_command"} else text
    conversation.record("user", resolved)

    def _default(cmd: str) -> ToolResult:
        """Default fallback - try LangChain agent, otherwise friendly echo."""
        agent = get_agent()
        if agent is not None:
            return _run_agent(agent, cmd)
        log.debug("No agent available; returning friendly echo for %r.", cmd)
        return ToolResult(
            success=True,
            message=(
                f"I heard: \"{cmd}\". The LangChain agent / Ollama LLM is "
                f"not reachable from the web backend yet. Start Ollama "
                f"(http://localhost:11434) to enable real answers here."
            ),
        )

    # Planner-first path. The planner turns multi-step missions into
    # a Task the executor runs in the background; the executor
    # broadcasts its own progress over SocketIO so the dashboard can
    # render a live mission panel.
    try:
        task = planner.plan(resolved, ctx=ctx)
    except Exception as exc:  # pragma: no cover - planner defensive
        log.warning("Planner failed in /api/chat: %s", exc)
        task = None

    if task is not None and task.steps:
        # Clarification request from the planner -> speak the question.
        if task.steps[0].tool_hint == "__clarify__":
            message = task.steps[0].description
            chat.add_message("assistant", message, {"clarification": True})
            socketio.emit("chat", chat.snapshot()[-1])
            socketio.emit("state", chat.set_state(thinking=False))
            return jsonify({"success": True, "message": message, "history": chat.snapshot()})

        # Otherwise run the task. We do NOT block the request; the
        # executor finishes in the background and pushes updates.
        ctx.last_goal = task.goal
        executor.run_async(task)
        # Tiny wait so the executor has a chance to mark itself as
        # "executing" before we return - the response then reflects
        # the live state for any client that polls immediately.
        executor.wait(timeout=0.2)
        snapshot = executor.state_snapshot()
        if snapshot.state == "completed":
            message = next(
                (s.message for s in snapshot.steps if s.status == "done" and s.message),
                "Done.",
            )
        elif snapshot.state == "failed":
            message = f"I could not complete the task: {snapshot.error or 'unknown error'}"
        elif snapshot.state == "cancelled":
            message = "Cancelled."
        else:
            message = "Working on it. Watch the mission panel for progress."
        chat.add_message("assistant", message, {"task_id": snapshot.current_step, "state": snapshot.state})
        socketio.emit("chat", chat.snapshot()[-1])
        socketio.emit("state", chat.set_state(thinking=False, last_tool=message[:60]))
        return jsonify({
            "success": True,
            "message": message,
            "history": chat.snapshot(),
            "task": snapshot.to_dict(),
        })

    # No plan produced -> legacy single-shot path. This handles free-
    # form questions ("what time is it?") that don't deserve a plan.
    try:
        result = router.dispatch(
            text,
            context={"router": router},
            default=_default,
        )
    except Exception as exc:
        log.exception("Router dispatch failed: %s", exc)
        result = ToolResult(success=False, message=f"Router error: {exc}")

    response_meta: Dict[str, Any] = {}
    if result.data:
        response_meta["data"] = result.data

    chat.add_message("assistant", result.message, response_meta)
    socketio.emit("chat", chat.snapshot()[-1])
    socketio.emit("state", chat.set_state(
        thinking=False,
        last_tool=result.message[:60] if result.success else None,
    ))

    return jsonify({
        "success": result.success,
        "message": result.message,
        "history": chat.snapshot(),
    })


@app.route("/api/listening", methods=["POST"])
def api_listening() -> Any:
    """Browser tells the server the mic started/stopped."""
    payload = request.get_json(silent=True) or {}
    listening = bool(payload.get("listening", False))
    state = chat.set_state(listening=listening)
    socketio.emit("state", state)
    return jsonify(state)


@app.route("/api/clear", methods=["POST"])
def api_clear() -> Any:
    chat.history.clear()
    socketio.emit("chat", None)
    socketio.emit("state", chat.set_state(thinking=False, last_tool=None))
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Custom commands - CRUD for user-defined voice commands
# ---------------------------------------------------------------------------
@app.route("/api/custom/commands", methods=["GET", "POST"])
def api_custom_commands() -> Any:
    store = get_command_store()
    if request.method == "GET":
        return jsonify({"commands": [c.to_dict() for c in store.list()]})
    payload = request.get_json(silent=True) or {}
    try:
        cmd = CustomCommand.from_dict(
            payload.get("name", "").strip() or "command",
            payload,
        )
        store.upsert_command(cmd)
        socketio.emit("custom:updated", {"kind": "command", "command": cmd.to_dict()})
        return jsonify({"ok": True, "command": cmd.to_dict()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/custom/commands/<name>", methods=["PUT", "DELETE"])
def api_custom_command_item(name: str) -> Any:
    store = get_command_store()
    if request.method == "DELETE":
        ok = store.delete(name)
        if ok:
            socketio.emit("custom:updated", {"kind": "command", "deleted": name})
        return jsonify({"ok": ok})
    payload = request.get_json(silent=True) or {}
    payload["name"] = name
    cmd = CustomCommand.from_dict(name, payload)
    store.upsert_command(cmd)
    socketio.emit("custom:updated", {"kind": "command", "command": cmd.to_dict()})
    return jsonify({"ok": True, "command": cmd.to_dict()})


# ---------------------------------------------------------------------------
# Custom modes - CRUD for user-defined behavioural modes
# ---------------------------------------------------------------------------
@app.route("/api/custom/modes", methods=["GET", "POST"])
def api_custom_modes() -> Any:
    store = get_mode_store()
    if request.method == "GET":
        return jsonify({"modes": [m.to_dict() for m in store.list()]})
    payload = request.get_json(silent=True) or {}
    try:
        mode = CustomMode.from_dict(
            payload.get("name", "").strip() or "mode",
            payload,
        )
        store.upsert_mode(mode)
        socketio.emit("custom:updated", {"kind": "mode", "mode": mode.to_dict()})
        return jsonify({"ok": True, "mode": mode.to_dict()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/custom/modes/<name>", methods=["PUT", "DELETE"])
def api_custom_mode_item(name: str) -> Any:
    store = get_mode_store()
    if request.method == "DELETE":
        ok = store.delete(name)
        if ok:
            socketio.emit("custom:updated", {"kind": "mode", "deleted": name})
        return jsonify({"ok": ok})
    payload = request.get_json(silent=True) or {}
    payload["name"] = name
    mode = CustomMode.from_dict(name, payload)
    store.upsert_mode(mode)
    socketio.emit("custom:updated", {"kind": "mode", "mode": mode.to_dict()})
    return jsonify({"ok": True, "mode": mode.to_dict()})


# ---------------------------------------------------------------------------
# Reminders - CRUD + auto-fire ticker
# ---------------------------------------------------------------------------
@app.route("/api/reminders", methods=["GET", "POST"])
def api_reminders() -> Any:
    store = get_reminder_store()
    if request.method == "GET":
        return jsonify({"reminders": [r.to_dict() for r in store.list_active()]})
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    fire_at = payload.get("fire_at")
    try:
        if not text or fire_at is None:
            raise ValueError("text and fire_at are required (fire_at is epoch seconds)")
        rem = store.add(text=text, fire_at=float(fire_at), recurring=payload.get("recurring"))
        socketio.emit("custom:updated", {"kind": "reminder", "reminder": rem.to_dict()})
        return jsonify({"ok": True, "reminder": rem.to_dict()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/reminders/<rid>", methods=["DELETE"])
def api_reminder_cancel(rid: str) -> Any:
    store = get_reminder_store()
    ok = store.cancel(rid)
    if ok:
        socketio.emit("custom:updated", {"kind": "reminder", "deleted": rid})
    return jsonify({"ok": ok})


# ---------------------------------------------------------------------------
# Terminal runner - direct exec (no LLM in the loop)
# ---------------------------------------------------------------------------
@app.route("/api/terminal", methods=["POST"])
def api_terminal() -> Any:
    from tools.terminal import TerminalTool
    payload = request.get_json(silent=True) or {}
    cmd = (payload.get("command") or "").strip()
    unsafe = bool(payload.get("unsafe", False))
    if not cmd:
        return jsonify({"ok": False, "error": "command is required"}), 400
    tool = TerminalTool()
    result = tool.execute(cmd, context={"unsafe": unsafe, "router": get_built_router()})
    return jsonify({"ok": result.success, "message": result.message, "data": result.data})


# ---------------------------------------------------------------------------
# PC control - direct invocation (no LLM in the loop)
# ---------------------------------------------------------------------------
@app.route("/api/pc", methods=["POST"])
def api_pc_control() -> Any:
    from tools.pc_control import PcControlTool
    payload = request.get_json(silent=True) or {}
    cmd = (payload.get("command") or "").strip()
    if not cmd:
        return jsonify({"ok": False, "error": "command is required"}), 400
    confirm = bool(payload.get("confirm", False))
    tool = PcControlTool()
    result = tool.execute(cmd, context={"confirm": confirm})
    return jsonify({"ok": result.success, "message": result.message})


# ---------------------------------------------------------------------------
# Hand-mouse control - start / stop / status
# ---------------------------------------------------------------------------
@app.route("/api/hand_mouse/start", methods=["POST"])
def api_hand_mouse_start() -> Any:
    tool = _get_hand_mouse_tool()
    if tool is None:
        return jsonify({
            "ok": False,
            "error": "Hand-mouse tool is not registered. Restart the server.",
        }), 503
    result = tool.start()
    socketio.emit("hand_mouse:status", tool.status())
    return jsonify({"ok": result.success, "message": result.message, "status": tool.status()})


@app.route("/api/hand_mouse/stop", methods=["POST"])
def api_hand_mouse_stop() -> Any:
    tool = _get_hand_mouse_tool()
    if tool is None:
        return jsonify({"ok": False, "error": "Hand-mouse tool is not registered."}), 503
    result = tool.stop()
    socketio.emit("hand_mouse:status", tool.status())
    return jsonify({"ok": result.success, "message": result.message, "status": tool.status()})


@app.route("/api/hand_mouse/status")
def api_hand_mouse_status() -> Any:
    tool = _get_hand_mouse_tool()
    if tool is None:
        return jsonify({
            "ok": False,
            "registered": False,
            "error": "Hand-mouse tool is not registered.",
        }), 503
    return jsonify({"ok": True, "registered": True, "status": tool.status()})


# ---------------------------------------------------------------------------
# Gmail OAuth - /api/auth/google opens browser, /callback catches the code
# ---------------------------------------------------------------------------
@app.route("/api/auth/google")
def api_auth_google_start() -> Any:
    try:
        from communication.gmail import GoogleOAuth
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    oauth = GoogleOAuth()
    if not oauth.configured:
        return jsonify({
            "ok": False,
            "error": (
                "Google OAuth is not configured. Add GOOGLE_CLIENT_ID and "
                "GOOGLE_CLIENT_SECRET to your .env, then restart the server."
            ),
        }), 400
    state = os.urandom(16).hex()
    url = oauth.build_auth_url(state)
    # Open in the user's browser automatically.
    try:
        webbrowser.open(url)
    except Exception:
        pass
    return jsonify({"ok": True, "auth_url": url, "state": state})


@app.route("/api/auth/google/status")
def api_auth_google_status() -> Any:
    try:
        from communication.gmail import GoogleOAuth
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    oauth = GoogleOAuth()
    token = oauth.store.load()
    email = token.email if token else None
    return jsonify({
        "ok": True,
        "configured": oauth.configured,
        "signed_in": oauth.is_authenticated(),
        "email": email,
    })


@app.route("/api/auth/google/callback")
def api_auth_google_callback() -> Any:
    """Receive the OAuth code. Exchanges it for tokens and shows a confirmation page."""
    code = request.args.get("code", "")
    if not code:
        return ("Missing code", 400)
    try:
        from communication.gmail import GoogleOAuth
        oauth = GoogleOAuth()
        token = oauth.exchange_code(code)
        return (
            "<h2 style='font-family:sans-serif;color:#1a7f3c'>"
            "Arvis is now signed in to Google as <b>%s</b>."
            "</h2><p>You can close this tab and return to Arvis.</p>"
            % (token.email or "your account")
        )
    except Exception as exc:
        log.exception("OAuth callback failed: %s", exc)
        return (f"<h2>Sign-in failed: {exc}</h2>", 500)


@app.route("/api/auth/google/logout", methods=["POST"])
def api_auth_google_logout() -> Any:
    try:
        from communication.gmail import GoogleOAuth
        GoogleOAuth().store.clear()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Telemetry broadcaster (background thread)
# ---------------------------------------------------------------------------
def _telemetry_loop() -> None:
    while True:
        try:
            snap = _system_snapshot()
            socketio.emit("system", snap)
            # Cheap status broadcast - only fires when something changed.
            tool = _get_hand_mouse_tool()
            if tool is not None:
                try:
                    status = tool.status()
                    # Only emit while the controller is actually live so
                    # the wire stays quiet when the feature is off.
                    if status.get("running"):
                        socketio.emit("hand_mouse:status", status)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover
            log.debug("Telemetry tick failed: %s", exc)
        time.sleep(2.0)


def _reminder_loop() -> None:
    """Poll the reminder store every second; fire anything past due."""
    while True:
        try:
            store = get_reminder_store()
            for rem in store.due():
                log.info("Reminder fired: %s", rem.text)
                socketio.emit("reminder:fired", rem.to_dict())
                # Inject an assistant message into the chat hub so the
                # browser speaks it back to the user.
                chat.add_message(
                    "assistant",
                    f"⏰ Reminder: {rem.text}",
                    {"reminder": rem.to_dict()},
                )
                socketio.emit("chat", chat.snapshot()[-1])
                store.mark_fired(rem.id)
        except Exception as exc:  # pragma: no cover
            log.debug("Reminder tick failed: %s", exc)
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# SocketIO events
# ---------------------------------------------------------------------------
@socketio.on("connect")
def _on_connect() -> None:
    log.info("Client connected: %s", request.sid)
    socketio.emit("state", chat.set_state(), to=request.sid)
    socketio.emit("system", _system_snapshot(), to=request.sid)
    tool = _get_hand_mouse_tool()
    if tool is not None:
        socketio.emit("hand_mouse:status", tool.status(), to=request.sid)
    # Replay history so a refresh doesn't lose the conversation.
    for msg in chat.snapshot():
        socketio.emit("chat", msg, to=request.sid)
    # Replay task + conversation state so the dashboard's mission
    # panel re-hydrates after a refresh.
    try:
        _, executor, _, conversation = get_task_layer()
        socketio.emit("task:state", executor.state_snapshot().to_dict(), to=request.sid)
        socketio.emit("conversation:state", conversation.snapshot().to_dict(), to=request.sid)
    except Exception:  # pragma: no cover - defensive
        log.debug("Initial task/conversation snapshot failed", exc_info=True)


@socketio.on("disconnect")
def _on_disconnect() -> None:
    log.info("Client disconnected: %s", request.sid)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    host = os.getenv("ARVIS_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("ARVIS_WEB_PORT", "5000"))
    log.info("Arvis web UI starting on http://%s:%d", host, port)
    threading.Thread(target=_telemetry_loop, daemon=True, name="telemetry").start()
    threading.Thread(target=_reminder_loop, daemon=True, name="reminders").start()
    try:
        socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    finally:
        # Release the camera + mouse controller cleanly so the OS does
        # not hold the webcam open after the server stops.
        tool = _get_hand_mouse_tool()
        if tool is not None:
            try:
                tool.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()