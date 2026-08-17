"""
app.py
~~~~~~

Integration layer that wires every new feature module into the existing
``main.py``-style assistant **without modifying** the legacy file.

The assistant's voice loop lives in :func:`run`. It mirrors the original
behaviour (wake word -> conversation mode -> LangChain agent -> TTS) but
*adds*:

* Vision module registration.
* Communication tools registration.
* Routine matching (string-match fallback that runs routines before the
  LangChain agent - so deterministic phrases always work).
* A shared :class:`CommandRouter` that the interactive routine builder
  uses for executing routine steps.

Drop-in usage
-------------

::

    $ python app.py

The script does **not** edit ``main.py``. It simply provides a newer entry
point that reuses the legacy tool definitions.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from communication import (
    register_discord_tool,
    register_email_tool,
    register_gmail_tool,
    register_slack_tool,
    register_telegram_tool,
    register_whatsapp_tool,
)
from core.autostart import launched_from_startup
from core.autostart_tool import register_autostart_tool
from core.config import get_config
from core.confirmation import ConfirmationManager
from core.conversation_manager import ConversationManager, UtteranceKind
from core.logger import configure_logging, get_logger
from core.planner import TaskPlanner
from core.router import CommandRouter, get_router
from core.task_context import TaskContext
from core.task_executor import TaskExecutor
from core.task_plan import Task
from core.verification import Verifier
from routines import RoutineManager, register_routines_tool
from smart_room import register_smart_room_tools
from tools.custom_commands import register_custom_command_tool
from tools.pc_control import register_pc_control_tool
from tools.reminders import register_reminder_tool
from tools.terminal import register_terminal_tool
from vision import register_vision_tools, register_hand_mouse_tools, register_eye_mouse_tools
from tools.typing import register_typing_tool
from core.intent_tools import register_intent_tool
from core.skill_manager import SkillManager
from core.context_engine import get_screen_cache, get_browser_cache, reset_caches

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging()
log = get_logger("arvis.app")

load_dotenv(override=False)

# ---------------------------------------------------------------------------
# Build the router
# ---------------------------------------------------------------------------
def build_router() -> CommandRouter:
    """Build a router with every new tool registered."""
    router = get_router()

    # Vision
    register_vision_tools(router)

    # Hand-mouse control (webcam gesture -> mouse events).
    # Always register the tool so the voice loop can start / stop it,
    # even if MediaPipe / pyautogui are missing - the tool itself
    # degrades to a friendly error message at runtime.
    register_hand_mouse_tools(router)

    # Eye-mouse control (webcam iris gaze -> mouse events).
    # Uses MediaPipe's FaceLandmarker with the iris refinement model.
    # Requires mediapipe>=0.10.14 (the modern Tasks API); on older
    # versions the tool reports a friendly install error at runtime.
    register_eye_mouse_tools(router)

    # Typing tool: paste arbitrary text into the focused window,
    # send special keys, paste from the clipboard.
    register_typing_tool(router)

    # Communication
    register_email_tool(router)
    register_gmail_tool(router)        # uses Gmail REST API when configured
    register_whatsapp_tool(router)
    register_telegram_tool(router)
    register_discord_tool(router)
    register_slack_tool(router)

    # Routines
    manager = RoutineManager()
    register_routines_tool(router, manager)

    # Custom commands, modes, reminders
    register_custom_command_tool(router)
    register_reminder_tool(router)

    # PC control (volume, brightness, lock, shutdown, media keys)
    register_pc_control_tool(router)

    # Safe terminal / shell runner
    register_terminal_tool(router)

    # Smart Room / IoT (ESP32 room controller on the local LAN)
    register_smart_room_tools(router)

    # Windows autostart toggle ("start at boot", "stop at boot", ...).
    register_autostart_tool(router)

    # Universal screen / visual actions / skills / browser agent.  All
    # four share the same :class:`SkillManager` and the same
    # :class:`ScreenContextCache` / :class:`BrowserContextCache` so
    # they cooperate across turns.
    skill_manager = SkillManager(router=router)
    register_intent_tool(router, skill_manager=skill_manager)

    # Legacy tools (kept exactly as-is) ----------------------------------
    try:
        from tools.time import get_time
        from tools.OCR import read_text_from_latest_image
        from tools.arp_scan import arp_scan_terminal
        from tools.duckduckgo import duckduckgo_search_tool
        from tools.matrix import matrix_mode
        from tools.screenshot import take_screenshot
        from tools.opener import open_anything

        # The legacy tools are decorated with ``@tool``; we wrap them in a
        # thin :class:`BaseTool` so they can be registered with the router.
        # Each adapter receives targeted keywords + a heuristic fallback so
        # the router no longer dumps every unknown command into get_time.
        from core.legacy_bridge import (
            langchain_tool_to_base,
            _is_open_intent,
            _is_time_query,
            _is_arp_query,
            _is_search_query,
            _is_matrix,
            _is_screenshot,
        )

        router.register(
            langchain_tool_to_base(
                get_time,
                keywords=("what time", "current time", "tell me the time", "time in ", "time at "),
                fallback_handler=_is_time_query,
                description="Tell the current time. Use for 'what time is it', 'time in Tokyo', etc.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                open_anything,
                keywords=(
                    "open ",
                    "launch ",
                    "start ",
                    "go to ",
                    "google ",
                    "youtube ",
                    "play ",
                    "search google ",
                    "search youtube ",
                    "open app ",
                    "launch app ",
                    "start app ",
                ),
                fallback_handler=_is_open_intent,
                description="Open a website, launch a desktop app, or run a Google/YouTube search.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                duckduckgo_search_tool,
                keywords=("search for", "look up", "search the web", "duckduckgo"),
                fallback_handler=_is_search_query,
                description="Search the web via DuckDuckGo.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                arp_scan_terminal,
                keywords=("arp scan", "scan the network", "who is on my network", "scan network"),
                fallback_handler=_is_arp_query,
                description="Scan the local network with ARP.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                matrix_mode,
                keywords=("matrix mode",),
                fallback_handler=_is_matrix,
                description="Toggle the Matrix-style screen effect.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                take_screenshot,
                keywords=("take a screenshot", "screenshot this", "capture the screen"),
                fallback_handler=_is_screenshot,
                description="Take a screenshot of the current screen.",
            ),
            priority=100,
        )
        router.register(
            langchain_tool_to_base(
                read_text_from_latest_image,
                keywords=("read text", "read the text", "ocr ", "read the image"),
                description="OCR the most recently captured image.",
            ),
            priority=100,
        )
    except Exception as exc:  # pragma: no cover - depends on optional deps
        log.warning("Could not register legacy tools: %s", exc)

    return router


# ---------------------------------------------------------------------------
# Voice I/O
# ---------------------------------------------------------------------------
recognizer = sr.Recognizer()


def _build_engine() -> pyttsx3.Engine:  # type: ignore[type-arg]
    """Build a pyttsx3 engine. Reads voice hint / rate / volume from env.

    Default voice: a female, charming English voice. The hint defaults
    to ``"zira"`` (Microsoft Zira Desktop - the built-in Windows
    female voice). If Zira is not installed we walk a short list of
    common female names and finally fall back to scanning every
    available voice for one whose name contains "female" / "woman"
    or a known-female identifier.

    Failures in the env-var parsing must never break TTS - we always return
    a working engine with sane defaults.
    """
    try:
        engine = pyttsx3.init()
    except Exception as exc:  # pragma: no cover - driver missing
        log.error("pyttsx3.init() failed: %s", exc)
        raise

    try:
        voice_hint = os.getenv("JARVIS_VOICE_HINT", "zira").lower().strip()
    except Exception:  # pragma: no cover - defensive
        voice_hint = "zira"

    if voice_hint:
        try:
            voices = list(engine.getProperty("voices") or [])
            matched = False
            for voice in voices:
                if voice_hint in (voice.name or "").lower():
                    try:
                        engine.setProperty("voice", voice.id)
                        matched = True
                        break
                    except Exception:  # pragma: no cover - defensive
                        continue
            if not matched:
                _apply_female_voice(engine, voice_hint)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not set voice hint %r: %s", voice_hint, exc)

    try:
        engine.setProperty("rate", int(os.getenv("JARVIS_SPEECH_RATE", "170")))
    except (TypeError, ValueError):
        engine.setProperty("rate", 170)
    try:
        engine.setProperty("volume", float(os.getenv("JARVIS_SPEECH_VOLUME", "1.0")))
    except (TypeError, ValueError):
        engine.setProperty("volume", 1.0)
    return engine


# Preferred female voice names to try, in order.  Microsoft Zira is
# the standard English female voice that ships with Windows; the
# other names cover common OneCore / third-party installs (en-US and
# en-GB).
_FEMALE_VOICE_CANDIDATES = (
    "zira",
    "aria",
    "jenny",
    "samantha",
    "victoria",
    "fiona",
    "serena",
    "ava",
    "allison",
    "susan",
    "kate",
    "hazel",
    "libby",
    "maisie",
    "michelle",
    "tracy",
    "moira",
    "tessa",
    "veena",
    "fema",  # some Microsoft packs use "Microsoft Server Speech ... Female"
)

# Tokens inside the voice name that strongly suggest a female voice
# even when the explicit candidate list does not match.
_FEMALE_NAME_TOKENS = (
    "female", "woman", "girl", "lady",
    "zira", "aria", "jenny", "samantha", "victoria", "fiona", "serena",
    "ava", "allison", "susan", "kate", "hazel", "libby", "michelle",
    "moira", "tessa", "veena",
)


def _apply_female_voice(engine: pyttsx3.Engine, original_hint: str) -> None:
    """Best-effort fallback that picks a female voice when *original_hint*
    is not installed.

    Strategy:

    1. Try every entry in :data:`_FEMALE_VOICE_CANDIDATES` against
       each available voice's name.
    2. If nothing matches, scan for any voice whose name contains a
       known female token.
    3. Last resort: do nothing - the engine will use the system default.
    """
    try:
        voices = list(engine.getProperty("voices") or [])
    except Exception:  # pragma: no cover - defensive
        return
    if not voices:
        return

    # 1. Explicit candidate names.
    for candidate in _FEMALE_VOICE_CANDIDATES:
        for voice in voices:
            name = (voice.name or "").lower()
            if candidate in name:
                try:
                    engine.setProperty("voice", voice.id)
                    log.info("Voice hint %r not found; using female voice %r", original_hint, voice.name)
                    return
                except Exception:  # pragma: no cover - defensive
                    continue

    # 2. Any voice whose name smells female.
    for voice in voices:
        name = (voice.name or "").lower()
        if any(token in name for token in _FEMALE_NAME_TOKENS):
            try:
                engine.setProperty("voice", voice.id)
                log.info("Voice hint %r not found; using female voice %r", original_hint, voice.name)
                return
            except Exception:  # pragma: no cover - defensive
                continue

    log.debug("No female voice found; engine will use the system default.")


_engine_lock = threading.Lock()
_engine: Optional[pyttsx3.Engine] = None  # type: ignore[type-arg]


def speak_text(text: str) -> None:
    """Speak *text* by delegating to :func:`core.speech.speak`.

    We do NOT build a competing pyttsx3 engine here - the Windows
    SAPI5 driver refuses two engines in the same process with
    "run loop already started". Use the single canonical engine in
    :mod:`core.speech` so there is exactly one TTS instance.
    """
    try:
        from core.speech import speak
        speak(text)
    except Exception as exc:  # pragma: no cover - defensive
        log.error("TTS failed: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _build_agent(router: CommandRouter) -> AgentExecutor:
    cfg = get_config()
    llm = ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_base_url, reasoning=False)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are F.R.I.D.A.Y. (Female Replacement Intelligent Digital Assistant Youth), an intelligent, conversational AI assistant inspired by the Marvel character. "
                "You can respond in natural, human-like language and use tools "
                "when needed. Keep responses conversational and concise. Refer to yourself as Friday when introducing yourself.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Hide legacy tools from the LLM. The legacy @tool functions expect
    # structured input the LLM cannot reliably fill; deterministic routing
    # already handles the user-facing cases for them.
    _LEGACY_NAMES = {
        "get_time",
        "open_anything",
        "arp_scan_terminal",
        "duckduckgo_search",
        "matrix_mode",
        "capture_screenshot",
        "read_latest_screenshot",
    }
    tools_for_llm = router.langchain_tools(exclude_predicate=lambda n: n in _LEGACY_NAMES)
    agent = create_tool_calling_agent(llm=llm, tools=tools_for_llm, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools_for_llm, verbose=False)


def _maybe_run_startup_tasks(cfg, router) -> None:
    """Run the once-per-boot actions.

    Triggered by either:
    * :func:`core.autostart.launched_from_startup` - we were launched
      through the Windows ``Run`` registry value with ``--startup``.
    * The very first launch of arvis on a fresh machine, when
      ``JARVIS_AUTO_REGISTER_STARTUP`` is true - so the autostart
      registration happens automatically the first time the user runs
      the app, instead of requiring an explicit voice command.

    The helper never raises - any error is logged and swallowed so a
    failed launcher never blocks the voice loop from coming up.
    """
    try:
        from_startup = launched_from_startup()
        if not from_startup and cfg.auto_register_startup:
            try:
                from core.autostart import is_enabled, enable

                if not is_enabled():
                    log.info("Auto-registering arvis in the Windows Run key.")
                    enable()
            except Exception as exc:  # pragma: no cover - depends on OS
                log.debug("Auto-register skipped: %s", exc)

        if not from_startup:
            # First-run path above already handled registration; no
            # apps / websites should be opened on a normal launch.
            return

        log.info(
            "Boot tasks: URLs=%s, apps=%s, hand_mouse=%s",
            cfg.startup_urls, cfg.startup_apps, cfg.startup_hand_mouse,
        )

        # Open the configured URLs through the same path the
        # ``open_anything`` tool uses, so platform quirks (Windows
        # ``start`` vs ``webbrowser``) are handled consistently.
        for url in cfg.startup_urls:
            try:
                from tools.opener import _open_url
                _open_url(url)
                log.info("Startup URL opened: %s", url)
                time.sleep(0.4)
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to open startup URL %s: %s", url, exc)

        # Launch the configured desktop apps.
        for app in cfg.startup_apps:
            try:
                from tools.opener import _launch_app
                ok, msg = _launch_app(app)
                log.info(
                    "Startup app %s: %s (%s)", app, msg, "ok" if ok else "fail"
                )
                time.sleep(0.4)
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to launch startup app %s: %s", app, exc)

        # Start the hand-mouse controller so arvis can drive the
        # cursor by itself, if the user opted in.
        if cfg.startup_hand_mouse:
            try:
                from vision import HandMouseTool

                tool = HandMouseTool()
                result = tool.start()
                log.info("Startup hand-mouse: %s", result.message)
            except Exception as exc:  # pragma: no cover
                log.warning("Startup hand-mouse failed: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Startup tasks failed: %s", exc)


def run() -> None:
    cfg = get_config()
    trigger = cfg.trigger_word
    timeout = cfg.conversation_timeout

    router = build_router()
    executor = _build_agent(router)
    log.info("Router has %d tools registered.", len(router.tools()))

    # On Windows boot, fire the configured startup URLs/apps and (if
    # requested) start the hand-mouse controller so arvis can drive the
    # cursor by itself. Also opportunistically self-register the Run
    # registry value the first time we run, so the user only has to say
    # "start at boot" once.
    _maybe_run_startup_tasks(cfg, router)

    # ------------------------------------------------------------------
    # New layers: conversation manager + planner + executor.
    # These cooperate on top of the legacy router. The mic loop now:
    #   1. listens for the wake word,
    #   2. begins a conversation session,
    #   3. classifies each utterance (NEW / FOLLOWUP / CANCEL / END / ...),
    #   4. feeds NEW + FOLLOWUP into the planner, which returns a Task,
    #   5. runs the Task on the executor thread (retry / replan / cancel),
    #   6. speaks short progress + the final completion message.
    # ------------------------------------------------------------------
    task_ctx = TaskContext()
    confirmation = ConfirmationManager()
    verifier = Verifier()
    planner = TaskPlanner(router=router)
    task_executor = TaskExecutor(
        router=router,
        ctx=task_ctx,
        confirmation=confirmation,
        verifier=verifier,
        planner=planner,
    )
    conversation = ConversationManager(ctx=task_ctx)

    mic = sr.Microphone()
    conversation_mode = False
    last_interaction_time = 0.0

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    if not conversation_mode:
                        log.info("Listening for wake word %r...", trigger)
                        audio = recognizer.listen(source, timeout=10)
                        transcript = recognizer.recognize_google(audio)
                        if trigger.lower() in transcript.lower():
                            log.info("Wake word detected: %s", transcript)
                            speak_text("At your service.")
                            conversation_mode = True
                            last_interaction_time = time.time()
                            conversation.begin_session()
                    else:
                        log.info("Listening for next command...")
                        audio = recognizer.listen(source, timeout=10)
                        command = recognizer.recognize_google(audio)
                        log.info("Command: %s", command)
                        last_interaction_time = time.time()

                        # Classify first; cancellation / end / wait
                        # take precedence over routing.
                        kind = conversation.classify(command)
                        if kind is UtteranceKind.CANCEL:
                            log.info("User cancelled.")
                            task_executor.cancel()
                            task_executor.wait(timeout=2.0)
                            conversation.end_session(reason="user cancelled")
                            task_ctx.reset()
                            speak_text("Cancelled.")
                            conversation_mode = False
                            continue
                        if kind is UtteranceKind.END_SESSION:
                            log.info("User ended session.")
                            task_executor.cancel()
                            task_executor.wait(timeout=2.0)
                            conversation.end_session(reason="user ended")
                            task_ctx.reset()
                            speak_text("Goodbye, sir.")
                            conversation_mode = False
                            continue
                        if kind is UtteranceKind.WAIT:
                            speak_text("Standing by.")
                            conversation.wait_for_followup()
                            continue

                        # Resolve any pronouns the user might have
                        # reused ("open that one", "type it again").
                        resolved = task_ctx.resolve(command) if kind in {
                            UtteranceKind.FOLLOWUP,
                            UtteranceKind.QUESTION,
                            UtteranceKind.NEW_COMMAND,
                        } else command
                        conversation.record("user", resolved)
                        conversation.set_state(conversation.state())  # ensure snapshot reflects role

                        # Try the planner -> executor path. If the
                        # planner decides the goal is single-step it
                        # returns a one-step Task; we still let the
                        # executor handle it (so retry / verify /
                        # confirmation logic is shared).
                        try:
                            task = planner.plan(resolved, ctx=task_ctx)
                        except Exception as exc:  # pragma: no cover - planner defensive
                            log.warning("Planner failed: %s", exc)
                            task = Task(goal=resolved, steps=[])
                        task_ctx.last_goal = task.goal

                        # If the planner produced a clarification,
                        # speak the question and stay in conversation.
                        if task.steps and task.steps[0].tool_hint == "__clarify__":
                            speak_text(task.steps[0].description)
                            conversation.set_state(conversation.state())
                            continue

                        # Special case: empty step list -> just route
                        # the raw command through the existing
                        # single-shot path (legacy LLM agent). This
                        # happens when the goal is a pure question or
                        # when the planner fell back to nothing.
                        if not task.steps or not task.steps[0].description.strip():
                            _fallback_single_shot(router, executor, resolved)
                            continue

                        # Multi-step (or planner-built single-step) ->
                        # run on the executor thread.
                        conversation.set_state(conversation.state())
                        task_executor.run_async(task)
                        # Brief wait so the executor has a chance to
                        # speak its first progress line, but don't
                        # block the mic loop forever - the executor
                        # runs in the background and we'll just wait
                        # for the next utterance to come in.
                        task_executor.wait(timeout=0.5)
                        # If it finished within the brief window
                        # (single-step plan or a quick fail), speak
                        # the final summary now.
                        if not task_executor.is_running():
                            snapshot = task_executor.state_snapshot()
                            if snapshot.state in {"completed", "failed", "cancelled"}:
                                # The executor already spoke its own
                                # summary; nothing more to do.
                                pass

                        # Follow-up timeout: stay in conversation
                        # mode until either (a) the user says
                        # goodbye/cancel, or (b) we go silent for the
                        # configured follow-up window.
                        if time.time() - last_interaction_time > timeout:
                            log.info("Conversation timeout - returning to wake word mode.")
                            conversation.end_session(reason="timeout")
                            task_ctx.reset()
                            conversation_mode = False

                except sr.WaitTimeoutError:
                    if conversation_mode and time.time() - last_interaction_time > timeout:
                        log.info("Timeout - returning to wake word mode.")
                        conversation.end_session(reason="timeout")
                        task_ctx.reset()
                        conversation_mode = False
                except sr.UnknownValueError:
                    log.warning("Could not understand audio.")
                except Exception as exc:
                    log.error("Error during recognition or tool call: %s", exc)
                    time.sleep(1)
    except Exception as exc:
        log.critical("Critical error in main loop: %s", exc)


def _fallback_single_shot(router: CommandRouter, executor: AgentExecutor, command: str) -> None:
    """Legacy single-shot path used when the planner has nothing to plan.

    Kept as a module-level helper so the run-loop can fall through to
    it for free-form questions and other non-mission utterances.
    """

    def _fallback(text: str):
        try:
            return _agent_default(executor, text)
        except Exception as exc:  # pragma: no cover
            log.exception("LLM fallback crashed: %s", exc)
            from core.base import ToolResult

            return ToolResult(success=False, message=f"LLM error: {exc}")

    try:
        result = router.dispatch(command, context={"router": router}, default=_fallback)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Dispatch raised: %s", exc)
        from core.base import ToolResult

        result = ToolResult(success=False, message=f"Dispatch error: {exc}")
    if not result.success:
        log.warning("Dispatch failed: %s", result.message)
    from core.speech import speak_brief

    if len(result.message) > 400:
        speak_brief(result.message, max_words=60)
    else:
        speak_text(result.message)


def _agent_default(executor: AgentExecutor, command: str):
    """Used by :func:`CommandRouter.dispatch` when no tool matches a command."""
    log.debug("Routing %r through LangChain agent.", command)
    from core.base import ToolResult
    try:
        response = executor.invoke({"input": command})
        content = response.get("output", "") if isinstance(response, dict) else str(response)
    except Exception as exc:
        log.exception("Agent invocation failed: %s", exc)
        return ToolResult(success=False, message=f"I could not reach my language model just now: {exc}")
    if not content:
        return ToolResult(success=False, message="My language model returned an empty response.")
    return ToolResult(success=True, message=content)


if __name__ == "__main__":
    run()
