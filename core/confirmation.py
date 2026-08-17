"""
core.confirmation
~~~~~~~~~~~~~~~~~

Risk classification + voice-based confirmation for risky tool calls.

Why this exists
---------------
The task executor can chain arbitrary tools together. Some of them
(send email, shutdown, run terminal, delete files, ...) are
destructive or external. We do NOT let the executor call them
silently - every confirmation-required step has to be approved by
the user first.

Risk levels
-----------
* ``SAFE``               - run silently.
* ``CONFIRMATION_REQUIRED`` - speak the action, listen for yes/no.
* ``HIGH_RISK``          - same as above PLUS a pause + require the
  literal word "yes" (not "okay" / "sure").

The mapping from tool name → risk level is hard-coded and extended
via env: ``JARVIS_CONFIRM_EXTRA=tool1,tool2,tool3``.

The confirmation flow opens its own short-lived
``speech_recognition.Microphone()`` listen (separate from the main
microphone loop) so the executor thread can block without deadlocking
the voice loop. The executor calls ``ask()`` synchronously and gets
back ``True`` / ``False``.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Iterable, Optional

from core.config import get_config
from core.logger import get_logger
from core.speech import speak
from core.task_plan import TaskStep

log = get_logger(__name__)


class RiskLevel(str, enum.Enum):
    SAFE = "safe"
    CONFIRMATION_REQUIRED = "confirmation_required"
    HIGH_RISK = "high_risk"


# Default mapping - covers every "destructive or external" tool
# registered with the CommandRouter in app.build_router. Anything not
# listed is SAFE.
_DEFAULT_RISK = {
    # External communication - always confirm.
    "send_email": RiskLevel.CONFIRMATION_REQUIRED,
    "gmail_tool": RiskLevel.CONFIRMATION_REQUIRED,
    "whatsapp_tool": RiskLevel.CONFIRMATION_REQUIRED,
    "telegram_tool": RiskLevel.CONFIRMATION_REQUIRED,
    "discord_tool": RiskLevel.CONFIRMATION_REQUIRED,
    "slack_tool": RiskLevel.CONFIRMATION_REQUIRED,
    # System / power - HIGH_RISK.
    "shutdown": RiskLevel.HIGH_RISK,
    "reboot": RiskLevel.HIGH_RISK,
    # Terminal / shell - HIGH_RISK because the legacy safe terminal
    # runner blocks truly destructive commands but we still want an
    # explicit "yes" before the executor runs them.
    "terminal_tool": RiskLevel.HIGH_RISK,
    # Anything whose name contains "delete" or "remove" is treated as
    # HIGH_RISK by the suffix check below.
}


_YES_WORDS = ("yes", "yeah", "yep", "yup", "sure", "okay", "ok", "do it", "go ahead", "confirm")
_NO_WORDS = ("no", "nope", "cancel", "stop", "don't", "abort", "never mind", "skip")


class ConfirmationManager:
    """Voice-based confirmation for risky steps.

    Thread-safe. The same instance can be shared between the executor
    thread and the conversation manager - ``ask()`` synchronises on
    an internal lock so two concurrent calls cannot both open the
    microphone.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # ``_last_utterance`` records the user's spoken reply for
        # logging / debugging. Reset to ``None`` after every ``ask``.
        self._last_utterance: Optional[str] = None
        # ``_confirm_callback`` lets tests / the web dashboard inject
        # a non-voice confirmation path (e.g. a "click yes on the
        # dashboard" stub). When set, ``ask`` returns the callback's
        # bool without opening the microphone.
        self._confirm_callback = None

    def classify(self, step: TaskStep) -> RiskLevel:
        """Map a ``TaskStep`` to its risk level.

        The check is name-based: tool_hint + any "name" / "action"
        argument. ``terminal_tool`` always escalates to HIGH_RISK.
        """
        cfg = get_config()
        hint = (step.tool_hint or "").strip().lower()
        name = str(
            (step.arguments or {}).get("name")
            or (step.arguments or {}).get("action")
            or ""
        ).lower()

        # Direct map.
        if hint in _DEFAULT_RISK:
            return _DEFAULT_RISK[hint]

        # Suffix / substring matches.
        if any(kw in hint for kw in ("delete", "remove", "drop", "wipe", "format")):
            return RiskLevel.HIGH_RISK
        if any(kw in hint for kw in ("shutdown", "reboot", "restart", "power off")):
            return RiskLevel.HIGH_RISK

        # Argument-driven matches.
        if name in {"shutdown", "reboot", "restart"}:
            return RiskLevel.HIGH_RISK
        if name in {"send", "delete", "remove"}:
            return RiskLevel.CONFIRMATION_REQUIRED

        # Env-driven extension - tools the user wants to be asked
        # about even though they are not in the default map.
        extra = getattr(cfg, "confirm_extra_tools", ())
        if hint in extra:
            return RiskLevel.CONFIRMATION_REQUIRED

        return RiskLevel.SAFE

    # ------------------------------------------------------------------
    # Voice confirmation
    # ------------------------------------------------------------------
    def ask(self, step: TaskStep) -> bool:
        """Run the voice confirmation flow for *step*.

        Returns ``True`` if the user approved, ``False`` otherwise.
        A timeout or recognition failure counts as "no" - the safe
        default is to NOT execute a risky action when the user did
        not clearly say yes.
        """
        risk = self.classify(step)
        if risk == RiskLevel.SAFE:
            return True

        # Test / web-UI hook.
        if self._confirm_callback is not None:
            try:
                return bool(self._confirm_callback(step, risk))
            except Exception as exc:  # pragma: no cover
                log.warning("Confirm callback raised: %s", exc)
                return False

        with self._lock:
            cfg = get_config()
            self._last_utterance = None

            prompt = self._format_prompt(step, risk)
            log.info("[CONFIRM] %s (risk=%s)", prompt, risk.value)
            speak(prompt)

            if risk == RiskLevel.HIGH_RISK:
                # Short delay so a background noise spike does NOT
                # trigger an accidental confirmation.
                time.sleep(cfg.confirmation_high_risk_pause_s)

            try:
                reply = self._listen(cfg.confirmation_listen_timeout_s)
            except Exception as exc:  # pragma: no cover - env dependent
                log.warning("[CONFIRM] listen failed: %s", exc)
                reply = ""
            self._last_utterance = reply
            approved = self._is_yes(reply, risk=risk)
            log.info("[CONFIRM] reply=%r approved=%s", reply, approved)
            if not approved:
                speak(self._denial_ack(risk))
            return approved

    @property
    def last_utterance(self) -> Optional[str]:
        return self._last_utterance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _format_prompt(self, step: TaskStep, risk: RiskLevel) -> str:
        desc = (step.description or step.tool_hint or "perform this action").strip()
        if risk == RiskLevel.HIGH_RISK:
            return (
                f"Warning, sir. I am about to {desc}. "
                f"This is a high-risk action. Please say yes to continue."
            )
        return f"I am about to {desc}. Should I continue?"

    def _denial_ack(self, risk: RiskLevel) -> str:
        if risk == RiskLevel.HIGH_RISK:
            return "Understood, sir. Cancelled."
        return "Understood, sir. Skipping that step."

    def _listen(self, timeout_s: float) -> str:
        """Open a short-lived STT listen and return the recognised text.

        Always opens a *new* Microphone() so we don't fight with the
        main microphone loop's input stream. If the recogniser is
        unavailable we return an empty string (which is interpreted
        as "no").
        """
        try:
            import speech_recognition as sr  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("SpeechRecognition unavailable: %s", exc)
            return ""

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                except Exception:
                    pass
                audio = recognizer.listen(source, timeout=timeout_s, phrase_time_limit=timeout_s)
        except Exception as exc:
            log.debug("Confirmation listen failed to capture audio: %s", exc)
            return ""

        try:
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

    @staticmethod
    def _is_yes(reply: str, *, risk: RiskLevel) -> bool:
        text = (reply or "").strip().lower()
        if not text:
            return False
        if any(w in text for w in _NO_WORDS):
            return False
        if not any(w in text for w in _YES_WORDS):
            return False
        if risk == RiskLevel.HIGH_RISK:
            # Require the literal word "yes" so a too-eager "okay"
            # does not bypass the HIGH_RISK gate.
            return "yes" in text.split() or text.startswith("yes")
        return True


__all__ = ["ConfirmationManager", "RiskLevel"]
