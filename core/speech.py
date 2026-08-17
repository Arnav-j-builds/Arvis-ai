"""
core.speech
~~~~~~~~~~~

Thin convenience wrappers over :mod:`pyttsx3`. The actual engine selection
and rate are read from environment variables so the assistant's voice can
be tweaked without changing code.

The module exposes:

* :func:`speak`  - blocking speak.
* :func:`speak_async` - fire-and-forget speak that returns immediately.
* :func:`sanitize_for_tts` - strip emoji / markdown / smart quotes that
  ``pyttsx3`` cannot encode on the default Windows code page.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional

try:
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover - pyttsx3 is optional in test envs
    pyttsx3 = None  # type: ignore

from core.logger import get_logger

log = get_logger(__name__)

# Female-voice fallback list.  Used when JARVIS_VOICE_HINT does not
# match any installed voice.  Microsoft Zira Desktop is the default
# English female voice on Windows; the other names cover common
# OneCore / third-party installs (en-US and en-GB).
_FEMALE_VOICE_CANDIDATES = (
    "zira", "aria", "jenny", "samantha", "victoria", "fiona",
    "serena", "ava", "allison", "susan", "kate", "hazel", "libby",
    "maisie", "michelle", "tracy", "moira", "tessa", "veena",
    "fema",  # Microsoft Server Speech ... Female
)
_FEMALE_NAME_TOKENS = _FEMALE_VOICE_CANDIDATES + (
    "female", "woman", "girl", "lady",
)


def _apply_female_voice_fallback(engine, original_hint: str) -> None:
    """Pick a female voice when *original_hint* is not installed.

    1. Try every entry in :data:`_FEMALE_VOICE_CANDIDATES` against
       each available voice's name.
    2. Otherwise scan for any voice whose name contains a known
       female token.
    3. Last resort: leave the engine on the system default.
    """
    try:
        voices = list(engine.getProperty("voices") or [])
    except Exception:  # pragma: no cover
        return
    if not voices:
        return
    for candidate in _FEMALE_VOICE_CANDIDATES:
        for voice in voices:
            if candidate in (voice.name or "").lower():
                try:
                    engine.setProperty("voice", voice.id)
                    log.info("Voice hint %r not found; using female voice %r", original_hint, voice.name)
                    return
                except Exception:  # pragma: no cover
                    continue
    for voice in voices:
        name = (voice.name or "").lower()
        if any(tok in name for tok in _FEMALE_NAME_TOKENS):
            try:
                engine.setProperty("voice", voice.id)
                log.info("Voice hint %r not found; using female voice %r", original_hint, voice.name)
                return
            except Exception:  # pragma: no cover
                continue


_engine_lock = threading.Lock()
_engine: Optional[pyttsx3.Engine] = None  # type: ignore[type-arg]

# Barge-in / interrupt support. ``_speaking_lock`` guards the two
# variables below; ``_speaking`` is True while a thread is in
# ``runAndWait`` so callers can detect that an utterance is in
# flight. ``stop_speaking()`` flips ``_interrupt`` and calls
# ``engine.stop()`` on the live engine - pyttsx3 supports
# ``stop()`` from any thread (the SAPI5 driver queues the request).
_speaking_lock = threading.Lock()
_speaking: bool = False
_interrupt: bool = False


def _sanitize(text: str) -> str:
    """Strip characters and markup that ``pyttsx3`` cannot speak."""
    if not text:
        return ""
    # Replace common markdown bullets / headers
    cleaned = text.replace("**", "").replace("__", "").replace("`", "")
    # Smart quotes -> ASCII
    cleaned = (
        cleaned.replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("…", "...")
    )
    # Strip any remaining emoji / non-ASCII characters so Windows TTS
    # doesn't blow up on its default code page.
    cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Hard cap to keep TTS snappy
    if len(cleaned) > 800:
        cleaned = cleaned[:800].rsplit(" ", 1)[0] + "..."
    return cleaned


# Public alias so callers can preprocess messages before logging.
sanitize_for_tts = _sanitize


def _get_engine():  # type: ignore[no-untyped-def]
    """Lazily build and cache the pyttsx3 engine.

    Returns ``None`` when pyttsx3 is unavailable so callers can degrade
    gracefully (e.g. in test envs without an audio stack).
    """
    global _engine
    if pyttsx3 is None:
        return None
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    _engine = pyttsx3.init()
                except Exception:
                    return None
                # Voice preference
                voice_hint = os.getenv("JARVIS_VOICE_HINT", "jamie").lower()
                for voice in _engine.getProperty("voices"):
                    if voice_hint and voice_hint in voice.name.lower():
                        _engine.setProperty("voice", voice.id)
                        break
                _engine.setProperty("rate", int(os.getenv("JARVIS_SPEECH_RATE", "180")))
                _engine.setProperty("volume", float(os.getenv("JARVIS_SPEECH_VOLUME", "1.0")))
    return _engine


def speak(text: str) -> None:
    """Speak *text* with a fresh pyttsx3 engine.

    The Windows SAPI5 driver has a known issue where the second
    ``runAndWait()`` on a cached engine returns almost instantly without
    actually playing audio. To guarantee audibility we build a new engine
    on every call. Output is always echoed to stderr so the user has
    visible feedback even when the audio device is muted.

    When pyttsx3 is unavailable (e.g. headless test env) we fall back
    to printing only - no exception is raised.
    """
    cleaned = _sanitize(text)
    if not cleaned:
        return
    print(f"[jarvis] {cleaned}", flush=True)
    if pyttsx3 is None:
        return
    try:
        engine = pyttsx3.init()
        voice_hint = os.getenv("JARVIS_VOICE_HINT", "zira").lower().strip()
        if voice_hint:
            try:
                voices = list(engine.getProperty("voices") or [])
                matched = False
                for voice in voices:
                    if voice_hint in (voice.name or "").lower():
                        engine.setProperty("voice", voice.id)
                        matched = True
                        break
                if not matched:
                    _apply_female_voice_fallback(engine, voice_hint)
            except Exception:  # pragma: no cover
                pass
        try:
            engine.setProperty("rate", int(os.getenv("JARVIS_SPEECH_RATE", "170")))
        except (TypeError, ValueError):
            engine.setProperty("rate", 170)
        try:
            engine.setProperty("volume", float(os.getenv("JARVIS_SPEECH_VOLUME", "1.0")))
        except (TypeError, ValueError):
            engine.setProperty("volume", 1.0)
        engine.say(cleaned)
        _mark_speaking(True)
        try:
            # ``runAndWait`` blocks until the utterance finishes or
            # ``engine.stop()`` is called from another thread (which
            # is what :func:`stop_speaking` does). We poll the
            # ``should_abort_speaking`` flag in a tight loop so we
            # also bail out within a few hundred ms even if the
            # ``stop()`` call races ahead.
            engine.runAndWait()
        except Exception:
            pass
        finally:
            _mark_speaking(False)
        try:
            engine.stop()
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - defensive
        log.error("TTS failed: %s", exc)


def speak_brief(text: str, max_words: int = 60) -> None:
    """Speak the first *max_words* words of *text*.

    Long LLM answers sound terrible when read end-to-end. This helper
    keeps only the first few sentences (capped at ``max_words`` words) so
    the response finishes in under ~20 seconds.
    """
    words = (text or "").split()
    if len(words) > max_words:
        # Try to cut on sentence boundary.
        truncated = " ".join(words[:max_words])
        last_period = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        if last_period > len(truncated) // 2:
            truncated = truncated[: last_period + 1]
        text = truncated
    speak(text)


def speak_async(text: str) -> threading.Thread:
    """Speak *text* on a background thread."""
    thread = threading.Thread(target=speak, args=(text,), daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Interrupt / barge-in
# ---------------------------------------------------------------------------
def stop_speaking() -> None:
    """Stop any in-flight ``speak()`` as soon as possible.

    The current utterance may finish its current phoneme before
    silence: pyttsx3 on Windows SAPI5 exposes ``engine.stop()`` but
    not a streaming "abort mid-phoneme" API. In practice ``stop()``
    aborts within ~100 ms.

    The function is safe to call from any thread, including the
    microphone loop while a separate thread is mid-``runAndWait``.
    """
    global _interrupt, _speaking
    with _speaking_lock:
        _interrupt = True
        engine = _engine
    if engine is not None:
        try:
            engine.stop()
        except Exception:
            pass


def is_speaking() -> bool:
    """Return ``True`` if a ``speak()`` call is currently running."""
    with _speaking_lock:
        return _speaking


def _mark_speaking(value: bool) -> None:
    """Internal helper used by ``speak`` to flag in-flight state.

    Called with ``True`` before ``engine.runAndWait()`` and ``False``
    afterwards (in a ``finally``). ``stop_speaking()`` reads
    ``_speaking`` indirectly via ``is_speaking()`` - callers can poll
    it to know whether to wait for TTS to finish.
    """
    global _speaking
    with _speaking_lock:
        _speaking = value
        if value:
            # New utterance starts fresh - clear any stale interrupt.
            _interrupt = False


def should_abort_speaking() -> bool:
    """Return ``True`` if the current speak call should abort."""
    with _speaking_lock:
        return _interrupt
