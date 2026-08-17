import os
import logging
import time
import pyttsx3
from dotenv import load_dotenv
import speech_recognition as sr
from langchain_ollama import ChatOllama, OllamaLLM

# from langchain_openai import ChatOpenAI # if you want to use openai
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# importing tools
from tools.time import get_time
from tools.OCR import read_text_from_latest_image
from tools.arp_scan import arp_scan_terminal
from tools.duckduckgo import duckduckgo_search_tool
from tools.matrix import matrix_mode
from tools.screenshot import take_screenshot
from tools.opener import open_anything

# Hand-mouse gesture control. Optional - the import is wrapped so the
# legacy main.py keeps working on systems without opencv / mediapipe /
# pyautogui installed.
try:
    from vision.hand_mouse import (
        HandMouseTool,
        HAS_CV2 as _HAND_HAS_CV2,
        HAS_PYAUTOGUI as _HAND_HAS_PYAUTOGUI,
        HAS_MEDIAPIPE as _HAND_HAS_MEDIAPIPE,
    )
    _hand_mouse_tool = HandMouseTool()
except Exception as _hand_mouse_import_error:  # pragma: no cover - defensive
    logging.warning("Hand-mouse module unavailable: %s", _hand_mouse_import_error)
    _hand_mouse_tool = None
    _HAND_HAS_CV2 = False
    _HAND_HAS_PYAUTOGUI = False
    _HAND_HAS_MEDIAPIPE = False

# Vision-mouse gaze controller. Optional, same defensive import pattern.
try:
    from vision.eye_mouse import EyeMouseTool
    _vision_mouse_tool = EyeMouseTool()
except Exception as _vision_mouse_import_error:  # pragma: no cover - defensive
    logging.warning("Vision-mouse module unavailable: %s", _vision_mouse_import_error)
    _vision_mouse_tool = None

# Typing tool: paste text / press keys / send Ctrl+V. Optional.
try:
    from tools.typing import TypingTool
    _typing_tool = TypingTool()
except Exception as _typing_import_error:  # pragma: no cover - defensive
    logging.warning("Typing module unavailable: %s", _typing_import_error)
    _typing_tool = None

load_dotenv()

MIC_INDEX = None
TRIGGER_WORD = "friday"
CONVERSATION_TIMEOUT = 30  # seconds of inactivity before exiting conversation mode

logging.basicConfig(level=logging.DEBUG)  # logging

# api_key = os.getenv("OPENAI_API_KEY") removed because it's not needed for ollama
# org_id = os.getenv("OPENAI_ORG_ID") removed because it's not needed for ollama

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=MIC_INDEX)

# Initialize LLM
llm = ChatOllama(model="minimax-m3:cloud", reasoning=False)

# llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, organization=org_id) for openai

# Tool list
tools = [get_time, arp_scan_terminal, read_text_from_latest_image, duckduckgo_search_tool, matrix_mode, take_screenshot, open_anything]

# ---------------------------------------------------------------------------
# Hand-mouse fast-path
# ---------------------------------------------------------------------------
# Voice commands that should drive the hand-mouse controller are matched
# here, before the slow LangChain agent runs. Returning early keeps the
# experience snappy and prevents the LLM from "explaining" the gesture
# instead of toggling it.
_HAND_MOUSE_START = (
    "start hand mouse",
    "start the hand mouse",
    "start hand tracking",
    "enable hand mouse",
    "enable hand control",
    "control my mouse with my hand",
    "control the mouse with my hand",
    "begin hand tracking",
    "begin hand mouse",
    "hand mouse on",
    "hand tracking on",
)
_HAND_MOUSE_STOP = (
    "stop hand mouse",
    "stop the hand mouse",
    "stop hand tracking",
    "disable hand mouse",
    "disable hand control",
    "hand mouse off",
    "hand tracking off",
    "release the mouse",
    "let go of the mouse",
)


def _maybe_handle_hand_mouse(command: str) -> str | None:
    """Run the hand-mouse command if it matches; return its spoken reply
    or ``None`` if it does not match (caller should fall through to LLM)."""
    if _hand_mouse_tool is None:
        return None
    lowered = (command or "").lower()
    if not lowered:
        return None
    if not ("hand mouse" in lowered or "hand tracking" in lowered or
            "control my mouse with my hand" in lowered or
            "control the mouse with my hand" in lowered):
        return None
    result = _hand_mouse_tool.execute(command)
    return result.message if result else None

# ---------------------------------------------------------------------------
# Vision-mouse / typing fast-path
# ---------------------------------------------------------------------------
_VISION_MOUSE_KEYWORDS = (
    "vision mouse",
    "vision tracking",
    # Eye-based fallbacks.
    "eye mouse",
    "eye tracking",
    "control my mouse with my eye",
    "control the mouse with my eye",
)


def _maybe_handle_vision_mouse(command: str) -> str | None:
    """Run the vision-mouse command if it matches; return its spoken reply.

    The user-facing phrasing is "vision mouse" / "vision tracking" so
    speech recognition reliably hears it - "eye tracking" gets
    misheard as "I tracking".
    """
    if _vision_mouse_tool is None:
        return None
    lowered = (command or "").lower()
    if not lowered:
        return None
    if not any(k in lowered for k in _VISION_MOUSE_KEYWORDS):
        return None
    result = _vision_mouse_tool.execute(command)
    return result.message if result else None


_TYPING_KEYWORDS = (
    "type ", "type this", "write ", "write this", "enter this", "input ",
    "press enter", "press tab", "press escape", "press esc",
    "press backspace", "press delete", "press space", "press spacebar",
    "press up", "press down", "press left", "press right",
    "press home", "press end", "press page up", "press page down",
    "press shift enter", "press ctrl enter", "press alt enter",
    "press alt tab", "press windows", "press super",
    "paste from clipboard", "paste clipboard", "paste the clipboard",
    "clear text", "select all",
)


def _maybe_handle_typing(command: str) -> str | None:
    """Run the typing tool if it matches; return its spoken reply."""
    if _typing_tool is None:
        return None
    lowered = (command or "").lower()
    if not lowered:
        return None
    if not any(k in lowered for k in _TYPING_KEYWORDS):
        return None
    result = _typing_tool.execute(command)
    return result.message if result else None

# Tool-calling prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are F.R.I.D.A.Y. (Female Replacement Intelligent Digital Assistant Youth), an intelligent, conversational AI assistant inspired by the Marvel character. Your goal is to be helpful, friendly, witty, and informative. You can respond in natural, human-like language and use tools when needed to answer questions more accurately. Always explain your reasoning simply when appropriate, and keep your responses conversational and concise. Address the user respectfully and refer to yourself as Friday when introducing yourself.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Agent + executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# TTS setup
_FEMALE_VOICE_CANDIDATES = (
    "zira", "aria", "jenny", "samantha", "victoria", "fiona",
    "serena", "ava", "allison", "susan", "kate", "hazel", "libby",
)
_FEMALE_NAME_TOKENS = _FEMALE_VOICE_CANDIDATES + (
    "female", "woman", "girl", "lady",
)


def _pick_voice(engine):
    """Pick a female, charming voice.  Honours the JARVIS_VOICE_HINT env
    var; falls back to the first installed female voice; finally
    leaves the engine on its system default if nothing matches."""
    hint = os.getenv("JARVIS_VOICE_HINT", "zira").lower().strip()
    voices = list(engine.getProperty("voices") or [])
    if hint:
        for v in voices:
            if hint in (v.name or "").lower():
                engine.setProperty("voice", v.id)
                return
    for candidate in _FEMALE_VOICE_CANDIDATES:
        for v in voices:
            name = (v.name or "").lower()
            if candidate in name:
                engine.setProperty("voice", v.id)
                return
    for v in voices:
        name = (v.name or "").lower()
        if any(tok in name for tok in _FEMALE_NAME_TOKENS):
            engine.setProperty("voice", v.id)
            return


def speak_text(text: str):
    try:
        engine = pyttsx3.init()
        _pick_voice(engine)
        try:
            engine.setProperty("rate", int(os.getenv("JARVIS_SPEECH_RATE", "170")))
        except (TypeError, ValueError):
            engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.3)
    except Exception as e:
        logging.error(f"❌ TTS failed: {e}")


# Main interaction loop
def write():
    conversation_mode = False
    last_interaction_time = None

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                try:
                    if not conversation_mode:
                        logging.info("🎤 Listening for wake word...")
                        audio = recognizer.listen(source, timeout=10)
                        transcript = recognizer.recognize_google(audio)
                        logging.info(f"🗣 Heard: {transcript}")

                        if TRIGGER_WORD.lower() in transcript.lower():
                            logging.info(f"🗣 Triggered by: {transcript}")
                            speak_text("At your service.")
                            conversation_mode = True
                            last_interaction_time = time.time()
                        else:
                            logging.debug("Wake word not detected, continuing...")
                    else:
                        logging.info("🎤 Listening for next command...")
                        audio = recognizer.listen(source, timeout=10)
                        command = recognizer.recognize_google(audio)
                        logging.info(f"📥 Command: {command}")

                        # Hand-mouse fast-path: route gesture commands
                        # to the local controller before paying the LLM
                        # round-trip.
                        hand_reply = _maybe_handle_hand_mouse(command)
                        if hand_reply is not None:
                            logging.info("🖐 Hand-mouse reply: %s", hand_reply)
                            print("F.R.I.D.A.Y.:", hand_reply)
                            speak_text(hand_reply)
                            last_interaction_time = time.time()
                            continue

                        # Vision-mouse fast-path: route gaze commands to
                        # the local controller before the LLM. Triggered
                        # by "vision mouse" / "vision tracking" so it
                        # isn't misheard as "I tracking".
                        vision_reply = _maybe_handle_vision_mouse(command)
                        if vision_reply is not None:
                            logging.info("👁 Vision-mouse reply: %s", vision_reply)
                            print("F.R.I.D.A.Y.:", vision_reply)
                            speak_text(vision_reply)
                            last_interaction_time = time.time()
                            continue

                        # Typing fast-path: paste text / press keys /
                        # Ctrl+V without round-tripping through the LLM.
                        typing_reply = _maybe_handle_typing(command)
                        if typing_reply is not None:
                            logging.info("⌨️ Typing reply: %s", typing_reply)
                            print("F.R.I.D.A.Y.:", typing_reply)
                            speak_text(typing_reply)
                            last_interaction_time = time.time()
                            continue

                        logging.info("🤖 Sending command to agent...")
                        response = executor.invoke({"input": command})
                        content = response["output"]
                        logging.info(f"✅ Agent responded: {content}")

                        print("F.R.I.D.A.Y.:", content)
                        speak_text(content)
                        last_interaction_time = time.time()

                        if time.time() - last_interaction_time > CONVERSATION_TIMEOUT:
                            logging.info("⌛ Timeout: Returning to wake word mode.")
                            conversation_mode = False

                except sr.WaitTimeoutError:
                    logging.warning("⚠️ Timeout waiting for audio.")
                    if (
                        conversation_mode
                        and time.time() - last_interaction_time > CONVERSATION_TIMEOUT
                    ):
                        logging.info(
                            "⌛ No input in conversation mode. Returning to wake word mode."
                        )
                        conversation_mode = False
                except sr.UnknownValueError:
                    logging.warning("⚠️ Could not understand audio.")
                except Exception as e:
                    logging.error(f"❌ Error during recognition or tool call: {e}")
                    time.sleep(1)

    except Exception as e:
        logging.critical(f"❌ Critical error in main loop: {e}")
    finally:
        # Always release the webcam / mouse controller on exit so the OS
        # does not lock the camera device.
        if _hand_mouse_tool is not None:
            try:
                _hand_mouse_tool.stop()
            except Exception:
                pass
        if _vision_mouse_tool is not None:
            try:
                _vision_mouse_tool.stop()
            except Exception:
                pass


if __name__ == "__main__":
    # Opportunistic autostart registration, mirroring app.py so users who
    # launch the assistant via ``python main.py`` (the legacy entry point)
    # also benefit from "open as soon as Windows starts" without having
    # to say the voice command first.
    try:
        from core.autostart import is_enabled, enable, launched_from_startup
        from core.config import get_config

        cfg = get_config()
        if cfg.auto_register_startup and not is_enabled() and not launched_from_startup():
            enable()
    except Exception as _autostart_err:  # pragma: no cover - defensive
        logging.debug("Auto-register skipped: %s", _autostart_err)

    write()
