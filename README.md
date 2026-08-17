# arvis Extensions — Vision, Communication, Routines

This document describes the three new feature areas added on top of the
existing voice assistant. **No file in the legacy project was rewritten**;
the new code lives in its own package and is registered through a thin
router.

---

## 1. Folder structure

```
jarvis-main/
├── main.py                       # UNCHANGED — original entry point
├── app.py                        # NEW drop-in entry point (registers everything)
├── requirements.txt              # extended with new dependencies
├── .env.example                  # NEW — every configurable value
├── core/                         # NEW — shared infrastructure
│   ├── __init__.py
│   ├── config.py                 # Config dataclass + get_config()
│   ├── logger.py                 # get_logger / configure_logging
│   ├── base.py                   # BaseTool, RoutineAction, ToolResult
│   ├── router.py                 # CommandRouter
│   ├── speech.py                 # speak / speak_async wrappers
│   └── legacy_bridge.py          # LangChain tool → BaseTool adapter
├── vision/                       # NEW
│   ├── __init__.py
│   ├── capture.py                # full screen / primary / region / active window
│   ├── webcam.py                 # single-frame webcam capture
│   ├── ocr.py                    # Tesseract + EasyOCR
│   ├── analyzer.py               # Ollama vision model + fallback
│   └── commands.py               # VisionTool (BaseTool)
├── communication/                # NEW
│   ├── __init__.py
│   ├── email.py
│   ├── whatsapp.py
│   ├── telegram.py
│   ├── discord.py
│   └── slack.py
├── routines/                     # NEW
│   ├── __init__.py
│   ├── manager.py                # CRUD + execution + persistence
│   └── commands.py               # RoutinesTool + interactive builder
├── storage/                      # NEW (auto-created)
│   ├── routines.json             # user routines
│   ├── contacts.json             # contact → phone mapping for WhatsApp
│   └── screenshots/              # PNGs captured by vision
├── tools/                        # UNCHANGED
├── tests/                        # NEW
│   ├── __init__.py
│   └── smoke.py                  # offline import + parser smoke tests
├── README.md / INTEGRATION.md
```

---

## 2. Required pip packages

```text
Pillow>=10.0.0
requests>=2.31.0
easyocr>=1.7.0            # optional, alternative to Tesseract
opencv-python>=4.8.0      # webcam capture
pywin32>=306              # active-window capture on Windows
pyobjc-framework-Quartz>=10.0   # active-window capture on macOS
```

Already required by the existing project: `mss`, `pytesseract`, `Pillow`,
`requests` are reused by the new modules. **Tesseract itself** must be
installed at the OS level (e.g. `brew install tesseract`,
`apt install tesseract-ocr`, or use the Windows installer).

Install everything:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the credentials you want to use.

---

## 3. Integration steps

The legacy project keeps working as-is. To enable the new features:

1. **Install the new packages** (`pip install -r requirements.txt`).
2. **Run the new entry point** instead of `main.py`:

   ```bash
   python app.py
   ```

   `app.py` imports the legacy tools (`tools/opener.py`, `tools/screenshot.py`,
   `tools/time.py`, …) and re-registers them through the new
   `core.router.CommandRouter`, then layers the vision, communication and
   routine tools on top. No edits are required to `main.py`.

3. **(Optional) Migrate to `app.py` permanently.** Replace your run script
   or update `run.bat`/`run.sh` to call `python app.py`.

4. **Add credentials** to `.env` for the providers you want to use.

> **Existing users** can keep invoking `python main.py`. The legacy file
> is untouched. New modules are entirely opt-in through `app.py`.

### How the router fits in

`core/router.py` exposes two methods used by `app.py`:

* `router.langchain_tools()` returns the LangChain-compatible tool list
  that is passed to the agent executor (so the LLM can still tool-call).
* `router.dispatch(command)` runs the deterministic keyword / `can_handle`
  fast-path before the LLM is consulted. This guarantees routines and
  vision shortcuts always work even when the LLM is offline.

---

## 4. New modules

### `core/`
* **`base.py`** — defines `BaseTool` (`can_handle`, `execute`, `safe_execute`,
  `as_langchain_tool`), `RoutineAction` and `ToolResult`. Pure data classes
  plus an abstract class; no I/O.
* **`config.py`** — frozen `Config` dataclass loaded from environment
  variables (one place for every magic string). `get_config()` is memoised.
* **`logger.py`** — `get_logger(name)` returns a configured logger; respects
  `JARVIS_LOG_LEVEL`. Quiets noisy libs.
* **`router.py`** — `CommandRouter` with priority-aware keyword routing,
  `can_handle` fallback, and routine-action execution.
* **`speech.py`** — `speak` / `speak_async` thin wrappers around `pyttsx3`,
  reads voice/rate/volume from env vars.
* **`legacy_bridge.py`** — wraps the existing `@tool`-decorated functions so
  the router can register them without rewriting them.

### `vision/`
* **`capture.py`** — `capture_full_screen`, `capture_primary_monitor`,
  `capture_active_window`, `capture_region`. Uses `mss`; Windows uses
  `pywin32` for active-window capture (with `Quartz` for macOS).
* **`webcam.py`** — single-frame webcam capture via `opencv-python` with a
  `PIL.ImageGrab` fallback.
* **`ocr.py`** — `extract_text(path, languages=None)` returns an `OCRResult`
  using Tesseract (`pytesseract`) by default. EasyOCR is supported when
  `JARVIS_OCR_ENGINE=easyocr`.
* **`analyzer.py`** — `analyze_image(path, prompt)` tries the configured
  vision model (auto-detection looks for `llava`, `bakllava`, `moondream`,
  `minicpm-v`, `qwen-vl`); falls back to OCR + caption if no vision model
  is available.
* **`commands.py`** — `VisionTool` (BaseTool). Dispatches by keywords
  (`"on my screen"`, `"describe the image"`, `"explain this error"`,
  `"what am i looking at"`, etc.).

### `communication/`
* **`email.py`** — IMAP/SMTP via standard library. Credentials read from
  `Config`. Supports `send`, `read latest`, `read unread`, `search`.
* **`whatsapp.py`** — Official Cloud API if `WHATSAPP_TOKEN` +
  `WHATSAPP_PHONE_ID` are present; otherwise opens
  `https://wa.me/<phone>?text=<msg>` in WhatsApp Web. Contact aliases
  resolve via `storage/contacts.json`.
* **`telegram.py`** — `sendMessage` + `getUpdates` over the Bot HTTP API.
* **`discord.py`** — Webhook URL OR Bot token. Webhook is preferred
  (simplest). Bot path enables reading the latest channel messages.
* **`slack.py`** — `chat.postMessage` via the Web API.

### `routines/`
* **`manager.py`** — `RoutineManager` loads/saves `storage/routines.json`,
  supports `upsert`, `delete`, `get`, `matches(command)`, `run(name)`.
* **`commands.py`** — `RoutinesTool` (BaseTool). Handles the interactive
  builder, list/edit/delete/show, and trigger matching.

---

## 5. Changes needed in existing files

**Zero mandatory changes.** Optional touches if you want to consolidate:

* `run.bat` / `run.sh`: replace `python main.py` with `python app.py`.
* `.env`: copy `.env.example` and fill in credentials.
* `requirements.txt`: the new dependencies were added, but everything that
  was previously installed still works.

---

## 6. Example commands

### Vision
| Voice                                              | Action                                                |
|----------------------------------------------------|-------------------------------------------------------|
| *"Jarvis, what is on my screen?"*                  | Captures primary monitor, asks the vision model.      |
| *"Read this page."*                                | Same, with OCR emphasis.                              |
| *"Explain this error."*                            | Captures screen, debug-flavoured prompt.              |
| *"What am I looking at?"*                          | Captures the foreground window.                       |
| *"Read the selected text."*                        | Captures screen, returns OCR only.                    |
| *"Describe the image."* / *"Take a photo."*        | Captures one webcam frame, analyses it.               |

### Communication
| Voice                                                  | Action                                                       |
|--------------------------------------------------------|--------------------------------------------------------------|
| *"Email John that I'll arrive at 5, body see you soon"* | Parses recipient/subject/body, sends via SMTP.               |
| *"Send WhatsApp to Mom saying I'll be late."*           | Cloud API or web.whatsapp.com fallback.                      |
| *"Read my newest email."*                              | Reads the most recent message via IMAP.                      |
| *"Send telegram to @alice saying meeting at 5."*       | Telegram Bot API.                                            |
| *"Message my Discord team deploy complete."*           | Webhook or bot.                                              |
| *"Send slack to #devs saying deploy complete."*        | `chat.postMessage`.                                          |

### Routines
| Voice                                    | Action                                                     |
|------------------------------------------|------------------------------------------------------------|
| *"I'm starting work."*                   | Runs the bundled routine that opens VSCode + Chrome + GH.  |
| *"Create a routine called Movie Time."*  | Interactive builder asks for trigger, actions, description.|
| *"List my routines."*                    | Prints every routine with its triggers.                    |
| *"Show routine starting work."*          | Dumps the routine's structure.                             |
| *"Edit routine Movie Time."*             | Reopens the builder with existing actions loaded.          |
| *"Delete routine Movie Time."*           | Removes the routine.                                       |

---

## 7. Example execution flow

### Routine flow (deterministic fast-path)

```
User: "I'm starting work"
   │
   ▼
app.py wake word → conversation_mode
   │
   ▼
router.dispatch("I'm starting work")
   │  (1) keyword match? no
   │  (2) RoutinesTool.can_handle() returns True because the routine
   │      "starting work" has trigger "starting work".
   │
   ▼
RoutinesTool.execute()
   │
   ▼
RoutineManager.run_routine(routine)
   │   open_app: vscode       → opener._launch_app
   │   open_app: chrome       → opener._launch_app
   │   open_url: github.com   → opener._open_url
   │   open_url: notion.so    → opener._open_url
   │   open_app: spotify      → opener._launch_app
   │   wait: 3                → time.sleep
   │   say: "Everything is ready, sir."
   │
   ▼
Result.message → speak_text("Routine 'starting work' finished successfully.")
```

### Vision flow

```
User: "Jarvis, what is on my screen?"
   │
   ▼
VisionTool.execute()
   │   match on _TRIGGERS_SCREEN → capture_primary_monitor()
   │   → analyze_image(path)
   │       ├─ OCR via Tesseract (always)
   │       └─ Ollama vision model (llava / qwen-vl / ...)
   │
   ▼
Result.message → speak_text(description)
```

### Email flow

```
User: "Email John that I'll arrive at 5, body see you soon"
   │
   ▼
EmailTool.execute()
   │   _parse_send_command extracts:
   │     recipients=["John"], subject="I'll arrive at 5",
   │     body="see you soon"
   │   SMTP login → send
   │
   ▼
Result.message → speak_text("Email sent to John with subject 'I'll arrive at 5'.")
```

---

## 8. Error handling

| Failure scenario                              | Behaviour                                                                                       |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| `.env` missing a credential                   | Tool returns a friendly message naming the variable. No crash, no leak.                         |
| Ollama not running                            | `analyze_image` returns the OCR-only caption; no exception.                                     |
| No vision model installed                     | `analyze_image` logs a warning and returns the OCR caption.                                    |
| Tesseract binary missing                      | `extract_text` raises `RuntimeError`; the router wraps it into a `ToolResult(success=False, ...)`. |
| Webcam device busy / disconnected             | `capture_webcam` tries OpenCV then PIL, finally raises `RuntimeError` with an actionable hint.  |
| Gmail blocks "less secure" logins              | The tool surfaces the SMTP auth error verbatim, advising to use an App Password.                |
| Telegram bot token invalid                    | TelegramTool returns `ToolResult(success=False, message="Telegram rejected the message: ...")`.  |
| Slack channel archived / bot not in channel   | SlackTool surfaces `{"ok": false, "error": "..."}` returned by Slack.                           |
| Routine references unknown action             | Router returns `ToolResult(success=False, message="Unknown routine action 'X'.")`.               |
| `routines.json` corrupt                       | Manager backs it up to `routines.json.bak` and starts fresh; logs a warning.                    |
| Speech recognition timeout                    | Loop continues; conversation mode times out after `JARVIS_CONVERSATION_TIMEOUT`.                |
| LangChain tool raises                         | `BaseTool.safe_execute` catches the exception and returns a `ToolResult(success=False, ...)`.   |

Every tool follows the same pattern:

```python
def safe_execute(...):
    try:
        ...
    except Exception as exc:
        return ToolResult(success=False, message=f"...")
```

so a failing feature never crashes the voice loop.

---

## 9. Future improvements

1. **Region selection UI** — overlay a translucent rectangle and capture
   the selected area instead of the full screen. The vision tool already
   honours `context["region"] = {"left":..., "top":..., "width":..., "height":...}`
   when supplied by such an overlay.
2. **Calendar / reminders** — read/write ICS feeds via the same
   `BaseTool` pattern.
3. **Plugin loader** — drop a `.py` file in `plugins/` and have the
   router auto-discover it.
4. **Routine conditions** — `if battery < 20%: ...` style gates on
   `RoutineAction.metadata`.
5. **Speech-only TTS** — swap `pyttsx3` for `piper` or `coqui-xtts` for
   natural voices.
6. **Vision cache** — debounce identical screenshots before calling the
   vision model.
7. **Contacts DB** — replace `storage/contacts.json` with a proper
   address book (vCard / Google People API).
8. **Email templates** — save frequently-used email bodies alongside
   routines.
9. **Multi-modal LLM** — when running `minimax-m3:cloud`, automatically
   pipe the screenshot into a multimodal chat model instead of the
   `/api/generate` endpoint.
10. **Routine dry-run** — preview the actions without executing them.
11. **Smoke-test CI** — `python tests/smoke.py` exercises every module
