# Arvis — Voice assistant extensions (Vision, Communication, Routines)

Arvis adds three feature areas on top of the existing voice assistant: vision (screen/webcam capture + image analysis), communication (email, WhatsApp, Telegram, Discord, Slack) and routines (user-defined multi-step automations). The new code lives alongside the legacy project and is opt-in via a new entry point `app.py` — no legacy files are rewritten.

Highlights

- Vision: screenshots, webcam capture, OCR (Tesseract/EasyOCR), and multimodal model analysis.
- Communication: send/read email, WhatsApp (Cloud API or web fallback), Telegram bot, Discord webhook/bot, Slack messages.
- Routines: create, edit, list and run named routines made of simple actions (open app, open URL, wait, speak, etc.).

Quick start

1. Clone and enter the repo:

```bash
git clone https://github.com/Arnav-j-builds/Arvis-ai.git
cd Arvis-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and fill in credentials you want to use:

```bash
cp .env.example .env
# edit .env
```

4. Run the new entry point to enable the extensions:

```bash
python app.py
```

Notes

- The original `main.py` is unchanged and will keep working. Use `app.py` to register the new tools and router.
- Tesseract OCR must be installed on your system for full OCR support (brew/apt/Windows installer).

Repository layout (important parts)

```
jarvis-main/
├── main.py                       # original entry point (left untouched)
├── app.py                        # new entry point that registers extended features
├── requirements.txt              # includes new dependencies (vision, OCR, comms)
├── .env.example                  # environment variables for credentials/config
├── core/                         # shared infrastructure (router, base types, config)
├── vision/                       # screen/webcam capture, OCR, image analysis
├── communication/                # email, whatsapp, telegram, discord, slack
├── routines/                     # routine manager + interactive builder
├── storage/                      # auto-created runtime storage (routines, contacts, screenshots)
├── tools/                        # legacy tools (kept for compatibility)
└── tests/                        # smoke tests
```

Core behavior

- CommandRouter: deterministic fast-path routing for keywords and routines before falling back to the LLM.
- BaseTool: common interface for tools (can_handle, execute, safe_execute) and a LangChain adapter for backwards compatibility.
- RoutineManager: persistent routines stored in `storage/routines.json` with upsert/delete/run functionality.

Common commands (voice examples)

Vision
- "Jarvis, what is on my screen?" — captures primary monitor and runs the vision analyzer.
- "Read this page." — captures screen and returns OCR text.
- "Describe the image." or "Take a photo." — takes one webcam frame and analyzes it.

Communication
- "Email John that I'll arrive at 5, body see you soon" — parses recipient, subject, body and sends via SMTP.
- "Send WhatsApp to Mom saying I'll be late." — uses Cloud API if configured, otherwise opens WhatsApp Web.
- "Send telegram to @alice saying meeting at 5." — uses Telegram Bot API.
- "Message my Discord team deploy complete." — sends via webhook or bot.

Routines
- "I'm starting work" — triggers a named routine that can open apps, URLs, wait and speak.
- "Create a routine called Movie Time" — interactive builder for creating routines.
- "List my routines" — prints all routines and triggers.

Error handling

- Missing credentials in `.env` produce friendly error messages naming the missing variables.
- If a vision model (e.g. Ollama) is not available, the analyzer falls back to OCR + caption.
- Tesseract binary missing raises a `RuntimeError` from the OCR module; the router wraps failures into a `ToolResult(success=False, ...)` to avoid crashing the loop.
- Corrupt `routines.json` is backed up and replaced with a fresh file; a warning is logged.

Developer notes

- Add credentials to `.env` only for the providers you plan to use.
- `app.py` imports legacy `tools/` functions and re-registers them through the router. No changes to the original `main.py` or legacy tools are required.
- Tests: run `python tests/smoke.py` to exercise imports and basic offline functionality.

Future ideas

- Region-selection UI for targeted screenshot capture.
- Calendar/reminders and better contacts DB (vCard / Google People API).
- Plugin loader, routine conditionals, and a vision cache to debounce identical screenshots.

If you want, I can further shorten, expand, or tailor this README for PyPI, a project website, or your repository's README badge/style preferences.
