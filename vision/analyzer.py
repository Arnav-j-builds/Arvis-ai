"""
vision.analyzer
~~~~~~~~~~~~~~~

Send an image to a vision-capable Ollama model and return the model's
natural-language description. When no vision model is available locally we
fall back to OCR + a generic caption so the assistant never silently fails.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from core.config import get_config
from core.logger import get_logger
from vision.ocr import OCRResult, extract_text

log = get_logger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Result of analysing an image."""

    description: str
    ocr: Optional[OCRResult]
    used_vision_model: bool
    model: Optional[str]


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _ollama_models(base_url: str) -> list[str]:
    """Return the names of every model Ollama has on disk."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("Could not enumerate Ollama models: %s", exc)
        return []


def _looks_like_vision_model(name: str) -> bool:
    lowered = name.lower()
    return any(tag in lowered for tag in ("llava", "bakllava", "vision", "moondream", "minicpm-v", "cogvlm", "qwen-vl"))


def pick_vision_model() -> Optional[str]:
    """Return the best vision model available on this Ollama install.

    Selection order:

    1. ``JARVIS_VISION_MODEL`` from config (if installed).
    2. First model on the Ollama server whose name contains a vision tag
       (e.g. ``llava``).
    3. ``None`` - caller must use the OCR/captioning fallback.
    """
    cfg = get_config()
    models = _ollama_models(cfg.ollama_base_url)
    if not models:
        return None

    preferred = cfg.vision_model.lower()
    for name in models:
        if name.lower() == preferred or name.lower().startswith(preferred + ":"):
            log.info("Using configured vision model %s", name)
            return name

    for name in models:
        if _looks_like_vision_model(name):
            log.info("Auto-detected vision model %s", name)
            return name

    log.warning("No vision model found on Ollama server %s", cfg.ollama_base_url)
    return None


def _ask_vision_model(image_path: Path, prompt: str, model: str) -> Optional[str]:
    cfg = get_config()
    encoded = _encode_image(image_path)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [encoded],
    }
    try:
        response = requests.post(
            f"{cfg.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("response", "").strip()
        return text or None
    except Exception as exc:  # pragma: no cover - depends on external service
        log.warning("Vision model call failed: %s", exc)
        return None


def _caption_fallback(ocr: Optional[OCRResult], image_path: Path) -> str:
    """Produce a description without a vision model."""
    if ocr and ocr.text:
        preview = ocr.text if len(ocr.text) <= 400 else ocr.text[:400] + "..."
        return (
            "I do not have a vision model installed, so I can only share the "
            "text I could read in the image:\n\n" + preview
        )
    return (
        f"I could not read any text in {image_path.name} and no vision model is "
        "available. Install `llava` (or any Ollama vision model) to enable image "
        "understanding."
    )


def analyze_image(
    image_path: Path | str,
    prompt: str = "Describe what is on the screen in clear, natural language.",
    force_fallback: bool = False,
) -> AnalysisResult:
    """Analyse *image_path*.

    The OCR pass is **always** run first so the user gets a textual fallback
    even if the vision model is missing. The vision model is only consulted
    when *force_fallback* is ``False`` and Ollama has a vision-capable
    model installed.
    """
    cfg = get_config()
    if not cfg.vision_enabled and not force_fallback:
        log.info("Vision disabled in config; running OCR only.")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # 1. Always try OCR (cheap, deterministic)
    try:
        ocr = extract_text(path)
    except Exception as exc:  # pragma: no cover - depends on tesseract install
        log.warning("OCR failed on %s: %s", path, exc)
        ocr = None

    # 2. Try vision model if enabled
    if not force_fallback:
        model = pick_vision_model()
        if model:
            description = _ask_vision_model(path, prompt, model)
            if description:
                # If OCR added additional structure, append it for context.
                if ocr and ocr.text and ocr.text.lower() not in description.lower():
                    description = f"{description}\n\nText I read in the image:\n{ocr.text}"
                return AnalysisResult(
                    description=description.strip(),
                    ocr=ocr,
                    used_vision_model=True,
                    model=model,
                )

    # 3. Fallback to captioning
    description = _caption_fallback(ocr, path)
    return AnalysisResult(description=description, ocr=ocr, used_vision_model=False, model=None)


__all__ = ["AnalysisResult", "analyze_image", "pick_vision_model"]
