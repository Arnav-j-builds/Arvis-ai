"""
vision.ocr
~~~~~~~~~~

Optical character recognition helpers.

We support two engines and pick the one configured via
``JARVIS_OCR_ENGINE`` (default: ``tesseract``):

* ``tesseract`` - uses ``pytesseract`` (already installed by the project).
* ``easyocr``   - uses the ``easyocr`` package which downloads its model on
  first use, so we lazy-import it and surface a clean error if it is not
  installed.

The module exposes a single function, :func:`extract_text`, that returns the
plain text recognised in the image. The caller does not need to know which
engine is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from core.config import get_config
from core.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class OCRResult:
    """The result of an OCR pass."""

    text: str
    engine: str
    confidence: Optional[float] = None
    language: Optional[str] = None


def _load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    return image


def _tesseract(image_path: Path, languages: Iterable[str]) -> OCRResult:
    import pytesseract

    image = _load_image(image_path)
    lang_str = "+".join(languages) if languages else "eng"
    text = pytesseract.image_to_string(image, lang=lang_str)

    # pytesseract.image_to_data gives us per-word confidences for richer logging.
    confidence: Optional[float] = None
    try:
        data = pytesseract.image_to_data(image, lang=lang_str, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
        if confs:
            confidence = round(sum(confs) / len(confs) / 100.0, 3)
    except Exception:  # pragma: no cover - data extraction is best-effort
        confidence = None

    return OCRResult(text=text.strip(), engine="tesseract", confidence=confidence, language=lang_str)


def _easyocr(image_path: Path, languages: Iterable[str]) -> OCRResult:
    try:
        import easyocr  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "EasyOCR is not installed. Run `pip install easyocr` to use the 'easyocr' engine."
        ) from exc

    reader = easyocr.Reader(list(languages) or ["en"], gpu=False)
    results: List[List] = reader.readtext(str(image_path), detail=1)  # type: ignore[assignment]
    if not results:
        return OCRResult(text="", engine="easyocr", confidence=0.0, language=",".join(languages))

    pieces = [text for (_bbox, text, _conf) in results]
    confidences = [float(conf) for (_bbox, _text, conf) in results]
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else None
    return OCRResult(text="\n".join(pieces).strip(), engine="easyocr", confidence=avg_conf, language=",".join(languages))


def extract_text(image_path: Path | str, languages: Optional[Iterable[str]] = None) -> OCRResult:
    """Run OCR on *image_path*.

    Parameters
    ----------
    image_path:
        File to read. The image is opened with :class:`PIL.Image` so PNG,
        JPG, BMP, TIFF, etc. are all accepted.
    languages:
        Optional list of BCP-47 codes (``"en"``, ``"de"``...). Defaults to
        the value of :attr:`Config.vision_ocr_languages`.
    """
    cfg = get_config()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    engine = (cfg.vision_ocr_engine or "tesseract").lower()
    langs = list(languages) if languages else list(cfg.vision_ocr_languages)

    log.info("Running OCR engine=%s langs=%s on %s", engine, langs, path)
    if engine == "easyocr":
        return _easyocr(path, langs)
    if engine == "tesseract":
        return _tesseract(path, langs)
    raise ValueError(f"Unknown OCR engine '{engine}'. Use 'tesseract' or 'easyocr'.")


__all__ = ["OCRResult", "extract_text"]
