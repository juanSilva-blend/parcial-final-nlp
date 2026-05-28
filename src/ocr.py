"""OCR local con Tesseract (paquete de español).

Se usa para (a) imágenes embebidas relevantes y (b) páginas escaneadas sin texto
nativo. Si Tesseract o el paquete 'spa' no están instalados, se degrada con
elegancia devolviendo cadena vacía y avisando una sola vez.
"""
from __future__ import annotations

import io

from . import config

_TESSERACT_OK: bool | None = None


def _check_tesseract() -> bool:
    global _TESSERACT_OK
    if _TESSERACT_OK is not None:
        return _TESSERACT_OK
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_OK = True
    except Exception as exc:  # binario ausente o sin permisos
        print(f"[ocr] Tesseract no disponible ({exc}). El OCR quedará vacío.")
        _TESSERACT_OK = False
    return _TESSERACT_OK


def ocr_image_bytes(img_bytes: bytes, lang: str = config.OCR_LANG) -> str:
    """OCR sobre los bytes de una imagen. Devuelve texto limpio (puede ser '')."""
    if not _check_tesseract():
        return ""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(img_bytes)) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            text = pytesseract.image_to_string(im, lang=lang)
        return _clean(text)
    except Exception as exc:
        print(f"[ocr] fallo al procesar imagen: {exc}")
        return ""


def ocr_image_path(path: str, lang: str = config.OCR_LANG) -> str:
    try:
        with open(path, "rb") as fh:
            return ocr_image_bytes(fh.read(), lang=lang)
    except OSError:
        return ""


def _clean(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()
