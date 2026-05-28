"""Extracción estructural de PDFs con PyMuPDF.

Produce, por página: líneas de texto con geometría y señales de formato (tamaño,
negrita), imágenes embebidas (guardadas en disco + OCR), y una bandera de
"escaneado". El texto se concatena en orden de lectura (columnas → y → x) para que
la detección de secciones reciba el contenido bien ordenado.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from . import config, ocr
from .models import ImageRef, Line, Page

# Bit de negrita en span["flags"] de PyMuPDF.
_BOLD_FLAG = 1 << 4
# Imágenes más pequeñas que esto (px) se consideran decorativas y se ignoran.
_MIN_IMG_DIM = 60


def _is_bold(span: dict) -> bool:
    if span.get("flags", 0) & _BOLD_FLAG:
        return True
    return "bold" in span.get("font", "").lower()


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _reading_order(lines: list[Line], page_width: float) -> list[Line]:
    """Ordena líneas respetando columnas. Agrupa por x0 si hay un hueco claro."""
    if not lines:
        return lines
    xs = sorted(ln.bbox[0] for ln in lines)
    # Detecta un corte de columnas: hueco grande entre x0 consecutivos.
    split = None
    for a, b in zip(xs, xs[1:]):
        if b - a > page_width * 0.18 and b > page_width * 0.4:
            split = (a + b) / 2
            break

    def key(ln: Line) -> tuple:
        col = 0 if (split is None or ln.bbox[0] < split) else 1
        return (col, round(ln.bbox[1], 1), ln.bbox[0])

    return sorted(lines, key=key)


def _extract_lines(page: "fitz.Page", page_no: int) -> list[Line]:
    data = page.get_text("dict")
    lines: list[Line] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 = bloque de texto
            continue
        for ln in block.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = _norm("".join(s.get("text", "") for s in spans)).strip()
            if not text:
                continue
            size = max((s.get("size", 0.0) for s in spans), default=0.0)
            bold = any(_is_bold(s) for s in spans)
            x0, y0, x1, y1 = ln["bbox"]
            lines.append(Line(text=text, bbox=(x0, y0, x1, y1),
                              size=size, bold=bold, page=page_no))
    return lines


def _extract_images(doc, page: "fitz.Page", page_no: int,
                    stem: str, assets_dir: Path) -> list[ImageRef]:
    images: list[ImageRef] = []
    seen: set[int] = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        if info.get("width", 0) < _MIN_IMG_DIM or info.get("height", 0) < _MIN_IMG_DIM:
            continue
        rects = page.get_image_rects(xref)
        bbox = tuple(rects[0]) if rects else (0.0, 0.0, 0.0, 0.0)
        ext = info.get("ext", "png")
        fname = f"{stem}_p{page_no + 1}_img{len(images) + 1}.{ext}"
        abs_path = assets_dir / fname
        img_bytes = info["image"]
        abs_path.write_bytes(img_bytes)
        ref = ImageRef(
            path=f"assets/{fname}",
            abs_path=str(abs_path),
            bbox=bbox,
            page=page_no,
            xref=xref,
            ocr_text=ocr.ocr_image_bytes(img_bytes),
        )
        images.append(ref)
    return images


def extract_pages(pdf_path: Path, assets_dir: Path | None = None) -> list[Page]:
    """Extrae todas las páginas de un PDF como objetos Page estructurados."""
    assets_dir = assets_dir or config.ASSETS_DIR
    assets_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem.replace(" ", "_")

    pages: list[Page] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            lines = _reading_order(_extract_lines(page, i), page.rect.width)
            images = _extract_images(doc, page, i, stem, assets_dir)
            raw_text = "\n".join(ln.text for ln in lines)

            is_scanned = len(raw_text.strip()) < config.SCANNED_TEXT_THRESHOLD
            if is_scanned:
                raw_text = _ocr_full_page(page)

            pages.append(Page(
                number=i,
                lines=lines,
                images=images,
                width=page.rect.width,
                height=page.rect.height,
                is_scanned=is_scanned,
                raw_text=raw_text,
            ))
    finally:
        doc.close()  # liberar memoria entre PDFs (RAM limitada)
    return pages


def _ocr_full_page(page: "fitz.Page") -> str:
    """Rasteriza una página escaneada y la pasa por OCR."""
    try:
        pix = page.get_pixmap(dpi=200)
        return ocr.ocr_image_bytes(pix.tobytes("png"))
    except Exception as exc:
        print(f"[pdf_extract] OCR de página completa falló: {exc}")
        return ""
