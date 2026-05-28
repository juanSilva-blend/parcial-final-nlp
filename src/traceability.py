"""Asociación de cada imagen con su contexto estructural (sección / tabla).

Heurística por proximidad espacial + referencias textuales:
  1. Sección contenedora = encabezado de sección más cercano por encima de la
     imagen (mismo lado horizontal), respetando el orden de páginas.
  2. Tabla relacionada = tabla en la misma página a poca distancia vertical y con
     solape horizontal.
  3. Referencias textuales ("ver Tabla 4", "véase la figura 2") halladas en el
     texto de la sección.
El resultado alimenta el bloque "Nota de trazabilidad" que md_builder escribe bajo
cada imagen.
"""
from __future__ import annotations

import re

from .models import ImageRef, ImageTrace, Page, SectionSpan, TableRef

_REF_RE = re.compile(
    r"(?:ver|v[ée]ase|consult\w*|seg[uú]n)\s+(?:la\s+|el\s+)?"
    r"(tabla|figura|secci[óo]n)\s+(\d+)",
    re.IGNORECASE,
)


def _x_overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0])


def _nearest_section(img: ImageRef, sections: list[SectionSpan]) -> SectionSpan | None:
    """Sección cuyo encabezado precede a la imagen más de cerca."""
    best: SectionSpan | None = None
    for sec in sections:
        if not sec.found:
            continue
        # En páginas anteriores: candidata. En la misma página: solo si está arriba.
        if sec.page < img.page:
            best = sec
        elif sec.page == img.page and sec.header_bbox[1] <= img.bbox[1]:
            best = sec
    return best


def _nearest_table(img: ImageRef, tables: list[TableRef]) -> TableRef | None:
    candidates = [t for t in tables if t.page == img.page and _x_overlap(t.bbox, img.bbox)]
    if not candidates:
        return None
    # Menor distancia vertical entre los bordes de imagen y tabla.
    def gap(t: TableRef) -> float:
        return min(abs(t.bbox[1] - img.bbox[3]), abs(img.bbox[1] - t.bbox[3]))
    best = min(candidates, key=gap)
    from . import config
    return best if gap(best) <= config.IMG_TABLE_MAX_GAP_PT else None


def _text_refs(section_text: str, ocr_text: str) -> list[str]:
    refs: list[str] = []
    for blob in (section_text, ocr_text):
        for kind, num in _REF_RE.findall(blob or ""):
            label = f"{kind.capitalize()} {num}"
            if label not in refs:
                refs.append(label)
    return refs


def link_images(pages: list[Page], sections: list[SectionSpan],
                tables_by_page: dict[int, list[TableRef]]) -> dict[str, ImageTrace]:
    """Devuelve {image.path: ImageTrace} para todas las imágenes del documento."""
    all_tables = [t for tl in tables_by_page.values() for t in tl]
    traces: dict[str, ImageTrace] = {}
    for pg in pages:
        for img in pg.images:
            sec = _nearest_section(img, sections)
            tbl = _nearest_table(img, all_tables)
            sec_text = sec.text if sec else ""
            traces[img.path] = ImageTrace(
                image=img,
                section_number=sec.number if sec else None,
                section_title=sec.title if sec else "",
                table_label=tbl.label if tbl else "",
                text_refs=_text_refs(sec_text, img.ocr_text),
            )
    return traces
