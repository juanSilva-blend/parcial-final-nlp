"""Chunking section-aware con metadata de trazabilidad.

Estrategia (documentada y justificable):
  - La frontera primaria de fragmentación son las 16 secciones normativas, que son
    límites semánticos naturales y trazables.
  - Las secciones largas se sub-dividen por párrafos hasta ~CHUNK_SIZE caracteres con
    un solape de CHUNK_OVERLAP para no perder contexto en los bordes.
  - Cada tabla se conserva intacta como su propio chunk (nunca se parte una tabla).
  - El texto OCR de cada imagen se indexa como chunk aparte, asociado a su sección.
Cada chunk lleva metadata {source_pdf, section_number, section_title, page,
has_table, image_refs} que permite citar la fuente exacta en cada respuesta.
"""
from __future__ import annotations

from . import config
from .models import Chunk, ImageTrace, SectionSpan, TableRef


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + p).strip()
        else:
            cur = (cur + "\n" + p).strip() if cur else p
        while len(cur) > size:
            chunks.append(cur[:size])
            cur = cur[size - overlap:] if overlap else cur[size:]
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if len(c) >= config.MIN_CHUNK_SIZE] or ([text] if text.strip() else [])


def _section_for_table(table: TableRef, spans: list[SectionSpan]) -> SectionSpan | None:
    best: SectionSpan | None = None
    for sec in spans:
        if not sec.found:
            continue
        if sec.page < table.page or (sec.page == table.page and sec.header_bbox[1] <= table.bbox[1]):
            best = sec
    return best


def chunk_document(spans: list[SectionSpan], tables_by_page: dict[int, list[TableRef]],
                   traces: dict[str, ImageTrace], source_pdf: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    # 1. Texto de cada sección encontrada.
    for sec in spans:
        if not sec.found or not sec.text.strip():
            continue
        pieces = _split_text(sec.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            header = f"Sección {sec.number}. {sec.title}\n"
            chunks.append(Chunk(
                chunk_id=f"{source_pdf}::s{sec.number:02d}::c{i:03d}",
                text=header + piece,
                source_pdf=source_pdf,
                section_number=sec.number,
                section_title=sec.title,
                page=sec.page + 1,
            ))

    # 2. Tablas intactas, asociadas a su sección.
    for tbl_list in tables_by_page.values():
        for tbl in tbl_list:
            sec = _section_for_table(tbl, spans)
            chunks.append(Chunk(
                chunk_id=f"{source_pdf}::{tbl.label.replace(' ', '')}",
                text=(f"Sección {sec.number}. {sec.title}\n" if sec else "") +
                     f"{tbl.label}\n{tbl.markdown}",
                source_pdf=source_pdf,
                section_number=sec.number if sec else None,
                section_title=sec.title if sec else "",
                page=tbl.page + 1,
                has_table=True,
            ))

    # 3. Texto OCR de imágenes relevantes.
    for path, trace in traces.items():
        ocr_text = trace.image.ocr_text.strip()
        if len(ocr_text) < config.MIN_CHUNK_SIZE:
            continue
        chunks.append(Chunk(
            chunk_id=f"{source_pdf}::img::{path.split('/')[-1]}",
            text=f"[Imagen, Sección {trace.section_number}] {ocr_text}",
            source_pdf=source_pdf,
            section_number=trace.section_number,
            section_title=trace.section_title,
            page=trace.image.page + 1,
            image_refs=[path],
        ))

    return chunks
