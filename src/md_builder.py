"""Ensamblaje del Markdown fiel a partir de las estructuras extraídas.

Por cada página se intercalan, en orden de lectura, los encabezados de sección, los
párrafos y listas, las tablas (renderizadas como tablas GFM) y las imágenes (con su
bloque "Nota de trazabilidad"). Las líneas que caen dentro del bbox de una tabla se
omiten para no duplicar contenido. Salida en UTF-8 normalizado (NFC).
"""
from __future__ import annotations

import re
import unicodedata

from .models import ImageTrace, Line, Page, SectionSpan, TableRef

_BULLET_RE = re.compile(r"^\s*([•·▪◦\-–\*]|\d+[\.\)])\s+")


def _is_inside(line: Line, table: TableRef, pad: float = 2.0) -> bool:
    lx0, ly0, lx1, ly1 = line.bbox
    tx0, ty0, tx1, ty1 = table.bbox
    cy = (ly0 + ly1) / 2
    return (ty0 - pad) <= cy <= (ty1 + pad) and (tx0 - pad) <= lx0 <= (tx1 + pad)


def _render_line(line: Line, header_map: dict) -> str:
    key = (line.page, line.bbox)
    if key in header_map:
        span: SectionSpan = header_map[key]
        return f"\n## Sección {span.number} — {span.title}\n"
    if _BULLET_RE.match(line.text):
        return "- " + _BULLET_RE.sub("", line.text, count=1)
    return line.text


def _render_table(table: TableRef) -> str:
    head = f"\n**{table.label}**\n\n" if table.label else "\n"
    return head + table.markdown + "\n"


def _render_image(trace: ImageTrace) -> str:
    img = trace.image
    sec = (f"{trace.section_number} — {trace.section_title}"
           if trace.section_number else "no asociada con certeza")
    lines = [
        f"\n![Imagen {img.path.split('/')[-1]}]({img.path})\n",
        "> **Nota de trazabilidad**",
        f"> - Sección asociada: {sec}",
        f"> - Página: {img.page + 1}",
    ]
    if trace.table_label:
        lines.append(f"> - Tabla relacionada: {trace.table_label}")
    if trace.text_refs:
        lines.append(f"> - Referencias en el texto: {', '.join(trace.text_refs)}")
    if img.ocr_text:
        snippet = img.ocr_text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"> - OCR (spa): \"{snippet}\"")
    lines.append("")
    return "\n".join(lines)


def build_markdown(pages: list[Page], sections: list[SectionSpan],
                   tables_by_page: dict[int, list[TableRef]],
                   traces: dict[str, ImageTrace], source_pdf: str,
                   n_found: int) -> str:
    header_map = {(s.page, s.header_bbox): s for s in sections if s.found}

    title = source_pdf.replace("_", " ")
    parts: list[str] = [
        f"# Ficha de Datos de Seguridad — {title}",
        "",
        f"<!-- Generado por el pipeline RAG-FDS. Secciones detectadas: {n_found}/16. "
        f"Documento fuente: {source_pdf} -->",
        "",
    ]

    for pg in pages:
        parts.append(f"<!-- página {pg.number + 1} -->")
        page_tables = tables_by_page.get(pg.number, [])

        # Elementos posicionables: (y0, x0, tipo, payload).
        elems: list[tuple[float, float, str, object]] = []
        for ln in pg.lines:
            if any(_is_inside(ln, t) for t in page_tables):
                continue  # ya forma parte de una tabla
            elems.append((ln.bbox[1], ln.bbox[0], "line", ln))
        for t in page_tables:
            elems.append((t.bbox[1], t.bbox[0], "table", t))
        for img in pg.images:
            elems.append((img.bbox[1], img.bbox[0], "image", img))

        elems.sort(key=lambda e: (round(e[0], 1), e[1]))

        for _, _, kind, payload in elems:
            if kind == "line":
                parts.append(_render_line(payload, header_map))
            elif kind == "table":
                parts.append(_render_table(payload))
            elif kind == "image":
                trace = traces.get(payload.path)
                if trace:
                    parts.append(_render_image(trace))
        parts.append("")

    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)  # colapsar líneas en blanco excesivas
    return unicodedata.normalize("NFC", text).strip() + "\n"
