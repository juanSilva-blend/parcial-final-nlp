"""Extracción de tablas con pdfplumber → Markdown.

Se prueban dos estrategias de detección (basada en líneas y basada en texto) y se
conserva la que produzca la tabla más consistente. Las tablas se renderizan como
tablas GFM; si una tabla no parsea limpia, el llamador puede conservarla como bloque
de texto (nunca se desplazan columnas).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pdfplumber

from . import config
from .models import TableRef

_LINE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
_TEXT_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text"}


def _clean_cell(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = text.replace("\n", " ").replace("|", "\\|").strip()
    return text


def _rows_to_markdown(rows: list[list]) -> str:
    rows = [r for r in rows if any(_clean_cell(c) for c in r)]
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    norm = [[_clean_cell(c) for c in r] + [""] * (ncols - len(r)) for r in rows]
    header = norm[0]
    body = norm[1:]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * ncols) + " |"]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _score(rows: list[list]) -> int:
    """Calidad de una tabla candidata: nº de celdas no vacías."""
    if not rows:
        return 0
    return sum(1 for r in rows for c in r if c not in (None, ""))


def _looks_tabular(rows: list[list]) -> bool:
    """Filtro anti-prosa para candidatas de la estrategia 'text'.

    La estrategia basada en texto puede fabricar tablas falsas a partir de párrafos
    (cortando palabras en columnas). Una tabla de datos real tiene celdas cortas
    (números, etiquetas), no oraciones; aquí se rechaza lo que parece prosa.
    """
    cleaned = [[_clean_cell(c) for c in r] for r in rows]
    cleaned = [r for r in cleaned if any(r)]
    if len(cleaned) < 2 or max(len(r) for r in cleaned) < 2:
        return False
    nonempty = [c for r in cleaned for c in r if c]
    if not nonempty:
        return False
    avg_words = sum(len(c.split()) for c in nonempty) / len(nonempty)
    return avg_words <= 3.0 and max(len(c) for c in nonempty) <= 90


def extract_tables(pdf_path: Path) -> dict[int, list[TableRef]]:
    """Devuelve {page_index: [TableRef, ...]} con etiquetas 'Tabla N' secuenciales."""
    result: dict[int, list[TableRef]] = {}
    counter = 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables: list[TableRef] = []
            # Tablas con bordes (estrategia 'lines'): se confían como tablas reales.
            # Si no hay, se intenta 'text' pero con filtro anti-prosa estricto.
            try:
                found_lines = page.find_tables(table_settings=_LINE_SETTINGS)
                if found_lines:
                    candidates = [(t, False) for t in found_lines]
                elif config.TABLE_TEXT_FALLBACK:
                    candidates = [(t, True) for t in
                                  page.find_tables(table_settings=_TEXT_SETTINGS)]
                else:
                    candidates = []
            except Exception:
                candidates = []

            for tbl, strict in candidates:
                try:
                    rows = tbl.extract()
                except Exception:
                    continue
                if _score(rows) < 2:  # ruido: ignorar
                    continue
                if strict and not _looks_tabular(rows):  # prosa disfrazada de tabla
                    continue
                md = _rows_to_markdown(rows)
                if not md:
                    continue
                counter += 1
                x0, y0, x1, y1 = tbl.bbox
                page_tables.append(TableRef(
                    markdown=md,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    page=i,
                    n_rows=len(rows),
                    n_cols=max((len(r) for r in rows), default=0),
                    label=f"Tabla {counter}",
                ))
            if page_tables:
                result[i] = page_tables
    return result
