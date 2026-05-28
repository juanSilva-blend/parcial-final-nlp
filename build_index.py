#!/usr/bin/env python3
"""Orquestación del pipeline: PDF → Markdown → chunks → embeddings → vector store.

Diseñado por fases para no agotar la RAM: primero se extrae y se generan TODOS los
chunks, luego se embeben con el embedder como único modelo cargado, y solo después
(en query/eval) se usa el LLM. Los PDFs se procesan uno a uno.

Uso:  python build_index.py
"""
from __future__ import annotations

import json
import sys
import time

from src import chunker, config, md_builder, pdf_extract, sections, tables, traceability
from src.embeddings import embed_texts
from src.models import Chunk
from src.vector_store import VectorStore


def process_pdf(pdf_path) -> tuple[list[Chunk], dict]:
    print(f"\n=== {pdf_path.name} ===")
    pages = pdf_extract.extract_pages(pdf_path)
    n_scanned = sum(1 for p in pages if p.is_scanned)
    print(f"  páginas: {len(pages)} (escaneadas: {n_scanned})")

    tables_by_page = tables.extract_tables(pdf_path)
    n_tables = sum(len(v) for v in tables_by_page.values())

    detected = sections.detect_sections(pages)
    full_sections, report = sections.validate(detected, pdf_path.name)
    print(f"  secciones detectadas: {len(report.found)}/16  "
          f"(faltan: {report.missing or 'ninguna'})")
    print(f"  tablas: {n_tables} · imágenes: {sum(len(p.images) for p in pages)}")

    traces = traceability.link_images(pages, full_sections, tables_by_page)

    md = md_builder.build_markdown(
        pages, full_sections, tables_by_page, traces,
        source_pdf=pdf_path.stem, n_found=len(report.found),
    )
    md_path = config.MD_DIR / f"{pdf_path.stem}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → {md_path.relative_to(config.ROOT)}")

    doc_chunks = chunker.chunk_document(full_sections, tables_by_page, traces, pdf_path.stem)
    print(f"  chunks: {len(doc_chunks)}")
    return doc_chunks, report.to_dict()


def main() -> int:
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No hay PDFs en {config.RAW_DIR}. Coloca las FDS de SIKA allí.")
        return 1

    t0 = time.time()
    all_chunks: list[Chunk] = []
    reports: list[dict] = []
    for pdf in pdfs:
        try:
            doc_chunks, report = process_pdf(pdf)
        except Exception as exc:  # un PDF roto no debe tumbar el lote
            print(f"  ERROR procesando {pdf.name}: {exc}")
            reports.append({"source_pdf": pdf.name, "error": str(exc)})
            continue
        all_chunks.extend(doc_chunks)
        reports.append(report)

    config.VALIDATION_REPORT.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte de validación → {config.VALIDATION_REPORT.relative_to(config.ROOT)}")

    if not all_chunks:
        print("No se generaron chunks; nada que indexar.")
        return 1

    print(f"\nEmbebiendo {len(all_chunks)} chunks con '{config.EMBED_MODEL}' …")
    vectors = embed_texts([c.text for c in all_chunks], batch_report=True)

    store = VectorStore()
    store.add(vectors, all_chunks)
    store.persist()
    print(f"Índice persistido: {len(store)} chunks → {config.INDEX_DIR.relative_to(config.ROOT)}")
    print(f"Tiempo total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
