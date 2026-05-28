#!/usr/bin/env python3
"""Genera BORRADORES de 'reference_answer' para el ground truth usando el propio RAG.

Acelera la construcción del ground truth: rellena cada 'reference_answer' vacío con la
respuesta del RAG (marcada verified=false) y registra en 'draft_sources' las fuentes
recuperadas. El usuario debe VERIFICAR cada borrador contra el PDF (o NotebookLM) y
poner verified=true. Las entradas ya verificadas (verified=true) no se sobrescriben.

Diseñado por fases para no agotar la RAM (GPU 4 GB): primero se embeben TODAS las
preguntas (embedder cargado una vez), luego se generan las respuestas (LLM cargado una
vez). Escribe de forma incremental para no perder progreso si se interrumpe.

Uso:  .venv/bin/python scripts/generate_draft_answers.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, llm
from src.embeddings import embed_texts
from src.rag import _PROMPT_TMPL, _SYSTEM, _format_context
from src.vector_store import VectorStore

GT_PATH = Path(__file__).resolve().parent.parent / "src" / "eval" / "ground_truth.json"


def _clean(text: str) -> str:
    text = re.sub(r"\[fuente:[^\]]*\]", "", text)   # quita marcadores de cita
    return re.sub(r"\s{2,}", " ", text).strip()


def main() -> int:
    data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    pending = [q for q in data["preguntas"]
               if not q.get("verified") and not (q.get("reference_answer") or "").strip()]
    if not pending:
        print("No hay borradores pendientes (todos verificados o ya con respuesta).")
        return 0

    store = VectorStore().load()

    # Fase 1: embeber todas las preguntas con el embedder cargado una sola vez.
    print(f"Embebiendo {len(pending)} preguntas con '{config.EMBED_MODEL}' …")
    qvecs = embed_texts([q["question"] for q in pending], keep_alive=config.OLLAMA_KEEP_ALIVE)

    # Fase 2: recuperar (numpy) + generar (LLM cargado una vez), escribiendo incremental.
    print(f"Generando borradores con '{config.LLM_MODEL}' …")
    for q, qvec in zip(pending, qvecs):
        hits = store.search(qvec, section_filter=q.get("expected_section"))
        prompt = _PROMPT_TMPL.format(context=_format_context(hits), question=q["question"])
        text = llm.generate(prompt, system=_SYSTEM)
        q["reference_answer"] = _clean(text)
        q["verified"] = False
        q["draft_sources"] = [
            f"{m['source_pdf']}, Sección {m['section_number']}, p.{m['page']}" for m, _ in hits
        ]
        GT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [{q['id']}] {q['reference_answer'][:90]}…")

    print(f"\nListo: {len(pending)} borradores escritos en {GT_PATH.name}.")
    print("Revísalos contra el PDF y pon verified=true en los correctos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
