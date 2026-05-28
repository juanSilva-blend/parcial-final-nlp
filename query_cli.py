#!/usr/bin/env python3
"""Consulta el RAG desde la terminal, mostrando respuesta + fuentes.

Uso:
  python query_cli.py "¿Qué EPP se requiere?"
  python query_cli.py                # modo interactivo
  python query_cli.py --section 8 "¿..."   # filtrar por sección
"""
from __future__ import annotations

import argparse
import sys

from src.rag import Answer, RagPipeline


def _print_answer(ans: Answer) -> None:
    print("\n" + "=" * 70)
    print("RESPUESTA:\n")
    print(ans.text)
    print("\n" + "-" * 70)
    print("FUENTES RECUPERADAS (trazabilidad):")
    for i, c in enumerate(ans.citations, 1):
        preview = c.text.replace("\n", " ")[:110]
        print(f"  [{i}] {c.label()}  (sim={c.score:.3f})")
        print(f"      {preview}…")
    print("=" * 70 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Consulta RAG sobre FDS de SIKA")
    ap.add_argument("question", nargs="*", help="pregunta (vacío = interactivo)")
    ap.add_argument("--section", type=int, default=None,
                    help="filtrar la recuperación a una sección (1-16)")
    ap.add_argument("-k", type=int, default=None, help="nº de fragmentos a recuperar")
    args = ap.parse_args()

    pipeline = RagPipeline()
    kwargs = {}
    if args.k:
        kwargs["k"] = args.k
    if args.section:
        kwargs["section_filter"] = args.section

    if args.question:
        _print_answer(pipeline.answer(" ".join(args.question), **kwargs))
        return 0

    print("Modo interactivo. Escribe tu pregunta (Ctrl-D o 'salir' para terminar).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"salir", "exit", "quit"}:
            break
        _print_answer(pipeline.answer(q, **kwargs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
