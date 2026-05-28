#!/usr/bin/env python3
"""Evaluación del RAG contra el ground truth.

Métricas:
  - retrieval_hit_rate: ¿se recuperó la sección esperada entre las fuentes citadas?
  - gen_cosine: similitud coseno (embeddings) entre la respuesta generada y la de
    referencia (solo entradas con reference_answer no vacío).
  - llm_judge: el propio LLM local puntúa corrección [0-1] y marca alucinación
    (solo entradas con reference_answer).

Genera un reporte JSON con ejemplos comparativos esperado-vs-generado para el informe.

Uso:  python -m src.eval.evaluate
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .. import config, llm
from ..embeddings import embed_texts
from ..rag import RagPipeline

GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"
REPORT_PATH = config.INDEX_DIR / "eval_report.json"

_JUDGE_SYSTEM = (
    "Eres un evaluador imparcial. Comparas una RESPUESTA con una REFERENCIA correcta "
    "y devuelves SOLO un objeto JSON: "
    '{\"correcta\": <0.0-1.0>, \"alucinacion\": <true|false>, \"justificacion\": \"...\"}. '
    "0 = totalmente incorrecta, 1 = equivalente a la referencia."
)


def _judge(question: str, generated: str, reference: str) -> dict:
    prompt = (f"PREGUNTA: {question}\n\nREFERENCIA: {reference}\n\n"
              f"RESPUESTA: {generated}\n\nDevuelve solo el JSON.")
    raw = llm.generate(prompt, system=_JUDGE_SYSTEM)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    num = re.search(r"0?\.\d+|[01](?:\.0)?", raw)
    return {"correcta": float(num.group(0)) if num else 0.0,
            "alucinacion": None, "justificacion": raw[:200]}


def run_eval(gt_path: Path = GT_PATH) -> dict:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    questions = data["preguntas"]
    pipeline = RagPipeline()

    results = []
    hits, total_with_section = 0, 0
    cosines: list[float] = []
    judge_scores: list[float] = []

    for q in questions:
        ans = pipeline.answer(q["question"])
        retrieved_sections = [c.section_number for c in ans.citations]
        expected = q.get("expected_section")

        hit = None
        if expected is not None:
            total_with_section += 1
            hit = expected in retrieved_sections
            hits += int(hit)

        entry = {
            "id": q["id"],
            "type": q.get("type"),
            "question": q["question"],
            "expected_section": expected,
            "retrieved_sections": retrieved_sections,
            "retrieval_hit": hit,
            "generated_answer": ans.text,
            "reference_answer": q.get("reference_answer", ""),
            "citations": [c.label() for c in ans.citations],
        }

        ref = (q.get("reference_answer") or "").strip()
        if ref:
            vecs = embed_texts([ans.text, ref])
            cos = float(np.dot(vecs[0], vecs[1]))
            cosines.append(cos)
            entry["gen_cosine"] = round(cos, 4)
            judged = _judge(q["question"], ans.text, ref)
            entry["llm_judge"] = judged
            if isinstance(judged.get("correcta"), (int, float)):
                judge_scores.append(float(judged["correcta"]))

        results.append(entry)
        flag = "—" if hit is None else ("✓" if hit else "✗")
        print(f"  [{q['id']}] hit={flag}  secc.esperada={expected}  "
              f"recuperadas={retrieved_sections}")

    summary = {
        "n_questions": len(questions),
        "retrieval_hit_rate": round(hits / total_with_section, 4) if total_with_section else None,
        "n_with_reference": len(cosines),
        "mean_gen_cosine": round(float(np.mean(cosines)), 4) if cosines else None,
        "mean_llm_judge": round(float(np.mean(judge_scores)), 4) if judge_scores else None,
    }
    report = {"summary": summary, "results": results}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESUMEN ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nReporte completo → {REPORT_PATH}")
    if summary["mean_gen_cosine"] is None:
        print("\nNota: completa 'reference_answer' en ground_truth.json para obtener "
              "similitud y juicio del LLM.")
    return report


if __name__ == "__main__":
    run_eval()
