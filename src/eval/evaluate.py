#!/usr/bin/env python3
"""Evaluación del RAG contra el ground truth.

El generador Y el juez usan el proveedor configurado en config.LLM_PROVIDER
(ollama local o gemini), de modo que cada sistema se evalúa completo tal como corre.
El reporte se guarda por proveedor: data/index/eval_report_<provider>.json.

Métricas:
  - retrieval_hit_rate: ¿se recuperó la sección esperada? (igual para ambos, no
    depende del generador → mide la precisión de la recuperación).
  - gen_cosine: similitud coseno (embeddings, objetiva, independiente del juez)
    entre la respuesta generada y la de referencia → calidad comparable entre sistemas.
  - llm_judge: el LLM del propio proveedor puntúa corrección [0-1] y marca alucinación.
  - n_hallucinations, mean_gen_seconds (latencia de generación).
  - Desglose por tipo de pregunta (factual / tecnica / trazabilidad).

Uso:  LLM_PROVIDER=ollama  python -m src.eval.evaluate
      LLM_PROVIDER=gemini GEMINI_API_KEY=... python -m src.eval.evaluate
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import config, llm
from ..embeddings import embed_texts
from ..rag import _PROMPT_TMPL, _SYSTEM, RagPipeline, _format_context

GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"
GEN_PROVIDER = config.LLM_PROVIDER.lower()
GEN_MODEL = config.GEMINI_LLM_MODEL if GEN_PROVIDER == "gemini" else config.LLM_MODEL
REPORT_PATH = config.INDEX_DIR / f"eval_report_{GEN_PROVIDER}.json"

_JUDGE_SYSTEM = (
    "Eres un evaluador imparcial. Comparas una RESPUESTA con una REFERENCIA correcta "
    "y devuelves SOLO un objeto JSON: "
    '{\"correcta\": <0.0-1.0>, \"alucinacion\": <true|false>, \"justificacion\": \"...\"}. '
    "0 = totalmente incorrecta, 1 = equivalente a la referencia."
)


def _judge(question: str, generated: str, reference: str) -> dict:
    prompt = (f"PREGUNTA: {question}\n\nREFERENCIA: {reference}\n\n"
              f"RESPUESTA: {generated}\n\nDevuelve solo el JSON.")
    raw = llm.generate(prompt, system=_JUDGE_SYSTEM)  # mismo proveedor que el generador
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    num = re.search(r"0?\.\d+|[01](?:\.0)?", raw)
    return {"correcta": float(num.group(0)) if num else 0.0,
            "alucinacion": None, "justificacion": raw[:200]}


def summarize(results: list[dict]) -> dict:
    """Calcula las métricas agregadas a partir de las entradas por pregunta.

    Reutilizable: permite recomputar el resumen tras parchear un resultado sin
    volver a generar todas las respuestas (que no dependen de la referencia)."""
    hits = tot = halluc = 0
    cos: list[float] = []
    jud: list[float] = []
    lat: list[float] = []
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "hits": 0, "sec": 0, "cos": [], "judge": []})
    for r in results:
        t = r.get("type", "?")
        agg[t]["n"] += 1
        lat.append(r.get("gen_seconds") or 0.0)
        exp, hit = r.get("expected_section"), r.get("retrieval_hit")
        if exp is not None and hit is not None:
            tot += 1
            hits += int(hit)
            agg[t]["sec"] += 1
            agg[t]["hits"] += int(hit)
        if r.get("gen_cosine") is not None:
            cos.append(r["gen_cosine"])
            agg[t]["cos"].append(r["gen_cosine"])
        j = r.get("llm_judge") or {}
        c = j.get("correcta")
        if isinstance(c, (int, float)):
            jud.append(float(c))
            agg[t]["judge"].append(float(c))
        if j.get("alucinacion") is True:
            halluc += 1

    def _mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    by_type = {t: {"n": a["n"],
                   "hit_rate": round(a["hits"] / a["sec"], 4) if a["sec"] else None,
                   "mean_cosine": _mean(a["cos"]),
                   "mean_judge": _mean(a["judge"])}
               for t, a in agg.items()}
    return {
        "n_questions": len(results),
        "retrieval_hit_rate": round(hits / tot, 4) if tot else None,
        "mean_gen_cosine": _mean(cos),
        "mean_llm_judge": _mean(jud),
        "n_hallucinations": halluc,
        "mean_gen_seconds": round(float(np.mean(lat)), 2) if lat else None,
        "by_type": by_type,
    }


def run_eval(gt_path: Path = GT_PATH) -> dict:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    questions = data["preguntas"]
    pipeline = RagPipeline()

    results = []

    print(f"Evaluando con proveedor='{GEN_PROVIDER}' modelo='{GEN_MODEL}' …")
    for q in questions:
        hits_list = pipeline.retrieve(q["question"])
        retrieved = [m["section_number"] for m, _ in hits_list]
        prompt = _PROMPT_TMPL.format(context=_format_context(hits_list), question=q["question"])
        t0 = time.perf_counter()
        text = llm.generate(prompt, system=_SYSTEM)
        gen_dt = time.perf_counter() - t0

        expected = q.get("expected_section")
        typ = q.get("type", "?")
        hit = expected in retrieved if expected is not None else None

        entry = {
            "id": q["id"], "type": typ, "product": q.get("product"),
            "question": q["question"], "expected_section": expected,
            "retrieved_sections": retrieved, "retrieval_hit": hit,
            "gen_seconds": round(gen_dt, 2),
            "generated_answer": text,
            "reference_answer": q.get("reference_answer", ""),
            "citations": [f"{m['source_pdf']}, Sección {m['section_number']}, p.{m['page']}"
                          for m, _ in hits_list],
        }

        ref = (q.get("reference_answer") or "").strip()
        if ref:
            vecs = embed_texts([text, ref])
            entry["gen_cosine"] = round(float(np.dot(vecs[0], vecs[1])), 4)
            entry["llm_judge"] = _judge(q["question"], text, ref)

        results.append(entry)
        flag = "—" if hit is None else ("✓" if hit else "✗")
        print(f"  [{q['id']}] hit={flag} cos={entry.get('gen_cosine','-')} "
              f"gen={gen_dt:.1f}s")

    summary = {"gen_provider": GEN_PROVIDER, "gen_model": GEN_MODEL, **summarize(results)}
    by_type = summary["by_type"]
    report = {"summary": summary, "results": results}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESUMEN ===")
    for k, v in summary.items():
        if k != "by_type":
            print(f"  {k}: {v}")
    print("  por tipo:")
    for t, s in by_type.items():
        print(f"    {t:12} hit={s['hit_rate']} cos={s['mean_cosine']} judge={s['mean_judge']} (n={s['n']})")
    print(f"\nReporte → {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_eval()
