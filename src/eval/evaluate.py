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


def run_eval(gt_path: Path = GT_PATH) -> dict:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    questions = data["preguntas"]
    pipeline = RagPipeline()

    results = []
    hits = tot = halluc = 0
    cosines: list[float] = []
    judge_scores: list[float] = []
    latencies: list[float] = []
    agg: dict[str, dict] = defaultdict(lambda: {"n": 0, "hits": 0, "sec": 0, "cos": [], "judge": []})

    print(f"Evaluando con proveedor='{GEN_PROVIDER}' modelo='{GEN_MODEL}' …")
    for q in questions:
        hits_list = pipeline.retrieve(q["question"])
        retrieved = [m["section_number"] for m, _ in hits_list]
        prompt = _PROMPT_TMPL.format(context=_format_context(hits_list), question=q["question"])
        t0 = time.perf_counter()
        text = llm.generate(prompt, system=_SYSTEM)
        gen_dt = time.perf_counter() - t0
        latencies.append(gen_dt)

        expected = q.get("expected_section")
        typ = q.get("type", "?")
        agg[typ]["n"] += 1
        hit = None
        if expected is not None:
            tot += 1
            hit = expected in retrieved
            hits += int(hit)
            agg[typ]["sec"] += 1
            agg[typ]["hits"] += int(hit)

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
            cos = float(np.dot(vecs[0], vecs[1]))
            cosines.append(cos)
            agg[typ]["cos"].append(cos)
            entry["gen_cosine"] = round(cos, 4)
            judged = _judge(q["question"], text, ref)
            entry["llm_judge"] = judged
            c = judged.get("correcta")
            if isinstance(c, (int, float)):
                judge_scores.append(float(c))
                agg[typ]["judge"].append(float(c))
            if judged.get("alucinacion") is True:
                halluc += 1

        results.append(entry)
        flag = "—" if hit is None else ("✓" if hit else "✗")
        print(f"  [{q['id']}] hit={flag} cos={entry.get('gen_cosine','-')} "
              f"gen={gen_dt:.1f}s")

    def _mean(xs):
        return round(float(np.mean(xs)), 4) if xs else None

    by_type = {t: {"n": a["n"],
                   "hit_rate": round(a["hits"] / a["sec"], 4) if a["sec"] else None,
                   "mean_cosine": _mean(a["cos"]),
                   "mean_judge": _mean(a["judge"])}
               for t, a in agg.items()}

    summary = {
        "gen_provider": GEN_PROVIDER,
        "gen_model": GEN_MODEL,
        "n_questions": len(questions),
        "retrieval_hit_rate": round(hits / tot, 4) if tot else None,
        "mean_gen_cosine": _mean(cosines),
        "mean_llm_judge": _mean(judge_scores),
        "n_hallucinations": halluc,
        "mean_gen_seconds": round(float(np.mean(latencies)), 2) if latencies else None,
        "by_type": by_type,
    }
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
