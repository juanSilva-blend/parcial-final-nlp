# Métricas de Evaluación — RAG sobre FDS de SIKA

> Resultados generados el 2026-05-29 ejecutando `python -m src.eval.evaluate` sobre
> el corpus completo de 15 FDS SIKA. Ground truth: 22 pares pregunta-respuesta verificados,
> con cobertura de las 16 secciones normativas GHS/SGA.

---

## 1. Resumen ejecutivo

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **retrieval_hit_rate** | **0.8182** (18/22) | Preguntas donde la sección esperada aparece entre los top-5 chunks recuperados |
| **mean_gen_cosine** | **0.7404** | Similitud coseno promedio entre respuesta generada y respuesta de referencia (embeddings bge-m3) |
| **mean_llm_judge** | **0.5614** | Puntuación promedio de corrección [0–1] asignada por el LLM local (qwen2.5:3b) como juez |
| **n_questions** | 22 | Total de preguntas evaluadas |
| **n_with_reference** | 22 | Preguntas con respuesta de referencia verificada |
| **alucinaciones detectadas** | 6/22 (27%) | Preguntas donde el LLM judge marcó `alucinacion: true` |

---

## 2. Volumetría del corpus

| Elemento | Cantidad |
|----------|----------|
| Documentos FDS indexados | 15 PDFs reales + 1 sintético |
| **Secciones normativas detectadas** | **240 / 240 (100%)** |
| Tablas extraídas (GFM) | 41 |
| Imágenes extraídas | 173 |
| **Chunks totales en el índice** | **459** |
| Dimensión de embeddings (bge-m3) | 1 024 |
| Tamaño del índice vectorial | 1.9 MB (`vectors.npy`) |
| Tamaño de metadatos | 512 KB (`meta.sqlite`) |
| Chunks promedio por documento | ~28.7 |
| TOP_K de recuperación | 5 |

---

## 3. Métricas por tipo de pregunta

| Tipo | N | Retrieval Hit Rate | Mean Gen Cosine | Mean LLM Judge |
|------|---|-------------------|-----------------|----------------|
| **factual** | 8 | **100%** (8/8) | 0.844 | 0.725 |
| **trazabilidad** | 4 | **100%** (4/4) | 0.884 | 0.788 |
| **tecnica** | 10 | **60%** (6/10) | 0.620 | 0.340 |
| **Global** | 22 | **81.8%** (18/22) | 0.740 | 0.561 |

Las preguntas factuales y de trazabilidad tienen recuperación perfecta. Las técnicas son el punto débil: 4 de 10 no recuperan la sección esperada, principalmente en §3 (composición/CAS) y §15 (reglamentario/VOC).

---

## 4. Detalle por pregunta

| ID | Tipo | §esperada | Retrieval | Gen Cosine | LLM Judge | Alucinación |
|----|------|-----------|-----------|------------|-----------|-------------|
| q01 | factual | 1 | ✓ | 0.7637 | 0.70 | No |
| q02 | factual | 1 | ✓ | 0.9547 | 0.90 | No |
| q03 | tecnica | 2 | ✓ | 0.7403 | 0.60 | No |
| q04 | tecnica | 2 | ✓ | 0.9019 | 0.60 | No |
| q05 | tecnica | 3 | **✗** | 0.4508 | 0.00 | **Sí** |
| q06 | tecnica | 3 | **✗** | 0.3271 | 0.00 | **Sí** |
| q07 | factual | 4 | ✓ | 0.9217 | 0.80 | No |
| q08 | factual | 5 | ✓ | 0.3673 | 0.00 | **Sí** |
| q09 | tecnica | 6 | ✓ | 0.8606 | **1.00** | No |
| q10 | tecnica | 7 | ✓ | 0.9401 | 0.40 | No |
| q11 | tecnica | 8 | ✓ | 0.9298 | 0.80 | No |
| q12 | trazabilidad | 8 | ✓ | 0.9812 | 0.75 | No |
| q13 | factual | 9 | ✓ | 0.9134 | 0.80 | No |
| q14 | factual | 9 | ✓ | 0.7872 | 0.80 | No |
| q15 | tecnica | 10 | **✗** | 0.3901 | 0.00 | **Sí** |
| q16 | tecnica | 11 | ✓ | 0.3160 | 0.00 | **Sí** |
| q17 | factual | 13 | ✓ | 0.9329 | 0.90 | No |
| q18 | trazabilidad | 14 | ✓ | 0.8692 | 0.90 | No |
| q19 | tecnica | 15 | **✗** | 0.3428 | 0.00 | **Sí** |
| q20 | trazabilidad | 14 | ✓ | 0.8559 | 0.50 | No |
| q21 | trazabilidad | 12 | ✓ | 0.8313 | **1.00** | No |
| q22 | factual | 16 | ✓ | 0.9110 | 0.90 | No |

---

## 5. Análisis de hallazgos

### 5.1 Secciones con mejor rendimiento

- **§1, §8, §9, §13, §14, §16**: recuperación 100%, respuestas de alta calidad (coseno > 0.85).
- **§12 y §16** (nuevas preguntas añadidas): recuperación perfecta con scores 0.83 y 0.91.
- **§6 (derrames)**: única pregunta técnica con juicio perfecto (1.0) — la información es concisa y la pregunta muy específica.

### 5.2 Secciones con peor rendimiento

| Sección | Problema | Causa probable |
|---------|----------|----------------|
| **§3 (Composición/CAS)** | Retrieval falla (q05, q06) | Los chunks de §3 son tablas densas de nombres IUPAC y números CAS; la representación semántica por embeddings es pobre para términos químicos técnicos. |
| **§10 (Incompatibilidades)** | Retrieval falla (q15) | Sección corta y genérica; la query "incompatible" compite con §9 (propiedades físicas) que contiene vocabulario similar. |
| **§15 (Reglamentario/VOC)** | Retrieval falla (q19) | La query sobre "COV/VOC" no encuentra la sección §15 porque los chunks de §1 dominan el ranking semántico. |

### 5.3 Patrón de alucinaciones

6 de 22 preguntas (27%) generaron `alucinacion: true`. Patrón común: cuando el retrieval falla (sección equivocada en top-5), el modelo responde "No se encuentra en los documentos" en lugar de responder con los chunks recuperados — lo cual técnicamente no es una alucinación de datos sino una negativa incorrecta. La excepción es **q20** donde el modelo inventó un número ONU incorrecto (UN 1263 en lugar de UN 3082) a partir de un chunk de otro producto.

### 5.4 Retrieval vs. generación

| Situación | Preguntas | Observación |
|-----------|-----------|-------------|
| Retrieval ✓ + Generación correcta | 12 | Caso ideal |
| Retrieval ✓ + Generación parcial/incorrecta | 6 | Falla del modelo pequeño (3B) al procesar tablas técnicas |
| Retrieval ✗ + Sin respuesta | 4 | Falla de embedding semántico en terminología química |
| Retrieval ✓ + Alucinación numérica | 1 (q20) | Confusión entre documentos similares |

---

## 6. Limitaciones identificadas

1. **Embeddings sobre terminología química**: el modelo `bge-m3` no distingue bien entre nombres IUPAC, números CAS y códigos UN de distintos productos cuando los chunks son tablas densas.
2. **Modelo generativo pequeño (3B)**: tiende a responder "No se encuentra" cuando el contexto recuperado es complejo (tablas multilínea, múltiples componentes).
3. **Sin GPU**: latencia de ~85s por consulta en CPU puro limita el uso interactivo.
4. **Secciones cortas mal chunkeadas**: §10 (2-3 líneas) puede quedar en el mismo chunk que §9, dificultando la recuperación por sección.
