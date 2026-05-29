# Informe técnico — RAG sobre FDS de SIKA

> Informe corto del proyecto. Las secciones de **resultados** se completan tras
> ejecutar `python build_index.py` y `python -m src.eval.evaluate` sobre los PDFs
> reales (los números provienen de `data/index/validation_report.json` y
> `data/index/eval_report.json`).

## 1. Objetivo

Construir un sistema RAG reproducible y de bajo costo para consultar Fichas de Datos
de Seguridad de SIKA, garantizando la fidelidad de la extracción a Markdown (16
secciones normativas) y la trazabilidad entre cada respuesta y su fragmento fuente,
sin depender de APIs pagas ni servicios externos.

## 2. Decisiones técnicas

| Decisión | Alternativa descartada | Justificación |
|----------|------------------------|---------------|
| Extracción híbrida PyMuPDF + pdfplumber + Tesseract | Docling / Marker | Más liviano, reproducible y sin descargar modelos pesados; la FDS digital tiene texto nativo regular. |
| Ollama por HTTP (sin torch) | LangChain / LlamaIndex + PyTorch | Elimina la dependencia más pesada; mantiene todo local. |
| Vector store numpy + SQLite | ChromaDB / FAISS | Corpus pequeño → coseno brute-force sub-ms; ~6-8 dependencias menos. |
| Indexar y consultar en fases | Cargar todo a la vez | Evita que embedder y LLM coexistan en ~3 GB de RAM libres. |
| Chunking section-aware | Chunking fijo por tokens | Fronteras semánticas naturales y trazabilidad por sección. |
| `bge-m3` + `qwen2.5:3b` (Q4) | Modelos 7B+ / fp16 | Caben en 4 GB de VRAM; buen español; bajo costo. |

## 3. Estrategia de chunking

- Partición primaria por las 16 secciones; sub-división por párrafos (~1000 chars,
  solape ~150) en secciones largas; tablas e imágenes (OCR) como chunks propios.
- Cada chunk conserva su sección, página y archivo fuente → toda respuesta es
  rastreable.

## 4. Trazabilidad

- **Documental:** cada imagen del `.md` lleva una "Nota de trazabilidad" (sección,
  página, tabla relacionada, referencias textuales, OCR).
- **De respuesta:** el RAG cita `[fuente: <pdf>, Sección N, p.X]` y se muestran los
  fragmentos recuperados con su similitud.

## 5. Resultados de extracción

Corpus: **15 FDS de SIKA** (data/raw/). Detalle por documento (de
`data/index/validation_report.json` y los `.md` generados):

| Documento | Secciones | Tablas | Imágenes | Chunks |
|-----------|-----------|--------|----------|--------|
| Esmalte Uretano AR Comp. B | 16/16 | 3 | 5 | 27 |
| FDS 20 - Esmalte Alquídico Serie 31 | 16/16 | 2 | 5 | 32 |
| FDS 22 - Esmalte Uretano AR Comp. B | 16/16 | 3 | 5 | 27 |
| FDS 26 - Esmalte Epóxico Aluminio | 16/16 | 5 | 6 | 35 |
| FDS 27 - Epoxi 100HS S300 CA | 16/16 | 2 | 5 | 23 |
| FDS 28 - 401 Pintura Texturizada | 16/16 | 3 | 2 | 26 |
| FDS 36 - Esmalte Uretano AR | 16/16 | 3 | 7 | 36 |
| FDS 69 - Esmalte Uretano Part A | 16/16 | 3 | 18 | 35 |
| FDS 70 - Esmalte Uretano Part B | 16/16 | 3 | 18 | 35 |
| FDS 71 - Sikafloor-2430 Part A | 16/16 | 3 | 19 | 32 |
| FDS 72 - Sikafloor-2430 Comp. B | 16/16 | 3 | 18 | 32 |
| FDS 73 - Sikafloor-161 Part A | 16/16 | 2 | 15 | 29 |
| FDS 74 - Sikafloor-161/264 Part B | 16/16 | 2 | 18 | 33 |
| FDS 86 - Sikafloor-510 Part A | 16/16 | 1 | 14 | 25 |
| FDS 87 - Esmalte Epóxico Part A | 16/16 | 3 | 18 | 32 |
| **TOTAL (15 docs)** | **240/240** | **41** | **173** | **459** |

- **Detección de secciones: 100% (16/16 en los 15 documentos), todas en orden.**
  Las FDS de SIKA usan encabezados regulares ("SECCIÓN N:" / "SECCION N:"), que la
  estrategia de dos pasadas reconoce con fiabilidad. Ningún documento resultó escaneado.
- **Tablas (41):** las tablas con bordes (composición §3, límites de exposición §8,
  transporte §14) se reconstruyen como tablas GFM. La numeración `Tabla N` es secuencial
  del pipeline.
- **Imágenes (173):** mayormente pictogramas SGA y diagramas NFPA/HMIS. Cada una lleva su
  "Nota de trazabilidad". Las imágenes que aparecen antes de la Sección 1 (logo Sika del
  encabezado) se marcan correctamente como "no asociada con certeza".

## 6. Resultados del RAG

Evaluación ejecutada sobre 22 pares pregunta-respuesta verificados que cubren las 16
secciones normativas. Modelo generativo: `qwen2.5:3b-instruct` (Q4, CPU). Embeddings:
`bge-m3` (1024-dim). Resultados de `data/index/eval_report.json`.

| Métrica | Valor |
|---------|-------|
| **retrieval_hit_rate** | **0.8182** (18/22) |
| **mean_gen_cosine** | **0.7404** |
| **mean_llm_judge** | **0.5614** |
| Alucinaciones detectadas | 6/22 (27%) |

Por tipo de pregunta: factual 100% hit / juicio 0.73 · trazabilidad 100% hit / juicio 0.79 ·
técnica 60% hit / juicio 0.34. Las secciones §3 (composición/CAS), §10 (incompatibilidades) y
§15 (COV/reglamentario) presentan fallos de recuperación por vocabulario técnico especializado.

### Ejemplos comparativos esperado vs. generado

| Pregunta | Respuesta esperada (referencia) | Respuesta del RAG | Fuente citada | Observación |
|----------|---------------------------------|-------------------|---------------|-------------|
| ¿Qué métodos de contención ante derrame del Sikafloor-2430 Part B? (q09) | Material absorbente no combustible: arena, tierra, diatomeas o vermiculita. | Material absorbente no combustible: arena, tierra, barro de diatomeas, vermiculita. | FDS 72 · Sección 6 · p.4 | **Correcta** — juicio 1.0, coseno 0.86 |
| ¿CL50 para peces del fumarato de dietilo en Sikafloor-510 Part A? (q21) | 2,4 mg/l, exposición 72 h. Sin datos de persistencia/bioacumulación. | La CL50 (Pez) es 2,4 mg/l. | FDS 86 · Sección 12 · p.7 | **Correcta** — juicio 1.0, coseno 0.83 |
| ¿Cómo eliminar residuos del Sikafloor-161 Part A? (q17) | Empresa autorizada; no contaminar desagües/suelos; vaciar envases; no reutilizar. | Empresa autorizada; no contaminar estanques/cursos; vaciar envases; no quemar. | FDS 73 · Sección 13 · p.8 | **Parcial** — juicio 0.9, omite desagües |
| ¿Cuáles son los componentes y CAS del Esmalte Uretano AR Comp. B? (q06) | Acetato de butilo (CAS 123-86-4), diisocianato hexametileno (CAS 822-06-0), etc. | "No se encuentra en los documentos." | Sección 1 y 9 (error de recuperación) | **Fallo** — retrieval §3 no recuperado; juicio 0.0 |
| ¿Número ONU y clase de transporte del Esmalte Epóxico Aluminio? (q20) | UN 3082, clase 9, grupo embalaje III. | "UN 1263, clase 3." | FDS 26 · Sección 14 · p.10 | **Alucinación numérica** — confusión con otro producto; juicio 0.5 |

### 6.1 Comparación de generadores: Ollama local vs Gemini API

Se evaluó el **mismo** sistema RAG (idéntica recuperación, embeddings `bge-m3` y los 22
pares del ground truth) cambiando solo el **generador** (y su juez): `qwen2.5:3b` local
en GPU vs `gemini-2.5-flash` por API. Reportes en `data/index/eval_report_ollama.json`
y `eval_report_gemini.json`.

| Métrica | Ollama `qwen2.5:3b` (local, GPU) | Gemini `2.5-flash` (API) |
|---------|----------------------------------|--------------------------|
| Hit-rate de recuperación | 0.8182 | 0.8182 |
| **Coseno gen-vs-ref (objetivo)** | 0.717 | **0.7769** |
| Juicio LLM (auto-juez) | 0.5177 | 0.70 |
| Alucinaciones (auto-juez) | 8 | 9 |
| **Latencia media de generación** | 20.33 s | **3.99 s** |

Por tipo de pregunta (coseno · juicio):

| Tipo | Ollama | Gemini |
|------|--------|--------|
| factual (n=8) | 0.822 · 0.69 | 0.826 · 0.84 |
| técnica (n=10) | 0.609 · 0.37 | 0.687 · 0.47 |
| trazabilidad (n=4) | 0.776 · 0.55 | 0.904 · 0.99 |

**Nota metodológica.** El *hit-rate* es idéntico porque la recuperación no depende del
generador → confirma que los fallos están en la recuperación (contaminación entre los 15
productos en §3, §10, §15), no en el modelo. Cada sistema **se juzga a sí mismo**, así que
`juicio` y `alucinaciones` NO son directamente comparables (Gemini, juez más estricto,
marca *más* alucinaciones pese a tener mejor coseno). Las métricas comparables y objetivas
son **coseno** (independiente del juez) y **latencia**.

#### Diferencias en las respuestas (esperada vs generada)

| Pregunta | Referencia | Ollama `qwen2.5:3b` | Gemini `2.5-flash` |
|----------|------------|---------------------|--------------------|
| q01 — fabricante y tel. emergencia (factual) | Sika Colombia S.A.S.; CISPROQUIM Bogotá 2886012… | Correcto, pero vuelca el bloque crudo (juicio 0.75, 14 s) | Correcto y redactado natural (juicio 1.0, 6.9 s) |
| q13 — inflamación y densidad (factual) | < 23 °C (ASTM D56); 0,9–1,4 kg/l | Correcto (juicio 0.8, 28 s) | Correcto y más limpio (cos 0.96, juicio 1.0, 3.2 s) |
| q11 — EPP manos/respiratoria (técnica) | Guantes químico-resistentes + protección respiratoria | Correcto (juicio 0.8) pero **38.7 s** | Correcto (juicio 1.0) en **3.5 s** |
| q06 — componentes y CAS (técnica) | Acetato de butilo (123-86-4)… | "No se encuentra" ❌ | "No se encuentra" ❌ |
| q20 — ONU/clase transporte (trazabilidad) | UN1263, "Pintura", clase 3, grupo III | Alucina "UN 000000613416" (código de producto); clase 3 ✓ pero ONU inventado (juicio 0.2) | "UN1263, clase 3" — **correcto** (juicio 1.0) |

Observaciones:
- **q06** falla igual en ambos: la Sección 3 del producto pedido no entró en el top-5
  (contaminación entre productos) → es un fallo de **recuperación**, no de generación.

#### Conclusión

- **Calidad:** Gemini supera al modelo local en la métrica objetiva (coseno +0.06) y
  produce respuestas más limpias y naturales; su auto-juicio también es mayor. El 3B
  comete alucinaciones numéricas más graves (inventa números ONU a partir de códigos).
- **Precisión de recuperación:** idéntica (0.8182) — el techo lo pone la recuperación,
  común a ambos; mejorarla (p. ej. filtro por producto) beneficia a los dos por igual.
- **Velocidad:** Gemini ~5× más rápido (3.99 s vs 20.33 s) y sin ocupar GPU/RAM locales.
- **Trade-off para el parcial:** la rúbrica premia la **baja dependencia y operación
  100% local/sin APIs**. Gemini mejora calidad y latencia pero añade dependencia de una
  API externa, clave y conectividad. El diseño deja ambos intercambiables vía
  `config.LLM_PROVIDER`: **Ollama por defecto** (local, reproducible, sin costo) y
  **Gemini opcional** cuando se prioriza calidad/velocidad.

## 7. Limitaciones y mitigaciones

- **PDFs escaneados:** sin texto nativo se degrada a OCR de página completa; la calidad
  depende de la resolución. Mitigación: rasterizado a 200 dpi + Tesseract `spa`; el
  modo se reporta por documento.
- **Tablas con celdas combinadas:** pdfplumber puede no reconstruirlas perfectamente.
  Mitigación: doble estrategia (líneas/texto) y, ante fallo, conservar como texto sin
  desplazar columnas.
- **Numeración de tablas:** la etiqueta `Tabla N` es secuencial del pipeline y puede no
  coincidir con la numeración interna del documento.
- **Detección de secciones:** encabezados con formato atípico podrían no detectarse; se
  registran en `validation_report.json` como `missing` en lugar de fusionarse en
  silencio.
- **Modelo pequeño (3B):** puede simplificar respuestas técnicas; se mitiga con
  `temperature=0`, prompt que prohíbe inventar y obligación de citar la fuente.
- **OCR en español:** errores en texto de baja resolución; afecta solo a contenido
  basado en imágenes.

## 8. Reproducibilidad

`bash setup.sh` (Tesseract+spa, Ollama, modelos, deps) → PDFs en `data/raw/` →
`python build_index.py` → `python query_cli.py "…"` / `streamlit run app_streamlit.py`
→ `python -m src.eval.evaluate`. Sin servicios externos ni claves de API.
