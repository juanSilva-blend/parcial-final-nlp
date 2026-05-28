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

## 6. Resultados del RAG *(completar)*

> Tomar de `data/index/eval_report.json` tras completar `reference_answer` en
> `ground_truth.json` (se recomienda NotebookLM para generar las referencias).

- **retrieval_hit_rate:** _…_
- **mean_gen_cosine:** _…_
- **mean_llm_judge:** _…_

### Ejemplos comparativos esperado vs. generado *(completar)*

| Pregunta | Respuesta esperada | Respuesta del RAG | Fuente citada | Observación |
|----------|--------------------|-------------------|---------------|-------------|
| _…_ | _…_ | _…_ | _Sección N, p.X_ | _correcta / parcial / alucinación_ |

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
