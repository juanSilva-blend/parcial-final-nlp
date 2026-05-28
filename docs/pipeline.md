# Documentación técnica del pipeline

Detalle de cada etapa, de PDF a respuesta citada. Las rutas y parámetros viven en
[`src/config.py`](../src/config.py).

## 1. Extracción estructural — `pdf_extract.py`

Con **PyMuPDF** (`fitz`) se recorre el PDF página a página:

- **Líneas de texto:** `page.get_text("dict")` entrega bloques → líneas → spans. Por
  línea se conserva el texto (normalizado NFC), el `bbox`, el **tamaño de fuente**
  máximo y si está en **negrita** (`flags & 2^4` o nombre de fuente con "bold"). Estas
  señales alimentan la detección de encabezados.
- **Orden de lectura:** `_reading_order` detecta columnas por un hueco grande en las
  `x0` y ordena `(columna, y, x)`. Evita mezclar texto en documentos a dos columnas.
- **Imágenes:** `page.get_images(full=True)` + `doc.extract_image(xref)`; el `bbox` se
  obtiene de `page.get_image_rects(xref)`. Se descartan imágenes < 60 px (decorativas).
  Cada imagen se guarda en `data/markdown/assets/` y se le aplica OCR.
- **Detección de escaneo:** si el texto nativo de la página es casi vacío
  (`< SCANNED_TEXT_THRESHOLD`), se rasteriza a 200 dpi y se procesa con OCR completo.
- La memoria se libera con `doc.close()` entre PDFs (RAM limitada).

## 2. Tablas — `tables.py`

Con **pdfplumber** se prueban dos estrategias de detección y se conserva la que
produce una tabla consistente:

1. `vertical/horizontal_strategy = "lines"` (tablas con bordes).
2. Si no hay tablas, `"text"` (tablas separadas por espaciado).

Las celdas se limpian (NFC, se escapan `|`, se colapsan saltos de línea) y se
renderizan como **tabla GFM**. Se ignoran tablas con < 2 celdas (ruido). Cada tabla
recibe una etiqueta secuencial `Tabla N` y conserva su `bbox` para la trazabilidad.
**Principio:** si una tabla no parsea limpia, se prefiere conservarla como bloque de
texto antes que desplazar columnas.

## 3. OCR — `ocr.py`

Wrapper de **Tesseract** con el paquete de español (`spa`). Acepta bytes o ruta de
imagen, convierte a RGB/escala de grises y limpia el texto resultante. Si Tesseract
no está instalado, degrada devolviendo `""` y avisa una sola vez (no rompe el lote).

## 4. Detección de las 16 secciones — `sections.py`

Núcleo de la fidelidad estructural. **Dos pasadas:**

1. **Candidatos:** regex unicode por línea
   `^\s*(?:SECCI[ÓO]N|SECTION)?\s*0?(num 1-16)\s*[.:\-–)]?\s+(título)$`.
   Cubre `SECCIÓN 1:`, `Sección 1.`, `1. IDENTIFICACIÓN`, `SECCION` sin tilde.
2. **Validación** para descartar falsos positivos del cuerpo:
   - El número debe avanzar de forma **creciente** (permite huecos por secciones
     faltantes, pero no retrocesos).
   - **Señal de formato:** la línea debe parecer encabezado (negrita, o fuente ≥
     mediana + 0.5, o empezar con "SECCIÓN").
   - **Coincidencia semántica del título** contra `SECTION_KEYWORDS` (p. ej. sección 4
     ↔ "primeros auxilios", 9 ↔ "propiedades/físicas/químicas", 14 ↔ "transporte"),
     salvo que sea un encabezado "SECCIÓN" explícito.

El contenido de cada sección es el texto entre su encabezado y el siguiente
(puede **cruzar páginas**). `validate()` completa las 16 (marca `found=False` las
ausentes) y emite un `ValidationReport` con `found / missing / out_of_order` que se
guarda en `data/index/validation_report.json` — insumo directo del informe de
limitaciones.

## 5. Trazabilidad imagen ↔ sección ↔ tabla — `traceability.py`

Para cada imagen:
- **Sección contenedora:** el encabezado de sección más cercano por encima de la
  imagen (respetando páginas y lado horizontal).
- **Tabla relacionada:** tabla de la misma página con solape horizontal y distancia
  vertical ≤ `IMG_TABLE_MAX_GAP_PT`.
- **Referencias textuales:** regex `(?:ver|véase|consulte|según) … (tabla|figura|sección) N`
  sobre el texto de la sección y el OCR de la imagen.

## 6. Ensamblaje del Markdown — `md_builder.py`

Por página, se intercalan en **orden de lectura** (`y`, luego `x`): encabezados de
sección (`## Sección N — Título`), párrafos, listas (viñetas detectadas por regex),
**tablas** (en su posición; las líneas que caen dentro del `bbox` de una tabla se
omiten para no duplicar) e **imágenes** con su bloque de trazabilidad:

```markdown
![Imagen ...](assets/...)

> **Nota de trazabilidad**
> - Sección asociada: 8 — Controles de exposición / protección individual
> - Página: 3
> - Tabla relacionada: Tabla 4
> - Referencias en el texto: Tabla 4
> - OCR (spa): "..."
```

Marcadores `<!-- página N -->` facilitan ubicar el contenido. Salida en UTF-8 NFC.

## 7. Chunking — `chunker.py`

Estrategia **section-aware** (documentada y justificable):

- **Frontera primaria:** las 16 secciones (límites semánticos naturales y trazables).
- Secciones largas se sub-dividen por párrafos hasta `CHUNK_SIZE` (~1000 chars) con
  solape `CHUNK_OVERLAP` (~150) para no perder contexto en los bordes.
- **Cada tabla es un chunk íntegro** (nunca se parte), asociado a su sección.
- El **OCR de cada imagen** se indexa como chunk aparte, ligado a su sección.
- Metadata por chunk: `{chunk_id, source_pdf, section_number, section_title, page,
  has_table, image_refs}`.

## 8. Embeddings y vector store — `embeddings.py`, `vector_store.py`

- `embed_texts` llama a Ollama `/api/embeddings` (`bge-m3`) y **L2-normaliza** cada
  vector, de modo que el coseno = producto punto.
- `VectorStore` guarda la matriz en `vectors.npy` y la metadata + texto en
  `meta.sqlite`. `search` calcula `vectors @ q`, aplica filtro opcional por sección y
  devuelve el top-k con su score.

## 9. RAG — `rag.py`, `llm.py`

1. Se embebe la pregunta y se recuperan los `TOP_K` chunks más similares.
2. Se arma el prompt con el contexto etiquetado por `[pdf · Sección N · p.X]`.
3. El **mensaje de sistema** obliga al modelo a: responder solo con el contexto,
   declarar "No se encuentra en los documentos" si falta, y **citar cada afirmación**
   como `[fuente: <documento>, Sección N, p.X]`. Generación determinista
   (`temperature=0`).
4. Se devuelve la respuesta + las citas (chunks con sección/página y score) para
   evidenciar la trazabilidad.

## 10. Evaluación — `src/eval/`

`ground_truth.json` contiene preguntas factuales, técnicas y de trazabilidad con su
`expected_section` y `reference_answer`. `evaluate.py` calcula:

- **retrieval_hit_rate:** ¿la sección esperada está entre las fuentes citadas?
- **gen_cosine:** similitud coseno respuesta-vs-referencia (entradas con referencia).
- **llm_judge:** el LLM local puntúa corrección [0-1] y marca alucinación.

Produce `data/index/eval_report.json` con ejemplos comparativos esperado-vs-generado.
