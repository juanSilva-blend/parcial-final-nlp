# RAG sobre Fichas de Datos de Seguridad — SIKA

Sistema **RAG (Recuperación Aumentada por Generación) totalmente local** para
consultar Fichas de Datos de Seguridad (FDS/SDS) del fabricante **SIKA**. Convierte
PDFs a Markdown fiel (16 secciones normativas, tablas, listas, imágenes con notas de
trazabilidad), indexa el contenido con embeddings locales y responde preguntas citando
el fragmento fuente.

**Sin APIs pagas. Sin servicios externos.** Modelos servidos localmente con Ollama;
extracción con PyMuPDF + pdfplumber + Tesseract; vector store propio (numpy + SQLite).

## Arquitectura (resumen)

```
PDF (data/raw/)
  │  PyMuPDF (texto+bbox+imágenes) · pdfplumber (tablas) · Tesseract (OCR spa)
  ▼
Detección de las 16 secciones  +  trazabilidad imagen↔sección↔tabla
  ▼
Markdown fiel (data/markdown/*.md)  +  validation_report.json
  ▼
Chunking section-aware (metadata de trazabilidad)
  ▼
Embeddings (Ollama bge-m3) → Vector store (numpy .npy + SQLite)
  ▼
Consulta → recuperación coseno top-k → LLM (Ollama qwen2.5:3b) con citas
```

Detalle completo en [docs/arquitectura.md](docs/arquitectura.md) y
[docs/pipeline.md](docs/pipeline.md).

## Requisitos

- Python 3.10+ (probado en 3.13).
- Linux con `sudo` para instalar Tesseract (incluye paquete de español) y Ollama.
- ~3 GB de descarga de modelos. GPU opcional (cuantización Q4 cabe en 4 GB VRAM).

## Instalación

```bash
bash setup.sh        # Tesseract+spa, Ollama, modelos, dependencias Python (en .venv)
```

Verificación rápida:

```bash
tesseract --list-langs | grep spa     # debe listar 'spa'
ollama list                            # debe mostrar bge-m3 y qwen2.5:3b-instruct
```

### Instalación sin privilegios de administrador (sin sudo)

Si no tienes `sudo`, todo funciona igual salvo Tesseract:

```bash
# 1. Python en un entorno virtual (evita el bloqueo PEP 668)
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

# 2. Ollama en $HOME (sin root)
ver=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4)
curl -fSL -o /tmp/ollama.tar.zst "https://github.com/ollama/ollama/releases/download/${ver}/ollama-linux-amd64.tar.zst"
mkdir -p "$HOME/.local" && tar --zstd -xf /tmp/ollama.tar.zst -C "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
# OLLAMA_FLASH_ATTENTION=0 es necesario en GPUs Turing (GTX 16xx): sin él, bge-m3
# devuelve embeddings NaN. OLLAMA_MAX_LOADED_MODELS=1 mantiene un solo modelo en RAM.
OLLAMA_FLASH_ATTENTION=0 OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
ollama pull bge-m3 && ollama pull qwen2.5:3b-instruct

# 3. Tesseract SÍ requiere sudo (única excepción). Sin él, el OCR de imágenes
#    queda vacío pero el resto del pipeline funciona:
sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
```

> Con un venv, antepón `.venv/bin/python` a los comandos (`.venv/bin/python build_index.py`)
> o activa el entorno con `source .venv/bin/activate`.

## Uso

```bash
# 1. Coloca los PDFs de SIKA en data/raw/
# 2. Construye el índice (extracción → markdown → chunks → embeddings)
python build_index.py

# 3. Consulta por terminal
python query_cli.py "¿Qué equipo de protección personal se requiere?"

# 4. Demo web
streamlit run app_streamlit.py

# 5. Evaluación contra ground truth
python -m src.eval.evaluate
```

## Modelos y fallbacks

| Rol        | Primario              | Fallback RAM-safe        |
|------------|-----------------------|--------------------------|
| Embeddings | `bge-m3` (1.2 GB)     | `nomic-embed-text`       |
| Generación | `qwen2.5:3b-instruct` | `qwen2.5:1.5b-instruct`  |

Para usar los fallbacks: `export EMBED_MODEL=nomic-embed-text LLM_MODEL=qwen2.5:1.5b-instruct`.

## Estructura

```
src/            extracción, secciones, OCR, trazabilidad, RAG, eval
build_index.py  orquestación (extracción → índice) por fases para no agotar la RAM
query_cli.py    consulta por terminal
app_streamlit.py demo web
data/raw/       PDFs de entrada · data/markdown/ salida · data/index/ vector store
docs/           arquitectura, pipeline e informe
```

## Solución de problemas

- **Embeddings NaN / error 500 en `/api/embeddings`** (GPUs Turing, GTX 16xx):
  arranca Ollama con `OLLAMA_FLASH_ATTENTION=0`.
- **`TesseractNotFoundError`**: instala `tesseract-ocr tesseract-ocr-spa`. Sin él, el
  OCR de imágenes queda vacío pero el resto del pipeline funciona.
- **"No se pudo contactar Ollama"**: asegúrate de que `ollama serve` esté corriendo y
  de haber hecho `ollama pull` de los modelos.

### Verificación rápida del sistema (sin PDFs reales)

```bash
.venv/bin/python scripts/make_sample_pdf.py    # genera una FDS sintética en data/raw/
.venv/bin/python build_index.py
.venv/bin/python query_cli.py "¿Qué EPP se requiere?"
```

## Limitaciones conocidas

Se documentan por documento en `data/index/validation_report.json` y se discuten en
[docs/informe.md](docs/informe.md): secciones no detectadas, PDFs escaneados, tablas
con celdas combinadas y calidad de OCR.
