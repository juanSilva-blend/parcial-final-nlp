"""Configuración central del pipeline RAG sobre Fichas de Datos de Seguridad (SIKA).

Todas las rutas, modelos y umbrales viven aquí para mantener el sistema
reproducible y fácil de auditar. No hay dependencias de servicios externos:
los modelos se sirven localmente con Ollama vía HTTP.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Rutas del proyecto -----------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # PDFs de entrada (los coloca el usuario)
MD_DIR = DATA / "markdown"        # .md generados
ASSETS_DIR = MD_DIR / "assets"    # imágenes extraídas
INDEX_DIR = DATA / "index"        # vectors.npy + meta.sqlite + reportes

VECTORS_PATH = INDEX_DIR / "vectors.npy"
META_DB_PATH = INDEX_DIR / "meta.sqlite"
VALIDATION_REPORT = INDEX_DIR / "validation_report.json"

for _d in (RAW_DIR, MD_DIR, ASSETS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Ollama -----------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Modelos primarios (caben en GTX 1650 4GB con cuantización Q4).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")            # multilingüe, 1024-dim
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct")  # buen español

# Fallbacks RAM-safe (exportar EMBED_MODEL / LLM_MODEL para activarlos).
FALLBACK_EMBED_MODEL = "nomic-embed-text"
FALLBACK_LLM_MODEL = "qwen2.5:1.5b-instruct"

# Parámetros de generación: deterministas para reproducibilidad y evaluación.
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX = 4096
# Durante un lote (indexación) el modelo se mantiene cargado para no hacer thrashing.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")
# En el camino de consulta, tras embeber la pregunta descargamos el embedder
# (keep_alive=0) para liberar la GPU/RAM antes de cargar el LLM (evita coexistencia).
OLLAMA_UNLOAD = "0"
REQUEST_TIMEOUT = 600  # segundos; los modelos pequeños en CPU pueden ser lentos

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 1000      # caracteres objetivo por sub-chunk
CHUNK_OVERLAP = 150    # solape entre sub-chunks de una misma sección
MIN_CHUNK_SIZE = 80    # descartar fragmentos triviales

# --- Recuperación -----------------------------------------------------------
TOP_K = 5

# --- OCR --------------------------------------------------------------------
OCR_LANG = "spa"
# Una página se considera escaneada si su texto nativo es casi vacío.
SCANNED_TEXT_THRESHOLD = 20  # nº de caracteres

# --- Tablas -----------------------------------------------------------------
# Estrategia primaria: detección por líneas (tablas con bordes), de alta precisión.
# La estrategia 'text' puede fabricar tablas falsas a partir de prosa (corta
# palabras en columnas), así que es opt-in. Actívala solo si tus documentos tienen
# tablas sin bordes que se pierden como texto.
TABLE_TEXT_FALLBACK = False

# --- Trazabilidad imagen ↔ tabla -------------------------------------------
IMG_TABLE_MAX_GAP_PT = 40.0  # distancia vertical máxima imagen-tabla (puntos PDF)
