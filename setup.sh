#!/usr/bin/env bash
# Instalación reproducible del entorno local (sin servicios externos pagos).
# Uso:  bash setup.sh
set -euo pipefail

echo "==> 1/4  Dependencias del sistema (Tesseract + paquete de español)"
if ! command -v tesseract >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
else
  echo "    tesseract ya instalado: $(tesseract --version | head -n1)"
fi
echo "    idiomas OCR disponibles:"
tesseract --list-langs 2>/dev/null | grep -i spa || \
  echo "    ADVERTENCIA: falta el paquete 'spa' (sudo apt-get install tesseract-ocr-spa)"

echo "==> 2/4  Ollama (motor local de modelos)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
  # --- Alternativa SIN sudo (instala en $HOME/.local) ---
  # ver=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4)
  # curl -fSL -o /tmp/ollama.tar.zst "https://github.com/ollama/ollama/releases/download/${ver}/ollama-linux-amd64.tar.zst"
  # mkdir -p "$HOME/.local" && tar --zstd -xf /tmp/ollama.tar.zst -C "$HOME/.local"
  # export PATH="$HOME/.local/bin:$PATH"   # añádelo a tu ~/.bashrc
  # nohup ollama serve >/tmp/ollama_serve.log 2>&1 &   # arrancar el servidor
else
  echo "    ollama ya instalado: $(ollama --version 2>/dev/null | head -n1)"
fi

echo "==> 3/4  Descarga de modelos (~3 GB)"
ollama pull bge-m3                 # embeddings multilingüe
ollama pull qwen2.5:3b-instruct    # LLM de generación
# Descomentar para los fallbacks RAM-safe:
# ollama pull nomic-embed-text
# ollama pull qwen2.5:1.5b-instruct

echo "==> 4/4  Dependencias de Python (en un entorno virtual)"
# Ubuntu/Debian modernos bloquean pip sobre el Python del sistema (PEP 668),
# por eso usamos un venv: es reproducible y no requiere privilegios.
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "==> NOTA para GPUs Turing (GTX 16xx): arranca el servidor de Ollama con"
echo "    OLLAMA_FLASH_ATTENTION=0 o bge-m3 devolverá embeddings NaN, p. ej.:"
echo "    OLLAMA_FLASH_ATTENTION=0 OLLAMA_MAX_LOADED_MODELS=1 ollama serve &"
echo ""

# ============================================================================
# (OPCIONAL) Generación con Gemini en lugar del modelo local
# ----------------------------------------------------------------------------
# El proveedor del LLM se elige con la variable LLM_PROVIDER (por defecto: ollama).
# El paquete 'google-genai' ya quedó instalado vía requirements.txt. Los embeddings
# SIEMPRE son locales (bge-m3 por Ollama), así que el servidor de Ollama debe estar
# corriendo también con Gemini.
#
# Para usar Gemini, exporta estas variables antes de build_index / query / eval:
#     export LLM_PROVIDER=gemini
#     export GEMINI_API_KEY=<tu_clave>
#     export GEMINI_LLM_MODEL=gemini-2.5-flash   # 'gemini-2.0-flash' quedó deprecado
# Ejemplo:
#     LLM_PROVIDER=gemini GEMINI_API_KEY=<clave> GEMINI_LLM_MODEL=gemini-2.5-flash \
#       .venv/bin/python query_cli.py "¿Qué EPP requiere el Esmalte Uretano AR Comp. B?"
# Para volver al modelo local:  export LLM_PROVIDER=ollama
#
# ⚠️  SEGURIDAD: este repositorio es PÚBLICO. NO dejes claves reales versionadas aquí;
#     si se commitean quedan expuestas (Google puede desactivarlas y otros gastan tu
#     cuota). Lo ideal es exportar GEMINI_API_KEY en tu shell o en un archivo .env
#     ignorado por git, y rotar la clave al terminar.
#     Clave usada en las pruebas (gemini-2.5-flash), funcional al 2026-05-29:
#       GEMINI_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ============================================================================
echo "==> (opcional) Gemini: export LLM_PROVIDER=gemini GEMINI_API_KEY=<clave> GEMINI_LLM_MODEL=gemini-2.5-flash"
echo ""
echo "==> Listo. Coloca los PDFs de SIKA en data/raw/ y ejecuta:"
echo "      .venv/bin/python build_index.py"
