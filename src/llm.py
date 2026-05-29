"""Cliente de generación: Ollama (local) o Gemini API.

Selección del proveedor mediante config.LLM_PROVIDER o la variable de entorno
LLM_PROVIDER="ollama"|"gemini". Los embeddings siempre son locales (bge-m3 via
Ollama) porque cambiar el modelo de embeddings invalida el índice vectorial.

Generación determinista (temperature=0) para reproducibilidad y evaluación.
"""
from __future__ import annotations

import time

import requests

from . import config


# ---------------------------------------------------------------------------
# Proveedor Ollama
# ---------------------------------------------------------------------------

def _generate_ollama(prompt: str, system: str | None, model: str,
                     temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"temperature": temperature, "num_ctx": config.LLM_NUM_CTX},
            },
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"[llm/ollama] Fallo al generar con '{model}' en {config.OLLAMA_URL} ({exc}). "
            f"Verifica 'ollama serve' y 'ollama pull {model}'.")
    return resp.json().get("message", {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# Proveedor Gemini
# ---------------------------------------------------------------------------

def _generate_gemini(prompt: str, system: str | None, model: str,
                     temperature: float) -> str:
    if not config.GEMINI_API_KEY:
        raise SystemExit(
            "[llm/gemini] Falta la clave de API. "
            "Exporta la variable de entorno:  export GEMINI_API_KEY=<tu_clave>")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise SystemExit(
            "[llm/gemini] Paquete no instalado. Ejecuta:  pip install google-genai")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    cfg = types.GenerateContentConfig(
        system_instruction=system or "", temperature=temperature)

    # Reintenta errores transitorios (503 sobrecarga / 429 cuota) con backoff.
    _TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=cfg)
            return response.text.strip()
        except Exception as exc:
            last_exc = exc
            if any(k in str(exc) for k in _TRANSIENT):
                time.sleep(2 ** attempt)   # 1, 2, 4, 8, 16 s
                continue
            raise SystemExit(f"[llm/gemini] Error en la llamada a la API: {exc}")
    raise SystemExit(
        f"[llm/gemini] La API sigue no disponible tras varios reintentos: {last_exc}")


# ---------------------------------------------------------------------------
# Interfaz pública — misma firma que antes, transparente para el resto del código
# ---------------------------------------------------------------------------

def generate(prompt: str, system: str | None = None, model: str | None = None,
             temperature: float | None = None, provider: str | None = None) -> str:
    """Genera una respuesta. El proveedor se elige con config.LLM_PROVIDER salvo que
    se pase 'provider' explícito (útil para fijar un juez distinto del generador)."""
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "gemini":
        model = model or config.GEMINI_LLM_MODEL
        return _generate_gemini(prompt, system, model, temperature)

    # Proveedor por defecto: ollama
    model = model or config.LLM_MODEL
    return _generate_ollama(prompt, system, model, temperature)
