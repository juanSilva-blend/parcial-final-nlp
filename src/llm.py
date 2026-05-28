"""Cliente de generación sobre Ollama (HTTP, sin torch).

Generación determinista (temperature=0) para reproducibilidad y evaluación. Usa
keep_alive=0 para descargar el modelo tras cada llamada y evitar que el LLM y el
embedder coexistan en memoria (la RAM es el recurso más escaso).
"""
from __future__ import annotations

import requests

from . import config


def generate(prompt: str, system: str | None = None, model: str | None = None,
             temperature: float | None = None) -> str:
    """Genera una respuesta a partir de un prompt (y un mensaje de sistema opcional)."""
    model = model or config.LLM_MODEL
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature
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
            f"[llm] Fallo al generar con '{model}' en {config.OLLAMA_URL} ({exc}). "
            f"Verifica 'ollama serve' y 'ollama pull {model}'.")
    return resp.json().get("message", {}).get("content", "").strip()
