"""Cliente de embeddings sobre Ollama (HTTP, sin torch).

Devuelve vectores L2-normalizados para que la similitud coseno se reduzca a un
producto punto. Comprueba que el modelo esté disponible y da un error accionable
(`ollama pull ...`) si no lo está.
"""
from __future__ import annotations

import numpy as np
import requests

from . import config


def _check_model(model: str) -> None:
    try:
        resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name", "").split(":")[0] for m in resp.json().get("models", [])}
        if model.split(":")[0] not in names:
            raise SystemExit(
                f"[embeddings] Modelo '{model}' no encontrado en Ollama. "
                f"Ejecuta:  ollama pull {model}")
    except requests.RequestException as exc:
        raise SystemExit(
            f"[embeddings] No se pudo contactar Ollama en {config.OLLAMA_URL} ({exc}). "
            f"¿Está corriendo 'ollama serve'?")


def embed_texts(texts: list[str], model: str | None = None,
                batch_report: bool = False, keep_alive: str | None = None) -> np.ndarray:
    """Embebe una lista de textos → matriz (n, dim) float32 L2-normalizada."""
    model = model or config.EMBED_MODEL
    keep_alive = config.OLLAMA_KEEP_ALIVE if keep_alive is None else keep_alive
    _check_model(model)
    vectors: list[list[float]] = []
    for i, text in enumerate(texts):
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": keep_alive},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
        if batch_report and (i + 1) % 25 == 0:
            print(f"  [embeddings] {i + 1}/{len(texts)}")
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(text: str, model: str | None = None) -> np.ndarray:
    """Embebe una consulta y descarga el embedder (libera GPU/RAM para el LLM)."""
    return embed_texts([text], model=model, keep_alive=config.OLLAMA_UNLOAD)[0]
