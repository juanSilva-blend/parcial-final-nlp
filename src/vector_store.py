"""Vector store mínimo y portable: numpy (.npy) + SQLite.

Decisión de arquitectura: en lugar de ChromaDB/FAISS (que arrastran muchas
dependencias transitivas), para un corpus de cientos de chunks basta una matriz
numpy normalizada + búsqueda coseno brute-force (producto punto), con la metadata en
SQLite de la stdlib. Resultado: latencia sub-ms, cero dependencias extra, totalmente
reproducible. La interfaz VectorStore permite cambiar de backend si hiciera falta.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from . import config
from .models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    idx INTEGER PRIMARY KEY,
    chunk_id TEXT,
    source_pdf TEXT,
    section_number INTEGER,
    section_title TEXT,
    page INTEGER,
    has_table INTEGER,
    image_refs TEXT,
    text TEXT
);
"""


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.metas: list[dict] = []

    # --- construcción -------------------------------------------------------
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors y chunks deben tener la misma longitud")
        rows = []
        for ch in chunks:
            m = ch.meta()
            m["text"] = ch.text
            rows.append(m)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metas.extend(rows)

    # --- persistencia -------------------------------------------------------
    def persist(self, vectors_path: Path | None = None,
                db_path: Path | None = None) -> None:
        vectors_path = vectors_path or config.VECTORS_PATH
        db_path = db_path or config.META_DB_PATH
        if self.vectors is None:
            raise RuntimeError("no hay vectores que persistir")
        np.save(vectors_path, self.vectors)

        db_path.unlink(missing_ok=True)
        con = sqlite3.connect(db_path)
        try:
            con.executescript(_SCHEMA)
            con.executemany(
                "INSERT INTO chunks (idx, chunk_id, source_pdf, section_number, "
                "section_title, page, has_table, image_refs, text) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [(i, m["chunk_id"], m["source_pdf"], m["section_number"],
                  m["section_title"], m["page"], int(m["has_table"]),
                  m["image_refs"], m["text"]) for i, m in enumerate(self.metas)],
            )
            con.commit()
        finally:
            con.close()

    def load(self, vectors_path: Path | None = None,
             db_path: Path | None = None) -> "VectorStore":
        vectors_path = vectors_path or config.VECTORS_PATH
        db_path = db_path or config.META_DB_PATH
        if not vectors_path.exists() or not db_path.exists():
            raise FileNotFoundError(
                "Índice no encontrado. Ejecuta primero: python build_index.py")
        self.vectors = np.load(vectors_path)
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            cur = con.execute("SELECT * FROM chunks ORDER BY idx")
            self.metas = [dict(r) for r in cur.fetchall()]
        finally:
            con.close()
        return self

    # --- consulta -----------------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int = config.TOP_K,
               section_filter: int | list[int] | None = None) -> list[tuple[dict, float]]:
        if self.vectors is None or not self.metas:
            return []
        scores = self.vectors @ query_vec  # coseno (vectores normalizados)

        mask = np.ones(len(self.metas), dtype=bool)
        if section_filter is not None:
            wanted = {section_filter} if isinstance(section_filter, int) else set(section_filter)
            mask = np.array([m["section_number"] in wanted for m in self.metas])
            if not mask.any():
                mask = np.ones(len(self.metas), dtype=bool)  # sin coincidencias → sin filtro

        idxs = np.where(mask)[0]
        order = idxs[np.argsort(-scores[idxs])][:k]
        return [(self.metas[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self.metas)
