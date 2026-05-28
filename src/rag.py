"""Pipeline RAG: recuperación coseno → prompt con citas → generación local.

El prompt obliga al modelo a responder SOLO con el contexto recuperado y a citar la
fuente de cada afirmación como [fuente: <pdf>, Sección N, p.X], lo que da
trazabilidad y mitiga alucinaciones. Devuelve la respuesta junto con los chunks
recuperados (con sección y página) para mostrar la procedencia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, llm
from .embeddings import embed_query
from .vector_store import VectorStore

_SYSTEM = (
    "Eres un asistente técnico experto en Fichas de Datos de Seguridad (FDS). "
    "Respondes en español, de forma precisa y concisa, usando ÚNICAMENTE la "
    "información del CONTEXTO proporcionado. Si el contexto no contiene la "
    "respuesta, dilo explícitamente ('No se encuentra en los documentos'). "
    "Cita la fuente de cada dato como [fuente: <documento>, Sección N, p.X]. "
    "No inventes valores ni secciones."
)

_PROMPT_TMPL = """CONTEXTO:
{context}

PREGUNTA: {question}

Responde en español citando la fuente de cada afirmación con el formato
[fuente: <documento>, Sección N, p.X]."""


@dataclass
class Citation:
    source_pdf: str
    section_number: int | None
    section_title: str
    page: int
    score: float
    text: str

    def label(self) -> str:
        sec = f"Sección {self.section_number}" if self.section_number else "s/sección"
        return f"{self.source_pdf}, {sec}, p.{self.page}"


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)


def _format_context(hits: list[tuple[dict, float]]) -> str:
    blocks = []
    for meta, _ in hits:
        sec = f"Sección {meta['section_number']}" if meta["section_number"] else "s/n"
        tag = f"[{meta['source_pdf']} · {sec} · p.{meta['page']}]"
        blocks.append(f"{tag}\n{meta['text']}")
    return "\n\n---\n\n".join(blocks)


class RagPipeline:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore().load()

    def retrieve(self, question: str, k: int = config.TOP_K,
                 section_filter: int | list[int] | None = None) -> list[tuple[dict, float]]:
        qvec = embed_query(question)
        return self.store.search(qvec, k=k, section_filter=section_filter)

    def answer(self, question: str, k: int = config.TOP_K,
               section_filter: int | list[int] | None = None) -> Answer:
        hits = self.retrieve(question, k=k, section_filter=section_filter)
        if not hits:
            return Answer(question=question,
                          text="No se encuentra en los documentos (índice vacío).")
        context = _format_context(hits)
        prompt = _PROMPT_TMPL.format(context=context, question=question)
        text = llm.generate(prompt, system=_SYSTEM)
        citations = [
            Citation(
                source_pdf=m["source_pdf"], section_number=m["section_number"],
                section_title=m["section_title"], page=m["page"],
                score=score, text=m["text"],
            )
            for m, score in hits
        ]
        return Answer(question=question, text=text, citations=citations)
