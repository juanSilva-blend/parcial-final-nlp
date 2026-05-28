"""Demo web del RAG (Streamlit).

Uso:  streamlit run app_streamlit.py
Muestra la respuesta generada y, debajo, los fragmentos fuente recuperados con su
sección y página para evidenciar la trazabilidad.
"""
from __future__ import annotations

import streamlit as st

from src import config
from src.rag import RagPipeline

st.set_page_config(page_title="RAG · FDS SIKA", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Cargando índice…")
def get_pipeline() -> RagPipeline:
    return RagPipeline()


st.title("🧪 RAG — Fichas de Datos de Seguridad (SIKA)")
st.caption(
    f"Modelos locales (Ollama): embeddings `{config.EMBED_MODEL}` · "
    f"generación `{config.LLM_MODEL}`. Sin servicios externos."
)

try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error("Índice no encontrado. Ejecuta `python build_index.py` con PDFs en data/raw/.")
    st.stop()

with st.sidebar:
    st.header("Opciones")
    k = st.slider("Fragmentos a recuperar (k)", 1, 10, config.TOP_K)
    use_filter = st.checkbox("Filtrar por sección")
    section = st.number_input("Sección (1-16)", 1, 16, 1) if use_filter else None
    st.markdown(f"**Chunks indexados:** {len(pipeline.store)}")

question = st.text_input("Pregunta", placeholder="¿Qué equipo de protección personal se requiere?")

if st.button("Consultar", type="primary") and question.strip():
    with st.spinner("Recuperando y generando…"):
        ans = pipeline.answer(question, k=k, section_filter=section)

    st.subheader("Respuesta")
    st.markdown(ans.text)

    st.subheader("Fuentes recuperadas (trazabilidad)")
    for i, c in enumerate(ans.citations, 1):
        with st.expander(f"[{i}] {c.label()} — similitud {c.score:.3f}"):
            st.text(c.text)
