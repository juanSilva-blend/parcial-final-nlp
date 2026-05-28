"""Estructuras de datos compartidas por el pipeline.

Centralizar los dataclasses evita imports circulares entre los módulos de
extracción, secciones, trazabilidad y chunking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# bbox = (x0, y0, x1, y1) en coordenadas PDF (origen arriba-izquierda en PyMuPDF).
BBox = tuple[float, float, float, float]


@dataclass
class Line:
    """Línea de texto con su geometría y señales de formato."""
    text: str
    bbox: BBox
    size: float          # tamaño de fuente máximo en la línea
    bold: bool
    page: int            # índice 0-based


@dataclass
class ImageRef:
    """Imagen embebida extraída de una página."""
    path: str            # ruta relativa al .md (assets/...)
    abs_path: str        # ruta absoluta en disco
    bbox: BBox
    page: int
    xref: int
    ocr_text: str = ""


@dataclass
class TableRef:
    """Tabla detectada por pdfplumber, ya renderizada a Markdown."""
    markdown: str
    bbox: BBox
    page: int
    n_rows: int
    n_cols: int
    label: str = ""      # p. ej. "Tabla 4" si se infiere


@dataclass
class Page:
    """Contenido estructurado de una página."""
    number: int          # índice 0-based
    lines: list[Line] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    is_scanned: bool = False
    raw_text: str = ""   # texto nativo concatenado en orden de lectura


@dataclass
class SectionSpan:
    """Una de las 16 secciones normativas localizada en el documento."""
    number: int          # 1..16
    title: str
    page: int            # página donde aparece el encabezado
    header_bbox: BBox
    text: str = ""       # contenido completo (puede cruzar páginas)
    found: bool = True


@dataclass
class ImageTrace:
    """Resultado de asociar una imagen a su contexto estructural."""
    image: ImageRef
    section_number: int | None
    section_title: str
    table_label: str = ""
    text_refs: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """Fragmento indexable con metadata de trazabilidad."""
    chunk_id: str
    text: str
    source_pdf: str
    section_number: int | None
    section_title: str
    page: int
    has_table: bool = False
    image_refs: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_pdf": self.source_pdf,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "page": self.page,
            "has_table": self.has_table,
            "image_refs": ",".join(self.image_refs),
        }
