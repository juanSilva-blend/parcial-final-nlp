"""Detección y validación de las 16 secciones normativas de una FDS (GHS/SGA).

Estrategia de dos pasadas para evitar falsos positivos (p. ej. "punto 9" en el
cuerpo del texto):
  1. Candidatos por regex sobre cada línea.
  2. Validación: el número debe llegar en orden creciente y el título debe
     coincidir con las palabras clave canónicas de la sección (o el encabezado
     debe empezar con "SECCIÓN"). Las líneas en negrita / fuente grande suman
     confianza.

La salida es una lista de 16 SectionSpan (las no encontradas quedan marcadas
found=False) y un ValidationReport con found / missing / out_of_order para el
informe de limitaciones.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .models import Line, Page, SectionSpan

# Títulos canónicos de las 16 secciones (referencia para el informe).
CANONICAL_TITLES: dict[int, str] = {
    1: "Identificación de la sustancia/mezcla y de la empresa",
    2: "Identificación de los peligros",
    3: "Composición/información sobre los componentes",
    4: "Primeros auxilios",
    5: "Medidas de lucha contra incendios",
    6: "Medidas en caso de vertido accidental",
    7: "Manipulación y almacenamiento",
    8: "Controles de exposición/protección individual",
    9: "Propiedades físicas y químicas",
    10: "Estabilidad y reactividad",
    11: "Información toxicológica",
    12: "Información ecológica",
    13: "Consideraciones relativas a la eliminación",
    14: "Información relativa al transporte",
    15: "Información reglamentaria",
    16: "Otra información",
}

# Palabras clave (sin tildes) para validar el título de cada sección.
SECTION_KEYWORDS: dict[int, set[str]] = {
    1: {"identificacion", "empresa", "sustancia", "mezcla", "producto"},
    2: {"peligros", "peligro"},
    3: {"composicion", "componentes", "ingredientes"},
    4: {"primeros", "auxilios"},
    5: {"incendios", "incendio", "lucha", "fuego"},
    6: {"vertido", "accidental", "derrame", "fuga"},
    7: {"manipulacion", "almacenamiento", "manejo"},
    8: {"exposicion", "proteccion", "controles", "individual", "epp"},
    9: {"propiedades", "fisicas", "quimicas", "fisicoquimicas"},
    10: {"estabilidad", "reactividad"},
    11: {"toxicologica", "toxicologia", "toxicidad"},
    12: {"ecologica", "ecologia", "ambiental"},
    13: {"eliminacion", "desecho", "disposicion", "residuos"},
    14: {"transporte"},
    15: {"reglamentaria", "regulatoria", "reglamentacion", "normativa", "legal"},
    16: {"otra", "informacion", "adicional"},
}

# Encabezado: opcional "SECCIÓN"/"SECTION", número 1-16, separador, título.
_HEADER_RE = re.compile(
    r"^\s*(?:SECCI[ÓO]N|SECTION)?\s*0?(?P<num>1[0-6]|[1-9])\s*[\.\:\-–\)]?\s+(?P<title>[A-Za-zÁÉÍÓÚÑáéíóúñ].{2,})$"
)


@dataclass
class ValidationReport:
    source_pdf: str
    found: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    out_of_order: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "n_found": len(self.found),
            "found": self.found,
            "missing": self.missing,
            "out_of_order": self.out_of_order,
            "complete": len(self.found) == 16 and not self.out_of_order,
        }


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


def _title_matches(num: int, title: str, explicit_section: bool) -> bool:
    """¿El título encaja con la sección 'num'? Acepta si es encabezado explícito."""
    if explicit_section:
        return True
    tokens = set(_strip_accents(title).split())
    return bool(tokens & SECTION_KEYWORDS.get(num, set()))


def _iter_lines(pages: list[Page]) -> list[Line]:
    out: list[Line] = []
    for pg in pages:
        out.extend(pg.lines)
    return out


def detect_sections(pages: list[Page]) -> list[SectionSpan]:
    """Localiza encabezados de sección en orden de lectura, validando coherencia."""
    lines = _iter_lines(pages)
    sizes = [ln.size for ln in lines if ln.size > 0]
    median_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0

    spans: list[SectionSpan] = []
    expected = 1  # el siguiente número de sección que esperamos ver
    for ln in lines:
        m = _HEADER_RE.match(ln.text)
        if not m:
            continue
        num = int(m.group("num"))
        title = m.group("title").strip(" .:-")
        explicit = bool(re.match(r"^\s*(?:SECCI[ÓO]N|SECTION)", ln.text, re.I))

        # El número debe avanzar de forma creciente (permite saltar huecos).
        if num < expected:
            continue
        # Señal de formato: encabezado en negrita o fuente mayor que la mediana.
        looks_like_header = ln.bold or ln.size >= median_size + 0.5 or explicit
        if not looks_like_header:
            continue
        if not _title_matches(num, title, explicit):
            continue

        spans.append(SectionSpan(
            number=num,
            title=title or CANONICAL_TITLES[num],
            page=ln.page,
            header_bbox=ln.bbox,
            found=True,
        ))
        expected = num + 1
        if num == 16:
            break

    _fill_section_text(spans, pages)
    return spans


def _fill_section_text(spans: list[SectionSpan], pages: list[Page]) -> None:
    """Asigna a cada sección el texto entre su encabezado y el siguiente."""
    if not spans:
        return
    flat: list[tuple[int, Line]] = []
    for idx, pg in enumerate(pages):
        for ln in pg.lines:
            flat.append((idx, ln))

    # Posición de cada encabezado dentro de 'flat' (match por página + bbox).
    def header_pos(span: SectionSpan) -> int:
        for pos, (_, ln) in enumerate(flat):
            if ln.page == span.page and ln.bbox == span.header_bbox:
                return pos
        return 0

    positions = [header_pos(s) for s in spans]
    for i, span in enumerate(spans):
        start = positions[i] + 1
        end = positions[i + 1] if i + 1 < len(spans) else len(flat)
        body = [flat[j][1].text for j in range(start, end)]
        span.text = "\n".join(body).strip()


def validate(spans: list[SectionSpan], source_pdf: str) -> tuple[list[SectionSpan], ValidationReport]:
    """Completa las 16 secciones (marca faltantes) y genera el reporte."""
    by_num = {s.number: s for s in spans}
    found = sorted(by_num.keys())

    # Detectar desorden respecto a la secuencia esperada 1..16.
    out_of_order: list[int] = []
    prev = 0
    for n in found:
        if n <= prev:
            out_of_order.append(n)
        prev = n

    full: list[SectionSpan] = []
    for n in range(1, 17):
        if n in by_num:
            full.append(by_num[n])
        else:
            full.append(SectionSpan(
                number=n, title=CANONICAL_TITLES[n], page=-1,
                header_bbox=(0, 0, 0, 0), text="", found=False,
            ))

    report = ValidationReport(
        source_pdf=source_pdf,
        found=found,
        missing=[n for n in range(1, 17) if n not in by_num],
        out_of_order=out_of_order,
    )
    return full, report
