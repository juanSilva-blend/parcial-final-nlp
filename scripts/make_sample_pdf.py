"""Genera una FDS sintética en español (16 secciones) para probar el pipeline E2E.

Uso:  .venv/bin/python scripts/make_sample_pdf.py
Crea data/raw/_SAMPLE_SIKA_test.pdf para verificar la instalación sin PDFs reales.
"""
import sys
from pathlib import Path

import fitz  # PyMuPDF

OUT = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "_SAMPLE_SIKA_test.pdf")

SECTIONS = [
    (1, "IDENTIFICACIÓN DE LA SUSTANCIA/MEZCLA Y DE LA EMPRESA",
     ["Nombre comercial: Sikaflex-11 FC+ (producto de demostración).",
      "Proveedor: SIKA Colombia S.A.S. Tel. emergencias: +57 1 234 5678.",
      "Usos recomendados: sellador y adhesivo elástico para construcción."]),
    (2, "IDENTIFICACIÓN DE LOS PELIGROS",
     ["Clasificación SGA: Sensibilizante cutáneo Categoría 1.",
      "Palabra de advertencia: Atención.",
      "Indicaciones de peligro: H317 Puede provocar una reacción alérgica en la piel."]),
    (3, "COMPOSICIÓN/INFORMACIÓN SOBRE LOS COMPONENTES",
     ["Polímero de poliuretano: 30-60%.",
      "Difenilmetano-4,4'-diisocianato (MDI), CAS 101-68-8: 1-5%.",
      "Negro de humo: < 1%."]),
    (4, "PRIMEROS AUXILIOS",
     ["Contacto con los ojos: enjuagar con abundante agua durante 15 minutos y consultar a un médico.",
      "Contacto con la piel: lavar con agua y jabón abundante.",
      "Ingestión: no inducir el vómito; buscar atención médica."]),
    (5, "MEDIDAS DE LUCHA CONTRA INCENDIOS",
     ["Medios de extinción adecuados: espuma resistente al alcohol, CO2, polvo químico seco.",
      "Medios no adecuados: chorro de agua de gran caudal."]),
    (6, "MEDIDAS EN CASO DE VERTIDO ACCIDENTAL",
     ["Recoger mecánicamente el producto derramado y depositarlo en recipientes adecuados.",
      "Evitar la entrada en desagües y cursos de agua."]),
    (7, "MANIPULACIÓN Y ALMACENAMIENTO",
     ["Almacenar en lugar fresco y seco entre 5 °C y 25 °C.",
      "Mantener los envases bien cerrados y alejados de la humedad."]),
    (8, "CONTROLES DE EXPOSICIÓN/PROTECCIÓN INDIVIDUAL",
     ["Protección de las manos: guantes de nitrilo resistentes a químicos.",
      "Protección de los ojos: gafas de seguridad con protección lateral.",
      "Protección respiratoria: usar mascarilla en lugares mal ventilados. Ver Tabla 1."]),
    (9, "PROPIEDADES FÍSICAS Y QUÍMICAS",
     ["Estado físico: pasta. Color: varios. Olor: característico.",
      "Densidad: 1.3 g/cm3. pH: no aplica. Punto de inflamación: > 100 °C."]),
    (10, "ESTABILIDAD Y REACTIVIDAD",
     ["Estable en condiciones normales de almacenamiento.",
      "Materiales incompatibles: agua, alcoholes y aminas. Evitar la humedad."]),
    (11, "INFORMACIÓN TOXICOLÓGICA",
     ["Puede provocar sensibilización cutánea por contacto repetido.",
      "La inhalación de vapores puede irritar las vías respiratorias."]),
    (12, "INFORMACIÓN ECOLÓGICA",
     ["No verter en el suelo, cursos de agua ni alcantarillado.",
      "El producto curado es inerte y no presenta toxicidad acuática conocida."]),
    (13, "CONSIDERACIONES RELATIVAS A LA ELIMINACIÓN",
     ["Eliminar el producto y su envase como residuo peligroso conforme a la normativa local.",
      "No reutilizar los envases vacíos."]),
    (14, "INFORMACIÓN RELATIVA AL TRANSPORTE",
     ["El producto no está clasificado como peligroso para el transporte.",
      "No tiene número ONU asignado."]),
    (15, "INFORMACIÓN REGLAMENTARIA",
     ["Clasificado conforme al Sistema Globalmente Armonizado (SGA).",
      "Cumple con la normativa colombiana de etiquetado de productos químicos."]),
    (16, "OTRA INFORMACIÓN",
     ["Ficha de demostración generada para pruebas del pipeline RAG.",
      "Revisión: 1.0. Fecha: 2026-05-28."]),
]


def main():
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    page.insert_text((72, 40), "FICHA DE DATOS DE SEGURIDAD (DEMO)", fontsize=14, fontname="hebo")
    for num, title, body in SECTIONS:
        if y > 760:
            page = doc.new_page()
            y = 60
        page.insert_text((72, y), f"SECCIÓN {num}: {title}", fontsize=12, fontname="hebo")
        y += 18
        for line in body:
            # envolver líneas largas manualmente
            while len(line) > 95:
                cut = line.rfind(" ", 0, 95)
                page.insert_text((80, y), line[:cut], fontsize=10, fontname="helv")
                line = line[cut + 1:]
                y += 13
            page.insert_text((80, y), line, fontsize=10, fontname="helv")
            y += 13
        y += 10
    doc.save(OUT)
    print("PDF generado:", OUT, "| páginas:", doc.page_count)
    doc.close()


if __name__ == "__main__":
    sys.exit(main())
