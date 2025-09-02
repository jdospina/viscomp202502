"""Punto de entrada para ejecutar las distintas secciones del tutorial."""

from image_processing import (
    basics,
    color_spaces,
    histograms,
    spatial_filtering,
    edge_detection,
    morphology,
)

SECTIONS = {
    "basics": basics.run,
    "color": color_spaces.run,
    "hist": histograms.run,
    "filter": spatial_filtering.run,
    "edges": edge_detection.run,
    "morph": morphology.run,
}


def run_all():
    """Ejecutar todas las secciones en orden."""
    for section in SECTIONS.values():
        section()


if __name__ == "__main__":
    run_all()
