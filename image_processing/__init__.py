"""Tutorial modular de procesamiento de imágenes."""
from .image_processor import ImageProcessor
from . import basics, color_spaces, histograms, spatial_filtering, edge_detection, morphology

__all__ = [
    "ImageProcessor",
    "basics",
    "color_spaces",
    "histograms",
    "spatial_filtering",
    "edge_detection",
    "morphology",
]
