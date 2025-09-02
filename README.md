# Tutorial de Procesamiento de Imágenes

Este repositorio contiene un tutorial modular para aprender las bases del
procesamiento digital de imágenes con Python. Cada módulo corresponde a una
sección temática que puede ejecutarse de forma independiente.

## Requisitos

- Python 3.11
- OpenCV
- NumPy
- Matplotlib

Instala dependencias con:

```bash
pip install opencv-python numpy matplotlib
```

## Uso

Ejecuta una sección específica desde la línea de comandos:

```bash
python -m image_processing.basics
python -m image_processing.color_spaces
```

También puedes ejecutar todas las secciones en secuencia:

```bash
python main.py
```

## Notebooks en Google Colab

En la carpeta `notebooks/` encontrarás versiones en Jupyter de cada módulo, listas para ejecutarse en Google Colab:

- `01_basics.ipynb`
- `02_color_spaces.ipynb`
- `03_histograms.ipynb`
- `04_spatial_filtering.ipynb`
- `05_edge_detection.ipynb`
- `06_morphology.ipynb`

## Contenido

- `image_processing/basics.py`: operaciones básicas de manipulación.
- `image_processing/color_spaces.py`: conversiones entre espacios de color y segmentación.
- `image_processing/histograms.py`: transformaciones de intensidad e histogramas.
- `image_processing/spatial_filtering.py`: filtrado espacial y convolución.
- `image_processing/edge_detection.py`: detección de bordes y contornos.
- `image_processing/morphology.py`: operaciones morfológicas.
