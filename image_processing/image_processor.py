"""Clase utilitaria para cargar y generar imágenes de ejemplo."""

import os
import cv2
import numpy as np


class ImageProcessor:
    """Clase para demostrar operaciones principales de procesamiento de imágenes."""

    def __init__(self):
        self.current_image = None

    def load_image(self, image_path: str | None = None, create_sample: bool = True):
        """Cargar una imagen desde disco o crear una imagen sintética.

        Parameters
        ----------
        image_path: str | None
            Ruta al archivo de imagen.
        create_sample: bool
            Si no se provee una ruta, genera una imagen sintética.
        """
        if image_path and os.path.exists(image_path):
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            print(f"Imagen cargada: {image_path}")
        elif create_sample:
            self.current_image = self._create_sample_image()
            print("Imagen sintética creada para demostración")
        else:
            raise ValueError("No se pudo cargar la imagen")

        print(f"Dimensiones: {self.current_image.shape}")
        return self.current_image

    def _create_sample_image(self):
        """Crear una imagen sintética con patrones simples."""
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        for i in range(300):
            img[i, :, 0] = int(255 * i / 300)  # Gradiente rojo

        img[50:150, 50:150, 2] = 255  # Rectángulo azul
        cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)  # Círculo verde

        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
