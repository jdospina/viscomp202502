"""Transformaciones de intensidad y análisis de histogramas."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar operaciones con histogramas y transformaciones de intensidad."""
    print("=" * 60)
    print("3. HISTOGRAMAS Y TRANSFORMACIONES DE INTENSIDAD")
    print("=" * 60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Histogramas y Transformaciones de Intensidad", fontsize=16)

    axes[0, 0].imshow(img_gray, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    hist_original = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    axes[1, 0].plot(hist_original)
    axes[1, 0].set_title("Histograma Original")
    axes[1, 0].set_xlim([0, 256])

    img_eq = cv2.equalizeHist(img_gray)
    axes[0, 1].imshow(img_eq, cmap="gray")
    axes[0, 1].set_title("Ecualizada")
    axes[0, 1].axis("off")

    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
    axes[1, 1].plot(hist_eq)
    axes[1, 1].set_title("Histograma Ecualizado")
    axes[1, 1].set_xlim([0, 256])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    axes[0, 2].imshow(img_clahe, cmap="gray")
    axes[0, 2].set_title("CLAHE")
    axes[0, 2].axis("off")

    hist_clahe = cv2.calcHist([img_clahe], [0], None, [256], [0, 256])
    axes[1, 2].plot(hist_clahe)
    axes[1, 2].set_title("Histograma CLAHE")
    axes[1, 2].set_xlim([0, 256])

    gamma = 0.5
    img_gamma = np.array(255 * (img_gray / 255) ** gamma, dtype=np.uint8)
    axes[0, 3].imshow(img_gamma, cmap="gray")
    axes[0, 3].set_title(f"Gamma = {gamma}")
    axes[0, 3].axis("off")

    hist_gamma = cv2.calcHist([img_gamma], [0], None, [256], [0, 256])
    axes[1, 3].plot(hist_gamma)
    axes[1, 3].set_title("Histograma Gamma")
    axes[1, 3].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
