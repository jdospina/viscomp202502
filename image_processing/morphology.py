"""Operaciones morfológicas básicas."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar operaciones morfológicas."""
    print("=" * 60)
    print("6. OPERACIONES MORFOLÓGICAS")
    print("=" * 60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Operaciones Morfológicas", fontsize=16)

    axes[0, 0].imshow(img_binary, cmap="gray")
    axes[0, 0].set_title("Imagen Binaria")
    axes[0, 0].axis("off")

    erosion = cv2.erode(img_binary, kernel, iterations=1)
    axes[0, 1].imshow(erosion, cmap="gray")
    axes[0, 1].set_title("Erosión")
    axes[0, 1].axis("off")

    dilation = cv2.dilate(img_binary, kernel, iterations=1)
    axes[0, 2].imshow(dilation, cmap="gray")
    axes[0, 2].set_title("Dilatación")
    axes[0, 2].axis("off")

    opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    axes[0, 3].imshow(opening, cmap="gray")
    axes[0, 3].set_title("Apertura")
    axes[0, 3].axis("off")

    closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    axes[1, 0].imshow(closing, cmap="gray")
    axes[1, 0].set_title("Cierre")
    axes[1, 0].axis("off")

    gradient = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
    axes[1, 1].imshow(gradient, cmap="gray")
    axes[1, 1].set_title("Gradiente")
    axes[1, 1].axis("off")

    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    axes[1, 2].imshow(tophat, cmap="gray")
    axes[1, 2].set_title("Top Hat")
    axes[1, 2].axis("off")

    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    axes[1, 3].imshow(blackhat, cmap="gray")
    axes[1, 3].set_title("Black Hat")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
