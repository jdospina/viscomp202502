"""Conversiones y segmentación en distintos espacios de color."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar conversiones entre espacios de color y segmentación."""
    print("=" * 60)
    print("2. ESPACIOS DE COLOR")
    print("=" * 60)

    processor = ImageProcessor()
    img_rgb = processor.load_image()

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Espacios de Color", fontsize=16)

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("RGB Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_gray, cmap="gray")
    axes[0, 1].set_title("Escala de Grises")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_hsv)
    axes[0, 2].set_title("HSV")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_hsv[:, :, 0], cmap="hsv")
    axes[1, 0].set_title("Canal H (Matiz)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img_hsv[:, :, 1], cmap="gray")
    axes[1, 1].set_title("Canal S (Saturación)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_hsv[:, :, 2], cmap="gray")
    axes[1, 2].set_title("Canal V (Valor)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    print("\n--- Ejemplo práctico: Segmentación por color ---")

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_green)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(mask_green, cmap="gray")
    axes[1].set_title("Máscara Verde")
    axes[1].axis("off")

    axes[2].imshow(result)
    axes[2].set_title("Solo Píxeles Verdes")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
