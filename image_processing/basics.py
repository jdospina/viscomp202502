"""Operaciones básicas de manipulación de imágenes."""

import cv2
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar operaciones básicas de manipulación de imágenes."""
    print("=" * 60)
    print("1. OPERACIONES BÁSICAS DE MANIPULACIÓN")
    print("=" * 60)

    processor = ImageProcessor()
    img = processor.load_image()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Operaciones Básicas de Manipulación", fontsize=16)

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    img_resized = cv2.resize(img, (200, 150))
    axes[0, 1].imshow(img_resized)
    axes[0, 1].set_title("Redimensionada (200x150)")
    axes[0, 1].axis("off")

    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    axes[0, 2].imshow(img_rotated)
    axes[0, 2].set_title("Rotación 45°")
    axes[0, 2].axis("off")

    img_cropped = img[50:200, 50:300]
    axes[1, 0].imshow(img_cropped)
    axes[1, 0].set_title("Recortada")
    axes[1, 0].axis("off")

    img_flipped = cv2.flip(img, 1)
    axes[1, 1].imshow(img_flipped)
    axes[1, 1].set_title("Volteo Horizontal")
    axes[1, 1].axis("off")

    img_vflipped = cv2.flip(img, 0)
    axes[1, 2].imshow(img_vflipped)
    axes[1, 2].set_title("Volteo Vertical")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
