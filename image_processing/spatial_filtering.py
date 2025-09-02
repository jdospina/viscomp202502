"""Filtrado espacial y convolución."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar operaciones de filtrado espacial."""
    print("=" * 60)
    print("4. FILTRADO ESPACIAL Y CONVOLUCIÓN")
    print("=" * 60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Filtrado Espacial", fontsize=16)

    axes[0, 0].imshow(img_gray, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    img_gaussian = cv2.GaussianBlur(img_gray, (15, 15), 0)
    axes[0, 1].imshow(img_gaussian, cmap="gray")
    axes[0, 1].set_title("Filtro Gaussiano")
    axes[0, 1].axis("off")

    kernel_avg = np.ones((5, 5), np.float32) / 25
    img_avg = cv2.filter2D(img_gray, -1, kernel_avg)
    axes[0, 2].imshow(img_avg, cmap="gray")
    axes[0, 2].set_title("Filtro Promedio")
    axes[0, 2].axis("off")

    img_median = cv2.medianBlur(img_gray, 5)
    axes[0, 3].imshow(img_median, cmap="gray")
    axes[0, 3].set_title("Filtro Mediano")
    axes[0, 3].axis("off")

    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    axes[1, 0].imshow(sobel_x, cmap="gray")
    axes[1, 0].set_title("Sobel X")
    axes[1, 0].axis("off")

    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    axes[1, 1].imshow(sobel_y, cmap="gray")
    axes[1, 1].set_title("Sobel Y")
    axes[1, 1].axis("off")

    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    axes[1, 2].imshow(laplacian, cmap="gray")
    axes[1, 2].set_title("Laplaciano")
    axes[1, 2].axis("off")

    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_gray, -1, kernel_sharpen)
    axes[1, 3].imshow(img_sharpened, cmap="gray")
    axes[1, 3].set_title("Realce de Bordes")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()

    print("\n--- Ejemplo de Convolución Manual ---")
    kernel_edge = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]], dtype=np.float32)
    roi = img_gray[100:103, 100:103].astype(np.float32)
    convolved_pixel = np.sum(roi * kernel_edge)

    print("ROI 3x3:")
    print(roi)
    print("\nKernel:")
    print(kernel_edge)
    print(f"\nResultado de convolución: {convolved_pixel:.2f}")

if __name__ == "__main__":
    run()
