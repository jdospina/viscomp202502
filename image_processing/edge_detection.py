"""Detección de bordes y contornos."""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .image_processor import ImageProcessor


def run():
    """Demostrar detección de bordes y contornos."""
    print("=" * 60)
    print("5. DETECCIÓN DE BORDES Y CONTORNOS")
    print("=" * 60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Detección de Bordes y Contornos", fontsize=16)

    axes[0, 0].imshow(img_gray, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    edges_canny = cv2.Canny(img_gray, 50, 150)
    axes[0, 1].imshow(edges_canny, cmap="gray")
    axes[0, 1].set_title("Canny (50, 150)")
    axes[0, 1].axis("off")

    edges_canny2 = cv2.Canny(img_gray, 100, 200)
    axes[0, 2].imshow(edges_canny2, cmap="gray")
    axes[0, 2].set_title("Canny (100, 200)")
    axes[0, 2].axis("off")

    contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    axes[1, 0].imshow(img_contours)
    axes[1, 0].set_title(f"Contornos ({len(contours)} encontrados)")
    axes[1, 0].axis("off")

    lines = cv2.HoughLines(edges_canny, 1, np.pi / 180, threshold=100)
    img_lines = img.copy()
    if lines is not None:
        for rho, theta in lines[:10]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    axes[1, 1].imshow(img_lines)
    axes[1, 1].set_title("Detección de Líneas (Hough)")
    axes[1, 1].axis("off")

    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)

    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(img_circles, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.circle(img_circles, (circle[0], circle[1]), 2, (255, 0, 0), 3)

    axes[1, 2].imshow(img_circles)
    axes[1, 2].set_title("Detección de Círculos (Hough)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
