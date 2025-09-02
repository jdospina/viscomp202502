"""
Tutorial: Operaciones Principales con Imágenes en Python
========================================================

Este tutorial cubre las operaciones fundamentales de procesamiento digital de imágenes
usando OpenCV, NumPy y Matplotlib. Está diseñado para estudiantes de visión por computador.

Autor: Tutorial para curso Visión por Computador 3009228
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure, filters
import os

# Configuración para mostrar imágenes
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

class ImageProcessor:
    """Clase para demostrar operaciones principales de procesamiento de imágenes"""

    def __init__(self):
        self.current_image = None

    def load_image(self, image_path=None, create_sample=True):
        """
        Cargar una imagen o crear una imagen de ejemplo
        """
        if image_path and os.path.exists(image_path):
            # Cargar imagen desde archivo
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            print(f"Imagen cargada: {image_path}")
        elif create_sample:
            # Crear imagen sintética para demostración
            self.current_image = self._create_sample_image()
            print("Imagen sintética creada para demostración")
        else:
            raise ValueError("No se pudo cargar la imagen")

        print(f"Dimensiones: {self.current_image.shape}")
        return self.current_image

    def _create_sample_image(self):
        """Crear una imagen sintética para demostración"""
        # Crear una imagen con diferentes patrones
        img = np.zeros((300, 400, 3), dtype=np.uint8)

        # Fondo gradiente
        for i in range(300):
            img[i, :, 0] = int(255 * i / 300)  # Rojo

        # Rectángulo azul
        img[50:150, 50:150, 2] = 255

        # Círculo verde
        cv2.circle(img, (300, 100), 50, (0, 255, 0), -1)

        # Agregar ruido
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

# =============================================================================
# 1. OPERACIONES BÁSICAS DE MANIPULACIÓN
# =============================================================================

def seccion_1_operaciones_basicas():
    """Demostrar operaciones básicas de manipulación de imágenes"""
    print("="*60)
    print("1. OPERACIONES BÁSICAS DE MANIPULACIÓN")
    print("="*60)

    processor = ImageProcessor()
    img = processor.load_image()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Operaciones Básicas de Manipulación', fontsize=16)

    # Imagen original
    axes[0,0].imshow(img)
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    # 1.1 Redimensionamiento
    img_resized = cv2.resize(img, (200, 150))
    axes[0,1].imshow(img_resized)
    axes[0,1].set_title('Redimensionada (200x150)')
    axes[0,1].axis('off')

    # 1.2 Rotación
    center = (img.shape[1]//2, img.shape[0]//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    axes[0,2].imshow(img_rotated)
    axes[0,2].set_title('Rotación 45°')
    axes[0,2].axis('off')

    # 1.3 Recorte (Cropping)
    img_cropped = img[50:200, 50:300]
    axes[1,0].imshow(img_cropped)
    axes[1,0].set_title('Recortada')
    axes[1,0].axis('off')

    # 1.4 Volteo horizontal
    img_flipped = cv2.flip(img, 1)
    axes[1,1].imshow(img_flipped)
    axes[1,1].set_title('Volteo Horizontal')
    axes[1,1].axis('off')

    # 1.5 Volteo vertical
    img_vflipped = cv2.flip(img, 0)
    axes[1,2].imshow(img_vflipped)
    axes[1,2].set_title('Volteo Vertical')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. ESPACIOS DE COLOR
# =============================================================================

def seccion_2_espacios_color():
    """Demostrar conversiones entre espacios de color"""
    print("="*60)
    print("2. ESPACIOS DE COLOR")
    print("="*60)

    processor = ImageProcessor()
    img_rgb = processor.load_image()

    # Conversiones a diferentes espacios de color
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Espacios de Color', fontsize=16)

    # RGB Original
    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title('RGB Original')
    axes[0,0].axis('off')

    # Escala de grises
    axes[0,1].imshow(img_gray, cmap='gray')
    axes[0,1].set_title('Escala de Grises')
    axes[0,1].axis('off')

    # HSV
    axes[0,2].imshow(img_hsv)
    axes[0,2].set_title('HSV')
    axes[0,2].axis('off')

    # Canales HSV individuales
    axes[1,0].imshow(img_hsv[:,:,0], cmap='hsv')
    axes[1,0].set_title('Canal H (Matiz)')
    axes[1,0].axis('off')

    axes[1,1].imshow(img_hsv[:,:,1], cmap='gray')
    axes[1,1].set_title('Canal S (Saturación)')
    axes[1,1].axis('off')

    axes[1,2].imshow(img_hsv[:,:,2], cmap='gray')
    axes[1,2].set_title('Canal V (Valor)')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

    # Ejemplo práctico: Segmentación por color en HSV
    print("\n--- Ejemplo práctico: Segmentación por color ---")

    # Definir rango para color verde en HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Crear máscara
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_green)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')

    axes[1].imshow(mask_green, cmap='gray')
    axes[1].set_title('Máscara Verde')
    axes[1].axis('off')

    axes[2].imshow(result)
    axes[2].set_title('Solo Píxeles Verdes')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. TRANSFORMACIONES DE INTENSIDAD E HISTOGRAMAS
# =============================================================================

def seccion_3_histogramas():
    """Demostrar operaciones con histogramas y transformaciones de intensidad"""
    print("="*60)
    print("3. HISTOGRAMAS Y TRANSFORMACIONES DE INTENSIDAD")
    print("="*60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3.1 Análisis de histograma
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Histogramas y Transformaciones de Intensidad', fontsize=16)

    # Imagen original y su histograma
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    hist_original = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    axes[1,0].plot(hist_original)
    axes[1,0].set_title('Histograma Original')
    axes[1,0].set_xlim([0, 256])

    # 3.2 Ecualización de histograma
    img_eq = cv2.equalizeHist(img_gray)
    axes[0,1].imshow(img_eq, cmap='gray')
    axes[0,1].set_title('Ecualizada')
    axes[0,1].axis('off')

    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
    axes[1,1].plot(hist_eq)
    axes[1,1].set_title('Histograma Ecualizado')
    axes[1,1].set_xlim([0, 256])

    # 3.3 CLAHE (Ecualización adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    axes[0,2].imshow(img_clahe, cmap='gray')
    axes[0,2].set_title('CLAHE')
    axes[0,2].axis('off')

    hist_clahe = cv2.calcHist([img_clahe], [0], None, [256], [0, 256])
    axes[1,2].plot(hist_clahe)
    axes[1,2].set_title('Histograma CLAHE')
    axes[1,2].set_xlim([0, 256])

    # 3.4 Corrección Gamma
    gamma = 0.5
    img_gamma = np.array(255 * (img_gray / 255) ** gamma, dtype=np.uint8)
    axes[0,3].imshow(img_gamma, cmap='gray')
    axes[0,3].set_title(f'Gamma = {gamma}')
    axes[0,3].axis('off')

    hist_gamma = cv2.calcHist([img_gamma], [0], None, [256], [0, 256])
    axes[1,3].plot(hist_gamma)
    axes[1,3].set_title('Histograma Gamma')
    axes[1,3].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. FILTRADO ESPACIAL Y CONVOLUCIÓN
# =============================================================================

def seccion_4_filtrado_espacial():
    """Demostrar operaciones de filtrado espacial"""
    print("="*60)
    print("4. FILTRADO ESPACIAL Y CONVOLUCIÓN")
    print("="*60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Filtrado Espacial', fontsize=16)

    # Imagen original
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    # 4.1 Filtro Gaussiano (suavizado)
    img_gaussian = cv2.GaussianBlur(img_gray, (15, 15), 0)
    axes[0,1].imshow(img_gaussian, cmap='gray')
    axes[0,1].set_title('Filtro Gaussiano')
    axes[0,1].axis('off')

    # 4.2 Filtro de promedio
    kernel_avg = np.ones((5,5), np.float32) / 25
    img_avg = cv2.filter2D(img_gray, -1, kernel_avg)
    axes[0,2].imshow(img_avg, cmap='gray')
    axes[0,2].set_title('Filtro Promedio')
    axes[0,2].axis('off')

    # 4.3 Filtro mediano (elimina ruido sal y pimienta)
    img_median = cv2.medianBlur(img_gray, 5)
    axes[0,3].imshow(img_median, cmap='gray')
    axes[0,3].set_title('Filtro Mediano')
    axes[0,3].axis('off')

    # 4.4 Detección de bordes - Sobel X
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    axes[1,0].imshow(sobel_x, cmap='gray')
    axes[1,0].set_title('Sobel X')
    axes[1,0].axis('off')

    # 4.5 Detección de bordes - Sobel Y
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    axes[1,1].imshow(sobel_y, cmap='gray')
    axes[1,1].set_title('Sobel Y')
    axes[1,1].axis('off')

    # 4.6 Laplaciano
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    axes[1,2].imshow(laplacian, cmap='gray')
    axes[1,2].set_title('Laplaciano')
    axes[1,2].axis('off')

    # 4.7 Realce de bordes
    kernel_sharpen = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_gray, -1, kernel_sharpen)
    axes[1,3].imshow(img_sharpened, cmap='gray')
    axes[1,3].set_title('Realce de Bordes')
    axes[1,3].axis('off')

    plt.tight_layout()
    plt.show()

    # Demostrar convolución paso a paso
    print("\n--- Ejemplo de Convolución Manual ---")

    # Kernel simple de detección de bordes
    kernel_edge = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)

    # Aplicar convolución manual en una región pequeña
    roi = img_gray[100:103, 100:103].astype(np.float32)
    convolved_pixel = np.sum(roi * kernel_edge)

    print(f"ROI 3x3:")
    print(roi)
    print(f"\nKernel:")
    print(kernel_edge)
    print(f"\nResultado de convolución: {convolved_pixel:.2f}")

# =============================================================================
# 5. DETECCIÓN DE BORDES Y CONTORNOS
# =============================================================================

def seccion_5_deteccion_bordes():
    """Demostrar detección de bordes y contornos"""
    print("="*60)
    print("5. DETECCIÓN DE BORDES Y CONTORNOS")
    print("="*60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Detección de Bordes y Contornos', fontsize=16)

    # Imagen original
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title('Original')
    axes[0,0].axis('off')

    # 5.1 Detector de Canny
    edges_canny = cv2.Canny(img_gray, 50, 150)
    axes[0,1].imshow(edges_canny, cmap='gray')
    axes[0,1].set_title('Canny (50, 150)')
    axes[0,1].axis('off')

    # 5.2 Canny con diferentes umbrales
    edges_canny2 = cv2.Canny(img_gray, 100, 200)
    axes[0,2].imshow(edges_canny2, cmap='gray')
    axes[0,2].set_title('Canny (100, 200)')
    axes[0,2].axis('off')

    # 5.3 Encontrar contornos
    contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    axes[1,0].imshow(img_contours)
    axes[1,0].set_title(f'Contornos ({len(contours)} encontrados)')
    axes[1,0].axis('off')

    # 5.4 Transformada de Hough - Detección de líneas
    lines = cv2.HoughLines(edges_canny, 1, np.pi/180, threshold=100)
    img_lines = img.copy()

    if lines is not None:
        for rho, theta in lines[:10]:  # Mostrar solo las primeras 10 líneas
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    axes[1,1].imshow(img_lines)
    axes[1,1].set_title('Detección de Líneas (Hough)')
    axes[1,1].axis('off')

    # 5.5 Detección de círculos
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)

    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(img_circles, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.circle(img_circles, (circle[0], circle[1]), 2, (255, 0, 0), 3)

    axes[1,2].imshow(img_circles)
    axes[1,2].set_title('Detección de Círculos (Hough)')
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. OPERACIONES MORFOLÓGICAS
# =============================================================================

def seccion_6_morfologia():
    """Demostrar operaciones morfológicas"""
    print("="*60)
    print("6. OPERACIONES MORFOLÓGICAS")
    print("="*60)

    processor = ImageProcessor()
    img = processor.load_image()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crear una imagen binaria
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Elemento estructural
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Operaciones Morfológicas', fontsize=16)

    # Imagen binaria original
    axes[0,0].imshow(img_binary, cmap='gray')
    axes[0,0].set_title('Imagen Binaria')
    axes[0,0].axis('off')

    # 6.1 Erosión
    erosion = cv2.erode(img_binary, kernel, iterations=1)
    axes[0,1].imshow(erosion, cmap='gray')
    axes[0,1].set_title('Erosión')
    axes[0,1].axis('off')

    # 6.2 Dilatación
    dilation = cv2.dilate(img_binary, kernel, iterations=1)
    axes[0,2].imshow(dilation, cmap='gray')
    axes[0,2].set_title('Dilatación')
    axes[0,2].axis('off')

    # 6.3 Apertura
    opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    axes[0,3].imshow(opening, cmap='gray')
    axes[0,3].set_title('Apertura')
    axes[0,3].axis('off')

    # 6.4 Cierre
    closing = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    axes[1,0].imshow(closing, cmap='gray')
    axes[1,0].set_title('Cierre')
    axes[1,0].axis('off')

    # 6.5 Gradiente morfológico
    gradient = cv2.morphologyEx(img_binary, cv2.MORPH_GRADIENT, kernel)
    axes[1,1].imshow(gradient, cmap='gray')
    axes[1,1].set_title('Gradiente')
    axes[1,1].axis('off')

    # 6.6 Top Hat
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    axes[1,2].imshow(tophat, cmap='gray')
    axes[1,2].set_title('Top Hat')
    axes[1,2].axis('off')

    # 6.7 Black Hat
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    axes[1,3].imshow(blackhat, cmap='gray')
    axes[1,3].set_title('Black Hat')
    axes[1,3].axis('off')

    plt.tight_layout()
    plt.show()

# =============================================================================
# 7. FUNCIÓN PRINCIPAL Y EJEMPLOS COMPLETOS
# =============================================================================

def ejemplo_pipeline_completo():
    """Ejemplo de un pipeline completo de procesamiento"""
    print("="*60)
    print("7. PIPELINE COMPLETO DE PROCESAMIENTO")
    print("="*60)

    processor = ImageProcessor()
    img_original = processor.load_image()

    print("Aplicando pipeline de procesamiento paso a paso...")

    # Paso 1: Conversión a escala de grises
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)

    # Paso 2: Reducción de ruido
    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Paso 3: Realce de contraste
    img_enhanced = cv2.equalizeHist(img_denoised)

    # Paso 4: Detección de bordes
    edges = cv2.Canny(img_enhanced, 50, 150)

    # Paso 5: Operaciones morfológicas para limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Paso 6: Encontrar contornos
    contours, _ = cv2.findContours(edges_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Paso 7: Filtrar contornos por área
    min_area = 100
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Visualizar resultados
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Pipeline Completo de Procesamiento', fontsize=16)

    stages = [
        (img_original, 'Original', None),
        (img_gray, 'Escala de Grises', 'gray'),
        (img_denoised, 'Reducción de Ruido', 'gray'),
        (img_enhanced, 'Realce de Contraste', 'gray'),
        (edges, 'Detección de Bordes', 'gray'),
        (edges_clean, 'Limpieza Morfológica', 'gray'),
    ]

    for i, (image, title, cmap) in enumerate(stages):
        row, col = i // 4, i % 4
        axes[row, col].imshow(image, cmap=cmap)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

    # Resultado final con contornos
    img_result = img_original.copy()
    cv2.drawContours(img_result, filtered_contours, -1, (0, 255, 0), 2)
    axes[1, 2].imshow(img_result)
    axes[1, 2].set_title(f'Contornos Finales ({len(filtered_contours)})')
    axes[1, 2].axis('off')

    # Análisis estadístico
    axes[1, 3].axis('off')
    stats_text = f"""Análisis Final:

Contornos encontrados: {len(contours)}
Contornos filtrados: {len(filtered_contours)}
Área mínima: {min_area} píxeles

Dimensiones originales:
{img_original.shape[0]} x {img_original.shape[1]}

Rango de intensidades:
Min: {np.min(img_gray)}
Max: {np.max(img_gray)}
Media: {np.mean(img_gray):.1f}"""

    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=10,
                   verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

def main():
    """Función principal que ejecuta todo el tutorial"""
    print("TUTORIAL: OPERACIONES PRINCIPALES CON IMÁGENES EN PYTHON")
    print("=" * 80)
    print("Este tutorial cubre las operaciones fundamentales de procesamiento digital")
    print("de imágenes usando OpenCV, NumPy y Matplotlib.")
    print("=" * 80)

    # Ejecutar todas las secciones
    try:
        seccion_1_operaciones_basicas()
        seccion_2_espacios_color()
        seccion_3_histogramas()
        seccion_4_filtrado_espacial()
        seccion_5_deteccion_bordes()
        seccion_6_morfologia()
        ejemplo_pipeline_completo()

        print("\n" + "="*60)
        print("TUTORIAL COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("Has aprendido las operaciones principales de procesamiento de imágenes:")
        print("✓ Operaciones básicas de manipulación")
        print("✓ Espacios de color y conversiones")
        print("✓ Histogramas y transformaciones de intensidad")
        print("✓ Filtrado espacial y convolución")
        print("✓ Detección de bordes y contornos")
        print("✓ Operaciones morfológicas")
        print("✓ Pipeline completo de procesamiento")
        print("\nPróximos pasos sugeridos:")
        print("• Experimentar con imágenes reales")
        print("• Ajustar parámetros según el tipo de imagen")
        print("• Combinar diferentes técnicas para casos específicos")
        print("• Explorar bibliotecas avanzadas como scikit-image")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        print("Asegúrate de tener instaladas las dependencias: opencv-python, numpy, matplotlib, scipy, scikit-image")

# =============================================================================
# 8. FUNCIONES AUXILIARES Y EJEMPLOS AVANZADOS
# =============================================================================

def crear_ejemplos_practicos():
    """Ejemplos prácticos adicionales para casos específicos"""
    print("="*60)
    print("8. EJEMPLOS PRÁCTICOS AVANZADOS")
    print("="*60)

    # 8.1 Segmentación por umbralización adaptativa
    def segmentacion_adaptativa():
        processor = ImageProcessor()
        img = processor.load_image()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Diferentes tipos de umbralización
        _, thresh_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        thresh_adaptive_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
        thresh_adaptive_gaussian = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 11, 2)
        thresh_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Técnicas de Umbralización', fontsize=16)

        axes[0,0].imshow(thresh_binary, cmap='gray')
        axes[0,0].set_title('Umbral Fijo (127)')
        axes[0,0].axis('off')

        axes[0,1].imshow(thresh_adaptive_mean, cmap='gray')
        axes[0,1].set_title('Adaptativo (Media)')
        axes[0,1].axis('off')

        axes[1,0].imshow(thresh_adaptive_gaussian, cmap='gray')
        axes[1,0].set_title('Adaptativo (Gaussiano)')
        axes[1,0].axis('off')

        axes[1,1].imshow(thresh_otsu, cmap='gray')
        axes[1,1].set_title('Otsu')
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.show()

    # 8.2 Análisis de textura
    def analisis_textura():
        processor = ImageProcessor()
        img = processor.load_image()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Filtros de Gabor para análisis de textura
        def aplicar_filtro_gabor(imagen, freq, theta):
            kernel_real, kernel_imag = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
            filtered_real = cv2.filter2D(imagen, cv2.CV_8UC3, kernel_real)
            filtered_imag = cv2.filter2D(imagen, cv2.CV_8UC3, kernel_imag)
            return np.sqrt(filtered_real**2 + filtered_imag**2)

        # Aplicar filtros con diferentes orientaciones
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Análisis de Textura con Filtros de Gabor', fontsize=16)

        axes[0,0].imshow(img_gray, cmap='gray')
        axes[0,0].set_title('Imagen Original')
        axes[0,0].axis('off')

        orientaciones = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        titulos = ['0°', '45°', '90°', '135°', '180°']

        for i, (theta, titulo) in enumerate(zip(orientaciones, titulos)):
            if i < 5:  # Solo mostrar 5 orientaciones
                filtered = aplicar_filtro_gabor(img_gray, 0.1, theta)
                row, col = (i+1) // 3, (i+1) % 3
                axes[row, col].imshow(filtered, cmap='gray')
                axes[row, col].set_title(f'Gabor {titulo}')
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    # 8.3 Detección de características con métodos modernos
    def deteccion_caracteristicas_modernas():
        processor = ImageProcessor()
        img = processor.load_image()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detectores de características
        # SIFT
        sift = cv2.SIFT_create()
        kp_sift, des_sift = sift.detectAndCompute(img_gray, None)

        # ORB
        orb = cv2.ORB_create()
        kp_orb, des_orb = orb.detectAndCompute(img_gray, None)

        # FAST
        fast = cv2.FastFeatureDetector_create()
        kp_fast = fast.detect(img_gray, None)

        # Visualizar características
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Detección de Características', fontsize=16)

        axes[0].imshow(img)
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')

        # SIFT
        img_sift = cv2.drawKeypoints(img_gray, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[1].imshow(img_sift, cmap='gray')
        axes[1].set_title(f'SIFT ({len(kp_sift)} puntos)')
        axes[1].axis('off')

        # ORB
        img_orb = cv2.drawKeypoints(img_gray, kp_orb, None, color=(0,255,0), flags=0)
        axes[2].imshow(img_orb, cmap='gray')
        axes[2].set_title(f'ORB ({len(kp_orb)} puntos)')
        axes[2].axis('off')

        # FAST
        img_fast = cv2.drawKeypoints(img_gray, kp_fast, None, color=(255,0,0), flags=0)
        axes[3].imshow(img_fast, cmap='gray')
        axes[3].set_title(f'FAST ({len(kp_fast)} puntos)')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

        return kp_sift, des_sift, kp_orb, des_orb

    # 8.4 Matching de características entre imágenes
    def matching_caracteristicas():
        # Para este ejemplo, crearemos dos versiones de la misma imagen
        processor = ImageProcessor()
        img1 = processor.load_image()
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        # Crear una segunda imagen con transformación
        center = (img1.shape[1]//2, img1.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 30, 0.8)  # Rotar 30° y escalar 0.8
        img2 = cv2.warpAffine(img1, rotation_matrix, (img1.shape[1], img1.shape[0]))
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Detectar características con ORB
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        # Matching de características
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Dibujar matches
        img_matches = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, matches[:20], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Matching de Características entre Imágenes', fontsize=16)

        axes[0,0].imshow(img1)
        axes[0,0].set_title('Imagen 1 (Original)')
        axes[0,0].axis('off')

        axes[0,1].imshow(img2)
        axes[0,1].set_title('Imagen 2 (Transformada)')
        axes[0,1].axis('off')

        axes[1,:].remove()
        ax_matches = fig.add_subplot(2, 1, 2)
        ax_matches.imshow(img_matches, cmap='gray')
        ax_matches.set_title(f'Matches encontrados: {len(matches)} (mostrando los 20 mejores)')
        ax_matches.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Características detectadas:")
        print(f"  Imagen 1: {len(kp1)} puntos")
        print(f"  Imagen 2: {len(kp2)} puntos")
        print(f"  Matches totales: {len(matches)}")
        print(f"  Distancia promedio: {np.mean([m.distance for m in matches]):.2f}")

    # Ejecutar ejemplos
    print("8.1 Segmentación por umbralización adaptativa")
    segmentacion_adaptativa()

    print("\n8.2 Análisis de textura con filtros de Gabor")
    analisis_textura()

    print("\n8.3 Detección de características modernas")
    deteccion_caracteristicas_modernas()

    print("\n8.4 Matching de características")
    matching_caracteristicas()

# =============================================================================
# 9. EJERCICIOS PRÁCTICOS PARA ESTUDIANTES
# =============================================================================

def ejercicios_para_estudiantes():
    """Propone ejercicios prácticos para que los estudiantes implementen"""
    print("="*60)
    print("9. EJERCICIOS PRÁCTICOS PARA ESTUDIANTES")
    print("="*60)

    ejercicios = [
        {
            "titulo": "Ejercicio 1: Contador de Objetos",
            "descripcion": """
Objetivo: Implementar un contador automático de objetos en una imagen.

Pasos a seguir:
1. Cargar una imagen con múltiples objetos similares
2. Convertir a escala de grises
3. Aplicar umbralización para segmentar objetos
4. Usar operaciones morfológicas para limpiar la imagen
5. Encontrar contornos y filtrar por área mínima
6. Contar y marcar cada objeto encontrado
7. Mostrar el resultado final con el conteo

Criterios de evaluación:
- Precisión en la detección (¿cuenta correctamente?)
- Calidad del preprocesamiento
- Robustez ante diferentes tipos de imagen""",
            "codigo_base": """
def contador_objetos(imagen_path):
    # TODO: Implementar contador de objetos
    img = cv2.imread(imagen_path)
    # Tu código aquí...
    return numero_objetos, imagen_resultado
"""
        },
        {
            "titulo": "Ejercicio 2: Detector de Bordes Personalizado",
            "descripción": """
Objetivo: Implementar tu propio detector de bordes combinando diferentes técnicas.

Pasos a seguir:
1. Implementar gradientes en X e Y usando convolución manual
2. Calcular magnitud y dirección del gradiente
3. Implementar supresión de no-máximos
4. Aplicar umbralización por histéresis
5. Comparar resultados con el detector de Canny de OpenCV

Criterios de evaluación:
- Implementación correcta de cada paso del algoritmo
- Calidad de los bordes detectados
- Análisis comparativo con Canny""",
            "codigo_base": """
def mi_detector_bordes(imagen, umbral_bajo, umbral_alto):
    # TODO: Implementar detector de bordes personalizado
    # Paso 1: Calcular gradientes
    # Paso 2: Magnitud y dirección
    # Paso 3: Supresión de no-máximos
    # Paso 4: Umbralización por histéresis
    return bordes_detectados
"""
        },
        {
            "titulo": "Ejercicio 3: Corrección de Iluminación",
            "descripcion": """
Objetivo: Corregir imágenes con iluminación no uniforme.

Pasos a seguir:
1. Detectar la iluminación de fondo usando filtro Gaussiano grande
2. Restar o dividir la imagen original por la iluminación estimada
3. Normalizar el resultado al rango [0, 255]
4. Comparar antes y después usando histogramas
5. Implementar al menos dos métodos diferentes

Criterios de evaluación:
- Efectividad de la corrección
- Preservación de detalles importantes
- Análisis cuantitativo de la mejora""",
            "codigo_base": """
def corregir_iluminacion(imagen, metodo='division'):
    # TODO: Implementar corrección de iluminación
    # Método 1: División por fondo estimado
    # Método 2: Sustracción de fondo
    # Método 3: Tu propuesta creativa
    return imagen_corregida
"""
        },
        {
            "titulo": "Ejercicio 4: Análisis de Formas",
            "descripcion": """
Objetivo: Clasificar objetos según su forma geométrica.

Pasos a seguir:
1. Segmentar objetos de la imagen
2. Para cada objeto, calcular descriptores de forma:
   - Área y perímetro
   - Relación de aspecto
   - Solidez (área/área del hull convexo)
   - Circularidad (4π·área/perímetro²)
3. Clasificar formas basándose en estos descriptores
4. Visualizar resultados con etiquetas

Criterios de evaluación:
- Precisión en el cálculo de descriptores
- Lógica de clasificación
- Presentación clara de resultados""",
            "codigo_base": """
def clasificar_formas(imagen):
    # TODO: Implementar clasificador de formas
    contornos = encontrar_contornos(imagen)
    formas_clasificadas = []

    for contorno in contornos:
        descriptores = calcular_descriptores(contorno)
        forma = clasificar_por_descriptores(descriptores)
        formas_clasificadas.append(forma)

    return formas_clasificadas, imagen_resultado
"""
        },
        {
            "titulo": "Ejercicio 5: Sistema de Medición",
            "descripcion": """
Objetivo: Crear un sistema para medir distancias reales en imágenes.

Pasos a seguir:
1. Detectar un objeto de referencia de tamaño conocido
2. Calcular la relación píxeles/unidad real
3. Permitir al usuario marcar dos puntos en la imagen
4. Calcular y mostrar la distancia real entre los puntos
5. Incluir manejo de errores y validaciones

Criterios de evaluación:
- Precisión de las mediciones
- Interfaz de usuario (aunque sea básica)
- Robustez del sistema de calibración""",
            "codigo_base": """
def sistema_medicion(imagen, tamaño_referencia_mm):
    # TODO: Implementar sistema de medición
    factor_escala = calibrar_escala(imagen, tamaño_referencia_mm)

    def medir_distancia(punto1, punto2):
        distancia_pixeles = calcular_distancia_euclidiana(punto1, punto2)
        distancia_real = distancia_pixeles * factor_escala
        return distancia_real

    return medir_distancia
"""
        }
    ]

    for i, ejercicio in enumerate(ejercicios, 1):
        print(f"\n{'-'*40}")
        print(f"EJERCICIO {i}: {ejercicio['titulo']}")
        print(f"{'-'*40}")
        print(ejercicio['descripcion'])
        if 'codigo_base' in ejercicio:
            print(f"\nCódigo base sugerido:")
            print(ejercicio['codigo_base'])

    print(f"\n{'='*60}")
    print("INSTRUCCIONES GENERALES PARA LOS EJERCICIOS:")
    print("="*60)
    print("""
1. Cada ejercicio debe incluir:
   - Código comentado y bien estructurado
   - Visualización de resultados (antes/después)
   - Análisis de parámetros utilizados
   - Discusión de limitaciones y posibles mejoras

2. Criterios de evaluación:
   - Correctitud técnica (40%)
   - Calidad del código y documentación (25%)
   - Análisis y discusión de resultados (20%)
   - Creatividad y mejoras adicionales (15%)

3. Entrega sugerida:
   - Notebook de Jupyter con código ejecutable
   - Imágenes de prueba utilizadas
   - Breve informe con conclusiones

4. Recursos adicionales recomendados:
   - Documentación oficial de OpenCV
   - Papers with Code para ideas avanzadas
   - Stack Overflow para resolver dudas específicas
""")

# =============================================================================
# 10. RECURSOS ADICIONALES Y REFERENCIAS
# =============================================================================

def recursos_adicionales():
    """Proporciona recursos adicionales para profundizar"""
    print("="*60)
    print("10. RECURSOS ADICIONALES Y REFERENCIAS")
    print("="*60)

    recursos = {
        "Libros Recomendados": [
            "• Szeliski, R. (2022). Computer Vision: Algorithms and Applications (2nd ed.)",
            "• Gonzalez, R. & Woods, R. (2017). Digital Image Processing (4th ed.)",
            "• Bradski, G. & Kaehler, A. (2008). Learning OpenCV",
            "• Prince, S. (2012). Computer Vision: Models, Learning, and Inference"
        ],
        "Cursos Online Gratuitos": [
            "• CS231n Stanford - Convolutional Neural Networks for Visual Recognition",
            "• First Principles of Computer Vision (Columbia University)",
            "• Computer Vision Basics (University at Buffalo)",
            "• PyImageSearch - Tutoriales prácticos de CV"
        ],
        "Bibliotecas Esenciales": [
            "• OpenCV - Biblioteca principal para visión por computador",
            "• scikit-image - Herramientas de procesamiento de imágenes",
            "• Pillow (PIL) - Manipulación básica de imágenes",
            "• Matplotlib - Visualización",
            "• NumPy/SciPy - Computación científica",
            "• SimpleITK - Procesamiento de imágenes médicas"
        ],
        "Datasets para Practicar": [
            "• CIFAR-10/100 - Clasificación de objetos",
            "• ImageNet - Base de datos masiva de imágenes",
            "• COCO - Detección y segmentación de objetos",
            "• Pascal VOC - Detección de objetos",
            "• CelebA - Análisis facial",
            "• MNIST - Reconocimiento de dígitos",
            "• Cityscapes - Segmentación de escenas urbanas"
        ],
        "Herramientas de Desarrollo": [
            "• Jupyter Lab/Notebook - Desarrollo interactivo",
            "• Google Colab - Entorno gratuito con GPU",
            "• Anaconda - Gestión de paquetes Python",
            "• Git - Control de versiones",
            "• Docker - Contenedores para reproducibilidad"
        ],
        "Conferencias Principales": [
            "• CVPR - Conference on Computer Vision and Pattern Recognition",
            "• ICCV - International Conference on Computer Vision",
            "• ECCV - European Conference on Computer Vision",
            "• NeurIPS - Neural Information Processing Systems",
            "• MICCAI - Medical Image Computing and Computer Assisted Intervention"
        ]
    }

    for categoria, items in recursos.items():
        print(f"\n{categoria.upper()}:")
        print("-" * len(categoria))
        for item in items:
            print(item)

    print(f"\n{'='*60}")
    print("PRÓXIMOS PASOS EN TU APRENDIZAJE:")
    print("="*60)
    print("""
1. NIVEL BÁSICO (Completado con este tutorial):
   ✓ Operaciones básicas con imágenes
   ✓ Espacios de color y filtrado
   ✓ Detección de bordes y contornos

2. NIVEL INTERMEDIO (Próximos 3-6 meses):
   → Geometría de múltiples vistas
   → Structure from Motion (SfM)
   → Calibración de cámaras
   → Visión estéreo y reconstrucción 3D

3. NIVEL AVANZADO (6-12 meses):
   → Deep Learning para CV (CNNs, ResNet, YOLO)
   → Segmentación semántica (U-Net, Mask R-CNN)
   → Generative Adversarial Networks (GANs)
   → Vision Transformers (ViT)

4. ESPECIALIZACIÓN (1+ años):
   → Visión médica computarizada
   → Vehículos autónomos
   → Realidad aumentada/virtual
   → Robótica e interacción humano-robot
""")

if __name__ == "__main__":
    main()
    crear_ejemplos_practicos()
    ejercicios_para_estudiantes()
    recursos_adicionales()