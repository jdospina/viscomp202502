"""
Tutorial: Registro de Imágenes (Image Registration)
====================================================

Este tutorial introduce los conceptos fundamentales del registro de imágenes
y proporciona implementaciones prácticas usando Python.

Autor: Tutorial para el curso Visión por Computador 3009228
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage

# Configuración para mejores visualizaciones
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.size'] = 10


# ============================================================================
# PARTE 1: TEORÍA Y CONCEPTOS FUNDAMENTALES
# ============================================================================

"""
¿QUÉ ES EL REGISTRO DE IMÁGENES?

El registro de imágenes es el proceso de alinear dos o más imágenes de la misma
escena. Este proceso busca encontrar una transformación geométrica que mapee
los píxeles de una imagen (imagen móvil) a las coordenadas correspondientes
en otra imagen (imagen fija o de referencia).

APLICACIONES:
- Imágenes médicas: alinear scans de diferentes modalidades (MRI, CT)
- Imágenes satelitales: monitorear cambios temporales
- Panorámicas: unir múltiples imágenes
- Visión estéreo: correspondencias entre vistas
- Seguimiento de objetos en video

COMPONENTES PRINCIPALES:
1. Imagen fija (reference/fixed): la imagen objetivo
2. Imagen móvil (moving): la imagen que queremos transformar
3. Transformación geométrica: función que mapea coordenadas
4. Métrica de similitud: mide qué tan bien están alineadas las imágenes
5. Optimizador: busca la mejor transformación

TIPOS DE TRANSFORMACIONES:
"""


# ============================================================================
# PARTE 2: GENERACIÓN DE IMÁGENES SINTÉTICAS
# ============================================================================

def crear_imagen_sintetica(size=256, tipo='patron'):
    """
    Crea imágenes sintéticas para demostración.
    
    Args:
        size: tamaño de la imagen (size x size)
        tipo: 'patron', 'cuadros', 'circulo', 'texto'
    
    Returns:
        imagen en escala de grises (numpy array)
    """
    imagen = np.zeros((size, size), dtype=np.uint8)
    
    if tipo == 'patron':
        # Patrón de tablero de ajedrez
        square_size = size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    imagen[i*square_size:(i+1)*square_size, 
                          j*square_size:(j+1)*square_size] = 255
        
        # Añadir círculo en el centro
        cv2.circle(imagen, (size//2, size//2), size//6, 128, -1)
        
        # Añadir rectángulo
        cv2.rectangle(imagen, (size//4, size//4), 
                     (3*size//4, size//3), 200, 3)
        
    elif tipo == 'cuadros':
        # Varios cuadrados de diferentes tamaños
        cv2.rectangle(imagen, (20, 20), (100, 100), 255, -1)
        cv2.rectangle(imagen, (size-120, 40), (size-40, 120), 180, -1)
        cv2.rectangle(imagen, (size//2-30, size-80), 
                     (size//2+30, size-20), 220, -1)
        
    elif tipo == 'circulo':
        # Patrón circular
        cv2.circle(imagen, (size//2, size//2), size//3, 255, -1)
        cv2.circle(imagen, (size//2, size//2), size//4, 0, -1)
        cv2.circle(imagen, (size//2, size//2), size//6, 200, -1)
        
    elif tipo == 'texto':
        # Añadir texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imagen, 'CV', (size//4, size//2), 
                   font, 3, 255, 5)
        cv2.circle(imagen, (3*size//4, size//4), 30, 255, -1)
        cv2.circle(imagen, (3*size//4, 3*size//4), 40, 200, -1)
    
    return imagen


def aplicar_transformacion(imagen, tipo, params):
    """
    Aplica una transformación geométrica a una imagen.
    
    Args:
        imagen: imagen de entrada
        tipo: 'traslacion', 'rotacion', 'afin', 'proyectiva'
        params: diccionario con parámetros de la transformación
    
    Returns:
        imagen transformada, matriz de transformación
    """
    h, w = imagen.shape[:2]
    center = (w / 2, h / 2)
    
    if tipo == 'traslacion':
        # Traslación pura: T = [1 0 tx; 0 1 ty]
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        M = np.float32([[1, 0, tx],
                        [0, 1, ty]])
        imagen_trans = cv2.warpAffine(imagen, M, (w, h))
        
    elif tipo == 'rotacion':
        # Rotación alrededor del centro
        angulo = params.get('angulo', 0)
        M = cv2.getRotationMatrix2D(center, angulo, 1.0)
        imagen_trans = cv2.warpAffine(imagen, M, (w, h))
        
    elif tipo == 'rigida':
        # Transformación rígida (rotación + traslación)
        angulo = params.get('angulo', 0)
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        M = cv2.getRotationMatrix2D(center, angulo, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        imagen_trans = cv2.warpAffine(imagen, M, (w, h))
        
    elif tipo == 'afin':
        # Transformación afín (6 parámetros)
        # Incluye rotación, traslación, escalado y cizallamiento
        angulo = params.get('angulo', 0)
        escala = params.get('escala', 1.0)
        tx = params.get('tx', 0)
        ty = params.get('ty', 0)
        shear = params.get('shear', 0)
        
        M = cv2.getRotationMatrix2D(center, angulo, escala)
        M[0, 2] += tx
        M[1, 2] += ty
        # Añadir cizallamiento
        M[0, 1] += shear
        imagen_trans = cv2.warpAffine(imagen, M, (w, h))
        
    elif tipo == 'proyectiva':
        # Transformación proyectiva/homografía (8 parámetros)
        # Define 4 puntos de origen y destino
        src_points = params.get('src_points', 
                               np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        dst_points = params.get('dst_points',
                               np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        imagen_trans = cv2.warpPerspective(imagen, M, (w, h))
    
    else:
        raise ValueError(f"Tipo de transformación '{tipo}' no reconocido")
    
    return imagen_trans, M


# ============================================================================
# PARTE 3: MÉTODOS DE REGISTRO BASADOS EN INTENSIDAD
# ============================================================================

def calcular_similitud(img1, img2, metrica='mse'):
    """
    Calcula una métrica de similitud entre dos imágenes.
    
    Args:
        img1, img2: imágenes a comparar
        metrica: 'mse', 'ncc', 'mi' (mutual information simplificada)
    
    Returns:
        valor de similitud
    """
    if metrica == 'mse':
        # Error Cuadrático Medio (menor es mejor)
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return mse
    
    elif metrica == 'ncc':
        # Coeficiente de Correlación Cruzada Normalizada (mayor es mejor)
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
        ncc = np.mean(img1_norm * img2_norm)
        return ncc
    
    elif metrica == 'mi':
        # Información Mutua simplificada (mayor es mejor)
        # Usando histograma 2D
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=20)
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Evitar log(0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        return mi
    
    else:
        raise ValueError(f"Métrica '{metrica}' no reconocida")


def registro_busqueda_exhaustiva(img_fija, img_movil, 
                                 rango_tx=(-20, 20), 
                                 rango_ty=(-20, 20),
                                 paso=1,
                                 metrica='mse'):
    """
    Registro por búsqueda exhaustiva de traslación.
    Prueba todas las combinaciones de traslación en el rango dado.
    
    Args:
        img_fija: imagen de referencia
        img_movil: imagen a registrar (ya desplazada)
        rango_tx, rango_ty: rangos de búsqueda (min, max)
        paso: incremento en la búsqueda
        metrica: métrica de similitud a usar
    
    Returns:
        mejor transformación, mejor valor de similitud, historia de búsqueda
    
    Nota: La transformación devuelta lleva img_movil -> img_fija
    """
    mejor_similitud = float('inf') if metrica == 'mse' else float('-inf')
    mejor_tx, mejor_ty = 0, 0
    historia = []
    
    # Búsqueda exhaustiva
    for tx in range(rango_tx[0], rango_tx[1] + 1, paso):
        for ty in range(rango_ty[0], rango_ty[1] + 1, paso):
            # Aplicar traslación para alinear con img_fija
            img_trans, _ = aplicar_transformacion(
                img_movil, 'traslacion', {'tx': tx, 'ty': ty}
            )
            
            # Calcular similitud
            sim = calcular_similitud(img_fija, img_trans, metrica)
            historia.append((tx, ty, sim))
            
            # Actualizar mejor resultado
            if metrica == 'mse':
                if sim < mejor_similitud:
                    mejor_similitud = sim
                    mejor_tx, mejor_ty = tx, ty
            else:  # ncc o mi
                if sim > mejor_similitud:
                    mejor_similitud = sim
                    mejor_tx, mejor_ty = tx, ty
    
    # Crear matriz de transformación óptima
    M_optima = np.float32([[1, 0, mejor_tx],
                           [0, 1, mejor_ty]])
    
    return M_optima, mejor_similitud, historia


# ============================================================================
# PARTE 4: MÉTODOS BASADOS EN CARACTERÍSTICAS
# ============================================================================

def registro_con_caracteristicas(img_fija, img_movil, 
                                metodo='orb',
                                max_features=500):
    """
    Registro usando detección y emparejamiento de características.
    
    Args:
        img_fija: imagen de referencia
        img_movil: imagen a registrar
        metodo: 'orb', 'sift', 'akaze'
        max_features: número máximo de características a detectar
    
    Returns:
        matriz de homografía, puntos correspondientes, imagen con matches
    """
    # Crear detector de características
    if metodo == 'orb':
        detector = cv2.ORB_create(max_features)
    elif metodo == 'sift':
        detector = cv2.SIFT_create(max_features)
    elif metodo == 'akaze':
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Método '{metodo}' no reconocido")
    
    # Detectar y computar descriptores
    kp1, des1 = detector.detectAndCompute(img_fija, None)
    kp2, des2 = detector.detectAndCompute(img_movil, None)
    
    if des1 is None or des2 is None:
        print("No se detectaron suficientes características")
        return None, None, None
    
    # Emparejar características
    if metodo == 'orb' or metodo == 'akaze':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # Aplicar ratio test de Lowe
    buenos_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                buenos_matches.append(m)
    
    print(f"Características detectadas: {len(kp1)} en img1, {len(kp2)} en img2")
    print(f"Matches buenos: {len(buenos_matches)}")
    
    if len(buenos_matches) < 4:
        print("No hay suficientes matches para calcular homografía")
        return None, None, None
    
    # Extraer puntos correspondientes
    pts_fija = np.float32([kp1[m.queryIdx].pt for m in buenos_matches])
    pts_movil = np.float32([kp2[m.trainIdx].pt for m in buenos_matches])
    
    # Calcular homografía con RANSAC
    H, mask = cv2.findHomography(pts_movil, pts_fija, cv2.RANSAC, 5.0)
    
    inliers = mask.ravel().tolist()
    print(f"Inliers después de RANSAC: {sum(inliers)}/{len(inliers)}")
    
    # Crear imagen de visualización de matches
    img_matches = cv2.drawMatches(
        img_fija, kp1, img_movil, kp2, buenos_matches, None,
        matchesMask=inliers,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return H, (pts_fija, pts_movil, inliers), img_matches


# ============================================================================
# PARTE 5: VISUALIZACIONES
# ============================================================================

def visualizar_registro(img_fija, img_movil, img_registrada, 
                       titulo="Proceso de Registro"):
    """
    Visualiza el proceso de registro: antes y después.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fila 1: Imágenes individuales
    axes[0, 0].imshow(img_fija, cmap='gray')
    axes[0, 0].set_title('Imagen Fija (Referencia)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_movil, cmap='gray')
    axes[0, 1].set_title('Imagen Móvil (Original)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_registrada, cmap='gray')
    axes[0, 2].set_title('Imagen Móvil (Registrada)')
    axes[0, 2].axis('off')
    
    # Fila 2: Comparaciones
    # Superposición antes del registro
    superposicion_antes = np.zeros((*img_fija.shape, 3), dtype=np.uint8)
    superposicion_antes[:, :, 0] = img_fija  # Rojo
    superposicion_antes[:, :, 1] = img_movil  # Verde
    
    axes[1, 0].imshow(superposicion_antes)
    axes[1, 0].set_title('Superposición Antes\n(Rojo: fija, Verde: móvil)')
    axes[1, 0].axis('off')
    
    # Superposición después del registro
    superposicion_despues = np.zeros((*img_fija.shape, 3), dtype=np.uint8)
    superposicion_despues[:, :, 0] = img_fija  # Rojo
    superposicion_despues[:, :, 1] = img_registrada  # Verde
    
    axes[1, 1].imshow(superposicion_despues)
    axes[1, 1].set_title('Superposición Después\n(Rojo: fija, Verde: registrada)')
    axes[1, 1].axis('off')
    
    # Diferencia absoluta
    diferencia = cv2.absdiff(img_fija, img_registrada)
    axes[1, 2].imshow(diferencia, cmap='hot')
    axes[1, 2].set_title('Diferencia Absoluta\n(más oscuro = mejor alineación)')
    axes[1, 2].axis('off')
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def visualizar_superficie_busqueda(historia, metrica='mse'):
    """
    Visualiza la superficie de similitud explorada en la búsqueda.
    """
    # Extraer datos
    tx_vals = [h[0] for h in historia]
    ty_vals = [h[1] for h in historia]
    sim_vals = [h[2] for h in historia]
    
    # Crear malla
    tx_unique = sorted(set(tx_vals))
    ty_unique = sorted(set(ty_vals))
    
    superficie = np.zeros((len(ty_unique), len(tx_unique)))
    for tx, ty, sim in historia:
        i = ty_unique.index(ty)
        j = tx_unique.index(tx)
        superficie[i, j] = sim
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Superficie 3D convertida a heatmap
    im = axes[0].imshow(superficie, extent=[min(tx_unique), max(tx_unique),
                                            min(ty_unique), max(ty_unique)],
                       origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Traslación en X (píxeles)')
    axes[0].set_ylabel('Traslación en Y (píxeles)')
    axes[0].set_title(f'Superficie de Similitud ({metrica.upper()})')
    plt.colorbar(im, ax=axes[0])
    
    # Marcar el óptimo
    if metrica == 'mse':
        idx_optimo = np.argmin(sim_vals)
    else:
        idx_optimo = np.argmax(sim_vals)
    
    tx_optimo, ty_optimo = tx_vals[idx_optimo], ty_vals[idx_optimo]
    axes[0].plot(tx_optimo, ty_optimo, 'r*', markersize=20, 
                label=f'Óptimo: ({tx_optimo}, {ty_optimo})')
    axes[0].legend()
    
    # Gráfica de convergencia (si hubiera iteraciones)
    axes[1].plot(sim_vals, 'b-', linewidth=2)
    axes[1].axhline(y=sim_vals[idx_optimo], color='r', linestyle='--',
                   label=f'Valor óptimo: {sim_vals[idx_optimo]:.2f}')
    axes[1].set_xlabel('Combinación de parámetros evaluada')
    axes[1].set_ylabel(f'Similitud ({metrica.upper()})')
    axes[1].set_title('Evolución de la Búsqueda')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def crear_tablero_comparativo(imagenes_dict):
    """
    Crea un tablero comparativo de múltiples transformaciones.
    
    Args:
        imagenes_dict: diccionario {'nombre': imagen}
    """
    n_imagenes = len(imagenes_dict)
    cols = min(4, n_imagenes)
    rows = (n_imagenes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_imagenes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (nombre, imagen) in enumerate(imagenes_dict.items()):
        axes[idx].imshow(imagen, cmap='gray')
        axes[idx].set_title(nombre)
        axes[idx].axis('off')
    
    # Ocultar ejes extras
    for idx in range(n_imagenes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# PARTE 6: EJEMPLOS PRÁCTICOS
# ============================================================================

def ejemplo_1_traslacion_simple():
    """
    Ejemplo 1: Registro de una traslación simple usando búsqueda exhaustiva.
    """
    print("\n" + "="*70)
    print("EJEMPLO 1: Registro de Traslación Simple")
    print("="*70)
    
    # Crear imagen sintética
    img_fija = crear_imagen_sintetica(256, 'patron')
    
    # Aplicar traslación conocida
    traslacion_real = {'tx': 15, 'ty': -10}
    img_movil, _ = aplicar_transformacion(img_fija, 'traslacion', traslacion_real)
    
    print(f"\nTraslación aplicada: tx={traslacion_real['tx']}, ty={traslacion_real['ty']}")
    
    # Registrar usando búsqueda exhaustiva
    print("\nRealizando búsqueda exhaustiva...")
    M_optima, similitud, historia = registro_busqueda_exhaustiva(
        img_fija, img_movil, 
        rango_tx=(-30, 30), 
        rango_ty=(-30, 30),
        paso=1,
        metrica='mse'
    )
    
    tx_encontrado = M_optima[0, 2]
    ty_encontrado = M_optima[1, 2]
    
    print(f"\nTraslación encontrada: tx={tx_encontrado:.1f}, ty={ty_encontrado:.1f}")
    print(f"Error: dx={abs(tx_encontrado - traslacion_real['tx']):.1f}, "
          f"dy={abs(ty_encontrado - traslacion_real['ty']):.1f}")
    
    # Aplicar transformación óptima
    img_registrada = cv2.warpAffine(img_movil, M_optima, 
                                    (img_fija.shape[1], img_fija.shape[0]))
    
    # Visualizar resultados
    fig1 = visualizar_registro(img_fija, img_movil, img_registrada,
                              "Ejemplo 1: Registro por Traslación")
    
    fig2 = visualizar_superficie_busqueda(historia, metrica='mse')
    
    return fig1, fig2


def ejemplo_2_rotacion():
    """
    Ejemplo 2: Registro de una rotación usando características.
    """
    print("\n" + "="*70)
    print("EJEMPLO 2: Registro de Rotación con Características")
    print("="*70)
    
    # Crear imagen sintética
    img_fija = crear_imagen_sintetica(256, 'cuadros')
    
    # Aplicar rotación
    angulo_real = 25
    img_movil, _ = aplicar_transformacion(img_fija, 'rotacion', 
                                         {'angulo': angulo_real})
    
    print(f"\nRotación aplicada: {angulo_real}°")
    
    # Registrar usando características
    print("\nDetectando y emparejando características...")
    H, puntos, img_matches = registro_con_caracteristicas(
        img_fija, img_movil, metodo='orb'
    )
    
    if H is not None:
        # Aplicar homografía
        h, w = img_fija.shape
        img_registrada = cv2.warpPerspective(img_movil, H, (w, h))
        
        # Visualizar
        fig1 = visualizar_registro(img_fija, img_movil, img_registrada,
                                  "Ejemplo 2: Registro de Rotación")
        
        # Visualizar matches
        fig2, ax = plt.subplots(1, 1, figsize=(15, 8))
        ax.imshow(img_matches)
        ax.set_title('Características Emparejadas (verde = inliers)')
        ax.axis('off')
        plt.tight_layout()
        
        return fig1, fig2
    else:
        print("No se pudo calcular la homografía")
        return None, None


def ejemplo_3_transformacion_afin():
    """
    Ejemplo 3: Registro de transformación afín compleja.
    """
    print("\n" + "="*70)
    print("EJEMPLO 3: Registro de Transformación Afín")
    print("="*70)
    
    # Crear imagen sintética
    img_fija = crear_imagen_sintetica(256, 'circulo')
    
    # Aplicar transformación afín (rotación + escalado + traslación)
    params_afin = {
        'angulo': 15,
        'escala': 0.9,
        'tx': 20,
        'ty': -15,
        'shear': 0.1
    }
    img_movil, _ = aplicar_transformacion(img_fija, 'afin', params_afin)
    
    print(f"\nTransformación aplicada:")
    print(f"  Rotación: {params_afin['angulo']}°")
    print(f"  Escala: {params_afin['escala']}")
    print(f"  Traslación: ({params_afin['tx']}, {params_afin['ty']})")
    
    # Registrar usando características
    print("\nRegistrando con detector ORB...")
    H, puntos, img_matches = registro_con_caracteristicas(
        img_fija, img_movil, metodo='orb', max_features=1000
    )
    
    if H is not None:
        # Aplicar homografía
        h, w = img_fija.shape
        img_registrada = cv2.warpPerspective(img_movil, H, (w, h))
        
        # Visualizar
        fig1 = visualizar_registro(img_fija, img_movil, img_registrada,
                                  "Ejemplo 3: Registro de Transformación Afín")
        
        return fig1
    else:
        print("No se pudo calcular la homografía")
        return None


def ejemplo_4_comparacion_metricas():
    """
    Ejemplo 4: Comparación de diferentes métricas de similitud.
    """
    print("\n" + "="*70)
    print("EJEMPLO 4: Comparación de Métricas de Similitud")
    print("="*70)
    
    # Crear imagen sintética
    img_fija = crear_imagen_sintetica(256, 'patron')
    
    # Aplicar traslación
    traslacion = {'tx': 12, 'ty': -8}
    img_movil, _ = aplicar_transformacion(img_fija, 'traslacion', traslacion)
    
    print(f"\nTraslación aplicada: tx={traslacion['tx']}, ty={traslacion['ty']}")
    
    # Probar diferentes métricas
    metricas = ['mse', 'ncc', 'mi']
    resultados = {}
    
    for metrica in metricas:
        print(f"\n--- Usando métrica: {metrica.upper()} ---")
        M_optima, similitud, historia = registro_busqueda_exhaustiva(
            img_fija, img_movil,
            rango_tx=(-20, 20),
            rango_ty=(-20, 20),
            paso=2,  # Paso mayor para ser más rápido
            metrica=metrica
        )
        
        tx_enc = M_optima[0, 2]
        ty_enc = M_optima[1, 2]
        error = np.sqrt((tx_enc - traslacion['tx'])**2 + 
                       (ty_enc - traslacion['ty'])**2)
        
        print(f"Traslación encontrada: ({tx_enc:.1f}, {ty_enc:.1f})")
        print(f"Error euclidiano: {error:.2f} píxeles")
        
        resultados[metrica] = {
            'M': M_optima,
            'error': error,
            'historia': historia
        }
    
    # Visualizar comparación
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metrica, res) in enumerate(resultados.items()):
        # Crear superficie
        historia = res['historia']
        tx_vals = [h[0] for h in historia]
        ty_vals = [h[1] for h in historia]
        sim_vals = [h[2] for h in historia]
        
        tx_unique = sorted(set(tx_vals))
        ty_unique = sorted(set(ty_vals))
        
        superficie = np.zeros((len(ty_unique), len(tx_unique)))
        for tx, ty, sim in historia:
            i = ty_unique.index(ty)
            j = tx_unique.index(tx)
            superficie[i, j] = sim
        
        im = axes[idx].imshow(superficie, 
                             extent=[min(tx_unique), max(tx_unique),
                                    min(ty_unique), max(ty_unique)],
                             origin='lower', cmap='viridis', aspect='auto')
        axes[idx].set_xlabel('tx')
        axes[idx].set_ylabel('ty')
        axes[idx].set_title(f'{metrica.upper()}\nError: {res["error"]:.2f} px')
        plt.colorbar(im, ax=axes[idx])
        
        # Marcar óptimo
        M = res['M']
        axes[idx].plot(M[0, 2], M[1, 2], 'r*', markersize=15)
        # Marcar verdadero
        axes[idx].plot(traslacion['tx'], traslacion['ty'], 'go', 
                      markersize=10, fillstyle='none', markeredgewidth=2)
    
    plt.suptitle('Comparación de Métricas de Similitud\n'
                '(* = encontrado, ○ = verdadero)', fontsize=14)
    plt.tight_layout()
    
    return fig


def ejemplo_5_transformaciones_multiples():
    """
    Ejemplo 5: Galería de diferentes tipos de transformaciones.
    """
    print("\n" + "="*70)
    print("EJEMPLO 5: Galería de Transformaciones")
    print("="*70)
    
    # Crear imagen base
    img_base = crear_imagen_sintetica(256, 'texto')
    
    # Diccionario para almacenar transformaciones
    imagenes = {'Original': img_base}
    
    # 1. Traslación
    img_tras, _ = aplicar_transformacion(img_base, 'traslacion', 
                                        {'tx': 30, 'ty': 20})
    imagenes['Traslación'] = img_tras
    
    # 2. Rotación
    img_rot, _ = aplicar_transformacion(img_base, 'rotacion', 
                                       {'angulo': 30})
    imagenes['Rotación 30°'] = img_rot
    
    # 3. Transformación rígida
    img_rig, _ = aplicar_transformacion(img_base, 'rigida',
                                       {'angulo': 20, 'tx': 15, 'ty': -10})
    imagenes['Rígida'] = img_rig
    
    # 4. Transformación afín
    img_afin, _ = aplicar_transformacion(img_base, 'afin',
                                        {'angulo': 15, 'escala': 0.8, 
                                         'tx': 10, 'ty': 5, 'shear': 0.2})
    imagenes['Afín'] = img_afin
    
    # 5. Transformación proyectiva
    h, w = img_base.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[20, 30], [w-10, 20], [w-30, h-20], [10, h-10]])
    img_proy, _ = aplicar_transformacion(img_base, 'proyectiva',
                                        {'src_points': src, 'dst_points': dst})
    imagenes['Proyectiva'] = img_proy
    
    # Visualizar
    fig = crear_tablero_comparativo(imagenes)
    plt.suptitle('Tipos de Transformaciones Geométricas', 
                fontsize=16, fontweight='bold', y=1.02)
    
    return fig


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "TUTORIAL: REGISTRO DE IMÁGENES EN PYTHON" + " "*18 + "║")
    print("║" + " "*68 + "║")
    print("║" + "  Este tutorial cubre los conceptos fundamentales del registro" + " "*6 + "║")
    print("║" + "  de imágenes, desde métodos basados en intensidad hasta" + " "*11 + "║")
    print("║" + "  técnicas basadas en características." + " "*33 + "║")
    print("╚" + "="*68 + "╝")
    
    # Ejecutar ejemplos
    print("\n🔹 Ejecutando ejemplos prácticos...")
    print("   (Esto puede tardar unos momentos)\n")
    
    try:
        # Ejemplo 1
        fig1a, fig1b = ejemplo_1_traslacion_simple()
        
        # Ejemplo 2
        fig2a, fig2b = ejemplo_2_rotacion()
        
        # Ejemplo 3
        fig3 = ejemplo_3_transformacion_afin()
        
        # Ejemplo 4
        fig4 = ejemplo_4_comparacion_metricas()
        
        # Ejemplo 5
        fig5 = ejemplo_5_transformaciones_multiples()
        
        print("\n" + "="*70)
        print("TUTORIAL COMPLETADO")
        print("="*70)
        print("\n✅ Todos los ejemplos se han ejecutado exitosamente.")
        print("📊 Las figuras se han generado y están listas para visualizar.")
        print("\n💡 Conceptos clave aprendidos:")
        print("   • Tipos de transformaciones geométricas")
        print("   • Métodos de registro basados en intensidad")
        print("   • Métodos basados en características")
        print("   • Métricas de similitud")
        print("   • Uso de RANSAC para estimación robusta")
        
        # Mostrar todas las figuras
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ============================================================================
# EJERCICIOS PROPUESTOS
# ============================================================================

"""
EJERCICIOS PARA PRACTICAR:

1. BÁSICO: Modifica el Ejemplo 1 para usar diferentes métricas de similitud
   (NCC o MI) y compara los resultados.

2. INTERMEDIO: Implementa un método de registro multi-escala (image pyramid)
   para mejorar la velocidad y robustez de la búsqueda exhaustiva.

3. INTERMEDIO: Añade ruido gaussiano a la imagen móvil antes del registro
   y observa cómo afecta a la precisión de cada método.

4. AVANZADO: Implementa el algoritmo de Lucas-Kanade para registro basado
   en gradientes (método iterativo).

5. AVANZADO: Crea un registro no-rígido usando B-splines o thin-plate splines
   para modelar deformaciones locales.

6. PROYECTO: Implementa un sistema completo de creación de panorámicas que:
   - Tome múltiples imágenes superpuestas
   - Las registre secuencialmente
   - Las fusione en una panorámica final

PREGUNTAS DE REFLEXIÓN:

1. ¿Qué ventajas y desventajas tienen los métodos basados en intensidad
   vs. los basados en características?

2. ¿Cómo afecta el tamaño del paso de búsqueda a la precisión y velocidad
   del registro?

3. ¿Por qué es importante usar RANSAC en el registro basado en características?

4. ¿En qué situaciones fallaría cada tipo de transformación?

5. ¿Cómo se podría extender este código para trabajar con imágenes 3D (volumes)?
"""
