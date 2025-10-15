"""
Script para generar y guardar todas las figuras del tutorial de registro de imágenes
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
import os

# Importar funciones del tutorial principal
import sys
sys.path.insert(0, '/home/claude')
from tutorial_registro_imagenes import (
    crear_imagen_sintetica,
    aplicar_transformacion,
    registro_busqueda_exhaustiva,
    registro_con_caracteristicas,
    visualizar_registro,
    visualizar_superficie_busqueda,
    crear_tablero_comparativo
)

# Crear directorio para figuras
os.makedirs('/mnt/user-data/outputs/figuras_registro', exist_ok=True)

print("🎨 Generando figuras del tutorial...")

# ============================================================================
# Figura 1: Ejemplo de traslación simple
# ============================================================================
print("\n📊 Generando Figura 1: Traslación Simple...")

img_fija = crear_imagen_sintetica(256, 'patron')
traslacion_real = {'tx': 15, 'ty': -10}
img_movil, _ = aplicar_transformacion(img_fija, 'traslacion', traslacion_real)

M_optima, similitud, historia = registro_busqueda_exhaustiva(
    img_fija, img_movil, 
    rango_tx=(-30, 30), 
    rango_ty=(-30, 30),
    paso=1,
    metrica='mse'
)

img_registrada = cv2.warpAffine(img_movil, M_optima, 
                                (img_fija.shape[1], img_fija.shape[0]))

fig1a = visualizar_registro(img_fija, img_movil, img_registrada,
                            "Ejemplo 1: Registro por Traslación")
fig1a.savefig('/mnt/user-data/outputs/figuras_registro/01_traslacion_registro.png', 
              dpi=150, bbox_inches='tight')
plt.close(fig1a)

fig1b = visualizar_superficie_busqueda(historia, metrica='mse')
fig1b.savefig('/mnt/user-data/outputs/figuras_registro/02_traslacion_superficie.png',
              dpi=150, bbox_inches='tight')
plt.close(fig1b)

# ============================================================================
# Figura 2: Ejemplo de rotación
# ============================================================================
print("📊 Generando Figura 2: Rotación...")

img_fija = crear_imagen_sintetica(256, 'cuadros')
angulo_real = 25
img_movil, _ = aplicar_transformacion(img_fija, 'rotacion', {'angulo': angulo_real})

H, puntos, img_matches = registro_con_caracteristicas(
    img_fija, img_movil, metodo='orb'
)

if H is not None:
    h, w = img_fija.shape
    img_registrada = cv2.warpPerspective(img_movil, H, (w, h))
    
    fig2a = visualizar_registro(img_fija, img_movil, img_registrada,
                                "Ejemplo 2: Registro de Rotación")
    fig2a.savefig('/mnt/user-data/outputs/figuras_registro/03_rotacion_registro.png',
                  dpi=150, bbox_inches='tight')
    plt.close(fig2a)
    
    fig2b, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.imshow(img_matches)
    ax.set_title('Características Emparejadas (verde = inliers)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    fig2b.savefig('/mnt/user-data/outputs/figuras_registro/04_rotacion_matches.png',
                  dpi=150, bbox_inches='tight')
    plt.close(fig2b)

# ============================================================================
# Figura 3: Transformación afín
# ============================================================================
print("📊 Generando Figura 3: Transformación Afín...")

img_fija = crear_imagen_sintetica(256, 'circulo')
params_afin = {
    'angulo': 15,
    'escala': 0.9,
    'tx': 20,
    'ty': -15,
    'shear': 0.1
}
img_movil, _ = aplicar_transformacion(img_fija, 'afin', params_afin)

H, puntos, img_matches = registro_con_caracteristicas(
    img_fija, img_movil, metodo='orb', max_features=1000
)

if H is not None:
    h, w = img_fija.shape
    img_registrada = cv2.warpPerspective(img_movil, H, (w, h))
    
    fig3 = visualizar_registro(img_fija, img_movil, img_registrada,
                               "Ejemplo 3: Registro de Transformación Afín")
    fig3.savefig('/mnt/user-data/outputs/figuras_registro/05_afin_registro.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

# ============================================================================
# Figura 4: Comparación de métricas
# ============================================================================
print("📊 Generando Figura 4: Comparación de Métricas...")

img_fija = crear_imagen_sintetica(256, 'patron')
traslacion = {'tx': 12, 'ty': -8}
img_movil, _ = aplicar_transformacion(img_fija, 'traslacion', traslacion)

metricas = ['mse', 'ncc', 'mi']
resultados = {}

for metrica in metricas:
    M_optima, similitud, historia = registro_busqueda_exhaustiva(
        img_fija, img_movil,
        rango_tx=(-20, 20),
        rango_ty=(-20, 20),
        paso=2,
        metrica=metrica
    )
    
    tx_enc = M_optima[0, 2]
    ty_enc = M_optima[1, 2]
    error = np.sqrt((tx_enc - traslacion['tx'])**2 + 
                   (ty_enc - traslacion['ty'])**2)
    
    resultados[metrica] = {
        'M': M_optima,
        'error': error,
        'historia': historia
    }

fig4, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (metrica, res) in enumerate(resultados.items()):
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
    
    M = res['M']
    axes[idx].plot(M[0, 2], M[1, 2], 'r*', markersize=15)
    axes[idx].plot(traslacion['tx'], traslacion['ty'], 'go', 
                  markersize=10, fillstyle='none', markeredgewidth=2)

plt.suptitle('Comparación de Métricas de Similitud\n'
            '(* = encontrado, ○ = verdadero)', fontsize=14)
plt.tight_layout()
fig4.savefig('/mnt/user-data/outputs/figuras_registro/06_comparacion_metricas.png',
             dpi=150, bbox_inches='tight')
plt.close(fig4)

# ============================================================================
# Figura 5: Galería de transformaciones
# ============================================================================
print("📊 Generando Figura 5: Galería de Transformaciones...")

img_base = crear_imagen_sintetica(256, 'texto')
imagenes = {'Original': img_base}

img_tras, _ = aplicar_transformacion(img_base, 'traslacion', 
                                    {'tx': 30, 'ty': 20})
imagenes['Traslación'] = img_tras

img_rot, _ = aplicar_transformacion(img_base, 'rotacion', 
                                   {'angulo': 30})
imagenes['Rotación 30°'] = img_rot

img_rig, _ = aplicar_transformacion(img_base, 'rigida',
                                   {'angulo': 20, 'tx': 15, 'ty': -10})
imagenes['Rígida'] = img_rig

img_afin, _ = aplicar_transformacion(img_base, 'afin',
                                    {'angulo': 15, 'escala': 0.8, 
                                     'tx': 10, 'ty': 5, 'shear': 0.2})
imagenes['Afín'] = img_afin

h, w = img_base.shape
src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
dst = np.float32([[20, 30], [w-10, 20], [w-30, h-20], [10, h-10]])
img_proy, _ = aplicar_transformacion(img_base, 'proyectiva',
                                    {'src_points': src, 'dst_points': dst})
imagenes['Proyectiva'] = img_proy

fig5 = crear_tablero_comparativo(imagenes)
plt.suptitle('Tipos de Transformaciones Geométricas', 
            fontsize=16, fontweight='bold', y=1.02)
fig5.savefig('/mnt/user-data/outputs/figuras_registro/07_galeria_transformaciones.png',
             dpi=150, bbox_inches='tight')
plt.close(fig5)

# ============================================================================
# Figura adicional: Diagrama conceptual
# ============================================================================
print("📊 Generando figura conceptual...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Proceso de registro
ax = axes[0, 0]
ax.text(0.5, 0.9, 'PROCESO DE REGISTRO', ha='center', va='top',
        fontsize=16, fontweight='bold', transform=ax.transAxes)

pasos = [
    '1. Imagen Fija (Referencia)',
    '2. Imagen Móvil (A registrar)',
    '3. Detección de características',
    '4. Emparejamiento',
    '5. Estimación de transformación',
    '6. Aplicar transformación',
    '7. Imagen registrada'
]

y_pos = 0.75
for paso in pasos:
    ax.text(0.1, y_pos, paso, fontsize=11, transform=ax.transAxes)
    if y_pos > 0.15:
        ax.arrow(0.5, y_pos - 0.02, 0, -0.06, transform=ax.transAxes,
                head_width=0.03, head_length=0.02, fc='blue', ec='blue')
    y_pos -= 0.1

ax.axis('off')

# Panel 2: Tipos de transformaciones
ax = axes[0, 1]
ax.text(0.5, 0.95, 'JERARQUÍA DE TRANSFORMACIONES', ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

transformaciones = [
    ('Traslación', '2 DOF', 'tx, ty'),
    ('Rotación', '1 DOF', 'θ'),
    ('Rígida (Euclidiana)', '3 DOF', 'tx, ty, θ'),
    ('Similitud', '4 DOF', 'tx, ty, θ, s'),
    ('Afín', '6 DOF', 'tx, ty, θ, sx, sy, shear'),
    ('Proyectiva', '8 DOF', 'homografía 3x3')
]

y_pos = 0.85
colors = plt.cm.viridis(np.linspace(0, 1, len(transformaciones)))

for i, (nombre, dof, params) in enumerate(transformaciones):
    ax.add_patch(Rectangle((0.1, y_pos - 0.08), 0.8, 0.08, 
                           facecolor=colors[i], alpha=0.3, 
                           transform=ax.transAxes))
    ax.text(0.15, y_pos - 0.04, f'{nombre}', fontsize=10, fontweight='bold',
           transform=ax.transAxes)
    ax.text(0.65, y_pos - 0.04, f'{dof}', fontsize=9,
           transform=ax.transAxes)
    y_pos -= 0.12

ax.axis('off')

# Panel 3: Métricas de similitud
ax = axes[1, 0]
ax.text(0.5, 0.95, 'MÉTRICAS DE SIMILITUD', ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

metricas_info = [
    ('MSE', 'Error Cuadrático Medio', 'Minimizar', 'Simple, sensible a outliers'),
    ('NCC', 'Correlación Cruzada', 'Maximizar', 'Invariante a brillo'),
    ('MI', 'Información Mutua', 'Maximizar', 'Multi-modal'),
    ('SSD', 'Suma de Diferencias²', 'Minimizar', 'Rápido, simple')
]

y_pos = 0.85
for metrica, nombre, objetivo, caracteristica in metricas_info:
    ax.text(0.05, y_pos, f'• {metrica}:', fontsize=11, fontweight='bold',
           transform=ax.transAxes)
    ax.text(0.15, y_pos - 0.04, f'{nombre}', fontsize=9,
           transform=ax.transAxes)
    ax.text(0.15, y_pos - 0.08, f'Objetivo: {objetivo}', fontsize=8,
           transform=ax.transAxes, style='italic')
    ax.text(0.15, y_pos - 0.12, f'{caracteristica}', fontsize=8,
           transform=ax.transAxes, color='gray')
    y_pos -= 0.2

ax.axis('off')

# Panel 4: Aplicaciones
ax = axes[1, 1]
ax.text(0.5, 0.95, 'APLICACIONES', ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

aplicaciones = [
    '🏥 Imágenes Médicas',
    '  • Fusión multi-modal (MRI, CT, PET)',
    '  • Seguimiento temporal',
    '',
    '🛰️ Teledetección',
    '  • Análisis de cambios',
    '  • Corrección geométrica',
    '',
    '📷 Fotografía',
    '  • Panorámicas',
    '  • HDR, focus stacking',
    '',
    '🤖 Robótica/Visión',
    '  • SLAM, localización',
    '  • Seguimiento de objetos'
]

y_pos = 0.85
for app in aplicaciones:
    if app:
        ax.text(0.05, y_pos, app, fontsize=10, transform=ax.transAxes)
    y_pos -= 0.055

ax.axis('off')

plt.tight_layout()
fig.savefig('/mnt/user-data/outputs/figuras_registro/00_conceptos_teoricos.png',
           dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n✅ Todas las figuras se han generado exitosamente!")
print(f"📁 Ubicación: /mnt/user-data/outputs/figuras_registro/")
print("\nFiguras generadas:")
print("  00_conceptos_teoricos.png - Diagrama conceptual")
print("  01_traslacion_registro.png - Ejemplo de traslación")
print("  02_traslacion_superficie.png - Superficie de búsqueda")
print("  03_rotacion_registro.png - Ejemplo de rotación")
print("  04_rotacion_matches.png - Matches de características")
print("  05_afin_registro.png - Transformación afín")
print("  06_comparacion_metricas.png - Comparación de métricas")
print("  07_galeria_transformaciones.png - Tipos de transformaciones")
