# 📚 Tutorial: Registro de Imágenes con Python

**Curso:** Visión por Computador 3009228  
**Universidad Nacional de Colombia**  
**Docente:** Juan David Ospina Arango

---

## 📖 Descripción

Este tutorial proporciona una introducción completa y práctica al **registro de imágenes** (image registration), cubriendo desde los fundamentos teóricos hasta implementaciones en Python con imágenes sintéticas.

## 🎯 Objetivos de Aprendizaje

Al completar este tutorial, serás capaz de:

✅ Entender los conceptos fundamentales del registro de imágenes  
✅ Conocer los diferentes tipos de transformaciones geométricas  
✅ Implementar métodos de registro basados en intensidad  
✅ Implementar métodos basados en características (SIFT, ORB)  
✅ Usar RANSAC para estimación robusta  
✅ Evaluar la calidad del registro con diferentes métricas  
✅ Aplicar estos conceptos a problemas reales

## 📁 Contenido del Tutorial

### Archivos Principales

1. **`tutorial_registro_imagenes.py`** (Script Python completo)
   - Implementación completa con funciones reutilizables
   - 5 ejemplos prácticos ejecutables
   - Código bien documentado y modular
   
2. **`Tutorial_Registro_Imagenes.ipynb`** (Jupyter Notebook interactivo)
   - Formato paso a paso ideal para aprendizaje
   - Celdas ejecutables para experimentación
   - Explicaciones teóricas integradas
   
3. **`Resumen_Registro_Imagenes.pdf`** (Documento de referencia)
   - Resumen teórico conciso
   - Tablas de referencia rápida
   - Ideal para consulta durante proyectos

### Figuras Generadas

📁 Carpeta: `figuras_registro/`

- `00_conceptos_teoricos.png` - Diagrama conceptual del proceso
- `01_traslacion_registro.png` - Ejemplo de registro por traslación
- `02_traslacion_superficie.png` - Superficie de búsqueda
- `03_rotacion_registro.png` - Ejemplo con rotación
- `04_rotacion_matches.png` - Visualización de características
- `05_afin_registro.png` - Transformación afín
- `06_comparacion_metricas.png` - Comparación de métricas MSE/NCC/MI
- `07_galeria_transformaciones.png` - Tipos de transformaciones

## 🚀 Cómo Usar Este Tutorial

### Opción 1: Script Python Completo (Recomendado para primera vez)

```bash
python tutorial_registro_imagenes.py
```

Esto ejecutará automáticamente todos los ejemplos y mostrará las visualizaciones.

### Opción 2: Jupyter Notebook (Recomendado para experimentación)

```bash
jupyter notebook Tutorial_Registro_Imagenes.ipynb
```

O usa Google Colab para ejecutarlo en la nube sin instalación local.

### Opción 3: Generar Solo las Figuras

```bash
python generar_figuras.py
```

Esto creará todas las figuras en la carpeta `figuras_registro/` sin mostrarlas en pantalla.

## 📦 Requisitos

### Bibliotecas Necesarias

```bash
pip install numpy opencv-python matplotlib scipy
```

O instala todo de una vez:

```bash
pip install --break-system-packages numpy opencv-python matplotlib scipy jupyter
```

### Versiones Recomendadas

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.3+
- SciPy 1.7+

## 📚 Estructura del Tutorial

### Módulo 1: Fundamentos (30 min)
- ¿Qué es el registro de imágenes?
- Componentes principales
- Aplicaciones en el mundo real

### Módulo 2: Transformaciones Geométricas (45 min)
- Jerarquía de transformaciones
- Traslación, rotación, rígida, afín, proyectiva
- Representación matricial

### Módulo 3: Métricas de Similitud (30 min)
- Error Cuadrático Medio (MSE)
- Correlación Cruzada Normalizada (NCC)
- Información Mutua (MI)

### Módulo 4: Métodos de Registro (60 min)
- Búsqueda exhaustiva
- Registro basado en características
- RANSAC para robustez

### Módulo 5: Ejemplos Prácticos (60 min)
- Ejemplo 1: Traslación simple
- Ejemplo 2: Rotación con características
- Ejemplo 3: Transformación afín compleja
- Ejemplo 4: Comparación de métricas
- Ejemplo 5: Galería de transformaciones

## 🎓 Ejemplos Incluidos

### Ejemplo 1: Registro de Traslación Simple
Búsqueda exhaustiva para encontrar una traslación 2D simple.
- **Método:** Búsqueda exhaustiva con MSE
- **Transformación:** Traslación (tx=15, ty=-10)
- **Tiempo:** ~10 segundos

### Ejemplo 2: Registro de Rotación
Uso de características ORB para registrar una imagen rotada.
- **Método:** ORB + RANSAC
- **Transformación:** Rotación de 25°
- **Tiempo:** ~2 segundos

### Ejemplo 3: Transformación Afín Compleja
Registro de una transformación que combina rotación, escala y traslación.
- **Método:** ORB + RANSAC con más características
- **Transformación:** Afín (6 DOF)
- **Tiempo:** ~3 segundos

### Ejemplo 4: Comparación de Métricas
Evalúa MSE, NCC y MI en la misma tarea.
- **Objetivo:** Comparar rendimiento de métricas
- **Resultado:** Visualización de superficies de similitud

### Ejemplo 5: Galería de Transformaciones
Visualiza todos los tipos de transformaciones geométricas.
- **Transformaciones:** Traslación, rotación, rígida, afín, proyectiva

## 💡 Ejercicios Propuestos

### Nivel Básico ⭐
1. Modifica los parámetros de búsqueda exhaustiva
2. Prueba diferentes tipos de imágenes sintéticas
3. Cambia las métricas de similitud

### Nivel Intermedio ⭐⭐
4. Implementa registro multi-escala (pirámides)
5. Añade ruido a las imágenes y evalúa robustez
6. Compara diferentes detectores (ORB vs SIFT vs AKAZE)

### Nivel Avanzado ⭐⭐⭐
7. Implementa Lucas-Kanade para registro iterativo
8. Crea un sistema de panorámicas automático
9. Desarrolla registro no-rígido con B-splines

## 🔬 Aplicaciones Prácticas

Este tutorial te preparará para trabajar en:

- **Imágenes Médicas**: Fusión de MRI, CT, PET
- **Teledetección**: Análisis de cambios en imágenes satelitales
- **Fotografía Computacional**: Creación de panorámicas, HDR
- **Robótica**: SLAM, localización y mapeo
- **Realidad Aumentada**: Alineamiento de contenido virtual

## 📖 Referencias y Recursos Adicionales

### Libros
- Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*
- Forsyth & Ponce (2012). *Computer Vision: A Modern Approach*

### Papers Clásicos
- Lowe (2004): "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT)
- Rublee et al. (2011): "ORB: An efficient alternative to SIFT or SURF"
- Fischler & Bolles (1981): "Random Sample Consensus" (RANSAC)

### Cursos Online
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- PyImageSearch: Computer Vision Tutorials

### Documentación
- [OpenCV Docs](https://docs.opencv.org/)
- [Scikit-image](https://scikit-image.org/)

## 🤝 Contribuciones y Feedback

Este tutorial fue desarrollado para el curso de Visión por Computador. 
Si encuentras errores o tienes sugerencias, por favor contacta al docente.

## 📝 Notas Importantes

- ⚠️ Las imágenes sintéticas son ideales para aprendizaje pero pueden no reflejar 
  todos los desafíos de imágenes reales
- 💻 Algunos ejemplos pueden tardar varios segundos en ejecutarse
- 🖼️ Las figuras se guardan automáticamente en alta resolución (150 DPI)

## ✅ Lista de Verificación

Antes de comenzar, asegúrate de:

- [ ] Tener Python 3.8+ instalado
- [ ] Instalar todas las bibliotecas necesarias
- [ ] Descargar todos los archivos del tutorial
- [ ] Tener ~100MB de espacio libre para figuras
- [ ] (Opcional) Jupyter Notebook instalado

## 🎯 Próximos Pasos

Después de completar este tutorial:

1. Experimenta con imágenes reales (no sintéticas)
2. Aplica el registro a tu proyecto del curso
3. Investiga métodos avanzados (Deep Learning para registro)
4. Explora aplicaciones específicas de tu interés

---

## 📧 Contacto

**Docente:** Juan David Ospina Arango  
**Email:** jdospina@unal.edu.co  
**Curso:** Visión por Computador 3009228  
**Universidad Nacional de Colombia**

---

**¡Disfruta aprendiendo sobre registro de imágenes!** 🚀📸

*Última actualización: Octubre 2025*
