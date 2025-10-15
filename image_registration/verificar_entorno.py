#!/usr/bin/env python3
"""
Script de verificación e instalación del entorno
Tutorial de Registro de Imágenes
"""

import sys
import subprocess

def verificar_python():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False
    else:
        print("✅ Versión de Python correcta")
        return True

def verificar_biblioteca(nombre, nombre_import=None):
    """Verifica si una biblioteca está instalada"""
    if nombre_import is None:
        nombre_import = nombre
    
    try:
        modulo = __import__(nombre_import)
        version = getattr(modulo, '__version__', 'desconocida')
        print(f"✅ {nombre}: {version}")
        return True
    except ImportError:
        print(f"❌ {nombre}: NO INSTALADO")
        return False

def instalar_bibliotecas():
    """Intenta instalar las bibliotecas faltantes"""
    print("\n🔧 Intentando instalar bibliotecas faltantes...")
    bibliotecas = ['numpy', 'opencv-python', 'matplotlib', 'scipy']
    
    cmd = [sys.executable, '-m', 'pip', 'install', '--break-system-packages'] + bibliotecas
    
    try:
        subprocess.check_call(cmd)
        print("✅ Instalación completada")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error durante la instalación")
        print("Intenta instalar manualmente con:")
        print(f"pip install --break-system-packages {' '.join(bibliotecas)}")
        return False

def verificar_archivos():
    """Verifica que todos los archivos del tutorial estén presentes"""
    import os
    
    archivos_requeridos = [
        'tutorial_registro_imagenes.py',
        'Tutorial_Registro_Imagenes.ipynb',
        'Resumen_Registro_Imagenes.pdf',
        'README.md',
        'generar_figuras.py'
    ]
    
    print("\n📁 Verificando archivos del tutorial:")
    todos_presentes = True
    
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            size = os.path.getsize(archivo) / 1024  # KB
            print(f"✅ {archivo} ({size:.1f} KB)")
        else:
            print(f"❌ {archivo}: FALTANTE")
            todos_presentes = False
    
    # Verificar carpeta de figuras
    if os.path.exists('figuras_registro'):
        n_figuras = len([f for f in os.listdir('figuras_registro') if f.endswith('.png')])
        print(f"✅ figuras_registro/ ({n_figuras} imágenes)")
    else:
        print(f"⚠️  figuras_registro/: No creada aún (se generará al ejecutar los ejemplos)")
    
    return todos_presentes

def main():
    """Función principal"""
    print("="*70)
    print("VERIFICACIÓN DEL ENTORNO - TUTORIAL REGISTRO DE IMÁGENES")
    print("="*70 + "\n")
    
    # Verificar Python
    print("1️⃣  Verificando Python:")
    python_ok = verificar_python()
    print()
    
    # Verificar bibliotecas
    print("2️⃣  Verificando bibliotecas:")
    bibliotecas = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy'
    }
    
    bibliotecas_ok = []
    for nombre, import_name in bibliotecas.items():
        ok = verificar_biblioteca(nombre, import_name)
        bibliotecas_ok.append(ok)
    
    todas_bibliotecas = all(bibliotecas_ok)
    
    # Si faltan bibliotecas, ofrecer instalar
    if not todas_bibliotecas:
        print("\n⚠️  Algunas bibliotecas no están instaladas")
        respuesta = input("¿Deseas instalarlas ahora? (s/n): ")
        if respuesta.lower() == 's':
            instalar_bibliotecas()
            print("\n🔄 Verificando instalación...")
            for nombre, import_name in bibliotecas.items():
                verificar_biblioteca(nombre, import_name)
    else:
        print("\n✅ Todas las bibliotecas están instaladas")
    
    # Verificar archivos
    archivos_ok = verificar_archivos()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    
    if python_ok and todas_bibliotecas and archivos_ok:
        print("🎉 ¡Todo está listo! Puedes comenzar con el tutorial.")
        print("\n📚 Para empezar:")
        print("   python tutorial_registro_imagenes.py")
        print("   o")
        print("   jupyter notebook Tutorial_Registro_Imagenes.ipynb")
    else:
        print("⚠️  Hay algunos problemas que resolver:")
        if not python_ok:
            print("   • Actualiza Python a la versión 3.8 o superior")
        if not todas_bibliotecas:
            print("   • Instala las bibliotecas faltantes")
        if not archivos_ok:
            print("   • Asegúrate de tener todos los archivos del tutorial")
    
    print("\n📖 Para más información, consulta el README.md")
    print("="*70)

if __name__ == "__main__":
    main()
