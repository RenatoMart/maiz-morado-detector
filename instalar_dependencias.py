"""Revisa lo que necesita el proyecto y descarga lo que falte. Funciona en Windows, Linux y macOS.

Uso (con cualquier Python 3.9 o más nuevo, no hace falta instalar nada antes):

    python instalar_dependencias.py                 revisa e instala lo mínimo para la aplicación de escritorio
    python instalar_dependencias.py --solo-revisar  solo muestra qué falta, sin instalar nada
    python instalar_dependencias.py --entrenar      añade lo necesario para entrenar y evaluar el modelo
    python instalar_dependencias.py --informe       añade lo necesario para generar el informe (carpeta DEMAS)

En Windows basta con hacer doble clic en `instalar_en_windows.bat`.
Solo usa la librería estándar de Python, así que puede ejecutarse antes de tener nada instalado.
"""
import argparse
import json
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ES_WINDOWS = platform.system() == "Windows"

# nombre en pip, versión mínima, para qué sirve
BASE = [
    ("ultralytics", "8.3.0", "detector YOLO"),
    ("torch", "2.2.0", "motor de cálculo del detector"),
    ("torchvision", "0.17.0", "complemento de torch para imágenes"),
    ("opencv-python-headless", "4.9.0", "lectura de imágenes y contorno de la mazorca"),
    ("numpy", "1.26.0", "cálculo numérico"),
    ("pillow", "10.0.0", "imágenes y la ventana de escritorio"),
    ("PyYAML", "6.0", "archivos de configuración"),
    ("matplotlib", "3.8.0", "tipos de letra y gráficos de las pruebas"),
]
ENTRENAR = [
    ("scikit-learn", "1.3.0", "regresión logística de la decisión"),
    ("pandas", "2.0.0", "tablas de resultados"),
    ("scipy", "1.11.0", "cálculo científico"),
    ("cleanlab", "2.5.0", "revisión de etiquetas dudosas"),
]
INFORME = [
    ("reportlab", "4.0.0", "crear el PDF del informe"),
    ("python-docx", "1.0.0", "crear el Word del informe"),
    ("fonttools", "4.40.0", "letras del logo en el informe"),
]
ARCHIVOS_MODELO = ["entrenamiento/modelo_final_v4/best.pt", "entrenamiento/modelo_final_v4/decision.json"]


def decir(texto=""):
    print(texto, flush=True)


def version_tupla(v):
    partes = []
    for p in v.replace("+", ".").split("."):
        digitos = "".join(c for c in p if c.isdigit())
        if not digitos:
            break
        partes.append(int(digitos))
    return tuple(partes)


def carpeta_entorno():
    """Usa .venv; si ese .venv se creó en otro sistema operativo (por ejemplo, copiado de Linux), usa uno propio."""
    nombre = ".venv"
    principal = AQUI / nombre
    if principal.exists() and not ejecutable(principal).exists():
        nombre = ".venv_" + platform.system().lower()
    return AQUI / nombre


def ejecutable(entorno):
    return entorno / ("Scripts/python.exe" if ES_WINDOWS else "bin/python")


def versiones_instaladas(py, nombres):
    """Pregunta al Python del entorno qué versión tiene de cada paquete (o None)."""
    codigo = (
        "import json,sys,importlib.metadata as m\n"
        "r={}\n"
        "for n in json.loads(sys.argv[1]):\n"
        "    try: r[n]=m.version(n)\n"
        "    except Exception: r[n]=None\n"
        "print(json.dumps(r))"
    )
    salida = subprocess.run([str(py), "-c", codigo, json.dumps(nombres)], capture_output=True, text=True)
    return json.loads(salida.stdout.strip().splitlines()[-1]) if salida.returncode == 0 and salida.stdout.strip() else {n: None for n in nombres}


def ejecutar(orden):
    decir("  > " + " ".join(str(o) for o in orden))
    return subprocess.run([str(o) for o in orden]).returncode


def main():
    ap = argparse.ArgumentParser(description="Revisa e instala lo que necesita el proyecto.")
    ap.add_argument("--solo-revisar", action="store_true", help="no instala nada, solo informa")
    ap.add_argument("--entrenar", action="store_true", help="incluye lo necesario para entrenar y evaluar")
    ap.add_argument("--informe", action="store_true", help="incluye lo necesario para generar el informe")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    decir("=" * 64)
    decir(" Revisión del equipo - Inspección de maíz morado")
    decir("=" * 64)
    decir(f" Sistema operativo : {platform.system()} {platform.release()} ({platform.machine()})")
    decir(f" Python            : {platform.python_version()}  ({sys.executable})")
    problemas = []

    if sys.version_info < (3, 9):
        decir("\n[ERROR] Hace falta Python 3.9 o más nuevo. Descárgalo de https://www.python.org/downloads/")
        decir("        En Windows, marca la casilla 'Add python.exe to PATH' al instalarlo.")
        return 1
    if sys.version_info >= (3, 15):
        decir("[AVISO] Este Python es muy nuevo y quizá PyTorch aún no tenga versión para él. Si falla, instala Python 3.12.")
    try:
        import tkinter  # noqa: F401

        decir(" Ventana (tkinter) : disponible")
    except ImportError:
        problemas.append("tkinter")
        decir(" Ventana (tkinter) : NO disponible")
        decir("   -> Windows: reinstala Python desde python.org y deja marcada la opción 'tcl/tk and IDLE'.")
        decir("   -> Linux:   sudo apt install python3-tk   (o el equivalente de tu distribución)")

    libre = shutil.disk_usage(AQUI).free / 1e9
    decir(f" Espacio libre     : {libre:.1f} GB")
    if libre < 4:
        decir("[AVISO] Queda poco espacio: las librerías pesan unos 3 GB en total.")

    paquetes = BASE + (ENTRENAR if a.entrenar else []) + (INFORME if a.informe else [])
    entorno = carpeta_entorno()
    py = ejecutable(entorno)
    decir(f"\n Entorno de trabajo: {entorno.name}")
    if not py.exists():
        if a.solo_revisar:
            decir("   (todavía no existe; se crearía al instalar)")
        else:
            decir("   Creándolo...")
            venv.EnvBuilder(with_pip=True).create(entorno)
    faltan = []
    if py.exists():
        inst = versiones_instaladas(py, [p[0] for p in paquetes])
    else:
        inst = {p[0]: None for p in paquetes}
    decir("\n Librerías:")
    for nombre, minima, para in paquetes:
        v = inst.get(nombre)
        if v and version_tupla(v) >= version_tupla(minima):
            decir(f"   [OK]    {nombre:<24} {v:<12} {para}")
        else:
            faltan.append((nombre, minima))
            decir(f"   [FALTA] {nombre:<24} {'sin instalar' if not v else 'versión ' + v + ' (se pide >= ' + minima + ')':<12} {para}")

    decir("\n Archivos del modelo:")
    for rel in ARCHIVOS_MODELO:
        existe = (AQUI / rel).exists()
        decir(f"   [{'OK' if existe else 'FALTA'}]    {rel}")
        if not existe:
            problemas.append(rel)

    if faltan and not a.solo_revisar:
        decir(f"\n Instalando {len(faltan)} librería(s). Puede tardar varios minutos y descargar más de 1 GB...")
        ejecutar([py, "-m", "pip", "install", "--upgrade", "pip"])
        pesadas = [f for f in faltan if f[0] in ("torch", "torchvision")]
        resto = [f for f in faltan if f not in pesadas]
        if pesadas:
            # en Linux la versión por defecto de torch incluye soporte de GPU y pesa mucho más; aquí solo se usa la CPU
            extra = ["--index-url", "https://download.pytorch.org/whl/cpu"] if not ES_WINDOWS and platform.system() != "Darwin" else []
            if ejecutar([py, "-m", "pip", "install", *extra, *[f"{n}>={v}" for n, v in pesadas]]) != 0 and extra:
                ejecutar([py, "-m", "pip", "install", *[f"{n}>={v}" for n, v in pesadas]])
        if resto:
            ejecutar([py, "-m", "pip", "install", *[f"{n}>={v}" for n, v in resto]])
        inst = versiones_instaladas(py, [p[0] for p in paquetes])
        faltan = [(n, m) for n, m, _ in paquetes if not inst.get(n) or version_tupla(inst[n]) < version_tupla(m)]
    elif faltan:
        decir("\n Para instalar lo que falta, ejecuta de nuevo sin --solo-revisar.")

    if py.exists() and not faltan:
        prueba = subprocess.run([str(py), "-c", "import ultralytics, cv2, numpy, PIL, yaml; print('ok')"], capture_output=True, text=True)
        decir("\n Prueba de importación: " + ("correcta" if prueba.stdout.strip() == "ok" else "FALLÓ\n" + prueba.stderr[-400:]))
        if prueba.stdout.strip() != "ok":
            problemas.append("importación")

    decir("\n" + "=" * 64)
    if faltan or problemas:
        decir(" Quedan pendientes: " + ", ".join([f[0] for f in faltan] + problemas))
        decir("=" * 64)
        return 1
    ruta = f"{entorno.name}\\Scripts\\python.exe" if ES_WINDOWS else f"{entorno.name}/bin/python"
    decir(" Todo listo. Para abrir la aplicación:")
    decir(f"   {ruta} -m app_escritorio.app" + ("    (o doble clic en abrir_app_windows.bat)" if ES_WINDOWS else ""))
    decir("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
