"""Utilidades compartidas: configuración, nombres de archivo y entrenamiento."""
import re
import shutil
import unicodedata
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ / "configs" / "proyecto.yaml"
DATOS = RAIZ / "data"
CORRIDAS = RAIZ / "runs"
MODELOS = RAIZ / "modelos"
SPLITS_CSV = DATOS / "splits.csv"
DIR_M1 = DATOS / "modelo1_clasificacion"
DIR_M2 = DATOS / "modelo2_deteccion"
EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def cargar_config(ruta=RUTA_CONFIG):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalizar_clase(nombre, config=None):
    """'Pigmentación Roja' y 'pigmentacion-roja' pasan a 'pigmentacion_roja'."""
    sin_tildes = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    clave = re.sub(r"[^a-z0-9]+", "_", sin_tildes.lower()).strip("_")
    if config:
        clave = config.get("alias", {}).get(clave, clave)
    return clave


def id_base(nombre):
    """Identificador estable: quita extensión y el sufijo '_jpg.rf.<hash>' de Roboflow."""
    tallo = Path(nombre).stem
    tallo = re.sub(r"\.rf\.[0-9A-Za-z]+$", "", tallo)
    return re.sub(r"[_.-](jpe?g|png|webp|bmp)$", "", tallo, flags=re.I)


def grado_por_nombre(nombre, config):
    """Grado de referencia según el prefijo del archivo, o None si no lo tiene."""
    minusculas = Path(nombre).name.lower()
    for prefijo, grado in config["datos"]["prefijos_grado"].items():
        if minusculas.startswith(prefijo):
            return grado
    return None


def listar_imagenes(carpeta):
    return sorted(
        p for p in Path(carpeta).rglob("*")
        if p.is_file() and p.suffix.lower() in EXT_IMAGEN
    )


def enlazar(origen, destino):
    """Enlace duro para no duplicar fotos en disco; copia si no es posible."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        destino.hardlink_to(origen)
    except OSError:
        shutil.copy2(origen, destino)


def resolver_experimento(experimentos, nombre):
    """Devuelve los argumentos del experimento; 'base' hereda de otro experimento."""
    if nombre not in experimentos:
        raise SystemExit(f"Experimento '{nombre}' no existe. Opciones: {sorted(experimentos)}")
    args = dict(experimentos[nombre])
    base = args.pop("base", None)
    if base:
        args = {**resolver_experimento(experimentos, base), **args}
    return args


def entrenar(args_experimento, data, grupo, nombre, semilla, **extra):
    """Entrena con Ultralytics y devuelve (modelo, ruta_best, carpeta_corrida)."""
    from ultralytics import YOLO

    args = {**args_experimento, **extra}
    modelo = YOLO(args.pop("model"))
    modelo.train(
        data=str(data),
        project=str(CORRIDAS / grupo),
        name=nombre,
        exist_ok=True,
        seed=semilla,
        deterministic=True,
        pretrained=True,
        plots=True,
        **args,
    )
    return modelo, Path(modelo.trainer.best), Path(modelo.trainer.save_dir)
