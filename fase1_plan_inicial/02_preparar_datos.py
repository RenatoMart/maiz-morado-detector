"""Une las fuentes de fotos y crea una sola división para los dos modelos.

Uso:
    python 02_preparar_datos.py --export data/roboflow/v4 \
        --primera RUTA/fotos_primera --descarte RUTA/fotos_descarte

--export    Dataset con cajas de Roboflow, en formato YOLO.
--primera   Fotos verificadas sin defectos: se usan en los dos modelos.
--descarte  Fotos de descarte sin cajas: solo para el modelo 1, porque el
            detector tomaría sus defectos sin marcar como fondo correcto.
--excluir   CSV con columna 'id' de imágenes en cuarentena tras la auditoría.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from maiz_morado.comun import (
    DATOS, DIR_M1, DIR_M2, SPLITS_CSV, cargar_config, enlazar, grado_por_nombre,
    id_base, listar_imagenes, normalizar_clase,
)
from maiz_morado.decision import PRIMERA, grado_por_regla

SPLITS_EXPORT = ("train", "valid", "val", "test")


def leer_nombres(export, config):
    with open(Path(export) / "data.yaml", encoding="utf-8") as f:
        nombres = yaml.safe_load(f)["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]
    return [normalizar_clase(n, config) for n in nombres]


def leer_cajas(ruta_label):
    """Devuelve [(id_clase, xc, yc, w, h)] y falla si encuentra polígonos."""
    if not ruta_label.exists():
        return []
    cajas = []
    for linea in ruta_label.read_text().splitlines():
        partes = linea.split()
        if not partes:
            continue
        if len(partes) != 5:
            raise SystemExit(f"{ruta_label}: se esperaba formato de cajas, no polígonos.")
        cajas.append((int(partes[0]), *map(float, partes[1:])))
    return cajas


def registros_export(export, nombres, config):
    filas = []
    for split in SPLITS_EXPORT:
        carpeta = Path(export) / split / "images"
        if not carpeta.is_dir():
            continue
        for imagen in listar_imagenes(carpeta):
            cajas = leer_cajas(carpeta.parent / "labels" / f"{imagen.stem}.txt")
            filas.append({
                "id": id_base(imagen.name),
                "imagen": str(imagen.resolve()),
                "grado_ref": grado_por_nombre(imagen.name, config),
                "anotada": True,
                "cajas": cajas,
                "clases": sorted({nombres[c[0]] for c in cajas}),
                "clases_lista": [nombres[c[0]] for c in cajas],
            })
    return filas


def registros_carpeta(carpeta, grado):
    return [{
        "id": id_base(imagen.name),
        "imagen": str(imagen.resolve()),
        "grado_ref": grado,
        "anotada": grado == "Primera",  # Primera verificada: se sabe que no tiene cajas
        "cajas": [],
        "clases": [],
        "clases_lista": [],
    } for imagen in listar_imagenes(carpeta)]


def unir(fuentes):
    """Quita repetidos por id; gana la primera fuente, que es la anotada."""
    vistos, unicos, repetidos = set(), [], 0
    for fila in fuentes:
        if fila["id"] in vistos:
            repetidos += 1
            continue
        vistos.add(fila["id"])
        unicos.append(fila)
    return unicos, repetidos


def clase_modelo1(fila, config):
    """Etiqueta del modelo 1, o None si las fuentes se contradicen o falta información."""
    sana = config["modelo1"]["nombre_clase_sana"]
    defecto = config["modelo1"]["nombre_clase_defecto"]
    referencia = fila["grado_ref"]
    if not fila["anotada"]:
        return defecto if referencia in ("Segunda", "Descarte") else None
    es_primera = grado_por_regla(fila["clases_lista"], config)[0] == PRIMERA
    if referencia == "Primera":
        return sana if es_primera else None
    if referencia in ("Segunda", "Descarte"):
        return None if es_primera else defecto
    return defecto if fila["cajas"] and not es_primera else None


def usar_en_detector(fila):
    """Una foto de Segunda o Descarte sin cajas enseñaría sus defectos como fondo."""
    return fila["anotada"] and not (fila["grado_ref"] in ("Segunda", "Descarte") and not fila["cajas"])


def estrato(fila, frecuencia):
    rara = min(fila["clases"], key=lambda c: frecuencia[c]) if fila["clases"] else "ninguna"
    return f"{fila['grado_ref'] or 'sin_ref'}|{rara}"


def agrupar_estratos_chicos(estratos, minimo=10):
    conteo = Counter(estratos)
    agrupados = [e if conteo[e] >= minimo else e.split("|")[0] for e in estratos]
    conteo = Counter(agrupados)
    return [e if conteo[e] >= minimo else "otros" for e in agrupados]


def _partir(indices, estratos, tamano, semilla):
    # Intenta por grado y defecto raro, luego solo por grado y, como último recurso, al azar.
    solo_grado = [e.split("|")[0] for e in estratos]
    for opcion in (estratos, solo_grado):
        try:
            return train_test_split(indices, test_size=tamano, random_state=semilla, stratify=opcion)
        except ValueError:
            continue
    print("AVISO: muy pocas imágenes para estratificar; se divide al azar.")
    return train_test_split(indices, test_size=tamano, random_state=semilla)


def dividir(filas, config):
    p = config["datos"]["proporciones"]
    semilla = config["datos"]["semilla"]
    por_nombre = Counter(c for f in filas for c in f["clases"])
    estratos = agrupar_estratos_chicos([estrato(f, por_nombre) for f in filas])
    indices = np.arange(len(filas))
    entrenamiento, resto = _partir(indices, estratos, p["val"] + p["test"], semilla)
    estratos_resto = agrupar_estratos_chicos([estratos[i] for i in resto], minimo=3)
    validacion, prueba = _partir(resto, estratos_resto, p["test"] / (p["val"] + p["test"]), semilla)
    for nombre, grupo in (("train", entrenamiento), ("val", validacion), ("test", prueba)):
        for i in grupo:
            filas[i]["split"] = nombre


def escribir_modelo1(filas):
    shutil.rmtree(DIR_M1, ignore_errors=True)
    for fila in filas:
        if fila["clase_m1"]:
            nombre = f"{fila['id']}{Path(fila['imagen']).suffix.lower()}"
            enlazar(fila["imagen"], DIR_M1 / fila["split"] / fila["clase_m1"] / nombre)


def escribir_modelo2(filas, nombres_originales):
    shutil.rmtree(DIR_M2, ignore_errors=True)
    for fila in filas:
        if not fila["usar_m2"]:
            continue
        nombre = f"{fila['id']}{Path(fila['imagen']).suffix.lower()}"
        enlazar(fila["imagen"], DIR_M2 / "images" / fila["split"] / nombre)
        etiqueta = DIR_M2 / "labels" / fila["split"] / f"{fila['id']}.txt"
        etiqueta.parent.mkdir(parents=True, exist_ok=True)
        etiqueta.write_text("".join(
            f"{c[0]} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f} {c[4]:.6f}\n" for c in fila["cajas"]
        ))
    (DIR_M2 / "data.yaml").write_text(yaml.safe_dump({
        "path": str(DIR_M2.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(nombres_originales)),
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")


def informar_tamanos(filas):
    """Lado corto de las cajas tras redimensionar: ayuda a elegir imgsz con datos."""
    relativos = []
    for fila in filas:
        if not fila["cajas"]:
            continue
        with Image.open(fila["imagen"]) as imagen:
            ancho, alto = imagen.size
        relativos += [min(c[3] * ancho, c[4] * alto) / max(ancho, alto) for c in fila["cajas"]]
    if not relativos:
        return
    print("\nLado corto de las cajas, en píxeles de entrada al modelo:")
    print("  imgsz  mínimo   p10  mediana")
    for imgsz in (640, 960, 1280):
        v = np.array(relativos) * imgsz
        print(f"  {imgsz:>5}  {v.min():6.1f} {np.percentile(v, 10):5.1f} {np.median(v):8.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True)
    parser.add_argument("--primera")
    parser.add_argument("--descarte")
    parser.add_argument("--excluir")
    args = parser.parse_args()
    config = cargar_config()

    with open(Path(args.export) / "data.yaml", encoding="utf-8") as f:
        crudos = yaml.safe_load(f)["names"]
    nombres_originales = [crudos[k] for k in sorted(crudos)] if isinstance(crudos, dict) else crudos
    nombres = leer_nombres(args.export, config)
    desconocidas = sorted(set(nombres) - set(config["clases"]))
    if desconocidas:
        raise SystemExit(f"Clases sin definir en configs/proyecto.yaml: {desconocidas}")

    fuentes = registros_export(args.export, nombres, config)
    if args.primera:
        fuentes += registros_carpeta(args.primera, "Primera")
    if args.descarte:
        fuentes += registros_carpeta(args.descarte, "Descarte")
    filas, repetidos = unir(fuentes)

    excluidas = 0
    if args.excluir:
        cuarentena = set(pd.read_csv(args.excluir)["id"].astype(str))
        excluidas = sum(f["id"] in cuarentena for f in filas)
        filas = [f for f in filas if f["id"] not in cuarentena]
    if len(filas) < 10:
        raise SystemExit(f"Solo hay {len(filas)} imágenes; no alcanza para dividir.")

    for fila in filas:
        fila["clase_m1"] = clase_modelo1(fila, config)
        fila["usar_m2"] = usar_en_detector(fila)
    dividir(filas, config)

    DATOS.mkdir(exist_ok=True)
    escribir_modelo1(filas)
    escribir_modelo2(filas, nombres_originales)

    tabla = pd.DataFrame([{
        "id": f["id"],
        "imagen": f["imagen"],
        "split": f["split"],
        "grado_ref": f["grado_ref"] or "",
        "clase_m1": f["clase_m1"] or "",
        "usar_m2": f["usar_m2"],
        "n_cajas": len(f["cajas"]),
        "clases": ";".join(f["clases"]),
    } for f in filas])
    tabla.to_csv(SPLITS_CSV, index=False)

    conflictos = tabla[tabla["clase_m1"].eq("")]
    if len(conflictos):
        ruta = DATOS / "conflictos_etiqueta.csv"
        conflictos.to_csv(ruta, index=False)
        print(f"{len(conflictos)} imágenes fuera del modelo 1 (contradicción o sin referencia): {ruta}")

    print(f"Imágenes: {len(tabla)} | repetidas omitidas: {repetidos} | en cuarentena: {excluidas}")
    print("\nModelo 1, imágenes por clase:")
    print(pd.crosstab(tabla["clase_m1"].replace("", "(excluida)"), tabla["split"]).to_string())
    print("\nModelo 2, imágenes por split:")
    usadas = tabla[tabla["usar_m2"]]
    print(pd.crosstab(usadas["n_cajas"].gt(0).map({True: "con defectos", False: "sin defectos"}), usadas["split"]).to_string())

    print("\nModelo 2, cajas por clase:")
    conteo = {(nombres[c[0]], f["split"]): 0 for f in filas for c in f["cajas"]}
    for f in filas:
        for c in f["cajas"]:
            conteo[(nombres[c[0]], f["split"])] += 1
    if conteo:
        cajas = pd.Series(conteo).unstack(fill_value=0)
        print(cajas.to_string())
        faltantes = [c for c in nombres if c not in cajas.index or cajas.loc[c].get("test", 0) == 0]
        if faltantes:
            print(f"AVISO: clases sin cajas en test, no se podrán evaluar: {faltantes}")
    informar_tamanos(filas)
    print(f"\nDivisión guardada en: {SPLITS_CSV}")


if __name__ == "__main__":
    main()
