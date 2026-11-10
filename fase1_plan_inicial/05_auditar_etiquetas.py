"""Busca etiquetas dudosas con validación cruzada y Confident Learning (cleanlab).

Cada imagen de train y val la evalúa un modelo que no la vio al entrenar. Test no
participa, para que siga siendo una evaluación limpia. El script no cambia ni
borra nada: genera una lista ordenada para que una persona la revise en Roboflow.

Uso:
    python 05_auditar_etiquetas.py --modelo 2 --exp R1 --folds 5
    python 05_auditar_etiquetas.py --modelo 1 --exp C1 --folds 5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from maiz_morado.comun import (
    CORRIDAS, DATOS, DIR_M2, SPLITS_CSV, cargar_config, enlazar, entrenar,
    resolver_experimento,
)
from maiz_morado.sistema import a_bgr, cargar_imagen

CONF_AUDITORIA = 0.01  # cleanlab necesita también las predicciones de baja confianza


def particiones(etiquetas, folds, semilla):
    conteo = pd.Series(etiquetas).value_counts()
    seguras = [e if conteo[e] >= folds else "otros" for e in etiquetas]
    if pd.Series(seguras).value_counts().min() < folds:
        seguras = ["todas"] * len(seguras)
    divisor = StratifiedKFold(n_splits=folds, shuffle=True, random_state=semilla)
    return list(divisor.split(np.zeros(len(seguras)), seguras))


def leer_etiqueta_absoluta(ruta_txt, ancho, alto):
    cajas, clases = [], []
    if ruta_txt.exists():
        for linea in ruta_txt.read_text().splitlines():
            c, xc, yc, w, h = map(float, linea.split())
            cajas.append([(xc - w / 2) * ancho, (yc - h / 2) * alto, (xc + w / 2) * ancho, (yc + h / 2) * alto])
            clases.append(int(c))
    return {
        "bboxes": np.array(cajas, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(clases, dtype=np.int64),
    }


def auditar_modelo2(tabla, experimento, folds, config, extra):
    from cleanlab.object_detection.filter import find_label_issues
    from cleanlab.object_detection.rank import get_label_quality_scores
    from ultralytics import YOLO

    with open(DIR_M2 / "data.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    nombres = data["names"]
    k_clases = len(nombres)
    tabla = tabla[tabla["usar_m2"] & tabla["split"].isin(["train", "val"])].reset_index(drop=True)
    carpeta_folds = DATOS / "folds_modelo2"
    shutil.rmtree(carpeta_folds, ignore_errors=True)
    carpeta_folds.mkdir(parents=True)

    def ruta_img(fila):
        return DIR_M2 / "images" / fila["split"] / f"{fila['id']}{Path(fila['imagen']).suffix.lower()}"

    etiquetas, predicciones = {}, {}
    estratos = [c.split(";")[0] if c else "ninguna" for c in tabla["clases"].fillna("")]
    for n, (idx_train, idx_val) in enumerate(particiones(estratos, folds, config["datos"]["semilla"]), 1):
        lista_train = carpeta_folds / f"fold{n}_train.txt"
        lista_val = carpeta_folds / f"fold{n}_val.txt"
        lista_train.write_text("\n".join(str(ruta_img(tabla.iloc[i])) for i in idx_train) + "\n")
        lista_val.write_text("\n".join(str(ruta_img(tabla.iloc[i])) for i in idx_val) + "\n")
        yaml_fold = carpeta_folds / f"fold{n}.yaml"
        yaml_fold.write_text(yaml.safe_dump({
            "path": str(DIR_M2.resolve()), "train": str(lista_train), "val": str(lista_val), "names": nombres,
        }, allow_unicode=True, sort_keys=False), encoding="utf-8")

        _, mejor, _ = entrenar(
            experimento, yaml_fold, "auditoria_modelo2", f"fold{n}",
            config["datos"]["semilla"], **extra,
        )
        modelo = YOLO(str(mejor))
        for i in idx_val:
            fila = tabla.iloc[i]
            imagen = cargar_imagen(ruta_img(fila))
            resultado = modelo.predict(a_bgr(imagen), conf=CONF_AUDITORIA, verbose=False)[0]
            cajas = resultado.boxes.xyxy.cpu().numpy()
            confianzas = resultado.boxes.conf.cpu().numpy()
            clases = resultado.boxes.cls.cpu().numpy().astype(int)
            por_clase = np.empty(k_clases, dtype=object)
            for k in range(k_clases):
                m = clases == k
                por_clase[k] = np.hstack([cajas[m], confianzas[m, None]]).astype(np.float32).reshape(-1, 5)
            ruta_txt = DIR_M2 / "labels" / fila["split"] / f"{fila['id']}.txt"
            etiquetas[i] = leer_etiqueta_absoluta(ruta_txt, imagen.width, imagen.height)
            predicciones[i] = por_clase

    orden = sorted(etiquetas)
    lista_etiquetas = [etiquetas[i] for i in orden]
    lista_predicciones = [predicciones[i] for i in orden]
    puntajes = get_label_quality_scores(lista_etiquetas, lista_predicciones, verbose=False)
    dudosas = find_label_issues(lista_etiquetas, lista_predicciones)
    salida = tabla.loc[orden, ["id", "imagen", "split", "grado_ref", "n_cajas", "clases"]].copy()
    salida["calidad_etiqueta"] = puntajes
    salida["dudosa"] = dudosas
    return salida.sort_values("calidad_etiqueta")


def auditar_modelo1(tabla, experimento, folds, config, extra):
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    from ultralytics import YOLO

    tabla = tabla[tabla["clase_m1"].fillna("").ne("") & tabla["split"].isin(["train", "val"])].reset_index(drop=True)
    sana = config["modelo1"]["nombre_clase_sana"]
    defecto = config["modelo1"]["nombre_clase_defecto"]
    clases = sorted([sana, defecto])
    y = np.array([clases.index(c) for c in tabla["clase_m1"]])
    probabilidades = np.zeros((len(tabla), 2))
    carpeta_folds = DATOS / "folds_modelo1"

    for n, (idx_train, idx_val) in enumerate(particiones(tabla["clase_m1"].tolist(), folds, config["datos"]["semilla"]), 1):
        shutil.rmtree(carpeta_folds, ignore_errors=True)
        for nombre_split, indices in (("train", idx_train), ("val", idx_val)):
            for i in indices:
                fila = tabla.iloc[i]
                nombre = f"{fila['id']}{Path(fila['imagen']).suffix.lower()}"
                enlazar(fila["imagen"], carpeta_folds / nombre_split / fila["clase_m1"] / nombre)
        _, mejor, _ = entrenar(
            experimento, carpeta_folds, "auditoria_modelo1", f"fold{n}",
            config["datos"]["semilla"], **extra,
        )
        modelo = YOLO(str(mejor))
        indice_modelo = {nombre: k for k, nombre in modelo.names.items()}
        for i in idx_val:
            resultado = modelo.predict(a_bgr(cargar_imagen(tabla.iloc[i]["imagen"])), verbose=False)[0]
            for k, nombre in enumerate(clases):
                probabilidades[i, k] = float(resultado.probs.data[indice_modelo[nombre]])
    shutil.rmtree(carpeta_folds, ignore_errors=True)

    salida = tabla[["id", "imagen", "split", "grado_ref", "clase_m1"]].copy()
    salida["p_sana"] = probabilidades[:, clases.index(sana)]
    salida["calidad_etiqueta"] = get_label_quality_scores(y, probabilidades)
    salida["dudosa"] = find_label_issues(y, probabilidades)
    return salida.sort_values("calidad_etiqueta")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", type=int, choices=[1, 2], required=True)
    parser.add_argument("--exp")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    if not SPLITS_CSV.exists():
        raise SystemExit("Primero ejecuta 02_preparar_datos.py.")
    config = cargar_config()
    tabla = pd.read_csv(SPLITS_CSV)
    extra = {k: getattr(args, k) for k in ("epochs", "imgsz", "batch", "device") if getattr(args, k) is not None}
    clave = f"modelo{args.modelo}"
    predeterminado = "R1" if args.modelo == 2 else "C1"
    experimento = resolver_experimento(config[clave]["experimentos"], args.exp or predeterminado)

    auditar = auditar_modelo2 if args.modelo == 2 else auditar_modelo1
    salida = auditar(tabla, experimento, args.folds, config, extra)

    CORRIDAS.mkdir(exist_ok=True)
    ruta = CORRIDAS / f"auditoria_modelo{args.modelo}.csv"
    salida.to_csv(ruta, index=False)
    print(f"Imágenes auditadas: {len(salida)} | dudosas: {int(salida['dudosa'].sum())}")
    print(f"Lista para revisar en Roboflow, de peor a mejor: {ruta}")
    print(salida.head(15)[["id", "calidad_etiqueta", "dudosa"]].to_string(index=False))


if __name__ == "__main__":
    main()
