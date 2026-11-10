"""Construye el dataset de entrenamiento con las anotaciones recientes de Roboflow.

Las anotaciones se leen de la sesión del navegador (datos/registros.json) y las fotos de
source.roboflow.com, así que no hace falta la clave de Roboflow.

Uso:
    python construir_dataset.py --desde 2026-11-05
"""
import argparse
import collections
import datetime as dt
import json
import random
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
OWNER = "8SYtx5mbSPPcVrmhV0JYbpn54pF2"
CLASES = ["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"]


def segundos(r):
    return float(r["last"]) / 1000 if r.get("last") else 0.0


def seleccionar(registros, desde, todas=False):
    """Usa las anotaciones recientes. Las Primera sin cajas son mazorcas limpias verificadas."""
    sel, omitidas = [], collections.Counter()
    for r in registros:
        primera, descarte = r["n"].startswith("primera"), r["n"].startswith("descarte")
        if not (primera or descarte):
            omitidas["nombre sin categoría"] += 1
        elif primera and not r["b"]:
            sel.append(r)
        elif todas and r["b"]:
            sel.append(r)
        elif r["b"] and segundos(r) >= desde and r.get("by") != "autolabelservice":
            sel.append(r)
        else:
            omitidas["anotación antigua, automática o vacía"] += 1
    return sel, omitidas


def preparar(r, lado):
    """Descarga la foto original, aplica la orientación EXIF y la reduce al lado indicado."""
    destino = DATOS / "img" / f"{r['id']}.jpg"
    if not destino.exists():
        tmp = DATOS / "tmp" / f"{r['id']}.jpg"
        url = f"https://source.roboflow.com/{OWNER}/{r['id']}/original.jpg"
        for _ in range(3):
            if subprocess.run(["curl", "-sf", "--max-time", "300", "-o", str(tmp), url]).returncode == 0:
                break
        else:
            return r["id"], "no se pudo descargar"
        try:
            im = ImageOps.exif_transpose(Image.open(tmp)).convert("RGB")
        except Exception as e:  # foto dañada
            tmp.unlink(missing_ok=True)
            return r["id"], f"imagen ilegible: {e}"
        if im.size != (r["w"], r["h"]):
            tmp.unlink(missing_ok=True)
            return r["id"], f"tamaño {im.size} distinto del anotado {(r['w'], r['h'])}"
        im.thumbnail((lado, lado), Image.LANCZOS)
        im.save(destino, quality=92)
        tmp.unlink(missing_ok=True)
    return r["id"], None


def huella(ruta):
    a = np.asarray(Image.open(ruta).convert("L").resize((17, 16)), float)
    return (a[:, 1:] > a[:, :-1]).flatten()


def clave_nombre(n):
    pre = n.split("__")[0]
    m = re.search(r"IMG_(\d+)", n)
    if m:
        return pre, "img", float(m.group(1)), 2
    m = re.search(r"(\d{8})_(\d{6})", n)
    if m:
        t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
        return pre, "ts", t, 40
    return pre, "x", 0.0, 0


def agrupar(items):
    """Une fotos seguidas y parecidas: probablemente son la misma mazorca vista de otro lado."""
    padre = {r["id"]: r["id"] for r in items}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    H = {r["id"]: huella(DATOS / "img" / f"{r['id']}.jpg") for r in items}
    K = {r["id"]: clave_nombre(r["n"]) for r in items}
    ids = [r["id"] for r in items]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ka, kb = K[a], K[b]
            if ka[0] == kb[0] and ka[1] == kb[1] and ka[1] != "x" and abs(ka[2] - kb[2]) <= ka[3] and (H[a] != H[b]).sum() <= 100:
                padre[raiz(a)] = raiz(b)
    grupos = collections.defaultdict(list)
    for i in ids:
        grupos[raiz(i)].append(i)
    return list(grupos.values())


def repartir(grupos, ref_de, semilla, prop):
    """Reparte grupos completos entre train, val y test dentro de cada categoría."""
    rnd, destino = random.Random(semilla), {}
    for ref in ("primera", "descarte"):
        gs = sorted((g for g in grupos if ref_de[g[0]] == ref), key=lambda g: sorted(g)[0])
        rnd.shuffle(gs)
        total = sum(len(g) for g in gs)
        cuota = {"val": prop[1] * total, "test": prop[2] * total}
        cont = collections.Counter()
        for g in gs:
            s = "val" if cont["val"] < cuota["val"] else "test" if cont["test"] < cuota["test"] else "train"
            cont[s] += len(g)
            for i in g:
                destino[i] = s
    return destino


def repartir_estable(grupos, previo, con_prueba):
    """Mantiene la división anterior; lo nuevo va a entrenamiento, salvo lo marcado con la etiqueta de prueba."""
    prio = {"test": 3, "val": 2, "train": 1}
    destino = {}
    for g in grupos:
        viejos = [previo[i] for i in g if i in previo]
        if viejos:
            s = max(viejos, key=prio.get)  # una mazorca nueva se queda con el conjunto de sus fotos anteriores
        elif any(i in con_prueba for i in g):
            s = "test"
        else:
            s = "train"
        for i in g:
            destino[i] = s
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registros", default=str(DATOS / "registros.json"))
    p.add_argument("--desde", default="2026-11-05", help="Solo anotaciones hechas desde esta fecha.")
    p.add_argument("--lado", type=int, default=1280)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--todas", action="store_true", help="Entrena con todas las fotos anotadas del Dataset, también las antiguas.")
    p.add_argument("--conservar", help="meta.json anterior: las fotos que ya estaban conservan su conjunto.")
    p.add_argument("--reservar-nuevas", type=int, default=0, help="Reserva al azar unas N fotos nuevas de Primera como prueba (por grupos de mazorca).")
    p.add_argument("--tag-prueba", default="prueba", help="Etiqueta de Roboflow para reservar fotos nuevas como prueba.")
    p.add_argument("--solo-seleccion", action="store_true", help="Descarga solo las fotos que se usarán para entrenar.")
    a = p.parse_args()

    (DATOS / "img").mkdir(parents=True, exist_ok=True)
    (DATOS / "tmp").mkdir(exist_ok=True)
    R = json.load(open(a.registros))
    desde = dt.datetime.fromisoformat(a.desde).timestamp()
    corr = json.load(open(AQUI / "correcciones.json")) if (AQUI / "correcciones.json").exists() else {}
    excluidas = set(corr.get("excluir", {}))
    R = [r for r in R if r["id"] not in excluidas]
    for r in R:
        c = corr.get("cambiar_clase", {}).get(r["id"])
        if c:
            for b in r["b"]:
                if b["l"] == c["de"]:
                    b["l"] = c["a"]
    if corr:
        print(f"Correcciones aplicadas: {len(excluidas)} fotos excluidas, {len(corr.get('cambiar_clase', {}))} con clase cambiada")
    sel, omitidas = seleccionar(R, desde, a.todas)
    print(f"{len(R)} registros -> {len(sel)} seleccionados para entrenar; omitidos: {dict(omitidas)}")
    if not a.solo_seleccion:
        print("Se descargan todas las fotos del Dataset; se entrena solo con las seleccionadas.")

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda r: preparar(r, a.lado), sel if a.solo_seleccion else R))
    fallos = {i: m for i, m in res if m}
    for i, m in list(fallos.items())[:10]:
        print("  omitida:", i, m)
    sel = [r for r in sel if r["id"] not in fallos]

    ref_de = {r["id"]: "primera" if r["n"].startswith("primera") else "descarte" for r in sel}
    grupos = agrupar(sel)
    if a.conservar:
        previo = {m["id"]: m["split"] for m in json.load(open(a.conservar))}
        con_prueba = {r["id"] for r in sel if a.tag_prueba in (r.get("tags") or [])}
        if a.reservar_nuevas and not con_prueba:
            candidatos = sorted((sorted(g) for g in grupos if not any(i in previo for i in g) and ref_de[g[0]] == "primera"), key=lambda g: g[0])
            random.Random(a.semilla).shuffle(candidatos)
            for g in candidatos:
                if len(con_prueba) >= a.reservar_nuevas:
                    break
                con_prueba |= set(g)
        destino = repartir_estable(grupos, previo, con_prueba)
        print(f"División conservada: {sum(i in previo for i in destino)} fotos ya conocidas, {len(con_prueba)} reservadas con la etiqueta '{a.tag_prueba}', {sum(i not in previo for i in destino)} nuevas")
    else:
        destino = repartir(grupos, ref_de, a.semilla, (0.7, 0.2, 0.1))
    gid = {i: k for k, g in enumerate(grupos) for i in g}

    for carpeta in ("images", "labels"):
        shutil.rmtree(DATOS / carpeta, ignore_errors=True)
    meta, cajas_clase, ignoradas = [], collections.Counter(), collections.Counter()
    for r in sel:
        s = destino[r["id"]]
        for c in ("images", "labels"):
            (DATOS / c / s).mkdir(parents=True, exist_ok=True)
        (DATOS / "images" / s / f"{r['id']}.jpg").hardlink_to(DATOS / "img" / f"{r['id']}.jpg")
        lineas, cajas = [], []
        vistas = set()
        for b in r["b"]:
            if b["l"] not in CLASES:
                ignoradas[b["l"]] += 1
                continue
            x1, y1 = max(0.0, b["x"] - b["w"] / 2), max(0.0, b["y"] - b["h"] / 2)
            x2, y2 = min(float(r["w"]), b["x"] + b["w"] / 2), min(float(r["h"]), b["y"] + b["h"] / 2)
            if (x2 - x1) / r["w"] < 0.002 or (y2 - y1) / r["h"] < 0.002:
                ignoradas["caja casi vacía"] += 1
                continue
            c = [CLASES.index(b["l"]), (x1 + x2) / 2 / r["w"], (y1 + y2) / 2 / r["h"], (x2 - x1) / r["w"], (y2 - y1) / r["h"]]
            t = tuple(round(v, 5) for v in c)
            if t in vistas:
                continue
            vistas.add(t)
            cajas.append(c)
            lineas.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(*c))
            cajas_clase[(s, b["l"])] += 1
        (DATOS / "labels" / s / f"{r['id']}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))
        meta.append({"id": r["id"], "n": r["n"], "ref": ref_de[r["id"]], "split": s, "grupo": gid[r["id"]], "w": r["w"], "h": r["h"], "cajas": cajas})
    (DATOS / "data.yaml").write_text(yaml.safe_dump({"path": str(DATOS), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(CLASES))}, allow_unicode=True, sort_keys=False))
    json.dump(meta, open(DATOS / "meta.json", "w"), ensure_ascii=False)

    cuenta = collections.Counter((m["split"], m["ref"]) for m in meta)
    print(f"\nImágenes: {len(meta)} | mazorcas probables (grupos): {len(grupos)} | cajas ignoradas: {dict(ignoradas)}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} primera {cuenta[(s, 'primera')]:3d}  descarte {cuenta[(s, 'descarte')]:3d}  cajas: " + ", ".join(f"{c.split()[0]} {cajas_clase[(s, c)]}" for c in CLASES))
    cruces = sum(1 for g in grupos if len({destino[i] for i in g}) > 1)
    print(f"Grupos repartidos entre conjuntos (debe ser 0): {cruces}")


if __name__ == "__main__":
    main()
