"""Prueba el modelo entrenado con las fotos de prueba y deja imágenes con el veredicto.

Uso: python probar.py --modelo modelo_final_final [--ablacion]
Salida en pruebas/: resultados.csv, laminas con aciertos y errores, ablacion.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from maiz_morado.medidas import CLASES, mascara_mazorca, rasgos, reducir, vector
from maiz_morado.veredicto import Decision, evaluar_imagen

COLOR = {0: "#3a86ff", 1: "#ff4fd8", 2: "#ff2a2a", 3: "#00e676", 4: "#ffe600"}
FUENTE = font_manager.findfont("DejaVu Sans")


def imagen(m, split):
    bgr = cv2.imread(str(AQUI / "datos" / "images" / split / f"{m['id']}.jpg"))
    return reducir(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def mascara(m, rgb):
    d = AQUI / "datos" / "mask"
    d.mkdir(exist_ok=True)
    f = d / f"{m['id']}.png"
    if f.exists():
        return (cv2.imread(str(f), 0) > 0).astype(np.uint8)
    x = mascara_mazorca(rgb)
    cv2.imwrite(str(f), x * 255)
    return x


def tarjeta(rgb, mascara_, v, m, ancho=380, alto=520):
    im = Image.fromarray(rgb)
    im.thumbnail((ancho, 360))
    e = im.width / rgb.shape[1]
    d = ImageDraw.Draw(im)
    for c, xc, yc, w, h, p in v["cajas"]:
        H, W = rgb.shape[:2]
        d.rectangle([(xc - w / 2) * W * e, (yc - h / 2) * H * e, (xc + w / 2) * W * e, (yc + h / 2) * H * e], outline=COLOR[int(c)], width=2)
    t = Image.new("RGB", (ancho, alto), "#14161a")
    t.paste(im, ((ancho - im.width) // 2, 0))
    d = ImageDraw.Draw(t)
    ok = (v["categoria"] == "Descarte") == (m["ref"] == "descarte")
    d.rectangle([0, 364, ancho, 364 + 34], fill="#2dc653" if ok else "#e5383b")
    d.text((8, 368), f"{'PASA' if v['pasa'] else 'DESCARTA'} ({v['categoria']})  p={v['p_descarte']:.2f}", fill="black", font=ImageFont.truetype(FUENTE, 17))
    y = 404
    for linea in [f"{m['n'].replace('primera_clase__', 'P ').replace('descarte__', 'D ')}  real: {m['ref']}",
                  f"defectos: {v['rasgos']['area_total'] * 100:.1f} % de la mazorca  |  {len(v['cajas'])} zonas"] + v["motivos"][:2]:
        d.text((8, y), linea[:52], fill="#e8eaed", font=ImageFont.truetype(FUENTE, 13))
        y += 20
    return t


def lamina(tarjetas, ruta, columnas=3):
    if not tarjetas:
        return
    f = (len(tarjetas) + columnas - 1) // columnas
    w, h = tarjetas[0].size
    L = Image.new("RGB", (columnas * w, f * h), "#14161a")
    for i, t in enumerate(tarjetas):
        L.paste(t, ((i % columnas) * w, (i // columnas) * h))
    L.save(ruta, quality=88)


def ablacion(yolo, decision, meta, filas_test, salida):
    """Reajusta la decisión sin algunas medidas para ver de cuáles depende realmente."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cache = AQUI / "datos" / "rasgos_train.json"
    if cache.exists():
        X, y = json.load(open(cache))
    else:
        X, y = [], []
        for m in [m for m in meta if m["split"] == "train"]:
            rgb = imagen(m, "train")
            X.append(vector(rasgos(rgb, mascara(m, rgb), m["cajas"])))
            y.append(int(m["ref"] == "descarte"))
        json.dump([X, y], open(cache, "w"))
    X, y = np.array(X), np.array(y)
    Xt = np.array([vector(f["v"]["rasgos"]) for f in filas_test])
    yt = np.array([int(f["m"]["ref"] == "descarte") for f in filas_test])
    ordn = decision.orden
    grupos = {"completo": ordn, "sin color": [c for c in ordn if c not in ("a_med", "rojo15", "rojo25")], "solo color": ["a_med", "rojo15", "rojo25"],
              "solo conteos": ["n_cajas", "n0", "n1", "n2", "n3", "n4"], "solo áreas": ["area_total", "a0", "a1", "a2", "a3", "a4"]}
    out = {}
    for nombre, cols in grupos.items():
        idx = [ordn.index(c) for c in cols]
        esc = StandardScaler().fit(X[:, idx])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(esc.transform(X[:, idx]), y)
        pred = (clf.predict_proba(esc.transform(Xt[:, idx]))[:, 1] >= 0.5).astype(int)
        out[nombre] = {"primera_ok": round(float((pred[yt == 0] == 0).mean()), 3), "descarte_ok": round(float((pred[yt == 1] == 1).mean()), 3)}
    json.dump(out, open(salida / "ablacion.json", "w"), indent=1, ensure_ascii=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default=str(AQUI / "modelo_final_final"))
    p.add_argument("--salida", default=str(AQUI / "pruebas"))
    p.add_argument("--ablacion", action="store_true")
    a = p.parse_args()
    modelo, salida = Path(a.modelo), Path(a.salida)
    salida.mkdir(exist_ok=True)
    from ultralytics import YOLO
    yolo, decision = YOLO(str(modelo / "best.pt")), Decision(modelo / "decision.json")
    meta = json.load(open(AQUI / "datos" / "meta.json"))
    filas = []
    for m in [m for m in meta if m["split"] == "test"]:
        rgb = imagen(m, "test")
        r = yolo.predict(str(AQUI / "datos" / "images" / "test" / f"{m['id']}.jpg"), imgsz=640, conf=decision.confianza, iou=decision.iou, device="cpu", verbose=False)[0]
        dets = [(int(c), *xywhn, float(q)) for c, xywhn, q in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
        v = evaluar_imagen(rgb, dets, decision, mascara(m, rgb))
        filas.append({"m": m, "v": v, "rgb": rgb})
    with open(salida / "resultados.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["foto", "real", "categoria", "pasa", "p_descarte", "defectos_pct", "zonas", "motivos"])
        for x in filas:
            w.writerow([x["m"]["n"], x["m"]["ref"], x["v"]["categoria"], x["v"]["pasa"], round(x["v"]["p_descarte"], 3), round(x["v"]["rasgos"]["area_total"] * 100, 1), len(x["v"]["cajas"]), "; ".join(x["v"]["motivos"])])
    ok = lambda x: (x["v"]["categoria"] == "Descarte") == (x["m"]["ref"] == "descarte")
    P = [x for x in filas if x["m"]["ref"] == "primera"]
    D = [x for x in filas if x["m"]["ref"] == "descarte"]
    err = [x for x in filas if not ok(x)]
    res = {"fotos": len(filas), "primera": len(P), "descarte": len(D), "primera_ok": sum(map(ok, P)), "descarte_ok": sum(map(ok, D)),
           "errores": [(x["m"]["n"], x["m"]["ref"], x["v"]["categoria"], round(x["v"]["p_descarte"], 2)) for x in err],
           "defectos_pct_mediano": {"primera": round(float(np.median([x["v"]["rasgos"]["area_total"] for x in P])) * 100, 1), "descarte": round(float(np.median([x["v"]["rasgos"]["area_total"] for x in D])) * 100, 1)}}
    tj = lambda xs: [tarjeta(x["rgb"], None, x["v"], x["m"]) for x in xs]
    lamina(tj(sorted([x for x in P if ok(x)], key=lambda x: x["v"]["p_descarte"])[:6]), salida / "aciertos_primera.jpg")
    lamina(tj(sorted([x for x in D if ok(x)], key=lambda x: -x["v"]["p_descarte"])[:6]), salida / "aciertos_descarte.jpg")
    lamina(tj(sorted(err, key=lambda x: abs(x["v"]["p_descarte"] - 0.5))[:6]), salida / "errores.jpg")
    cercanos = sorted([x for x in D if ok(x)], key=lambda x: x["v"]["p_descarte"])[:6]
    lamina(tj(cercanos), salida / "descarte_acertado_por_poco.jpg")
    if a.ablacion:
        res["ablacion"] = ablacion(yolo, decision, meta, filas, salida)
    json.dump(res, open(salida / "resumen.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
