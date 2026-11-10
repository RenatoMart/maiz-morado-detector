"""Después de entrenar: mide el detector, ajusta la decisión por proporciones y exporta para la web.

Uso: python pos.py --run runs/final --salida modelo_final
Genera modelo_final/{best.pt, best.onnx, decision.json, RESULTADO.md}.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from maiz_morado.medidas import CLASES, ORDEN, mascara_mazorca, rasgos, reducir, vector

CONFS = [0.05, 0.10, 0.15, 0.25, 0.35]


def cargar(m):
    bgr = cv2.imread(str(AQUI / "datos" / "images" / m["split"] / f"{m['id']}.jpg"))
    return reducir(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def mascara_cacheada(m, rgb):
    d = AQUI / "datos" / "mask"
    d.mkdir(exist_ok=True)
    f = d / f"{m['id']}.png"
    if f.exists():
        return (cv2.imread(str(f), 0) > 0).astype(np.uint8)
    x = mascara_mazorca(rgb)
    cv2.imwrite(str(f), x * 255)
    return x


def metricas_detector(modelo, split, **kw):
    v = modelo.val(data=str(AQUI / "datos" / "data.yaml"), split=split, batch=8, device="cpu", workers=0, plots=False, verbose=False, **kw)
    return {"map50": float(v.box.map50), "map50_95": float(v.box.map), "precision": float(v.box.mp), "recall": float(v.box.mr),
            "ap50_por_clase": {CLASES[int(c)]: round(float(a), 3) for c, a in zip(v.box.ap_class_index, v.box.ap50)}}


def detectar(modelo, imagenes, ruta_de, **kw):
    """Detecciones por imagen con confianza mínima 0.05, para filtrarlas después."""
    out = {}
    for m in imagenes:
        r = modelo.predict(str(ruta_de(m)), conf=min(CONFS), device="cpu", verbose=False, **kw)[0]
        out[m["id"]] = [(int(c), *xywhn, float(p)) for c, xywhn, p in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
    return out


def evaluar(clf, esc, datos, thr, regla_zonas=2):
    """Devuelve (aciertos Primera, aciertos Descarte, exactitud balanceada, categorías predichas)."""
    y = np.array([d["y"] for d in datos])
    p = clf.predict_proba(esc.transform([d["x"] for d in datos]))[:, 1]
    pred = (p >= thr).astype(int)
    cat = ["Descarte" if q else ("Segunda" if d["z"] >= regla_zonas else "Primera") for q, d in zip(pred, datos)]
    return float((pred[y == 0] == 0).mean()) if (y == 0).any() else None, float((pred[y == 1] == 1).mean()) if (y == 1).any() else None, float(balanced_accuracy_score(y, pred)), cat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--salida", default=str(AQUI / "modelo_final"))
    p.add_argument("--min-descarte", type=float, default=0.0, help="Exige al menos esta fracción de Descarte rechazado en validación.")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--conf-fija", type=float, default=0.0, help="Usa solo esta confianza mínima y deja que la validación escoja únicamente el umbral.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--maximo", type=int, default=0, help="Limita imágenes por conjunto (solo para pruebas).")
    a = p.parse_args()
    run, salida = Path(a.run), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    mejor = run / "weights" / "best.pt"
    if not mejor.exists():
        sys.exit(f"No existe {mejor}")
    from ultralytics import YOLO

    t0 = time.time()
    modelo = YOLO(str(mejor))
    meta = json.load(open(AQUI / "datos" / "meta.json"))
    por = {s: [m for m in meta if m["split"] == s] for s in ("train", "val", "test")}
    if a.maximo:
        por = {s: [m for r in ("primera", "descarte") for m in [x for x in v if x["ref"] == r][: a.maximo // 2]] for s, v in por.items()}
    R = {"imagenes": {s: len(v) for s, v in por.items()}}

    print("1/4 métricas del detector", flush=True)
    kw = {"imgsz": a.imgsz, "iou": a.iou}
    R["detector"] = {s: metricas_detector(modelo, s, **kw) for s in ("val", "test")}

    print("2/4 medidas con las cajas humanas del conjunto de entrenamiento", flush=True)
    img, masc = {}, {}
    for s in por:
        for m in por[s]:
            img[m["id"]] = cargar(m)
            masc[m["id"]] = mascara_cacheada(m, img[m["id"]])
    train = [{"x": vector(rasgos(img[m["id"]], masc[m["id"]], m["cajas"])), "y": int(m["ref"] == "descarte")} for m in por["train"]]
    X, y = np.array([d["x"] for d in train]), np.array([d["y"] for d in train])
    esc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(esc.transform(X), y)

    print("3/4 decisión con las cajas que dibuja el detector", flush=True)
    det = {s: detectar(modelo, por[s], lambda m, s=s: AQUI / "datos" / "images" / s / f"{m['id']}.jpg", **kw) for s in ("val", "test")}

    def armar(s, conf):
        out = []
        for m in por[s]:
            cajas = [c for c in det[s][m["id"]] if c[5] >= conf]
            r = rasgos(img[m["id"]], masc[m["id"]], cajas)
            out.append({"x": vector(r), "y": int(m["ref"] == "descarte"), "z": r["n4"], "n2": r["n2"], "n_pudricion": r["n3"]})
        return out

    candidatos = []
    for conf in ([a.conf_fija] if a.conf_fija else CONFS):
        v = armar("val", conf)
        for thr in np.arange(0.05, 0.81, 0.05):
            pv, dv, bal, _ = evaluar(clf, esc, v, float(thr))
            candidatos.append({"conf": conf, "thr": round(float(thr), 2), "bal_val": bal, "primera_val": pv, "descarte_val": dv})
    validos = [c for c in candidatos if a.min_descarte and c["descarte_val"] is not None and c["descarte_val"] >= a.min_descarte]
    mejor_cfg = max(validos, key=lambda c: (c["primera_val"], c["bal_val"])) if validos else max(candidatos, key=lambda c: c["bal_val"])
    R["criterio_de_eleccion"] = f"máxima Primera correcta con Descarte rechazado >= {a.min_descarte:.0%} en validación" if validos else "máxima exactitud balanceada en validación"
    t = armar("test", mejor_cfg["conf"])
    pt, dt_, bal_t, cat = evaluar(clf, esc, t, mejor_cfg["thr"])
    base = armar("test", 0.25)
    pb, db, bb, _ = evaluar(clf, esc, base, mejor_cfg["thr"])
    regla5 = [int(d["n2"] > 5 or d["n_pudricion"] > 0 or d["z"] >= 4) for d in base]
    yt = np.array([d["y"] for d in base])
    R["decision"] = {"elegida_en_validacion": mejor_cfg, "test": {"primera_ok": pt, "descarte_ok": dt_, "exactitud_balanceada": bal_t},
                     "test_con_confianza_0.25": {"primera_ok": pb, "descarte_ok": db},
                     "regla_de_conteo_test": {"primera_ok": float((np.array(regla5)[yt == 0] == 0).mean()) if (yt == 0).any() else None, "descarte_ok": float((np.array(regla5)[yt == 1] == 1).mean()) if (yt == 1).any() else None},
                     "categorias_test": {c: cat.count(c) for c in ("Primera", "Segunda", "Descarte")}}

    print("4/4 exportación", flush=True)
    shutil.copy2(mejor, salida / "best.pt")
    try:
        onnx = modelo.export(format="onnx", imgsz=640, simplify=False, device="cpu")
        shutil.copy2(onnx, salida / "best.onnx")
        R["onnx"] = "ok"
    except Exception as e:
        R["onnx"] = f"no se pudo exportar: {str(e)[:120]}"
    for f in ("results.png", "confusion_matrix.png", "BoxPR_curve.png"):
        if (run / f).exists():
            shutil.copy2(run / f, salida / f)
    json.dump({"clases": list(CLASES), "orden_rasgos": ORDEN, "media": esc.mean_.tolist(), "escala": esc.scale_.tolist(),
               "coef": clf.coef_[0].tolist(), "intercepto": float(clf.intercept_[0]), "confianza_minima": mejor_cfg["conf"],
               "umbral_descarte": mejor_cfg["thr"], "zonas_para_segunda": 2, "imgsz": a.imgsz, "iou_nms": a.iou,
               "nota": "p = sigmoide(coef . ((x - media) / escala) + intercepto). Descarte si p >= umbral; si no, Segunda con 2 o más zonas sin granos, y Primera en otro caso. Los rasgos se calculan sobre la mazorca segmentada."},
              open(salida / "decision.json", "w"), indent=1, ensure_ascii=False)
    R["segundos"] = round(time.time() - t0)
    json.dump(R, open(salida / "resultado.json", "w"), indent=1, ensure_ascii=False)
    d, dd = R["detector"]["test"], R["decision"]
    (salida / "RESULTADO.md").write_text(
        f"# Resultado del entrenamiento\n\nImágenes: {R['imagenes']}\n\n## Detector (conjunto de prueba)\n- mAP@50: {d['map50']:.3f}  |  mAP@50-95: {d['map50_95']:.3f}\n- Precisión: {d['precision']:.3f}  |  Recall: {d['recall']:.3f}\n- Por clase (AP@50): {d['ap50_por_clase']}\n\n"
        f"## Decisión por proporciones (prueba)\n- Configuración elegida en validación: {mejor_cfg}\n- Primera correcta: {pt}  |  Descarte correcto: {dt_}\n- Regla de conteo, para comparar: {dd['regla_de_conteo_test']}\n- Categorías predichas: {dd['categorias_test']}\n\nONNX: {R['onnx']}\n\nAdvertencia: validación y prueba son pocas imágenes, así que las cifras son orientativas.\n")
    print(json.dumps(R["decision"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
