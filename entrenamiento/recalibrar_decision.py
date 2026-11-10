"""Reajusta la decisión usando las cajas que produce el detector (no las dibujadas por personas).

La decisión se ajustaba con cajas humanas y se aplicaba a cajas del detector, que son más grandes y
confunden la punta con pudrición. Aquí se ajusta con lo que el detector entrega de verdad, se elige el
umbral con validación cruzada por grupos de mazorca y se mide una sola vez en la prueba.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from maiz_morado.medidas import CLASES, ORDEN, mascara_mazorca, rasgos, reducir, vector


def imagen(m):
    bgr = cv2.imread(str(AQUI / "datos/images" / m["split"] / f"{m['id']}.jpg"))
    return reducir(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def mascara(m, rgb):
    f = AQUI / "datos/mask" / f"{m['id']}.png"
    if f.exists():
        return (cv2.imread(str(f), 0) > 0).astype(np.uint8)
    x = mascara_mazorca(rgb)
    cv2.imwrite(str(f), x * 255)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pesos", default=str(AQUI / "runs/ajuste/weights/best.pt"))
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--min-descarte", type=float, default=0.95)
    p.add_argument("--salida", default=str(AQUI / "modelo_final_v4"))
    a = p.parse_args()
    from ultralytics import YOLO
    y = YOLO(a.pesos)
    meta = json.load(open(AQUI / "datos/meta.json"))
    cache = AQUI / "datos/dets_ajuste.json"
    dets = json.load(open(cache)) if cache.exists() else {}
    for i, m in enumerate(meta):
        if m["id"] in dets:
            continue
        r = y.predict(str(AQUI / "datos/images" / m["split"] / f"{m['id']}.jpg"), imgsz=640, conf=a.conf, iou=a.iou, device="cpu", verbose=False)[0]
        dets[m["id"]] = [[int(c), *xywhn, float(q)] for c, xywhn, q in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
        if i % 100 == 0:
            print(f"detecciones {i}/{len(meta)}", flush=True)
    json.dump(dets, open(cache, "w"))
    X, Xh, y_, split, grupo, ids = [], [], [], [], [], []
    for m in meta:
        rgb = imagen(m)
        k = mascara(m, rgb)
        X.append(vector(rasgos(rgb, k, dets[m["id"]])))
        Xh.append(vector(rasgos(rgb, k, m["cajas"])))
        y_.append(int(m["ref"] == "descarte"))
        split.append(m["split"])
        grupo.append(m["grupo"])
        ids.append(m["n"])
    X, Xh, y_, split, grupo = np.array(X), np.array(Xh), np.array(y_), np.array(split), np.array(grupo)
    tr, te = split != "test", split == "test"

    def ajustar(Xa, ya):
        esc = StandardScaler().fit(Xa)
        return esc, LogisticRegression(max_iter=3000, class_weight="balanced").fit(esc.transform(Xa), ya)

    def prob(esc, clf, Xa):
        return clf.predict_proba(esc.transform(Xa))[:, 1]

    def medir(pred, yv):
        return (float((pred[yv == 0] == 0).mean()), float((pred[yv == 1] == 1).mean()))

    # A: como estaba (ajustada con cajas humanas de entrenamiento), umbral elegido en validación
    escA, clfA = ajustar(Xh[split == "train"], y_[split == "train"])
    pv = prob(escA, clfA, X[split == "val"])
    yv = y_[split == "val"]
    cand = []
    for t in np.arange(0.05, 0.96, 0.05):
        pa, da = medir((pv >= t).astype(int), yv)
        cand.append((t, pa, da))
    ok = [c for c in cand if c[2] >= a.min_descarte] or cand
    tA = max(ok, key=lambda c: (c[1], -c[0]))[0]
    pA = medir((prob(escA, clfA, X[te]) >= tA).astype(int), y_[te])
    # B: ajustada con las cajas del detector en entrenamiento y validación; umbral por validación cruzada por grupos
    oof = np.zeros(tr.sum())
    Xt, yt, gt = X[tr], y_[tr], grupo[tr]
    for i, (a_, b_) in enumerate(GroupKFold(5).split(Xt, yt, gt)):
        e, c = ajustar(Xt[a_], yt[a_])
        oof[b_] = prob(e, c, Xt[b_])
    cand = []
    for t in np.arange(0.05, 0.96, 0.05):
        pa, da = medir((oof >= t).astype(int), yt)
        cand.append((t, pa, da))
    ok = [c for c in cand if c[2] >= a.min_descarte] or cand
    tB = max(ok, key=lambda c: (c[1], -c[0]))[0]
    escB, clfB = ajustar(Xt, yt)
    pB_prob = prob(escB, clfB, X[te])
    pB = medir((pB_prob >= tB).astype(int), y_[te])
    print(f"\nPrueba ({te.sum()} fotos: {int((y_[te] == 0).sum())} Primera, {int((y_[te] == 1).sum())} Descarte)")
    print(f"A  cajas humanas, umbral {tA:.2f}: Primera {pA[0]:.1%} | Descarte {pA[1]:.1%}")
    print(f"B  cajas del detector, umbral {tB:.2f}: Primera {pB[0]:.1%} | Descarte {pB[1]:.1%}   (validación cruzada: Primera {max(ok, key=lambda c: (c[1], -c[0]))[1]:.1%}, Descarte {max(ok, key=lambda c: (c[1], -c[0]))[2]:.1%})")
    err = [(ids[i], int(y_[i]), round(float(p), 2)) for i, p in zip(np.where(te)[0], pB_prob) if int(p >= tB) != y_[i]]
    print("errores de B:", [(n.replace("primera_clase__", "P ").replace("descarte__", "D ").replace(".jpg", ""), "Descarte" if r else "Primera", p) for n, r, p in err])
    print("coeficientes de B:", {k: round(float(c), 2) for k, c in zip(ORDEN, clfB.coef_[0])})
    out = Path(a.salida)
    out.mkdir(exist_ok=True)
    json.dump({"clases": list(CLASES), "orden_rasgos": ORDEN, "media": escB.mean_.tolist(), "escala": escB.scale_.tolist(), "coef": clfB.coef_[0].tolist(),
               "intercepto": float(clfB.intercept_[0]), "confianza_minima": a.conf, "umbral_descarte": round(float(tB), 2), "zonas_para_segunda": 2, "imgsz": 640, "iou_nms": a.iou,
               "nota": "Decisión ajustada con las cajas que produce el detector en entrenamiento y validación."}, open(out / "decision.json", "w"), indent=1, ensure_ascii=False)
    for f in ("best.pt", "best.onnx", "results.png", "confusion_matrix.png"):
        if (AQUI / "modelo_final_v3" / f).exists():
            shutil.copy2(AQUI / "modelo_final_v3" / f, out / f)
    json.dump({"A_cajas_humanas": {"umbral": float(tA), "primera_ok": pA[0], "descarte_ok": pA[1]}, "B_cajas_detector": {"umbral": float(tB), "primera_ok": pB[0], "descarte_ok": pB[1]}, "errores_B": err}, open(out / "resultado_decision.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
