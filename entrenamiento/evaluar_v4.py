"""Métricas finales de la versión 4: detector, ablación, acierto por fondo y errores. Usa solo cachés y la prueba una vez."""
import csv, json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
from maiz_morado.medidas import ORDEN, rasgos, reducir, vector
from recalibrar_decision import imagen, mascara

meta = json.load(open(AQUI / "datos/meta.json")); dets = json.load(open(AQUI / "datos/dets_ajuste.json"))
R = {}
from ultralytics import YOLO
y = YOLO(str(AQUI / "runs/ajuste/weights/best.pt"))
R["detector"] = {}
for s in ("val", "test"):
    v = y.val(data=str(AQUI / "datos/data.yaml"), split=s, imgsz=640, iou=0.5, batch=8, device="cpu", workers=0, plots=False, verbose=False)
    R["detector"][s] = {"imagenes": sum(m["split"] == s for m in meta), "map50": round(float(v.box.map50), 4), "map50_95": round(float(v.box.map), 4), "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
                        "ap50": {n: round(float(a), 3) for n, a in zip(["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"], [0] * 5)}}
    ap = {int(c): float(a) for c, a in zip(v.box.ap_class_index, v.box.ap50)}
    R["detector"][s]["ap50"] = {n: round(ap.get(i, 0.0), 3) for i, n in enumerate(["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"])}
X = []; Y = []; S = []; G = []; N = []
for m in meta:
    rgb = imagen(m); k = mascara(m, rgb)
    X.append(vector(rasgos(rgb, k, dets[m["id"]]))); Y.append(int(m["ref"] == "descarte")); S.append(m["split"]); G.append(m["grupo"]); N.append(m["n"])
X, Y, S, G = np.array(X), np.array(Y), np.array(S), np.array(G); tr, te = S != "test", S == "test"

def correr(cols, min_d=0.95):
    idx = [ORDEN.index(c) for c in cols]; Xt, yt, gt = X[tr][:, idx], Y[tr], G[tr]; oof = np.zeros(len(yt))
    for a, b in GroupKFold(5).split(Xt, yt, gt):
        e = StandardScaler().fit(Xt[a]); c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(e.transform(Xt[a]), yt[a]); oof[b] = c.predict_proba(e.transform(Xt[b]))[:, 1]
    cand = [(t, float(((oof >= t) == 0)[yt == 0].mean()), float(((oof >= t) == 1)[yt == 1].mean())) for t in np.arange(0.05, 0.96, 0.05)]
    ok = [c for c in cand if c[2] >= min_d] or cand; t = max(ok, key=lambda c: (c[1], -c[0]))[0]
    e = StandardScaler().fit(Xt); c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(e.transform(Xt), yt); p = c.predict_proba(e.transform(X[te][:, idx]))[:, 1]
    pred = (p >= t).astype(int); yy = Y[te]
    return {"umbral": round(float(t), 2), "primera_ok": int((pred[yy == 0] == 0).sum()), "primera": int((yy == 0).sum()), "descarte_ok": int((pred[yy == 1] == 1).sum()), "descarte": int((yy == 1).sum())}, p, t

todas = list(ORDEN)
R["ablacion"] = {n: correr(c)[0] for n, c in {"completo": todas, "sin color": [c for c in todas if c not in ("a_med", "rojo15", "rojo25")], "solo areas": ["area_total", "a0", "a1", "a2", "a3", "a4"], "solo conteos": ["n_cajas", "n0", "n1", "n2", "n3", "n4"], "solo color": ["a_med", "rojo15", "rojo25"]}.items()}
# conteo simple con las cajas del detector a confianza 0.25 como comparación
base = [int(any(d[5] >= 0.25 for d in dets[m["id"]])) for m in meta]; b = np.array(base)[te]; yy = Y[te]
R["regla_contar_cajas_0.25"] = {"primera_ok": int((b[yy == 0] == 0).sum()), "descarte_ok": int((b[yy == 1] == 1).sum())}
_, p, t = correr(todas); pred = (p >= t).astype(int)
def fondo(m):
    im = np.asarray(Image.open(AQUI / f"datos/images/{m['split']}/{m['id']}.jpg").convert("RGB").resize((64, 64)), float)
    bb = np.concatenate([im[:6].reshape(-1, 3), im[-6:].reshape(-1, 3), im[:, :6].reshape(-1, 3), im[:, -6:].reshape(-1, 3)]).mean(0); r, g, bl = bb; sat = (bb.max() - bb.min()) / max(bb.max(), 1)
    return "césped" if g > r + 8 and g > bl + 25 else "tela blanca o gris" if sat < 0.12 else "malla o yute"
ms = [m for m in meta if m["split"] == "test"]; fon = {}
for m, pr in zip(ms, pred):
    k = fon.setdefault(fondo(m), [0, 0, 0, 0])
    if m["ref"] == "primera": k[1] += 1; k[0] += int(pr == 0)
    else: k[3] += 1; k[2] += int(pr == 1)
R["por_fondo_prueba"] = fon
tot = {}
for m in meta: tot.setdefault(fondo(m), {"primera": 0, "descarte": 0})[m["ref"]] += 1
R["fondo_todo_el_conjunto"] = tot
R["errores"] = [(m["n"], m["ref"], round(float(q), 2)) for m, pr, q in zip(ms, pred, p) if int(pr) != int(m["ref"] == "descarte")]
R["conteos"] = {"imagenes": len(meta), "grupos": len(set(G.tolist())), "primera": sum(m["ref"] == "primera" for m in meta), "descarte": sum(m["ref"] == "descarte" for m in meta), "primera_sin_cajas": sum(m["ref"] == "primera" and not m["cajas"] for m in meta),
               "split": {s: sum(m["split"] == s for m in meta) for s in ("train", "val", "test")}, "cajas": {s: sum(len(m["cajas"]) for m in meta if m["split"] == s) for s in ("train", "val", "test")}}
json.dump(R, open(AQUI / "modelo_final_v4/metricas.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(R, indent=1, ensure_ascii=False))
