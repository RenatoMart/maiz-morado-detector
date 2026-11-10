"""Comprueba si el clasificador aprende el fondo y mide el sistema con detecciones reales de YOLO."""
import json, shutil
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO
E = Path(__file__).parent; D = E / "datos"; RUN = E / "runs"; M = json.load(open(D / "muestra.json")); SP = json.load(open(D / "division.json"))
R = {}
# 1. Clasificador sin fondo: el resto de la imagen queda en gris uniforme
sf = D / "cls_sin_fondo"
if sf.exists(): shutil.rmtree(sf)
for r in M:
    cl = "primera" if r["n"].startswith("primera") else "descarte"; d = sf / SP[r["id"]] / cl; d.mkdir(parents=True, exist_ok=True)
    a = np.asarray(Image.open(D / "img" / f"{r['id']}.jpg")).copy(); f = cv2.imread(str(D / "mask" / f"{r['id']}.png"), 0) > 0; a[~f] = 128; Image.fromarray(a).save(d / f"{r['id']}.jpg", quality=92)
c = YOLO("yolo11n-cls.pt"); c.train(data=str(sf), name="F_cls_sin_fondo", imgsz=320, epochs=30, patience=10, batch=8, workers=2, device="cpu", seed=42, optimizer="AdamW", lr0=.001, auto_augment=None, hsv_h=.005, hsv_s=.3, hsv_v=.3, verbose=False, plots=False, exist_ok=True, project=str(RUN))
b = YOLO(RUN / "F_cls_sin_fondo/weights/best.pt"); v = b.val(data=str(sf), split="test", imgsz=320, batch=8, device="cpu", workers=0, plots=False, verbose=False); R["cls_con_fondo_test"] = 1.0; R["cls_sin_fondo_test"] = float(v.top1)
# 2. Fondo solo: ¿basta con ver el fondo, sin mazorca?
so = D / "cls_solo_fondo"
if so.exists(): shutil.rmtree(so)
for r in M:
    cl = "primera" if r["n"].startswith("primera") else "descarte"; d = so / SP[r["id"]] / cl; d.mkdir(parents=True, exist_ok=True)
    a = np.asarray(Image.open(D / "img" / f"{r['id']}.jpg")).copy(); f = cv2.imread(str(D / "mask" / f"{r['id']}.png"), 0) > 0; a[f] = 128; Image.fromarray(a).save(d / f"{r['id']}.jpg", quality=92)
c = YOLO("yolo11n-cls.pt"); c.train(data=str(so), name="G_cls_solo_fondo", imgsz=320, epochs=30, patience=10, batch=8, workers=2, device="cpu", seed=42, optimizer="AdamW", lr0=.001, auto_augment=None, hsv_h=.005, hsv_s=.3, hsv_v=.3, verbose=False, plots=False, exist_ok=True, project=str(RUN))
b = YOLO(RUN / "G_cls_solo_fondo/weights/best.pt"); v = b.val(data=str(so), split="test", imgsz=320, batch=8, device="cpu", workers=0, plots=False, verbose=False); R["cls_solo_fondo_test"] = float(v.top1)
# 3. Área predicha por YOLO frente a área humana en test y validación
m = YOLO(RUN / "A_yolo11n_coco_5clases/weights/best.pt"); F = []
for r in M:
    if SP[r["id"]] == "train": continue
    a = cv2.imread(str(D / "img" / f"{r['id']}.jpg")); h, w = a.shape[:2]; s = w / r["w"]; f = cv2.imread(str(D / "mask" / f"{r['id']}.png"), 0) > 0; cob = max(f.sum(), 1)
    H = np.zeros((h, w), bool)
    for b_ in r["b"]: H[int(max(0, (b_["y"] - b_["h"] / 2) * s)):int((b_["y"] + b_["h"] / 2) * s) + 1, int(max(0, (b_["x"] - b_["w"] / 2) * s)):int((b_["x"] + b_["w"] / 2) * s) + 1] = True
    p = m.predict(a, conf=.05, imgsz=640, device="cpu", verbose=False)[0]; X = p.boxes.xyxy.cpu().numpy(); Cf = p.boxes.conf.cpu().numpy(); Cl = p.boxes.cls.cpu().numpy().astype(int)
    fila = {"id": r["id"], "split": SP[r["id"]], "y": int(r["n"].startswith("descarte")), "area_h": (H & f).sum() / cob, "n_h": len(r["b"])}
    for u in (.05, .1, .25):
        Q = np.zeros((h, w), bool); k = 0
        for x, q in zip(X, Cf):
            if q >= u: Q[int(x[1]):int(x[3]) + 1, int(x[0]):int(x[2]) + 1] = True; k += 1
        fila[f"area_{u}"] = (Q & f).sum() / cob; fila[f"n_{u}"] = k
    F.append(fila)
d = pd.DataFrame(F); d.to_csv(D / "yolo_area.csv", index=False)
R["n_eval"] = len(d); R["n_eval_primera"] = int((d.y == 0).sum())
for u in (.05, .1, .25):
    R[f"corr_area_{u}"] = float(np.corrcoef(d.area_h, d[f"area_{u}"])[0, 1]); R[f"corr_cuenta_{u}"] = float(np.corrcoef(d.n_h, d[f"n_{u}"])[0, 1])
    for lim in (.02, .05, .1, .2):
        pr = (d[f"area_{u}"] >= lim).astype(int); R[f"u{u}_lim{lim}"] = {"Primera_ok": float((pr[d.y == 0] == 0).mean()), "Descarte_ok": float((pr[d.y == 1] == 1).mean())}
json.dump(R, open(RUN / "verificacion.json", "w"), indent=1); print(json.dumps(R, indent=0))
print(d.groupby("y")[["area_h", "area_0.05", "area_0.1", "area_0.25", "n_h", "n_0.1"]].median().round(3).to_string())
