"""Piloto en CPU con la misma división: compara pesos, versión, clases y clasificación directa."""
import json, shutil, time, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml, cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
EST = Path(__file__).parent; D = EST / "datos"; RUN = EST / "runs"; RUN.mkdir(exist_ok=True)
M = json.load(open(D / "muestra.json")); C = ["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"]
ref = ["P" if r["n"].startswith("primera") else "D" for r in M]; idx = np.arange(len(M))
tr, re = train_test_split(idx, test_size=.3, random_state=42, stratify=ref)
va, te = train_test_split(re, test_size=.5, random_state=42, stratify=[ref[i] for i in re])
SP = {**{int(i): "train" for i in tr}, **{int(i): "val" for i in va}, **{int(i): "test" for i in te}}
json.dump({M[i]["id"]: s for i, s in SP.items()}, open(D / "division.json", "w"))
print({s: sum(v == s for v in SP.values()) for s in ("train", "val", "test")}, "Primera en test", sum(ref[i] == "P" for i in te), flush=True)
def arbol(nombre, una):
    b = D / nombre
    if b.exists(): shutil.rmtree(b)
    for i, r in enumerate(M):
        s = SP[i]; (b / "images" / s).mkdir(parents=True, exist_ok=True); (b / "labels" / s).mkdir(parents=True, exist_ok=True)
        (b / "images" / s / f"{r['id']}.jpg").hardlink_to(D / "img" / f"{r['id']}.jpg")
        (b / "labels" / s / f"{r['id']}.txt").write_text("".join(f"{0 if una else C.index(x['l'])} {x['x']/r['w']:.6f} {x['y']/r['h']:.6f} {x['w']/r['w']:.6f} {x['h']/r['h']:.6f}\n" for x in r["b"]))
    y = b / "data.yaml"; y.write_text(yaml.safe_dump({"path": str(b.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "defecto"} if una else dict(enumerate(C))}, allow_unicode=True)); return y
y5, y1 = arbol("yolo5", False), arbol("yolo1", True)
cl = D / "cls"
if cl.exists(): shutil.rmtree(cl)
for i, r in enumerate(M):
    d = cl / SP[i] / ("primera" if ref[i] == "P" else "descarte"); d.mkdir(parents=True, exist_ok=True); (d / f"{r['id']}.jpg").hardlink_to(D / "img" / f"{r['id']}.jpg")
COMUN = dict(imgsz=640, epochs=40, patience=15, batch=8, workers=2, device="cpu", seed=42, cache="ram", optimizer="AdamW", lr0=.001, cos_lr=True, hsv_h=.005, hsv_s=.3, hsv_v=.3, degrees=10, fliplr=.5, flipud=.5, mosaic=.5, close_mosaic=10, verbose=False, plots=False, exist_ok=True, project=str(RUN))
EXP = [("A_yolo11n_coco_5clases", "yolo11n.pt", y5), ("B_yolo11n_desde_cero_5clases", "yolo11n.yaml", y5), ("C_yolo26n_coco_5clases", "yolo26n.pt", y5), ("D_yolo11n_coco_1clase", "yolo11n.pt", y1)]
res = []
for nom, peso, y in EXP:
    t = time.time(); m = YOLO(peso); m.train(data=str(y), name=nom, pretrained=peso.endswith(".pt"), **COMUN); mejor = YOLO(RUN / nom / "weights/best.pt")
    v = mejor.val(data=str(y), split="test", imgsz=640, batch=8, device="cpu", workers=0, plots=False, verbose=False)
    f = {"exp": nom, "min": round((time.time() - t) / 60, 1), "map50": float(v.box.map50), "map": float(v.box.map), "p": float(v.box.mp), "r": float(v.box.mr), "por_clase": {mejor.names[int(c)]: round(float(a), 3) for c, a in zip(v.box.ap_class_index, v.box.ap50)}}
    res.append(f); print(json.dumps(f, ensure_ascii=False), flush=True); json.dump(res, open(RUN / "detectores.json", "w"), indent=1, ensure_ascii=False)
t = time.time(); c = YOLO("yolo11n-cls.pt"); c.train(data=str(cl), name="E_yolo11n_cls", imgsz=320, epochs=30, patience=10, batch=8, workers=2, device="cpu", seed=42, optimizer="AdamW", lr0=.001, auto_augment=None, hsv_h=.005, hsv_s=.3, hsv_v=.3, verbose=False, plots=False, exist_ok=True, project=str(RUN))
b = YOLO(RUN / "E_yolo11n_cls/weights/best.pt"); v = b.val(data=str(cl), split="test", imgsz=320, batch=8, device="cpu", workers=0, plots=False, verbose=False)
print(json.dumps({"exp": "E_yolo11n_cls", "min": round((time.time() - t) / 60, 1), "top1_test": float(v.top1)}), flush=True); print("FIN", flush=True)
