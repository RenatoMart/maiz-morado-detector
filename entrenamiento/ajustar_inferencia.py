"""Prueba parámetros de inferencia sobre un modelo ya entrenado (sin reentrenar) y mide el mAP.

Uso: python ajustar_inferencia.py --pesos runs/final/weights/best.pt --split val
"""
import argparse
import json
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pesos", default=str(AQUI / "runs/final/weights/best.pt"))
    p.add_argument("--split", default="val")
    p.add_argument("--hilos", type=int, default=3)
    p.add_argument("--salida", default=str(AQUI / "logs/ajuste_inferencia.json"))
    p.add_argument("--configs", help="JSON con una lista de configuraciones; por defecto se prueba una barrida.")
    a = p.parse_args()
    torch.set_num_threads(a.hilos)
    from ultralytics import YOLO
    y = YOLO(a.pesos)
    base = dict(imgsz=640, iou=0.7)
    barrida = [("base", {}), ("iou 0.6", dict(iou=0.6)), ("iou 0.5", dict(iou=0.5)), ("iou 0.4", dict(iou=0.4)),
               ("imgsz 512", dict(imgsz=512)), ("imgsz 576", dict(imgsz=576)), ("imgsz 704", dict(imgsz=704)), ("imgsz 768", dict(imgsz=768)),
               ("sin separar por clase al unir cajas", dict(agnostic_nms=True)), ("TTA (giros y escalas)", dict(augment=True))]
    cfgs = json.loads(a.configs) if a.configs else barrida
    R = []
    for nombre, extra in cfgs:
        t = time.time()
        k = {**base, **extra}
        v = y.val(data=str(AQUI / "datos/data.yaml"), split=a.split, batch=8, device="cpu", workers=0, plots=False, verbose=False, project=str(AQUI / "runs/tmp_val"), name="x", exist_ok=True, **k)
        f = {"config": nombre, "params": k, "map50": round(float(v.box.map50), 4), "map50_95": round(float(v.box.map), 4), "precision": round(float(v.box.mp), 3), "recall": round(float(v.box.mr), 3), "seg": round(time.time() - t)}
        R.append(f)
        json.dump(R, open(a.salida, "w"), indent=1, ensure_ascii=False)
        print(f"{nombre:38s} mAP50 {f['map50']:.3f}  mAP50-95 {f['map50_95']:.3f}  P {f['precision']:.3f}  R {f['recall']:.3f}  ({f['seg']} s)", flush=True)


if __name__ == "__main__":
    main()
