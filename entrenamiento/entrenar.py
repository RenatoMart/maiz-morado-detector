"""Entrena YOLO11 con el dataset de entrenamiento/datos. Si el vigía lo reinicia, retoma desde last.pt."""
import argparse
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nombre", default="final")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--modelo", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--partir-de", help="Pesos desde los que continuar (ajuste fino), por ejemplo runs/final/weights/best.pt.")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--nuevo", action="store_true", help="Ignora el avance guardado y empieza de cero.")
    a = p.parse_args()

    from ultralytics import YOLO

    ultimo = AQUI / "runs" / a.nombre / "weights" / "last.pt"
    mejor = AQUI / "runs" / a.nombre / "weights" / "best.pt"
    # Si ya existe un avance guardado se retoma, tanto tras un reinicio del vigía como tras volver a
    # lanzar todo (por ejemplo, si se apagó la laptop). Con --nuevo se empieza de cero.
    if ultimo.exists() and not a.nuevo:
        print(f"Reanudando desde {ultimo}", flush=True)
        try:
            YOLO(str(ultimo)).train(resume=True)
        except Exception as e:
            if "finished" in str(e).lower() and mejor.exists():
                print("El entrenamiento ya había terminado.", flush=True)
                return
            raise
        return

    ajuste = bool(a.partir_de)
    YOLO(a.partir_de or a.modelo).train(
        data=str(AQUI / "datos" / "data.yaml"), project=str(AQUI / "runs"), name=a.nombre, exist_ok=True,
        epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device="cpu",
        seed=a.semilla, optimizer="AdamW", lr0=a.lr0, cos_lr=True, fraction=a.fraction, warmup_epochs=1 if ajuste else 3,
        hsv_h=0.005, hsv_s=0.3, hsv_v=0.3, degrees=10, fliplr=0.5, flipud=0.5, mosaic=0.5, close_mosaic=5 if ajuste else 10,
        cache="ram", plots=True, verbose=False,
    )


if __name__ == "__main__":
    sys.exit(main())
