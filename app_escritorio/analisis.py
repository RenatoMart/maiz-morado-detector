"""Análisis de una foto con el modelo entrenado (el mismo resultado que muestra la web).

Usa el detector YOLO (best.pt) y la decisión por proporciones de `maiz_morado`.
"""
from pathlib import Path

import cv2
import numpy as np

from maiz_morado.medidas import CLASES, MENSAJE_SIN_MAZORCA, hay_mazorca, mascara_mazorca, reducir
from maiz_morado.veredicto import Decision, evaluar_imagen

AQUI = Path(__file__).resolve().parent
MODELO = AQUI.parent / "entrenamiento" / "modelo_final_v4"  # best.pt y decision.json del modelo final


class Analizador:
    """Carga el modelo una sola vez y analiza fotos."""

    def __init__(self, carpeta_modelo: Path = MODELO):
        from ultralytics import YOLO  # se importa aquí porque tarda unos segundos

        self.decision = Decision(carpeta_modelo / "decision.json")
        self.yolo = YOLO(str(carpeta_modelo / "best.pt"))

    def analizar(self, ruta: str) -> dict:
        bgr = cv2.imread(ruta)
        if bgr is None:
            raise ValueError("No se pudo leer la imagen. Usa un archivo JPG o PNG.")
        rgb = reducir(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not hay_mazorca(rgb)[0]:
            return {"sin_mazorca": True, "mensaje": MENSAJE_SIN_MAZORCA}
        r = self.yolo.predict(ruta, imgsz=640, conf=self.decision.confianza, iou=self.decision.iou, device="cpu", verbose=False)[0]
        dets = [(int(c), *xywhn, float(p)) for c, xywhn, p in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
        mascara = mascara_mazorca(rgb)
        v = evaluar_imagen(rgb, dets, self.decision, mascara)
        return {
            "categoria": v["categoria"],
            "p_descarte": float(v["p_descarte"]),
            "motivos": v["motivos"],
            "area_defectos_pct": round(float(v["rasgos"]["area_total"]) * 100, 1),
            "mazorca_pct_imagen": round(float(np.mean(mascara)) * 100, 1),
            "porcentaje_por_clase": v["porcentaje_por_clase"],
            "zonas": [{"clase": CLASES[int(c)], "x": xc - w / 2, "y": yc - h / 2, "ancho": w, "alto": h, "confianza": float(p)} for c, xc, yc, w, h, p in v["cajas"]],
        }
