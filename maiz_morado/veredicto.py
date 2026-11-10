"""Veredicto por proporciones: la lógica que usará también el sistema web.

Se alimenta con las cajas del detector y la mazorca segmentada, y con el archivo decision.json
que genera entrenamiento/pos.py. No depende de YOLO.
"""
import json
import math

import numpy as np

from .medidas import CLASES, ORDEN, rasgos, vector

NOMBRES = {"n_cajas": "cantidad de zonas marcadas", "area_total": "proporción de la mazorca con defectos", "a_med": "rojez general de la mazorca",
           "rojo15": "granos rojizos", "rojo25": "granos muy rojos", "a0": "área de granos quebrados", "a1": "área de pérdida extensa",
           "a2": "área de pigmentación roja", "a3": "área de pudrición", "a4": "área sin granos", "n0": "granos quebrados", "n1": "pérdida extensa",
           "n2": "zonas rojas", "n3": "zonas de pudrición", "n4": "zonas sin granos"}


class Decision:
    def __init__(self, ruta):
        d = json.load(open(ruta))
        self.orden, self.confianza, self.umbral, self.zonas_segunda = d["orden_rasgos"], d["confianza_minima"], d["umbral_descarte"], d["zonas_para_segunda"]
        self.media, self.escala, self.coef = np.array(d["media"]), np.array(d["escala"]), np.array(d["coef"])
        self.intercepto = d["intercepto"]
        self.iou = d.get("iou_nms", 0.7)
        assert self.orden == ORDEN, "El orden de los rasgos no coincide con el entrenamiento"

    def probabilidad(self, r):
        z = (np.array(vector(r)) - self.media) / self.escala
        return 1 / (1 + math.exp(-float(self.coef @ z + self.intercepto))), z * self.coef


def _texto(clave, valor):
    if clave.startswith("a") and clave != "a_med" or clave in ("area_total", "rojo15", "rojo25"):
        return f"{valor * 100:.1f} %"
    return f"{valor:.0f}" if clave.startswith("n") else f"{valor:.1f}"


def evaluar_imagen(rgb, detecciones, decision, mascara):
    """detecciones = [(clase, xc, yc, ancho, alto, confianza)]. Devuelve el veredicto y su explicación."""
    cajas = [d for d in detecciones if d[5] >= decision.confianza]
    r = rasgos(rgb, mascara, cajas)
    p, aportes = decision.probabilidad(r)
    if p >= decision.umbral:
        categoria = "Descarte"
    elif r["n4"] >= decision.zonas_segunda:
        categoria = "Segunda"
    else:
        categoria = "Primera"
    orden = np.argsort(-aportes)
    motivos = [f"{NOMBRES[ORDEN[i]]}: {_texto(ORDEN[i], r[ORDEN[i]])}" for i in orden[:3] if aportes[i] > 0.3] if categoria == "Descarte" else []
    return {"categoria": categoria, "pasa": categoria != "Descarte", "p_descarte": p, "rasgos": r, "motivos": motivos, "cajas": cajas,
            "porcentaje_por_clase": {CLASES[i]: round(r[f"a{i}"] * 100, 1) for i in range(len(CLASES))}}
