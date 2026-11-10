"""Ejecuta los dos modelos sobre una imagen y dibuja el resultado."""
from dataclasses import dataclass

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

from .comun import normalizar_clase
from .decision import DESCARTE, PRIMERA, SEGUNDA, Decision, decidir

COLOR_GRADO = {PRIMERA: "#2DC653", SEGUNDA: "#F2B705", DESCARTE: "#E5383B"}
COLOR_REVISAR = "#3A86FF"
ANCHO_PANEL = 560
ALTO_MAXIMO = 1400


@dataclass
class Deteccion:
    clase: str
    confianza: float
    caja: tuple  # x1, y1, x2, y2 en píxeles de la imagen original


def cargar_imagen(ruta):
    """Abre la foto respetando la orientación EXIF de los teléfonos."""
    with Image.open(ruta) as imagen:
        return ImageOps.exif_transpose(imagen).convert("RGB")


def a_bgr(imagen):
    return np.array(imagen)[:, :, ::-1].copy()


class Sistema:
    def __init__(self, config, ruta_modelo2, ruta_modelo1=None, device=None):
        self.config = config
        self.device = device
        self.modelo2 = YOLO(str(ruta_modelo2))
        self.modelo1 = YOLO(str(ruta_modelo1)) if ruta_modelo1 else None

    def detectar(self, imagen, conf=None):
        conf = self.config["fusion"]["conf_defectos"] if conf is None else conf
        resultado = self.modelo2.predict(
            a_bgr(imagen), conf=conf, device=self.device, verbose=False
        )[0]
        return [
            Deteccion(
                normalizar_clase(resultado.names[int(c)], self.config),
                float(p),
                tuple(float(v) for v in xyxy),
            )
            for xyxy, p, c in zip(
                resultado.boxes.xyxy.tolist(),
                resultado.boxes.conf.tolist(),
                resultado.boxes.cls.tolist(),
            )
        ]

    def probabilidad_sana(self, imagen):
        if self.modelo1 is None:
            return None
        resultado = self.modelo1.predict(a_bgr(imagen), device=self.device, verbose=False)[0]
        nombre_sana = self.config["modelo1"]["nombre_clase_sana"]
        indice = next(i for i, n in resultado.names.items() if n == nombre_sana)
        return float(resultado.probs.data[indice])

    def analizar(self, imagen):
        detecciones = self.detectar(imagen)
        p_sana = self.probabilidad_sana(imagen)
        decision = decidir([d.clase for d in detecciones], p_sana, self.config)
        return detecciones, p_sana, decision


def _fuente(tamano):
    # DejaVu viene con matplotlib, instalado por Ultralytics, e incluye tildes y "ñ".
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), tamano)


def _partir(dibujo, texto, fuente, ancho):
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if dibujo.textlength(prueba, font=fuente) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    return lineas + [actual] if actual else lineas


def dibujar(imagen, detecciones, p_sana, decision: Decision, config):
    """Devuelve la foto con las zonas marcadas y un panel que las explica."""
    catalogo = config["clases"]
    escala = min(1.0, ALTO_MAXIMO / imagen.height)
    foto = imagen.resize((round(imagen.width * escala), round(imagen.height * escala)))
    alto = max(foto.height, 900)
    lienzo = Image.new("RGB", (foto.width + ANCHO_PANEL, alto), "#14161A")
    lienzo.paste(foto, (0, 0))
    dibujo = ImageDraw.Draw(lienzo)
    fuente_caja = _fuente(20)

    numerada = sorted(detecciones, key=lambda d: (d.clase, -d.confianza))
    for numero, det in enumerate(numerada, 1):
        color = catalogo.get(det.clase, {}).get("color", "#FFFFFF")
        caja = [v * escala for v in det.caja]
        dibujo.rectangle(caja, outline=color, width=4)
        etiqueta = f"{numero}"
        x, y = caja[0], max(0, caja[1] - 28)
        dibujo.rectangle([x, y, x + 12 + dibujo.textlength(etiqueta, font=fuente_caja), y + 28], fill=color)
        dibujo.text((x + 6, y + 2), etiqueta, fill="#000000", font=fuente_caja)

    x = foto.width + 28
    ancho = ANCHO_PANEL - 56
    y = 28
    color = COLOR_GRADO.get(decision.grado, COLOR_REVISAR)
    dibujo.rectangle([x, y, x + ancho, y + 120], fill=color)
    dibujo.text((x + 18, y + 14), decision.accion, fill="#000000", font=_fuente(40))
    dibujo.text((x + 18, y + 72), f"Grado: {decision.grado}", fill="#000000", font=_fuente(28))
    y += 144

    def escribir(texto, fuente, relleno="#E8EAED", separacion=8):
        nonlocal y
        for linea in _partir(dibujo, texto, fuente, ancho):
            dibujo.text((x, y), linea, fill=relleno, font=fuente)
            y += fuente.size + 6
        y += separacion

    for motivo in decision.motivos:
        escribir(motivo, _fuente(24))
    for aviso in decision.avisos:
        escribir(f"Aviso: {aviso}", _fuente(22), "#F2B705")
    if p_sana is not None:
        escribir(f"Modelo 1, probabilidad de mazorca sana: {p_sana:.0%}", _fuente(20), "#AEB4BD")

    y += 10
    if numerada:
        escribir(f"Zonas detectadas: {len(numerada)}", _fuente(28), separacion=12)
    for numero, det in enumerate(numerada, 1):
        info = catalogo.get(det.clase, {"nombre": det.clase, "significado": "", "color": "#FFFFFF"})
        escribir(f"{numero}. {info['nombre']} ({det.confianza:.0%})", _fuente(23), info["color"], 0)
        escribir(info["significado"], _fuente(19), "#C9CDD3", 12)
        if y > alto - 110 and numero < len(numerada):
            escribir(f"… y {len(numerada) - numero} zona(s) más en la foto.", _fuente(21))
            break

    aviso_final = "Apoyo a la inspección: la decisión final es de la persona."
    dibujo.text((x, alto - 42), aviso_final, fill="#AEB4BD", font=_fuente(18))
    return lienzo
