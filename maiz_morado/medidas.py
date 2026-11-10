"""Mide qué proporción de la mazorca ocupa cada defecto y qué tan rojiza es.

Entradas en coordenadas normalizadas (0 a 1): cajas = [(clase, xc, yc, ancho, alto), ...].
Las cinco clases siguen el orden de CLASES, el mismo del entrenamiento.
"""
import cv2
import numpy as np

CLASES = ("granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos")
ORDEN = ["n_cajas", "n0", "n1", "n2", "n3", "n4", "area_total", "a0", "a1", "a2", "a3", "a4", "a_med", "rojo15", "rojo25"]


def reducir(rgb, lado=800):
    h, w = rgb.shape[:2]
    e = lado / max(h, w)
    return cv2.resize(rgb, (round(w * e), round(h * e)), interpolation=cv2.INTER_AREA) if e < 1 else rgb


def _lab(rgb):
    x = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return x[..., 0] * 100 / 255, x[..., 1] - 128


def mascara_mazorca(rgb):
    """Contorno de la mazorca con GrabCut, sin entrenar nada. Devuelve 1 dentro y 0 fuera."""
    h, w = rgb.shape[:2]
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    L, _ = _lab(rgb)
    m[L < np.percentile(L, 35)] = cv2.GC_PR_FGD
    k = max(4, int(0.02 * min(h, w)))
    m[:k] = m[-k:] = cv2.GC_BGD
    m[:, :k] = m[:, -k:] = cv2.GC_BGD
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
        f = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        f = (L < np.percentile(L, 35)).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, et, st, _ = cv2.connectedComponentsWithStats(f)
    if n > 1:
        f = (et == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if f.mean() < 0.01:  # recorte fallido: se usa la zona oscura completa
        f = (L < np.percentile(L, 35)).astype(np.uint8)
    return f


def _union(cajas, h, w, clase=None):
    u = np.zeros((h, w), np.uint8)
    for c, xc, yc, bw, bh, *_ in cajas:
        if clase is None or int(c) == clase:
            u[int(max(0, (yc - bh / 2) * h)):int(min(h, (yc + bh / 2) * h)) + 1, int(max(0, (xc - bw / 2) * w)):int(min(w, (xc + bw / 2) * w)) + 1] = 1
    return u


def rasgos(rgb, mascara, cajas):
    """Devuelve el diccionario de medidas que usa la regla de decisión."""
    h, w = mascara.shape
    cob = max(int(mascara.sum()), 1)
    r = {"n_cajas": float(len(cajas)), "area_total": float((_union(cajas, h, w) & mascara).sum() / cob)}
    for i in range(len(CLASES)):
        r[f"n{i}"] = float(sum(1 for c in cajas if int(c[0]) == i))
        r[f"a{i}"] = float((_union(cajas, h, w, i) & mascara).sum() / cob)
    L, A = _lab(rgb)
    v = (mascara == 1) & (L > 8) & (L < 70)
    r["a_med"] = float(np.median(A[v])) if v.any() else 0.0
    r["rojo15"] = float((A[v] > 15).mean()) if v.any() else 0.0
    r["rojo25"] = float((A[v] > 25).mean()) if v.any() else 0.0
    return r


def vector(r):
    return [r[k] for k in ORDEN]


# ---- ¿hay una mazorca en la foto? ----
# El detector solo busca defectos: sin esta comprobación, una foto sin maíz "no tiene defectos" y saldría como primera.
# La mazorca de maíz morado es una mancha oscura, alargada y con textura (los granos). Los umbrales se calibraron con
# las 751 fotos del conjunto (todas se aceptan) y con recortes de fondo, capturas y fotos de otros objetos (casi todos se rechazan).
MENSAJE_SIN_MAZORCA = "No se reconoce una mazorca de maíz morado en la foto."


def hay_mazorca(rgb):
    """Devuelve (hay, medidas). rgb: imagen uint8 (alto, ancho, 3)."""
    h, w = rgb.shape[:2]
    e = 320 / max(h, w)
    im = cv2.resize(rgb, None, fx=e, fy=e, interpolation=cv2.INTER_AREA) if e < 1 else rgb
    im = cv2.GaussianBlur(im, (0, 0), 1.5)
    L = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    oscuro = (L < 42).astype(np.uint8)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, etiquetas, est, _ = cv2.connectedComponentsWithStats(oscuro)
    if n <= 1:
        return False, {"fraccion": 0.0}
    k = 1 + int(np.argmax(est[1:, cv2.CC_STAT_AREA]))
    comp = (etiquetas == k).astype(np.uint8)
    fraccion = float(comp.sum() / comp.size)
    contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (rw, rh), _ = cv2.minAreaRect(max(contornos, key=cv2.contourArea))
    alargada = max(rw, rh) / max(1.0, min(rw, rh))
    relleno = float(comp.sum() / max(1.0, rw * rh))
    gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    media = cv2.blur(gris, (5, 5))
    varianza = cv2.blur(gris * gris, (5, 5)) - media * media
    interior = cv2.erode(comp, np.ones((9, 9), np.uint8)) > 0
    textura = float(np.sqrt(np.maximum(varianza[interior], 0)).mean()) if interior.any() else 0.0
    borde = float((comp[0].sum() + comp[-1].sum() + comp[:, 0].sum() + comp[:, -1].sum()) / (2 * sum(comp.shape)))
    hay = 0.06 <= fraccion <= 0.90 and relleno >= 0.40 and textura >= 5.0 and alargada >= 1.05 and not (fraccion > 0.35 and borde > 0.25)
    return bool(hay), {"fraccion": fraccion, "alargada": alargada, "relleno": relleno, "textura": textura, "borde": borde}
