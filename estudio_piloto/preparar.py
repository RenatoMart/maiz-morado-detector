"""Descarga la muestra, reduce fotos, recorta la mazorca y calcula rasgos de cajas y color."""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).parent / "datos"
IMG = BASE / "img"; IMG.mkdir(parents=True, exist_ok=True); (BASE / "tmp").mkdir(exist_ok=True); MASK = BASE / "mask"; MASK.mkdir(exist_ok=True)
OWNER = "8SYtx5mbSPPcVrmhV0JYbpn54pF2"
R = json.load(open(BASE / "dataset_estudio.json"))
HUMANA = "g6puQD"
sel = [r for r in R if str(r["by"]).startswith(HUMANA) or (r["n"].startswith("primera") and not r["b"])]
print("muestra", len(sel))
CLASES = ["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"]

def bajar(r):
    dest = IMG / f"{r['id']}.jpg"
    if dest.exists(): return
    t = BASE / "tmp" / f"{r['id']}.jpg"
    subprocess.run(["curl", "-sf", "--max-time", "180", "-o", str(t), f"https://source.roboflow.com/{OWNER}/{r['id']}/original.jpg"], check=True)
    im = ImageOps.exif_transpose(Image.open(t)).convert("RGB")
    if im.size != (r["w"], r["h"]): print("AVISO tamaño", r["n"], im.size, (r["w"], r["h"]))
    im.thumbnail((1024, 1024), Image.LANCZOS); im.save(dest, quality=92); t.unlink()
with ThreadPoolExecutor(5) as ex: list(ex.map(bajar, sel))

def lab(a):
    x = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(float)
    return x[..., 0] * 100 / 255, x[..., 1] - 128, x[..., 2] - 128

def mascara(a, r):
    """GrabCut iniciado con un rectángulo central y las cajas de defectos como mazorca probable."""
    h, w = a.shape[:2]; m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    L, A, B = lab(a)
    oscuro = L < np.percentile(L, 35)
    m[oscuro] = cv2.GC_PR_FGD
    borde = max(4, int(.02 * min(h, w)))
    m[:borde] = m[-borde:] = cv2.GC_BGD; m[:, :borde] = m[:, -borde:] = cv2.GC_BGD
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), m, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    f = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)); f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, et, st, _ = cv2.connectedComponentsWithStats(f)
    if n > 1:
        i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); f = (et == i).astype(np.uint8)
    return f

def union(cajas, w, h):
    m = np.zeros((h, w), np.uint8)
    for b in cajas:
        m[int(max(0, b["y"] - b["h"] / 2)):int(b["y"] + b["h"] / 2) + 1, int(max(0, b["x"] - b["w"] / 2)):int(b["x"] + b["w"] / 2) + 1] = 1
    return m

filas = []
for r in sel:
    a = np.asarray(Image.open(IMG / f"{r['id']}.jpg")); h, w = a.shape[:2]; s = w / r["w"]
    f = mascara(a, r); cv2.imwrite(str(MASK / f"{r['id']}.png"), f * 255)
    L, A, B = lab(a); v = (f == 1) & (L > 8) & (L < 70)
    cob = max(int(f.sum()), 1)
    fila = {"id": r["id"], "n": r["n"], "ref": "Primera" if r["n"].startswith("primera") else "Descarte", "split_rf": r["split"], "t": r["t"],
            "frac_mazorca_foto": f.mean(), "a_med": float(np.median(A[v])) if v.any() else 0, "b_med": float(np.median(B[v])) if v.any() else 0,
            "rojo_a15": float((A[v] > 15).mean()) if v.any() else 0, "rojo_a25": float((A[v] > 25).mean()) if v.any() else 0, "n_cajas": len(r["b"])}
    bs = [{**b, "x": b["x"] * s, "y": b["y"] * s, "w": b["w"] * s, "h": b["h"] * s} for b in r["b"]]
    todas = union(bs, w, h); fila["area_total"] = (todas & f).sum() / cob; fila["area_bruta"] = todas.sum() / cob
    for i, c in enumerate(CLASES):
        q = [b for b in bs if b["l"] == c]; u = union(q, w, h)
        fila[f"n_{i}"] = len(q); fila[f"area_{i}"] = (u & f).sum() / cob; fila[f"max_{i}"] = max([b["w"] * b["h"] / cob for b in q], default=0)
    filas.append(fila)
pd.DataFrame(filas).to_csv(BASE / "rasgos.csv", index=False)
json.dump(sel, open(BASE / "muestra.json", "w"))
df = pd.DataFrame(filas); print(df.groupby("ref")[["frac_mazorca_foto", "a_med", "rojo_a15", "rojo_a25", "n_cajas", "area_total", "area_2"]].median().round(3).to_string()); print(df.ref.value_counts().to_string())
# lámina de control de máscaras
ids = list(df.sort_values("frac_mazorca_foto").id); elegidas = ids[::max(1, len(ids) // 24)][:24]
W = 6; T = 260; H = (len(elegidas) + W - 1) // W; lam = Image.new("RGB", (W * T, H * T), "#111")
for k, i in enumerate(elegidas):
    a = np.asarray(Image.open(IMG / f"{i}.jpg")).copy(); f = cv2.imread(str(MASK / f"{i}.png"), 0)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(a, cs, -1, (255, 255, 0), 6)
    p = Image.fromarray(a); p.thumbnail((T, T)); lam.paste(p, ((k % W) * T, (k // W) * T))
lam.save(BASE / "control_mascaras.jpg", quality=88)
