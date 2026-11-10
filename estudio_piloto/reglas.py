"""Compara reglas y modelos de tabla usando las cajas humanas, con validación cruzada repetida."""
import json, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import importlib.util
BASE = Path(__file__).parent / "datos"; R = json.load(open(BASE / "dataset_estudio.json")); M = {r["id"] for r in json.load(open(BASE / "muestra.json"))}
extra = [r for r in R if r["n"].startswith("primera") and r["id"] not in M]
print("Primera antiguas con cajas:", len(extra))
OWNER = "8SYtx5mbSPPcVrmhV0JYbpn54pF2"; IMG = BASE / "img"; MASK = BASE / "mask"
def bajar(r):
    d = IMG / f"{r['id']}.jpg"
    if d.exists(): return
    t = BASE / "tmp" / f"{r['id']}.jpg"; subprocess.run(["curl", "-sf", "--max-time", "180", "-o", str(t), f"https://source.roboflow.com/{OWNER}/{r['id']}/original.jpg"], check=True)
    im = ImageOps.exif_transpose(Image.open(t)).convert("RGB"); im.thumbnail((1024, 1024), Image.LANCZOS); im.save(d, quality=92); t.unlink()
with ThreadPoolExecutor(5) as e: list(e.map(bajar, extra))
def lab(a):
    x = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(float); return x[..., 0] * 100 / 255, x[..., 1] - 128, x[..., 2] - 128
def mascara(a):
    h, w = a.shape[:2]; m = np.full((h, w), cv2.GC_PR_BGD, np.uint8); L, _, _ = lab(a); m[L < np.percentile(L, 35)] = cv2.GC_PR_FGD
    k = max(4, int(.02 * min(h, w))); m[:k] = m[-k:] = cv2.GC_BGD; m[:, :k] = m[:, -k:] = cv2.GC_BGD
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65)); cv2.setNumThreads(1); cv2.grabCut(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), m, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    f = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8); f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)); f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, et, st, _ = cv2.connectedComponentsWithStats(f)
    return (et == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) if n > 1 else f
def union(c, w, h):
    m = np.zeros((h, w), np.uint8)
    for b in c: m[int(max(0, b["y"] - b["h"] / 2)):int(b["y"] + b["h"] / 2) + 1, int(max(0, b["x"] - b["w"] / 2)):int(b["x"] + b["w"] / 2) + 1] = 1
    return m
CL = ["granos quebrados o deteriorados", "perdida extensa de granos", "pigmentacion roja", "pudricion", "zona sin granos"]
F = []
for r in [x for x in R if x["id"] in M] + extra:
    a = np.asarray(Image.open(IMG / f"{r['id']}.jpg")); h, w = a.shape[:2]; s = w / r["w"]
    p = MASK / f"{r['id']}.png"
    if p.exists(): f = (cv2.imread(str(p), 0) > 0).astype(np.uint8)
    else: f = mascara(a); cv2.imwrite(str(p), f * 255)
    L, A, B = lab(a); v = (f == 1) & (L > 8) & (L < 70); cob = max(int(f.sum()), 1); bs = [{**b, **{k: b[k] * s for k in "xywh"}} for b in r["b"]]
    t = {"id": r["id"], "n": r["n"], "y": int(r["n"].startswith("descarte")), "reciente": str(r["by"]).startswith("g6puQD"), "n_cajas": len(bs), "a_med": float(np.median(A[v])) if v.any() else 0, "rojo15": float((A[v] > 15).mean()) if v.any() else 0, "rojo25": float((A[v] > 25).mean()) if v.any() else 0, "area_total": (union(bs, w, h) & f).sum() / cob}
    for i, c in enumerate(CL):
        q = [b for b in bs if b["l"] == c]; t[f"n{i}"] = len(q); t[f"a{i}"] = (union(q, w, h) & f).sum() / cob
    F.append(t)
d = pd.DataFrame(F); d.to_csv(BASE / "rasgos_todos.csv", index=False); y = d.y.values; print(d.groupby("y").size().to_string(), "recientes", d.groupby("y").reciente.sum().to_dict())
def regla_cuenta(lim):
    def g(x): return ((x.n3 > 0) | (x.n1 > 0) | (x.n2 > lim) | (x.n0 > 5) | (x.n4 >= 2)).astype(int).values
    return g
fijas = {"Regla estricta: cualquier caja": lambda x: (x.n_cajas > 0).astype(int).values, "Regla conteo, 3 rojos": regla_cuenta(3), "Regla conteo, 5 rojos": regla_cuenta(5)}
def med(v, p):
    return {"exact_bal": balanced_accuracy_score(y, p), "Primera_ok": (p[y == 0] == 0).mean(), "Descarte_ok": (p[y == 1] == 1).mean()}
filas = [{"metodo": k, **med(y, g(d)), "auc": np.nan} for k, g in fijas.items()]
Xc = {"Solo conteos": ["n_cajas", "n0", "n1", "n2", "n3", "n4"], "Solo áreas": ["area_total", "a0", "a1", "a2", "a3", "a4"], "Áreas + conteos": ["n_cajas", "n0", "n1", "n2", "n3", "n4", "area_total", "a0", "a1", "a2", "a3", "a4"], "Áreas + conteos + color": ["n_cajas", "n0", "n1", "n2", "n3", "n4", "area_total", "a0", "a1", "a2", "a3", "a4", "a_med", "rojo15", "rojo25"], "Solo color": ["a_med", "rojo15", "rojo25"]}
mods = {"Logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), "Random Forest": lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=2), "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=.05, max_iter=120, class_weight="balanced", random_state=0)}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=42)
for nx, cols in Xc.items():
    X = d[cols].values
    for nm, fm in mods.items():
        s = np.zeros(len(y)); n = np.zeros(len(y))
        for a, b in cv.split(X, y):
            m = fm().fit(X[a], y[a]); s[b] += m.predict_proba(X[b])[:, 1]; n[b] += 1
        s /= n; filas.append({"metodo": f"{nm} | {nx}", **med(y, (s >= .5).astype(int)), "auc": roc_auc_score(y, s)})
# Umbral simple de área por validación cruzada
for col in ["area_total", "a2"]:
    x = d[col].values; p = np.zeros(len(y)); n = np.zeros(len(y))
    for a, b in cv.split(x.reshape(-1, 1), y):
        ts = np.unique(np.r_[0, np.quantile(x[a], np.linspace(0, 1, 40))]); best = max(ts, key=lambda t: balanced_accuracy_score(y[a], x[a] > t)); p[b] += x[b] > best; n[b] += 1
    filas.append({"metodo": f"Umbral único aprendido en {col}", **med(y, (p / n >= .5).astype(int)), "auc": roc_auc_score(y, x)})
t = pd.DataFrame(filas).sort_values("exact_bal", ascending=False).round(3); t.to_csv(BASE / "comparacion_reglas.csv", index=False); print(t.head(26).to_string(index=False))
rf = RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(d[Xc["Áreas + conteos"]].values, y); print(pd.Series(rf.feature_importances_, Xc["Áreas + conteos"]).sort_values(ascending=False).round(3).head(8).to_string())
print(d.groupby("y")[["n_cajas", "area_total", "a2", "n2", "a3", "n3", "a0", "n0", "a4", "n4", "a_med", "rojo15"]].median().round(3).T.to_string())
