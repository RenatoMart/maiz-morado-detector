"""Registra cada 30 s el estado del equipo y del entrenamiento. No toca el entrenamiento.

Escribe logs/vigilante.log (avisos legibles), logs/vigilante.jsonl (todas las mediciones) y
logs/estado.json (el último estado, con el tiempo estimado que falta).
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import psutil
import yaml

AQUI = Path(__file__).resolve().parent
LOGS = AQUI / "logs"


def leer(ruta):
    try:
        return Path(ruta).read_text().strip()
    except OSError:
        return ""


def temperatura():
    try:
        return max(x.current for g in psutil.sensors_temperatures().values() for x in g)
    except Exception:
        return 0.0


def progreso(run):
    f = run / "results.csv"
    if not f.exists():
        return {"epocas_hechas": 0, "epocas_total": 0, "min_desde_ultima": None, "eta_min": None, "map50": None}
    filas = list(csv.DictReader(open(f)))
    total = 0
    try:
        total = int(yaml.safe_load(open(run / "args.yaml"))["epochs"])
    except Exception:
        pass
    t = [float(x["time"]) for x in filas if x.get("time")]
    d = [t[i] - t[i - 1] for i in range(1, len(t))][-6:]
    eta = (total - len(filas)) * (sum(d) / len(d)) / 60 if d and total else None
    map50 = None
    if filas:
        try:
            map50 = float(filas[-1]["metrics/mAP50(B)"])
        except (KeyError, ValueError):
            pass
    return {"epocas_hechas": len(filas), "epocas_total": total, "s_epoca": round(sum(d) / len(d), 1) if d else None,
            "min_desde_ultima": round((time.time() - f.stat().st_mtime) / 60, 1), "eta_min": round(eta) if eta else None, "map50": map50}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pid", type=int, required=True, help="Termina cuando este proceso termine.")
    p.add_argument("--cada", type=int, default=30)
    a = p.parse_args()
    LOGS.mkdir(exist_ok=True)
    run, avisado = Path(a.run), set()
    log = open(LOGS / "vigilante.log", "a", buffering=1)
    jsonl = open(LOGS / "vigilante.jsonl", "a", buffering=1)

    def aviso(clave, texto):
        if clave not in avisado:
            avisado.add(clave)
            log.write(f"{time.strftime('%F %T')} AVISO {texto}\n")

    log.write(f"{time.strftime('%F %T')} vigilante iniciado; sigue el proceso {a.pid}\n")
    while psutil.pid_exists(a.pid):
        v, sw = psutil.virtual_memory(), psutil.swap_memory()
        f = psutil.cpu_freq()
        e = {"hora": time.strftime("%F %T"), "ram_pct": v.percent, "ram_disp_mb": round(v.available / 2**20), "swap_pct": sw.percent,
             "cpu_pct": psutil.cpu_percent(), "mhz": round(f.current) if f else 0, "temp_c": temperatura(), "carga": os.getloadavg()[0],
             "cargador": leer("/sys/class/power_supply/ACAD/online"), "bateria_pct": leer("/sys/class/power_supply/BAT1/capacity"),
             "energia": leer("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"), **progreso(run)}
        jsonl.write(json.dumps(e) + "\n")
        (LOGS / "estado.json.tmp").write_text(json.dumps(e, indent=1))
        os.replace(LOGS / "estado.json.tmp", LOGS / "estado.json")
        if e["ram_disp_mb"] < 400:
            aviso(("ram", e["hora"][:13]), f"poca RAM libre: {e['ram_disp_mb']} MB")
        if e["temp_c"] >= 92:
            aviso(("temp", e["hora"][:13]), f"CPU a {e['temp_c']:.0f} °C")
        if e["cargador"] != "1":
            aviso("cargador", "sin cargador: el entrenamiento irá más lento y la batería bajará")
        if e["energia"] and e["energia"] != "performance":
            aviso(("energia", e["energia"]), f"energía en '{e['energia']}'")
        if e["min_desde_ultima"] and e["min_desde_ultima"] > 20:
            aviso(("sinprog", int(e["min_desde_ultima"]) // 20), f"{e['min_desde_ultima']} min sin avanzar de época")
        time.sleep(a.cada)
    log.write(f"{time.strftime('%F %T')} el proceso {a.pid} terminó; fin del vigilante\n")


if __name__ == "__main__":
    main()
