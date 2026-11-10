"""Entrena el modelo 1: ¿la mazorca completa se ve sana o defectuosa?

Uso:
    python 03_entrenar_modelo1_sana.py --exp C1 --publicar
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import json
import shutil

from maiz_morado.comun import DIR_M1, MODELOS, cargar_config, entrenar, resolver_experimento


def main():
    config = cargar_config()
    experimentos = config["modelo1"]["experimentos"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="C1", choices=sorted(experimentos))
    parser.add_argument("--epochs", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--imgsz", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--batch", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--device", help="Ejemplos: 0, cpu.")
    parser.add_argument("--publicar", action="store_true", help="Copia best.pt a modelos/.")
    args = parser.parse_args()

    if not DIR_M1.is_dir():
        raise SystemExit("Primero ejecuta 02_preparar_datos.py.")

    argumentos = resolver_experimento(experimentos, args.exp)
    for clave in ("epochs", "imgsz", "batch", "device"):
        valor = getattr(args, clave)
        if valor is not None:
            argumentos[clave] = valor

    modelo, mejor, carpeta = entrenar(
        argumentos, DIR_M1, "modelo1", args.exp, config["datos"]["semilla"]
    )
    # La selección del modelo usa val; test queda reservado para 07_evaluar_sistema.py.
    metricas = modelo.val(split="val", plots=False)
    resumen = {"experimento": args.exp, "top1_val": float(metricas.top1), "pesos": str(mejor)}
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(json.dumps(resumen, indent=2))

    if args.publicar:
        MODELOS.mkdir(exist_ok=True)
        shutil.copy2(mejor, MODELOS / "modelo1_sana.pt")
        print(f"Publicado en: {MODELOS / 'modelo1_sana.pt'}")


if __name__ == "__main__":
    main()
