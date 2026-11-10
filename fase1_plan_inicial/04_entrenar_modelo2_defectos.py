"""Entrena el modelo 2: localiza y nombra los defectos con cajas.

Uso:
    python 04_entrenar_modelo2_defectos.py --exp E1
    python 04_entrenar_modelo2_defectos.py --exp R1 --publicar
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import json
import shutil

from maiz_morado.comun import DIR_M2, MODELOS, cargar_config, entrenar, resolver_experimento


def main():
    config = cargar_config()
    experimentos = config["modelo2"]["experimentos"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="R1", choices=sorted(experimentos))
    parser.add_argument("--epochs", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--imgsz", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--batch", type=int, help="Reemplaza el valor del experimento.")
    parser.add_argument("--device", help="Ejemplos: 0, cpu.")
    parser.add_argument("--publicar", action="store_true", help="Copia best.pt a modelos/.")
    args = parser.parse_args()

    data = DIR_M2 / "data.yaml"
    if not data.exists():
        raise SystemExit("Primero ejecuta 02_preparar_datos.py.")

    argumentos = resolver_experimento(experimentos, args.exp)
    for clave in ("epochs", "imgsz", "batch", "device"):
        valor = getattr(args, clave)
        if valor is not None:
            argumentos[clave] = valor

    modelo, mejor, carpeta = entrenar(
        argumentos, data, "modelo2", args.exp, config["datos"]["semilla"]
    )
    # La selección del modelo usa val; test queda reservado para 07_evaluar_sistema.py.
    metricas = modelo.val(data=str(data), split="val", plots=False)
    resumen = {
        "experimento": args.exp,
        "pesos": str(mejor),
        "map50_val": float(metricas.box.map50),
        "map50_95_val": float(metricas.box.map),
        "precision_val": float(metricas.box.mp),
        "recall_val": float(metricas.box.mr),
        "ap50_por_clase": {
            modelo.names[int(c)]: float(v)
            for c, v in zip(metricas.box.ap_class_index, metricas.box.ap50)
        },
        "alcanza_85": bool(metricas.box.map50 >= 0.85),
    }
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    if args.publicar:
        MODELOS.mkdir(exist_ok=True)
        shutil.copy2(mejor, MODELOS / "modelo2_defectos.pt")
        print(f"Publicado en: {MODELOS / 'modelo2_defectos.pt'}")


if __name__ == "__main__":
    main()
