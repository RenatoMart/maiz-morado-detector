"""Descarga desde Roboflow la versión anotada del dataset.

Uso:
    export ROBOFLOW_API_KEY="tu_clave_privada"
    python 01_descargar_roboflow.py [--version 4]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import os

from maiz_morado.comun import DATOS, cargar_config


def main():
    config = cargar_config()["roboflow"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=config["version"])
    parser.add_argument("--formato", default=config["formato"])
    args = parser.parse_args()

    clave = os.environ.get("ROBOFLOW_API_KEY")
    if not clave:
        raise SystemExit(
            "Falta ROBOFLOW_API_KEY. La clave privada está en "
            "https://app.roboflow.com/settings/api"
        )

    from roboflow import Roboflow

    proyecto = Roboflow(api_key=clave).workspace(config["workspace"]).project(config["proyecto"])
    destino = DATOS / "roboflow" / f"v{args.version}"
    proyecto.version(args.version).download(
        model_format=args.formato, location=str(destino), overwrite=True
    )
    print(f"Dataset descargado en: {destino}")


if __name__ == "__main__":
    main()
