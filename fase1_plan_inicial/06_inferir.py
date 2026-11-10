"""Analiza fotos: marca las zonas, explica qué significan y dice si pasa o se descarta.

Uso:
    python 06_inferir.py foto.jpg --mostrar
    python 06_inferir.py carpeta_con_fotos/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from maiz_morado.comun import CORRIDAS, EXT_IMAGEN, MODELOS, cargar_config, listar_imagenes
from maiz_morado.sistema import Sistema, cargar_imagen, dibujar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrada", help="Una foto o una carpeta de fotos.")
    parser.add_argument("--modelo1", default=str(MODELOS / "modelo1_sana.pt"))
    parser.add_argument("--modelo2", default=str(MODELOS / "modelo2_defectos.pt"))
    parser.add_argument("--sin-modelo1", action="store_true", help="Usa solo el detector y la regla.")
    parser.add_argument("--salida", default=str(CORRIDAS / "inferencia"))
    parser.add_argument("--mostrar", action="store_true", help="Abre el resultado en pantalla.")
    parser.add_argument("--device")
    args = parser.parse_args()

    entrada = Path(args.entrada)
    if entrada.is_dir():
        fotos = listar_imagenes(entrada)
    elif entrada.suffix.lower() in EXT_IMAGEN and entrada.exists():
        fotos = [entrada]
    else:
        raise SystemExit(f"No se encontró una imagen válida en: {entrada}")
    for ruta, nombre in ((args.modelo2, "modelo 2"), (None if args.sin_modelo1 else args.modelo1, "modelo 1")):
        if ruta and not Path(ruta).exists():
            raise SystemExit(f"No existe el {nombre}: {ruta}. Entrénalo con --publicar o indica la ruta.")

    config = cargar_config()
    sistema = Sistema(config, args.modelo2, None if args.sin_modelo1 else args.modelo1, args.device)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    resultados = []
    for foto in fotos:
        imagen = cargar_imagen(foto)
        detecciones, p_sana, decision = sistema.analizar(imagen)
        ruta_salida = salida / f"{foto.stem}_resultado.png"
        dibujar(imagen, detecciones, p_sana, decision, config).save(ruta_salida)
        resultados.append({
            "foto": str(foto),
            "resultado": str(ruta_salida),
            "accion": decision.accion,
            "grado": decision.grado,
            "motivos": decision.motivos,
            "avisos": decision.avisos,
            "p_sana": p_sana,
            "zonas": [
                {**asdict(d), "significado": config["clases"].get(d.clase, {}).get("significado", "")}
                for d in detecciones
            ],
        })
        print(f"{foto.name}: {decision.accion} ({decision.grado}), {len(detecciones)} zona(s) -> {ruta_salida}")
        if args.mostrar:
            subprocess.Popen(["xdg-open", str(ruta_salida)])

    resumen = salida / "resumen.json"
    resumen.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen: {resumen}")


if __name__ == "__main__":
    main()
