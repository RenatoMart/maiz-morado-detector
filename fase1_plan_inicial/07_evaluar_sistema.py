"""Evalúa una sola vez sobre test: detector, regla y sistema completo.

Compara la decisión con la regla aplicada a las cajas anotadas (como el artículo)
y mide lo más costoso: mazorcas de Descarte que el sistema deja pasar.

Uso:
    python 07_evaluar_sistema.py
    python 07_evaluar_sistema.py --sin-modelo1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # para importar el paquete maiz_morado

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from maiz_morado.comun import CORRIDAS, DIR_M1, DIR_M2, MODELOS, SPLITS_CSV, cargar_config, normalizar_clase
from maiz_morado.decision import DESCARTE, PRIMERA, REVISAR, SEGUNDA, decidir, grado_por_regla
from maiz_morado.sistema import Sistema, cargar_imagen

REALES = [PRIMERA, SEGUNDA, DESCARTE]
PREDICHOS = [PRIMERA, SEGUNDA, DESCARTE, REVISAR]


def metricas(reales, predichos):
    matriz = confusion_matrix(reales, predichos, labels=PREDICHOS)[:3]
    total = int(matriz.sum())
    aciertos = int(sum(matriz[i, i] for i in range(3)))
    recall = {
        g: (float(matriz[i, i] / matriz[i].sum()) if matriz[i].sum() else None)
        for i, g in enumerate(REALES)
    }
    i_descarte = REALES.index(DESCARTE)
    return {
        "mazorcas": total,
        "exactitud": aciertos / total if total else None,
        "recall": recall,
        "descarte_que_paso": int(matriz[i_descarte, 0] + matriz[i_descarte, 1]),
        "enviadas_a_revision": int(matriz[:, 3].sum()),
        "matriz": pd.DataFrame(matriz, index=REALES, columns=PREDICHOS),
    }


def imprimir(titulo, m):
    print(f"\n{titulo}")
    print(f"  Mazorcas: {m['mazorcas']} | exactitud: {m['exactitud']:.3f}")
    print(f"  Descarte que pasó como aceptable: {m['descarte_que_paso']} | a revisión: {m['enviadas_a_revision']}")
    print("  Recall: " + ", ".join(f"{g} {'n/d' if v is None else f'{v:.3f}'}" for g, v in m["recall"].items()))
    print(m["matriz"].to_string())


def serializable(m):
    return {**m, "matriz": m["matriz"].to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo1", default=str(MODELOS / "modelo1_sana.pt"))
    parser.add_argument("--modelo2", default=str(MODELOS / "modelo2_defectos.pt"))
    parser.add_argument("--sin-modelo1", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()

    config = cargar_config()
    tabla = pd.read_csv(SPLITS_CSV)
    with open(DIR_M2 / "data.yaml", encoding="utf-8") as f:
        nombres = yaml.safe_load(f)["names"]
    nombres = {int(k): normalizar_clase(v, config) for k, v in nombres.items()}

    sistema = Sistema(config, args.modelo2, None if args.sin_modelo1 else args.modelo1, args.device)
    prueba = tabla[(tabla["split"] == "test") & tabla["usar_m2"]]
    if prueba.empty:
        raise SystemExit("No hay imágenes anotadas en test.")

    det = sistema.modelo2.val(data=str(DIR_M2 / "data.yaml"), split="test", plots=False, device=args.device)
    detector = {
        "map50": float(det.box.map50),
        "map50_95": float(det.box.map),
        "precision": float(det.box.mp),
        "recall": float(det.box.mr),
        "ap50_por_clase": {nombres[int(c)]: float(v) for c, v in zip(det.box.ap_class_index, det.box.ap50)},
        "recall_por_clase": {nombres[int(c)]: float(v) for c, v in zip(det.box.ap_class_index, det.box.r)},
        "inferencia_ms": float(det.speed["inference"]),
    }
    print("\nDetector en test:")
    print(json.dumps(detector, indent=2, ensure_ascii=False))

    conf_base = min(0.25, config["fusion"]["conf_defectos"])
    casos, referencia = [], []
    defectuosa = config["modelo1"]["nombre_clase_defecto"]
    for fila in tabla[tabla["split"] == "test"].itertuples():
        imagen = cargar_imagen(fila.imagen)
        detecciones = sistema.detectar(imagen, conf=conf_base)
        p_sana = sistema.probabilidad_sana(imagen)
        if fila.usar_m2:
            etiqueta = DIR_M2 / "labels" / "test" / f"{fila.id}.txt"
            casos.append({
                "id": fila.id,
                "clases_reales": [nombres[int(l.split()[0])] for l in etiqueta.read_text().splitlines() if l.strip()],
                "detecciones": detecciones,
                "p_sana": p_sana,
            })
        if isinstance(fila.clase_m1, str):
            grado = decidir(
                [d.clase for d in detecciones if d.confianza >= config["fusion"]["conf_defectos"]],
                p_sana, config,
            ).grado
            referencia.append({"id": fila.id, "ref_defectuosa": fila.clase_m1 == defectuosa, "predicho": grado})

    def evaluar(conf, max_zonas):
        reales, predichos, detalle = [], [], []
        for caso in casos:
            real, _ = grado_por_regla(caso["clases_reales"], config, max_zonas)
            clases = [d.clase for d in caso["detecciones"] if d.confianza >= conf]
            decision = decidir(clases, caso["p_sana"], config, max_zonas)
            reales.append(real)
            predichos.append(decision.grado)
            detalle.append({"id": caso["id"], "real": real, "predicho": decision.grado, "n_detecciones": len(clases)})
        return metricas(reales, predichos), detalle

    conf = config["fusion"]["conf_defectos"]
    limite = config["regla"]["max_zonas_segunda"]
    principal, detalle = evaluar(conf, limite)
    imprimir(f"Sistema completo (conf={conf}, máximo {limite} zonas para Segunda):", principal)

    # Referencia del productor: incluye fotos de descarte sin cajas, y por eso
    # usa más mazorcas para medir cuántas con defecto pasarían como Primera.
    ref = pd.DataFrame(referencia)
    ref_resumen = {}
    if not ref.empty:
        con_defecto = ref[ref["ref_defectuosa"]]
        sanas = ref[~ref["ref_defectuosa"]]
        ref_resumen = {
            "con_defecto": len(con_defecto),
            "con_defecto_como_primera": int((con_defecto["predicho"] == PRIMERA).sum()),
            "sanas": len(sanas),
            "sanas_como_primera": int((sanas["predicho"] == PRIMERA).sum()),
        }
        print("\nContra la referencia del productor:")
        print(json.dumps(ref_resumen, indent=2, ensure_ascii=False))
    if not args.sin_modelo1:
        top1 = float(sistema.modelo1.val(data=str(DIR_M1), split="test", plots=False, device=args.device).top1)
        ref_resumen["modelo1_top1_test"] = top1
        print(f"Modelo 1 en test, exactitud: {top1:.3f}")

    # Estabilidad de la regla, como pide el artículo. Las zonas van de 3 a 6, sin
    # cambiar la verdad de referencia, que se recalcula con el mismo límite.
    filas_barrido = []
    for valor_conf in (0.25, 0.50):
        for zonas in (3, 4, 5, 6):
            m, _ = evaluar(valor_conf, zonas)
            filas_barrido.append({
                "conf": valor_conf,
                "max_zonas": zonas,
                "exactitud": m["exactitud"],
                "recall_descarte": m["recall"][DESCARTE],
                "descarte_que_paso": m["descarte_que_paso"],
                "a_revision": m["enviadas_a_revision"],
            })
    barrido = pd.DataFrame(filas_barrido)
    print("\nEstabilidad de la regla:")
    print(barrido.to_string(index=False))

    salida = Path(CORRIDAS) / "evaluacion"
    salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detalle).to_csv(salida / "por_mazorca.csv", index=False)
    barrido.to_csv(salida / "barrido_regla.csv", index=False)
    (salida / "resumen.json").write_text(json.dumps(
        {"detector": detector, "sistema": serializable(principal), "referencia_productor": ref_resumen}, indent=2, ensure_ascii=False
    ), encoding="utf-8")
    print(f"\nResultados guardados en: {salida}")


if __name__ == "__main__":
    main()
