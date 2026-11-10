"""Regla de calidad y combinación de los dos modelos. Sin dependencias de YOLO."""
from collections import Counter
from dataclasses import dataclass, field

PRIMERA, SEGUNDA, DESCARTE, REVISAR = "Primera", "Segunda", "Descarte", "Revisar"
ACCIONES = {
    PRIMERA: "PASA",
    SEGUNDA: "PASA",
    DESCARTE: "DESCARTADA",
    REVISAR: "REVISIÓN MANUAL",
}


@dataclass
class Decision:
    grado: str
    motivos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)

    @property
    def accion(self):
        return ACCIONES[self.grado]


def grado_por_regla(clases, config, max_zonas=None):
    """Aplica la regla a una lista de clases normalizadas y explica el resultado."""
    conteo = Counter(clases)
    catalogo = config["clases"]
    criticas = [c for c in conteo if catalogo.get(c, {}).get("critica")]
    if criticas:
        nombres = ", ".join(catalogo[c]["nombre"] for c in sorted(criticas))
        return DESCARTE, [f"Defecto crítico: {nombres}."]

    regla = config["regla"]
    excedidas = [
        f"{conteo[clase]} de {catalogo[clase]['nombre'].lower()} (máximo {maximo})"
        for clase, maximo in regla.get("limites_descarte", {}).items()
        if conteo[clase] > maximo
    ]
    if excedidas:
        return DESCARTE, [f"Supera el límite: {'; '.join(excedidas)}."]

    limite = regla["max_zonas_segunda"] if max_zonas is None else max_zonas
    zonas = conteo[regla["clase_conteo"]]
    if zonas > limite:
        return DESCARTE, [f"{zonas} zonas sin granos (máximo {limite} para Segunda)."]
    if zonas > regla.get("max_zonas_primera", 0):
        return SEGUNDA, [f"{zonas} zonas sin granos, sin defectos críticos."]
    if conteo:
        return PRIMERA, ["Defectos menores dentro de los límites tolerados."]
    return PRIMERA, ["Ningún defecto detectado."]


def decidir(clases, p_sana, config, max_zonas=None):
    """Combina el detector (clases) con la probabilidad de 'sana' del modelo 1.

    Una mazorca solo pasa como Primera si ningún modelo sospecha. El modelo 1
    nunca borra un defecto localizado por el modelo 2: solo agrega una alerta.
    """
    grado, motivos = grado_por_regla(clases, config, max_zonas)
    if p_sana is None:
        return Decision(grado, motivos)

    fusion = config["fusion"]
    if grado == PRIMERA and p_sana < fusion["p_sana_minima"]:
        return Decision(
            REVISAR,
            [f"El modelo 1 sospecha defecto (p_sana={p_sana:.2f}), "
             "pero el modelo 2 no localizó ninguna zona."],
        )
    avisos = []
    if grado != PRIMERA and p_sana >= fusion["p_sana_contradice"]:
        avisos.append(
            f"El modelo 1 la ve sana (p_sana={p_sana:.2f}); confirma las zonas marcadas."
        )
    return Decision(grado, motivos, avisos)
