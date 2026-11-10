"""Regla del productor y combinación de modelos.

Ejecutar con: python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maiz_morado.comun import cargar_config, grado_por_nombre, id_base, normalizar_clase
from maiz_morado.decision import DESCARTE, PRIMERA, REVISAR, SEGUNDA, decidir, grado_por_regla

CONFIG = cargar_config()
ZONA = "zona_sin_granos"


ROJO = "pigmentacion_roja"
QUEBRADO = "granos_quebrados_o_deteriorados"


class ReglaProductor(unittest.TestCase):
    def test_limites(self):
        casos = [
            ([], PRIMERA),
            ([ZONA], PRIMERA),
            ([ZONA] * 2, SEGUNDA),
            ([ZONA] * 3, SEGUNDA),
            ([ZONA] * 4, DESCARTE),
            ([ROJO], PRIMERA),
            ([ROJO] * 5, PRIMERA),
            ([ROJO] * 6, DESCARTE),
            ([QUEBRADO] * 5, PRIMERA),
            ([QUEBRADO] * 6, DESCARTE),
            (["pudricion"], DESCARTE),
            (["perdida_extensa_de_granos"], DESCARTE),
        ]
        for clases, esperado in casos:
            with self.subTest(clases=clases):
                self.assertEqual(grado_por_regla(clases, CONFIG)[0], esperado)

    def test_limite_configurable(self):
        self.assertEqual(grado_por_regla([ZONA] * 4, CONFIG, max_zonas=6)[0], SEGUNDA)


class Fusion(unittest.TestCase):
    def test_primera_requiere_que_ningun_modelo_sospeche(self):
        self.assertEqual(decidir([], 0.95, CONFIG).grado, PRIMERA)
        self.assertEqual(decidir([], 0.30, CONFIG).grado, REVISAR)

    def test_modelo1_no_borra_un_defecto_localizado(self):
        decision = decidir(["pudricion"], 0.99, CONFIG)
        self.assertEqual(decision.grado, DESCARTE)
        self.assertEqual(decision.accion, "DESCARTADA")
        self.assertTrue(decision.avisos)

    def test_sin_modelo1_usa_solo_la_regla(self):
        self.assertEqual(decidir([], None, CONFIG).grado, PRIMERA)


class Nombres(unittest.TestCase):
    def test_id_base(self):
        esperado = "primera_clase__IMG_6649"
        self.assertEqual(id_base("primera_clase__IMG_6649.jpg"), esperado)
        self.assertEqual(id_base("primera_clase__IMG_6649_jpg.rf.a1B2c3.jpg"), esperado)

    def test_grado_por_prefijo(self):
        self.assertEqual(grado_por_nombre("primera_clase__IMG_1.jpg", CONFIG), PRIMERA)
        self.assertEqual(grado_por_nombre("descarte__20261017_122850.jpg", CONFIG), DESCARTE)
        self.assertIsNone(grado_por_nombre("IMG_6440.jpg", CONFIG))

    def test_normalizar_clase(self):
        self.assertEqual(normalizar_clase("Pigmentación Roja"), "pigmentacion_roja")
        self.assertEqual(normalizar_clase("ausencia de granos", CONFIG), ZONA)


if __name__ == "__main__":
    unittest.main()
