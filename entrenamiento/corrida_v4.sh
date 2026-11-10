#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY evaluar_v4.py > logs/evaluar_v4.log 2>&1; echo "evaluar rc=$?" > logs/corrida_v4.txt
$PY probar.py --modelo modelo_final_v4 --salida pruebas_v4 > logs/probar_v4.log 2>&1; echo "probar rc=$?" >> logs/corrida_v4.txt
