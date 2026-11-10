#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY pos.py --run runs/ajuste --salida modelo_final_v2 --iou 0.5 --min-descarte 0.95 > logs/pos_v2.log 2>&1
echo "pos v2 rc=$?" > logs/corrida_v2.txt
$PY probar.py --modelo modelo_final_v2 --salida pruebas_v2 --ablacion > logs/probar_v2.log 2>&1
echo "probar v2 rc=$?" >> logs/corrida_v2.txt
$PY probar.py --modelo modelo_final_final --salida pruebas_antes > logs/probar_antes.log 2>&1
echo "probar antes rc=$?" >> logs/corrida_v2.txt
