#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY pos.py --run runs/ajuste --salida modelo_final_v3 --iou 0.5 --min-descarte 0.95 --conf-fija 0.05 > logs/pos_v3.log 2>&1
echo "pos v3 rc=$?" > logs/corrida_v3.txt
$PY probar.py --modelo modelo_final_v3 --salida pruebas_v3 --ablacion > logs/probar_v3.log 2>&1
echo "probar v3 rc=$?" >> logs/corrida_v3.txt
