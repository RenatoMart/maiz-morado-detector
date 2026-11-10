#!/bin/bash
# Entrena con vigilancia y, al terminar, mide, ajusta la decisión y exporta.
# Uso: ./todo.sh NOMBRE EPOCAS [opciones extra de entrenar.py]
# Se recomienda lanzarlo con:  systemd-inhibit --what=idle:sleep --why="Entrenamiento" ./todo.sh final 100
cd "$(dirname "$0")" || exit 1
PY=../.venv/bin/python
NOMBRE=${1:-final}
EPOCAS=${2:-100}
shift 2
mkdir -p logs
LOG=logs/todo.log
echo "$(date '+%F %T') inicio: $NOMBRE, $EPOCAS épocas" >> $LOG

$PY vigilante.py --run runs/$NOMBRE --pid $$ > /dev/null 2>&1 &

../../DEMAS/go/pipeline_go/bin/vigia --progreso runs/$NOMBRE/results.csv --sin-progreso 30m --gracia 40m \
    --ram-critica 0.97 --reintentos 8 -- \
    $PY entrenar.py --nombre "$NOMBRE" --epochs "$EPOCAS" "$@"
RC=$?
echo "$(date '+%F %T') entrenamiento terminó con código $RC" >> $LOG

if [ -f runs/$NOMBRE/weights/best.pt ]; then
    $PY pos.py --run runs/$NOMBRE --salida modelo_final_$NOMBRE >> logs/pos_$NOMBRE.log 2>&1
    echo "$(date '+%F %T') pos-proceso terminó con código $?" >> $LOG
else
    echo "$(date '+%F %T') no hay best.pt; no se ejecuta el pos-proceso" >> $LOG
fi
echo "$(date '+%F %T') fin" >> $LOG
