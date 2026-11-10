# Inspección de maíz morado: código en Python

Detector de defectos visibles en mazorcas (YOLOv11) y decisión de calidad por la **proporción de la mazorca afectada**
(Primera, Segunda o Descarte). Esta carpeta tiene todo lo necesario para entrenar, evaluar y probar el modelo.

> **Cómo abrirla:** en Visual Studio Code, `Archivo > Abrir carpeta…` y elige **esta carpeta (`PYTHON`)**.
> VS Code usará solo el entorno `.venv` de aquí. Todos los comandos de abajo se escriben desde esta carpeta.

## Instalación (Windows, Linux o macOS)

1. Instala **Python 3.9 o más nuevo** desde https://www.python.org/downloads/ (en Windows marca *Add python.exe to PATH*).
2. **Windows:** haz doble clic en `instalar_en_windows.bat`. **Linux o macOS:** `python3 instalar_dependencias.py`.
   El script revisa tu equipo, crea un entorno propio (`.venv`) y descarga solo lo que falte (unos 3 GB, solo CPU).
   Con `--solo-revisar` únicamente informa, sin instalar nada. Con `--entrenar` o `--informe` añade lo necesario para esas tareas.
3. Abre la aplicación: **Windows** `abrir_app_windows.bat` · **Linux/macOS** `.venv/bin/python -m app_escritorio.app`.

En Windows los comandos de más abajo se escriben con `.venv\Scripts\python.exe` en lugar de `.venv/bin/python`.
Los archivos `.sh` (entrenamiento vigilado) son solo para Linux; para entrenar en Windows usa `entrenar.py` y `pos.py` directamente.

## Qué hay aquí

```
PYTHON/
├── app_escritorio/        Versión de escritorio de la web (ventana con el mismo diseño y el mismo modelo)
├── maiz_morado/           Paquete con la lógica compartida (lo importan todos los demás)
│   ├── medidas.py           contorno de la mazorca (GrabCut) y las 15 medidas de proporción
│   ├── veredicto.py         decisión final: probabilidad de descarte → categoría y motivos
│   ├── decision.py          regla de conteo del plan inicial (se usa de comparación)
│   ├── sistema.py, comun.py utilidades del plan inicial
├── entrenamiento/         ★ El entrenamiento final y todo lo que se hizo con él
│   ├── construir_dataset.py  arma el dataset (grupos de mazorca, división train/val/test)
│   ├── entrenar.py           entrena yolo11n (o continúa desde pesos previos)
│   ├── pos.py                mide el detector, ajusta la decisión y exporta a ONNX
│   ├── recalibrar_decision.py ajuste final de la decisión con las cajas del detector
│   ├── probar.py             prueba el modelo con las fotos de prueba y deja láminas de ejemplo
│   ├── evaluar_v4.py, ajustar_inferencia.py  evaluaciones y barrido de parámetros
│   ├── todo.sh, vigilante.py entrenamiento vigilado (reanuda solo si se cuelga)
│   ├── datos/                 el dataset: images/, labels/, meta.json, data.yaml (751 imágenes)
│   ├── runs/                  corridas de entrenamiento (pesos, curvas)
│   ├── modelo_final_v4/       ★ MODELO FINAL: best.pt, best.onnx, decision.json, métricas
│   └── modelo_final_v2, v3, ajuste, final/, pruebas_*  versiones anteriores y sus pruebas
├── estudio_piloto/        Prueba inicial con 57 imágenes (comparó YOLO11, YOLO26, desde cero, una clase)
├── fase1_plan_inicial/    Scripts 01 a 07 del primer plan (dos modelos: mazorca sana y defectos)
├── configs/proyecto.yaml  Configuración del plan inicial (clases, regla del productor)
├── tests/                 Pruebas automáticas de la regla de decisión
├── instalar_dependencias.py  Revisa el equipo e instala lo que falte (funciona en Windows)
├── instalar_en_windows.bat / abrir_app_windows.bat   Atajos de doble clic para Windows
├── fotos_de_ejemplo/      9 fotos para probar la aplicación (incluye un caso en que el sistema falla)
├── requirements.txt       Librerías necesarias (versiones del entorno de desarrollo)
└── .venv/                 Entorno de Python ya instalado (Python 3.14)
```

## Qué ejecutar

**Ver el sistema funcionando (lo más rápido para presentar)**

```bash
.venv/bin/python -m app_escritorio.app
```
Abre una ventana igual a la web: eliges una foto y muestra si pasa o se descarta, el porcentaje de la mazorca afectado,
las zonas marcadas y qué defectos se encontraron. Hay fotos listas para probar en `~/Descargas/fotos_de_ejemplo`.

**Probar el modelo con el conjunto de prueba (80 fotos que el modelo no vio)**

```bash
cd entrenamiento
../.venv/bin/python probar.py --modelo modelo_final_v4 --salida pruebas_v4
```
Resultado esperado: 30 de 30 mazorcas de primera aceptadas y 49 de 50 de descarte rechazadas.

**Repetir el entrenamiento** (en CPU toma varias horas)

```bash
cd entrenamiento
../.venv/bin/python construir_dataset.py --todas      # arma el dataset
./todo.sh final 100                                    # entrena, mide, ajusta la decisión y exporta
```
`todo.sh` usa un vigía hecho en Go (carpeta `DEMAS/go`, que no está en este repositorio). Sin él, entrena directamente con
`../.venv/bin/python entrenar.py --nombre final --epochs 100` y luego `../.venv/bin/python pos.py --run runs/final --salida modelo_final_final`.

**Pruebas automáticas**

```bash
.venv/bin/python -m unittest discover tests
```

## Cómo funciona (resumen)

1. **Detector YOLOv11n** (`best.pt`): marca cajas sobre 5 tipos de defecto: granos quebrados o deteriorados, pérdida extensa
   de granos, pigmentación roja, pudrición y zona sin granos.
2. **Contorno de la mazorca** (GrabCut, `medidas.py`): permite medir qué fracción de la mazorca cubre cada defecto.
3. **15 medidas** (áreas por clase, número de zonas y color) → **regresión logística** (`decision.json`) → probabilidad de descarte.
4. **Categoría** (`veredicto.py`): Descarte si la probabilidad es 0.60 o más; si no, Segunda con 2 o más zonas sin granos, y Primera en otro caso.

Resultados del conjunto de prueba (80 imágenes): detector con mAP@50 de 15.1 %; categoría con 30 de 30 primeras y
49 de 50 descartes correctos. La categoría «Segunda» no pudo evaluarse (no hay mazorcas confirmadas de esa clase).
Detalles en el informe (carpeta `DEMAS/informe`).

## Qué incluye este repositorio y qué no

- **Incluye:** todo el código, el modelo final (`entrenamiento/modelo_final_v4/`), las anotaciones (`entrenamiento/datos/labels/`,
  `meta.json` y `registros.json`) y las fotos de ejemplo.
- **No incluye** (por peso): las fotos del dataset (≈ 480 MB), los registros de entrenamiento, las versiones anteriores del modelo y
  los pesos base de YOLO (`yolo11n.pt` se descarga solo). La carpeta `DEMAS` (informe, guía, diapositivas y herramienta en Go) tampoco está aquí.
- **Reconstruir el dataset:** `python entrenamiento/construir_dataset.py --todas` vuelve a descargar las fotos desde Roboflow
  usando `entrenamiento/datos/registros.json` (no hace falta clave de API).

## Notas

- `entrenamiento/datos` y `entrenamiento/runs` pesan unos 500 MB y no se suben a git (ver `.gitignore`).
- Las corridas guardadas en `entrenamiento/runs` conservan rutas del lugar anterior; sirven para consultar, no para reanudar.
  Para reanudar un entrenamiento, empieza uno nuevo con `--partir-de` (ver `entrenar.py`).
