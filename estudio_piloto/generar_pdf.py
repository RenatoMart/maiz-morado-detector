"""Genera el informe. Vuelve a ejecutarlo cuando terminen los entrenamientos para actualizar resultados."""
import json
from datetime import date
from pathlib import Path
from matplotlib import font_manager
from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

E = Path(__file__).parent; SAL = E.parent.parent / "DEMAS" / "informe" / "Estudio_modelos_maiz_morado.pdf"
d = Path(font_manager.findfont("DejaVu Sans")).parent
pdfmetrics.registerFont(TTFont("D", d / "DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DB", d / "DejaVuSans-Bold.ttf")); pdfmetrics.registerFontFamily("D", normal="D", bold="DB", italic="D", boldItalic="DB")
MOR, TIN, GRI, CLA = colors.HexColor("#4A2C6F"), colors.HexColor("#1F2328"), colors.HexColor("#5C6470"), colors.HexColor("#F2EEF7")
AMB = colors.HexColor("#FFF6DD"); VER = colors.HexColor("#E7F4EA")
B = ParagraphStyle("b", fontName="D", fontSize=9.2, leading=13, textColor=TIN, alignment=TA_LEFT, spaceAfter=4)
T = ParagraphStyle("t", parent=B, fontName="DB", fontSize=18, leading=22, textColor=MOR, spaceAfter=2)
S = ParagraphStyle("s", parent=B, fontSize=9, textColor=GRI, spaceAfter=8)
H = ParagraphStyle("h", parent=B, fontName="DB", fontSize=11.5, leading=15, textColor=MOR, spaceBefore=9, spaceAfter=4)
C = ParagraphStyle("c", parent=B, fontSize=8.1, leading=10.6, spaceAfter=0)
CH = ParagraphStyle("ch", parent=C, fontName="DB", textColor=colors.white)
N = ParagraphStyle("n", parent=B, fontSize=7.8, leading=10.4, textColor=GRI)
W = A4[0] - 36 * mm
def tabla(filas, anchos, resaltar=None, color=VER):
    x = [[Paragraph(str(c), CH if i == 0 else C) for c in f] for i, f in enumerate(filas)]
    t = Table(x, colWidths=[a * W for a in anchos], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), MOR), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#D9D4E2")), ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    for r in resaltar or []: st.append(("BACKGROUND", (0, r), (-1, r), color))
    t.setStyle(TableStyle(st)); return t
def cuadro(txt, fondo=CLA):
    t = Table([[Paragraph(txt, B)]], colWidths=[W]); t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fondo), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])); return t
def lista(items):
    return ListFlowable([ListItem(Paragraph(i, B), leftIndent=11) for i in items], bulletType="bullet", start="•", leftIndent=11, bulletFontSize=8)
def pct(v): return f"{v*100:.1f} %"

det = {}
if (E / "runs/detectores.json").exists(): det = {x["exp"]: x for x in json.load(open(E / "runs/detectores.json"))}
log = (E / "runs_log.txt").read_text(errors="ignore") if (E / "runs_log.txt").exists() else ""
clsr = next((json.loads(l) for l in log.splitlines() if l.startswith('{"exp": "E_')), None)
FIN = "FIN" in log.splitlines() and clsr is not None
def fila_det(k, nom, para):
    x = det.get(k)
    return [nom, para, f'{x["min"]:.0f} min' if x else "En curso", pct(x["map50"]) if x else "En curso", pct(x["p"]) if x else "En curso", pct(x["r"]) if x else "En curso"]
s = []
s += [Paragraph("Detección de defectos en maíz morado", T), Paragraph(f"Estudio de modelos, resultados y plan recomendado · {date.today():%d/%m/%Y}" + ("" if FIN else " · <b>versión preliminar</b>"), S)]
s += [cuadro("<b>Conclusión.</b> Proponemos un sistema por etapas: <b>YOLO11 preentrenado</b> marca los defectos, se mide el <b>área de la mazorca</b> y se calcula qué <b>proporción</b> ocupa cada defecto. Un modelo sencillo entrenado con esas medidas y las reglas del productor dan el veredicto. En las pruebas, YOLO11 preentrenado fue el mejor detector, y el área coincidió mejor que el conteo incluso con cajas reales de YOLO. Son pocas fotos: debe confirmarse con las 1000 fotos finales.")]
s += [Paragraph("1. El proyecto y por qué se hace", H), Paragraph("Hoy la calidad de la mazorca se decide a simple vista, y eso depende de cada evaluador. El sistema analiza una foto, marca las zonas con defectos, explica qué significan e indica si la mazorca pasa o se descarta. Los defectos son: zona sin granos, pudrición, pigmentación roja, granos quebrados y pérdida extensa. La persona conserva la decisión final.", B)]
s += [Paragraph("2. Qué probamos y qué salió", H)]
s += [tabla([["Prueba", "Qué probamos", "Imágenes", "Resultado"],
 ["Recorte de mazorca", "Método clásico (GrabCut), sin entrenar", "82 recortadas; 24 revisadas a ojo", "Buen contorno en casi todas; falló en casos con hojas y sombras. Falta medir el error"],
 ["Rojo general", "Color dentro de la mazorca", "2 fotos: muy roja y Primera", "Rojo notorio: 47.9 % contra 0.2 % de píxeles. Solo es un indicio"],
 ["Forma de decidir", "Reglas y modelos con las cajas humanas, en validación cruzada", "117: 66 Descarte y 51 Primera", "Ver tabla siguiente"],
 ["Modelos YOLO", "Pesos, versión, número de clases y clasificador", "82: 57 entrenar, 12 validar, 13 probar", "Ganó YOLO11n preentrenado. Ver sección 3"]], [.17, .31, .22, .30])]
s += [Spacer(1, 6), Paragraph("<b>Forma de decidir: Primera o Descarte</b>", B)]
s += [tabla([["Método", "Primera correcta", "Descarte correcto", "Promedio"],
 ["Cualquier caja descarta", "28 %", "100 %", "64 %"], ["Contar cajas (hasta 3 rojos)", "67 %", "92 %", "80 %"], ["Contar cajas (hasta 5 rojos)", "76 %", "92 %", "84 %"], ["Un solo límite de área total", "92 %", "79 %", "86 %"],
 ["Logística con áreas", "100 %", "85 %", "92 %"], ["Logística con áreas y conteos", "96 %", "92 %", "94 %"], ["<b>Logística con áreas, conteos y color</b>", "<b>100 %</b>", "<b>92 %</b>", "<b>96 %</b>"], ["Gradient Boosting con áreas, conteos y color", "98 %", "100 %", "99 %"]], [.49, .17, .18, .16], resaltar=[7])]
s += [Spacer(1, 5), cuadro("<b>Cuidado: estas cifras están infladas.</b> Usan cajas dibujadas por personas, no por YOLO. Además, 35 de las 51 Primera son anotaciones antiguas sin revisar, y solo 5 son recientes, así que el modelo puede estar aprendiendo el estilo de anotar. Sirven para comparar métodos, no como exactitud final.", AMB)]
s += [Paragraph("3. Modelos YOLO", H), Paragraph("Entrenamiento corto en CPU: 640 píxeles, hasta 40 épocas y una prueba final de 13 fotos, 3 de ellas de Primera. Con tan pocas fotos el mAP sale bajo y es muy inestable, así que importa la comparación y no el número. Los tiempos son de esta prueba en CPU y no sirven para calcular el entrenamiento final." + ("" if FIN else " <b>Estos entrenamientos aún no terminan.</b>"), B)]
s += [tabla([["Modelo", "Pregunta", "Tiempo", "mAP@50", "Precisión", "Recall"], fila_det("A_yolo11n_coco_5clases", "YOLO11n preentrenado, 5 clases", "¿Cómo rinde la base?"), fila_det("B_yolo11n_desde_cero_5clases", "YOLO11n desde cero", "¿Conviene entrenar sin pesos previos?"), fila_det("C_yolo26n_coco_5clases", "YOLO26n preentrenado", "¿La versión nueva rinde más?"), fila_det("D_yolo11n_coco_1clase", "YOLO11n, 1 clase \"defecto\"", "¿Encuentra más si no separa tipos?"), ["YOLO11n clasificador", "¿Basta clasificar la foto completa?", f'{clsr["min"]:.0f} min' if clsr else "En curso", "No aplica", "No aplica", "No aplica"]], [.27, .29, .09, .12, .12, .11], resaltar=[1])]
s += [Spacer(1, 5), lista(["<b>Pesos preentrenados:</b> YOLO11n logró 11.8 % de mAP@50, contra 3.5 % desde cero. Desde cero acertó muy pocas cajas (0.7 % de precisión).", "<b>YOLO11 contra YOLO26:</b> en la versión n ganó YOLO11, 11.8 % contra 5.0 %. Con tan pocas fotos no basta para descartar YOLO26, así que se comparan de nuevo con los datos finales.", "<b>Una clase contra cinco:</b> 10.5 % contra 11.8 %. Casi igual. Separar los 5 defectos no perjudicó y da más información.", "<b>Clasificador:</b> 100 % en 13 fotos, pero viendo <b>solo el fondo</b>, sin la mazorca, ya acierta 92 %. El fondo delata la categoría. Sin fondo también dio 100 %, porque esta muestra compara Primera sana con Descarte muy dañado: es el caso fácil."])]
s += [Spacer(1, 5), Paragraph("<b>Sistema con cajas reales de YOLO11n</b> (25 fotos de validación y prueba: 20 Descarte y 5 Primera)", B)]
s += [tabla([["Medida", "Resultado", "Qué significa"], ["Parecido con las cajas humanas", "Área: 0.89 · Conteo: 0.57", "El área coincide mejor que el conteo, con el modelo real (confianza 0.05)"], ["Área de defectos de 10 % o más, confianza 0.05", "Primera 5/5 · Descarte 20/20", "Muy prometedor, pero el límite se eligió con estas mismas fotos y las Primera son fáciles"], ["Mismo límite, confianza 0.25 (la del informe)", "Primera 5/5 · Descarte 0/20", "Con este modelo corto, esa confianza deja escapar el Descarte; hay que recalibrarla"]], [.38, .24, .38], resaltar=[1])]
s += [Paragraph("Referencia: el modelo RF-DETR XL actual de Roboflow (v4) marca mAP@50 de 7.1 %, medido con solo 5 fotos de prueba.", N)]
s += [Paragraph("4. Qué aprendimos", H), lista(["<b>Las proporciones funcionan mejor que contar.</b> Si una persona dibuja 2 cajas y el modelo 1, el área total casi no cambia, pero el conteo sí.", "<b>La confianza importa.</b> Con área, una confianza baja funcionó mucho mejor que la de 0.25, al menos en este modelo corto.", "<b>Aprender los pesos supera a una regla fija.</b> El área total fue la medida que más influyó en el Random Forest, seguida por la pudrición. La pigmentación roja pesó poco por sí sola.", "<b>El rojo necesita dos medidas.</b> Los granos sueltos se miden con cajas; cuando toda la mazorca es rojiza, se mide con el color.", "<b>No hace falta otro modelo para empezar a medir la mazorca.</b> El método clásico puede bastar. Si falla, se entrena un segmentador.", "<b>El fondo es un riesgo comprobado.</b> Un clasificador acertó 92 % viendo solo el fondo, sin la mazorca. Hay que fotografiar ambas categorías en los mismos fondos."])]
s += [Paragraph("5. Sistema propuesto y por qué", H)]
s += [tabla([["Etapa", "Qué se usa", "Por qué"],
 ["1. Marcar defectos", "YOLO11 preentrenado (COCO), comparado con YOLO26", "Fue el mejor en la prueba y es lo que plantea el informe. Marca las zonas y aprovecha pesos previos"],
 ["2. Medir la mazorca", "GrabCut; si falla, YOLO11n-seg", "Dio buen contorno sin entrenar. Permite medir proporciones reales, aunque la mazorca esté inclinada"],
 ["3. Calcular medidas", "% de área por defecto sin contar dos veces lo encimado, conteos y color", "Reúne tamaño, tipo y rojo general, sin depender de cuántas cajas se dibujen"],
 ["4. Decidir", "Regresión logística y reglas fijas del productor", "Es simple, explica cada peso y necesita pocos datos. Gradient Boosting se compara al final"],
 ["5. Mostrar", "Zonas, significado, % y veredicto", "La persona entiende el motivo y puede corregirlo"]], [.2, .36, .44])]
s += [Spacer(1, 5), Paragraph("<b>Por qué no las otras opciones:</b> un clasificador solo no marca zonas y aprendió del fondo. Entrenar YOLO desde cero rindió 3 veces menos en esta prueba. Boosting y árboles no sirven para leer píxeles, pero sí para decidir con las medidas de la etapa 3.", B)]
s += [Paragraph("6. Plan con 500 fotos marcadas y 500 de Primera", H)]
s += [tabla([["Paso", "Qué hacer", "Condición para avanzar"],
 ["1. Fijar el criterio", "Confirmar con el productor los límites y si los defectos se suman", "Guía de anotación actualizada"],
 ["2. Corregir datos", "Revisar las anotaciones antiguas, marcar zonas rojas faltantes y evitar cajas duplicadas", "Criterio uniforme en ambas categorías"],
 ["3. Equilibrar fotos", "Incluir Primera y Descarte en todos los fondos y teléfonos; agrupar fotos de una misma mazorca", "Ningún fondo identifica una categoría"],
 ["4. Dividir", "70 % entrenar, 20 % validar y 10 % prueba (unas 100 fotos), con todas las clases", "La prueba no se toca hasta el final"],
 ["5. Entrenar", "YOLO11s y YOLO26s a 960 píxeles en Colab o Kaggle; también YOLO11n a 640, el base del informe", "Elegir con validación"],
 ["6. Auditar", "Listar etiquetas dudosas con cleanlab y corregirlas a mano", "Errores revisados por una persona"],
 ["7. Ajustar la decisión", "Entrenar la logística y calibrar límites de área y color", "Validado por el productor"],
 ["8. Evaluar una vez", "mAP (meta del informe: 85 %), exactitud por categoría, Descarte que se escapa y la mejor confianza", "Resultado final para el informe"]], [.2, .52, .28])]
s += [Spacer(1, 4), Paragraph("<b>Qué medir como éxito:</b> sobre todo, cuántas mazorcas de Descarte pasan por error. El informe la considera la falla más costosa.", B)]
img = E / "datos/control_mascaras.jpg"
if img.exists():
    a, h = PI.open(img).size; w = W * .82
    s += [KeepTogether([Paragraph("Control visual del recorte automático (contorno amarillo)", H), Image(str(img), width=w, height=w * h / a), Paragraph("Algunos contornos incluyen sombras u hojas. Por eso hay que medir su error antes de adoptarlo.", N)])]
s += [Paragraph("7. Límites y cambios en el informe", H), lista(["<b>Límites:</b> pocos datos, una prueba muy pequeña, pocos casos difíciles de Primera con defectos menores, un solo lugar, y solo se mide la cara visible. La luz y el teléfono pueden alterar el color; conviene usar una referencia de color en fotos nuevas.", "<b>Cambios en el informe:</b> actualizar las clases (de 3 a 5), las Tablas 10 y 11 con la regla por áreas, la cantidad real de fotos y la justificación sobre versiones nuevas de YOLO, además de describir la medición de área y color."])]
s += [Paragraph("Fuentes principales", H), Paragraph("Ultralytics, comparación YOLO26 y YOLO11 (docs.ultralytics.com/compare/yolo26-vs-yolo11). Sapkota et al., SAM3 frente a YOLO ajustado (arXiv 2512.11884). Gu et al. 2021, gravedad por proporción de lesión en trigo (doi 10.3389/fpls.2020.599886). Anotaciones del proyecto en Roboflow; datos y scripts en la carpeta <b>estudio/</b>.", N)]
def pie(c, doc):
    c.saveState(); c.setFont("D", 7.5); c.setFillColor(GRI); c.drawString(18 * mm, 10 * mm, "Postcosecha del maíz morado"); c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Página {doc.page}"); c.restoreState()
SimpleDocTemplate(str(SAL), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="Estudio de modelos - Maíz morado", author="Postcosecha del maíz morado").build(s, onFirstPage=pie, onLaterPages=pie)
print(SAL)
