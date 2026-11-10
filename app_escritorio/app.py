"""Versión de escritorio de la web de inspección de maíz morado (Tkinter).

Uso (desde la carpeta PYTHON):  python -m app_escritorio.app
"""
import platform
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageTk  # noqa: E402

AQUI = Path(__file__).resolve().parent
ASSETS = AQUI / "assets"

# Paleta del logo de la UPN (la misma de la web)
AMARILLO, TINTA, FONDO, BLANCO, SUAVE, CLARO = "#FBC105", "#0F0F0F", "#FFFAEB", "#FFFFFF", "#5B5440", "#FFF3C4"
VEREDICTOS = {
    "Primera": ("PASA: PRIMERA CALIDAD", "No se encontraron defectos que superen lo tolerado.", "#D9F3E0", "#14532D"),
    "Segunda": ("PASA: SEGUNDA CALIDAD", "Tiene zonas sin granos, pero dentro de lo tolerado.", "#FFE9A8", "#6B4200"),
    "Descarte": ("SE DESCARTA", "Los defectos superan lo tolerado para vender la mazorca.", "#FFD9D6", "#8E1B1B"),
}
CLASES = {
    "zona sin granos": ("Zona sin granos", "Hueco donde faltan uno o varios granos.", "#FFE600"),
    "pudricion": ("Pudrición", "Tejido blando u oscuro, a veces con moho. Descarta la mazorca.", "#00E676"),
    "pigmentacion roja": ("Pigmentación roja", "Granos rojos en lugar de morados.", "#FF2A2A"),
    "granos quebrados o deteriorados": ("Granos quebrados o deteriorados", "Granos rotos o dañados, a menudo blanquecinos.", "#3A86FF"),
    "perdida extensa de granos": ("Pérdida extensa de granos", "Zona grande sin granos, de un cuarto o más de la mazorca.", "#FF4FD8"),
}
# Tipo de letra según el sistema: Windows trae Segoe UI; en Linux se usa DejaVu Sans
LETRA = {"Windows": "Segoe UI", "Darwin": "Helvetica Neue"}.get(platform.system(), "DejaVu Sans")
TITULO = (LETRA, 20, "bold")
TEXTO = (LETRA, 10)
NEGRITA = (LETRA, 10, "bold")


def boton(padre, texto, comando, negro=True):
    """Botón con el estilo de la web: negro con texto amarillo, o claro con borde."""
    fg, bg = (AMARILLO, TINTA) if negro else (TINTA, BLANCO)
    b = tk.Button(padre, text=texto, command=comando, font=NEGRITA, fg=fg, bg=bg, activeforeground=TINTA, activebackground=AMARILLO if negro else CLARO,
                  relief="solid", bd=2, highlightthickness=0, padx=22, pady=8, cursor="hand2")
    return b


def titulo_paso(padre, numero, texto):
    fila = tk.Frame(padre, bg=FONDO)
    tk.Label(fila, text=numero, font=TITULO, fg=AMARILLO, bg=TINTA, width=2).pack(side="left")
    tk.Label(fila, text=texto, font=TITULO, fg=TINTA, bg=FONDO).pack(side="left", padx=12)
    return fila


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UPN · Inspección de maíz morado")
        self.geometry("1220x780")
        self.minsize(980, 640)
        self.configure(bg=FONDO)
        try:
            self.iconphoto(True, tk.PhotoImage(file=str(ASSETS / "icono.png")))
        except tk.TclError:
            pass
        self.analizador = None
        self.error_modelo = None
        self.fotos = []  # evita que Tk descarte las imágenes
        self.cola = queue.Queue()  # los hilos de trabajo dejan aquí lo que debe hacer la ventana
        self._cabecera()
        cuerpo = tk.Frame(self, bg=FONDO)
        cuerpo.pack(fill="both", expand=True, padx=28, pady=22)
        cuerpo.columnconfigure(0, weight=5, uniform="c")
        cuerpo.columnconfigure(1, weight=7, uniform="c")
        cuerpo.rowconfigure(0, weight=1)
        self._panel_foto(cuerpo)
        self._panel_resultado(cuerpo)
        self._pie()
        self.estado("Cargando el modelo…")
        threading.Thread(target=self._cargar_modelo, daemon=True).start()
        self.after(100, self._atender_cola)

    # ---------- estructura ----------
    def _cabecera(self):
        barra = tk.Frame(self, bg=TINTA)
        barra.pack(fill="x")
        logo = ImageTk.PhotoImage(Image.open(ASSETS / "logo_cabecera.png").resize((160, 96), Image.LANCZOS))
        self.fotos.append(logo)
        tk.Label(barra, image=logo, bg=TINTA).pack(side="left", padx=24, pady=8)
        tk.Label(barra, text="INSPECCIÓN DE MAÍZ MORADO", font=(LETRA, 14, "bold"), fg=AMARILLO, bg=TINTA).pack(side="right", padx=28)
        tk.Frame(self, bg=AMARILLO, height=5).pack(fill="x")

    def _pie(self):
        tk.Frame(self, bg=TINTA, height=2).pack(fill="x", side="bottom")
        self.pie = tk.Label(self, text="", font=(LETRA, 9), fg=SUAVE, bg=FONDO, anchor="w")
        self.pie.pack(fill="x", side="bottom", padx=28, pady=6)

    def _panel_foto(self, padre):
        izq = tk.Frame(padre, bg=FONDO)
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        titulo_paso(izq, "1", "Elige una foto").pack(anchor="w", pady=(0, 14))
        caja = tk.Frame(izq, bg=BLANCO, highlightbackground=TINTA, highlightthickness=2)
        caja.pack(fill="x")
        self.vista = tk.Canvas(caja, width=380, height=290, bg=BLANCO, highlightthickness=0)
        self.vista.pack(padx=18, pady=(18, 8))
        flecha = ImageTk.PhotoImage(Image.open(ASSETS / "flecha.png").resize((60, 78), Image.LANCZOS))
        self.fotos.append(flecha)
        self.vista.create_image(190, 120, image=flecha)
        self.vista.create_text(190, 205, text="Elige la foto de una mazorca", font=(LETRA, 12, "bold"), fill=TINTA)
        self.vista.create_text(190, 232, text="JPG o PNG, una sola mazorca por foto", font=TEXTO, fill=SUAVE)
        self.boton_subir = boton(caja, "Subir foto", self.elegir)
        self.boton_subir.pack(pady=(4, 18))
        for linea in ("Una sola mazorca por foto, bien iluminada.", "Acércate hasta que la mazorca ocupe buena parte de la imagen.", "Sin otras mazorcas ni objetos grandes alrededor."):
            tk.Label(izq, text="•  " + linea, font=TEXTO, fg=SUAVE, bg=FONDO, anchor="w", wraplength=400, justify="left").pack(fill="x", pady=(8, 0))

    def _panel_resultado(self, padre):
        der = tk.Frame(padre, bg=FONDO)
        der.grid(row=0, column=1, sticky="nsew")
        titulo_paso(der, "2", "Resultado").pack(anchor="w", pady=(0, 14))
        marco = tk.Frame(der, bg=FONDO)
        marco.pack(fill="both", expand=True)
        self.lienzo = tk.Canvas(marco, bg=FONDO, highlightthickness=0)
        barra = ttk.Scrollbar(marco, orient="vertical", command=self.lienzo.yview)
        self.lienzo.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.lienzo.pack(side="left", fill="both", expand=True)
        self.res = tk.Frame(self.lienzo, bg=FONDO)
        self.ventana_res = self.lienzo.create_window((0, 0), window=self.res, anchor="nw")
        self.res.bind("<Configure>", lambda e: self.lienzo.configure(scrollregion=self.lienzo.bbox("all")))
        self.lienzo.bind("<Configure>", lambda e: self.lienzo.itemconfigure(self.ventana_res, width=e.width))
        for evento, delta in (("<Button-4>", -1), ("<Button-5>", 1)):
            self.lienzo.bind_all(evento, lambda e, d=delta: self.lienzo.yview_scroll(d * 2, "units"))
        self.lienzo.bind_all("<MouseWheel>", lambda e: self.lienzo.yview_scroll(-1 * (e.delta // 120) * 2, "units"))
        self._mensaje("Aquí verás la foto con las zonas marcadas, el porcentaje de la mazorca afectado y si pasa o se descarta.")

    # ---------- acciones ----------
    def _atender_cola(self):
        while not self.cola.empty():
            self.cola.get_nowait()()
        self.after(100, self._atender_cola)

    def estado(self, texto):
        self.pie.configure(text=texto)

    def _cargar_modelo(self):
        try:
            from app_escritorio.analisis import Analizador

            self.analizador = Analizador()
            self.cola.put(lambda: self.estado("Modelo cargado. Resultados preliminares: es un apoyo a la inspección; la decisión final es de la persona."))
        except Exception as e:  # noqa: BLE001
            self.error_modelo = str(e)
            self.cola.put(lambda: self.estado(f"No se pudo cargar el modelo: {e}"))

    def elegir(self):
        inicial = next((p for p in (AQUI.parent / "fotos_de_ejemplo", Path.home() / "Descargas" / "fotos_de_ejemplo", Path.home() / "Downloads" / "fotos_de_ejemplo") if p.exists()), Path.home())
        ruta = filedialog.askopenfilename(title="Elige la foto de una mazorca", initialdir=inicial,
                                          filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Todos", "*.*")])
        if ruta:
            self.analizar(ruta)

    def analizar(self, ruta):
        if self.analizador is None:
            self._error(self.error_modelo or "El modelo todavía se está cargando. Espera unos segundos e inténtalo de nuevo.")
            return
        self.boton_subir.configure(state="disabled")
        self._vaciar()
        self._mensaje("Analizando la mazorca… puede tardar unos segundos.", fondo=AMARILLO)
        self.estado("Analizando…")

        def trabajo():
            try:
                datos = self.analizador.analizar(ruta)
                self.cola.put(lambda: self._mostrar(ruta, datos))
            except Exception as e:  # noqa: BLE001
                self.cola.put(lambda: self._error(str(e)))

        threading.Thread(target=trabajo, daemon=True).start()

    # ---------- dibujo del resultado ----------
    def _vaciar(self):
        for w in self.res.winfo_children():
            w.destroy()
        self.lienzo.yview_moveto(0)

    def _mensaje(self, texto, fondo=BLANCO):
        tk.Label(self.res, text=texto, font=TEXTO, fg=TINTA, bg=fondo, wraplength=560, justify="left", anchor="w", padx=20, pady=18,
                 highlightbackground=TINTA, highlightthickness=2).pack(fill="x")

    def _error(self, texto):
        self._vaciar()
        self.boton_subir.configure(state="normal")
        tk.Label(self.res, text=texto, font=NEGRITA, fg="#8E1B1B", bg="#FFD9D6", wraplength=560, justify="left", anchor="w", padx=20, pady=16,
                 highlightbackground=TINTA, highlightthickness=2).pack(fill="x")
        self.estado("No se pudo analizar la foto.")

    def _seccion(self, texto):
        tk.Label(self.res, text=texto.upper(), font=(LETRA, 15, "bold"), fg=TINTA, bg=FONDO, anchor="w").pack(fill="x", pady=(22, 0))
        tk.Frame(self.res, bg=AMARILLO, height=4).pack(fill="x", pady=(2, 8))

    def _sin_mazorca(self, ruta, mensaje):
        banner = tk.Frame(self.res, bg="#FFE9A8", highlightbackground=TINTA, highlightthickness=2)
        banner.pack(fill="x")
        tk.Label(banner, text="NO HAY UNA MAZORCA", font=(LETRA, 22, "bold"), fg="#6B4200", bg="#FFE9A8", anchor="w").pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(banner, text=mensaje, font=NEGRITA, fg="#6B4200", bg="#FFE9A8", anchor="w").pack(fill="x", padx=18, pady=(0, 14))
        base = Image.open(ruta).convert("RGB")
        base.thumbnail((620, 460))
        foto = ImageTk.PhotoImage(base)
        self.fotos.append(foto)
        marco = tk.Frame(self.res, bg=BLANCO, highlightbackground=TINTA, highlightthickness=2)
        marco.pack(fill="x", pady=(16, 0))
        tk.Label(marco, image=foto, bg=BLANCO).pack(pady=6)
        self._seccion("Para que funcione")
        for linea in ("Fotografía una mazorca de maíz morado, sola y bien iluminada.", "Acércate hasta que ocupe buena parte de la imagen.", "Evita fondos muy oscuros o con otros objetos grandes."):
            tk.Label(self.res, text="•  " + linea, font=TEXTO, fg=TINTA, bg=FONDO, anchor="w", wraplength=560, justify="left").pack(fill="x", pady=(4, 0))
        self.estado(f"No se reconoció una mazorca en {Path(ruta).name}.")

    def _mostrar(self, ruta, d):
        self.boton_subir.configure(state="normal")
        self._vaciar()
        if d.get("sin_mazorca"):
            self._sin_mazorca(ruta, d["mensaje"])
            return
        titulo, detalle, fondo, tinta = VEREDICTOS[d["categoria"]]
        banner = tk.Frame(self.res, bg=fondo, highlightbackground=TINTA, highlightthickness=2)
        banner.pack(fill="x")
        tk.Label(banner, text=titulo, font=(LETRA, 22, "bold"), fg=tinta, bg=fondo, anchor="w").pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(banner, text=detalle, font=NEGRITA, fg=tinta, bg=fondo, anchor="w").pack(fill="x", padx=18, pady=(0, 14))

        # foto con las zonas marcadas
        base = Image.open(ruta).convert("RGB")
        base.thumbnail((620, 460))
        dib = ImageDraw.Draw(base)
        for z in d["zonas"]:
            color = CLASES.get(z["clase"], ("", "", "#FFFFFF"))[2]
            x0, y0 = z["x"] * base.width, z["y"] * base.height
            dib.rectangle([x0, y0, x0 + z["ancho"] * base.width, y0 + z["alto"] * base.height], outline=color, width=3)
        foto = ImageTk.PhotoImage(base)
        self.fotos.append(foto)
        marco = tk.Frame(self.res, bg=BLANCO, highlightbackground=TINTA, highlightthickness=2)
        marco.pack(fill="x", pady=(16, 0))
        tk.Label(marco, image=foto, bg=BLANCO).pack(pady=6)

        self._seccion("Por qué")
        tk.Label(self.res, text=f"Defectos en {d['area_defectos_pct']} % de la mazorca.", font=(LETRA, 13, "bold"), fg=TINTA, bg=FONDO, anchor="w").pack(fill="x")
        tk.Label(self.res, text=f"La mazorca ocupa {d['mazorca_pct_imagen']} % de la foto.", font=TEXTO, fg=SUAVE, bg=FONDO, anchor="w").pack(fill="x")
        for m in d["motivos"]:
            tk.Label(self.res, text="•  " + m, font=TEXTO, fg=TINTA, bg=FONDO, anchor="w", wraplength=560, justify="left").pack(fill="x", pady=(4, 0))
        if d["mazorca_pct_imagen"] < 4:
            tk.Label(self.res, text="La mazorca ocupa muy poco de la foto; acércate para un resultado más fiable.", font=NEGRITA, fg="#6B4200", bg="#FFE9A8",
                     wraplength=560, justify="left", anchor="w", padx=12, pady=8, highlightbackground=TINTA, highlightthickness=2).pack(fill="x", pady=(10, 0))

        clases = sorted(((k, v) for k, v in d["porcentaje_por_clase"].items() if v > 0), key=lambda kv: -kv[1])
        if clases:
            self._seccion("Qué se encontró")
            for clave, pct in clases:
                nombre, significado, color = CLASES.get(clave, (clave, "", "#CCCCCC"))
                fila = tk.Frame(self.res, bg=FONDO)
                fila.pack(fill="x", pady=(8, 0))
                tk.Frame(fila, bg=color, width=16, height=16, highlightbackground=TINTA, highlightthickness=2).pack(side="left")
                tk.Label(fila, text=nombre, font=NEGRITA, fg=TINTA, bg=FONDO).pack(side="left", padx=8)
                tk.Label(fila, text=f"{pct} %", font=(LETRA, 13, "bold"), fg=TINTA, bg=FONDO).pack(side="right")
                barra = tk.Frame(self.res, bg=CLARO, height=12, highlightbackground=TINTA, highlightthickness=2)
                barra.pack(fill="x", pady=(3, 0))
                tk.Frame(barra, bg=TINTA).place(relx=0, rely=0, relheight=1, relwidth=min(1.0, pct / 100))
                tk.Label(self.res, text=significado, font=(LETRA, 9), fg=SUAVE, bg=FONDO, anchor="w").pack(fill="x")

        if d["zonas"]:
            self._seccion(f"Zonas marcadas ({len(d['zonas'])})")
            estilo = ttk.Style()
            estilo.configure("Treeview.Heading", font=NEGRITA)
            tabla = ttk.Treeview(self.res, columns=("zona", "conf"), show="headings", height=min(8, len(d["zonas"])))
            tabla.heading("zona", text="Zona")
            tabla.heading("conf", text="Confianza")
            tabla.column("conf", width=110, anchor="e")
            for z in d["zonas"][:40]:
                tabla.insert("", "end", values=(CLASES.get(z["clase"], (z["clase"],))[0], f"{round(z['confianza'] * 100)} %"))
            tabla.pack(fill="x")

        tk.Label(self.res, text="Apoyo a la inspección: es un resultado automático y preliminar. La decisión final es de la persona.", font=(LETRA, 9), fg=SUAVE,
                 bg=FONDO, anchor="w", wraplength=560, justify="left").pack(fill="x", pady=(22, 10))
        self.estado(f"Listo: {Path(ruta).name}")


if __name__ == "__main__":
    Aplicacion().mainloop()
