# El Niño 1997-98 vs. hoy: video para redes sociales (MP4)
# Datos: el mismo archivo .bin de la web (NOAA OISST v2.1, anomalía frente a 1991-2020)
# Orden del video: 1) lo último (gancho)  2) 1997-98  3) El Niño actual  4) cierre igual al gancho
# El cierre es idéntico al inicio, así el video se repite en bucle sin salto.
# Instalar una sola vez:
#   pip install numpy pandas matplotlib cartopy imageio-ffmpeg

import gzip
import json
import struct
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

# ---------- PARÁMETROS (edita solo esto) ----------
DATOS = "mapa-temperatura-fen-datos.bin"   # el .bin de tu web: archivo local o su URL
ANIO_ACTUAL = 2026          # año en que empezó El Niño actual (se compara con 1997)
INICIO_1997 = "1997-01-01"  # desde qué fecha se muestra cada evento
FIN_1997 = "1998-06-30"     # hasta qué fecha se muestra 1997-98
FORMATO = "4:5"             # "4:5" = LinkedIn | "9:16" = Reels, TikTok y Shorts
ESCALA = (-5, 5)            # mínimo y máximo de la escala en °C (igual que la web)
LINEA_ECUADOR = True        # False = quita la línea punteada del Ecuador
REGIONES = False            # True = cajas Niño 1+2 y Niño 3.4 en el mapa grande
MINIMAPA = True             # False = quita el mini mapa de la zona Niño 1+2
SUAVIZAR = True             # False = curva con tramos rectos (sin redondear)
AUTOR = "Elaboración: Ing. Javier More"
GANCHO = 5                  # segundos iniciales con el dato más reciente
SEG_SEMANA_1997 = 0.15      # velocidad del tramo 1997-98: más alto = más lento
SEG_SEMANA_HOY = 0.25       # velocidad del tramo actual
PAUSA_FINAL = 5             # segundos del cierre
SALIDA = "el_nino_1997_vs_hoy"   # se crea el_nino_1997_vs_hoy.mp4
FPS = 30

# ---------- 1. LECTURA DEL .BIN ----------
def leer(origen):
    if origen.startswith("http"):
        pedido = urllib.request.Request(origen, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(pedido, timeout=120) as r:
            crudo = r.read()
    else:
        with open(origen, "rb") as f:
            crudo = f.read()
    if crudo[:2] == b"\x1f\x8b":                 # viene comprimido (gzip)
        crudo = gzip.decompress(crudo)
    if crudo[:4] != b"FEN1":
        raise SystemExit("Ese archivo no es el .bin de la web (formato FEN1).")
    largo = struct.unpack("<I", crudo[4:8])[0]
    return json.loads(crudo[8:8 + largo]), np.frombuffer(crudo, np.int8, offset=8 + largo)

meta, cuerpo = leer(DATOS)
semanas = meta["semanas"]                        # [inicio, fin, días, preliminar]
W = len(semanas)

def capa(inicio, info):
    """Cada semana viene como diferencia con la anterior, en décimas de °C."""
    celdas = info["nlat"] * info["nlon"]
    bloque = cuerpo[inicio:inicio + W * celdas].reshape(W, celdas).astype(np.int16)
    vacio = bloque[0] == -128                    # tierra: sin dato
    bloque[bloque == -128] = 0
    campo = np.cumsum(bloque, axis=0, dtype=np.int32).astype(np.float32) * meta["escala"]
    campo[:, vacio] = np.nan
    lat = info["lat0"] + info["paso"] * np.arange(info["nlat"])
    lon = info["lon0"] + info["paso"] * np.arange(info["nlon"])
    return lon, lat, campo.reshape(W, info["nlat"], info["nlon"]), inicio + W * celdas

LONg, LATg, GRILLA, siguiente = capa(0, meta["grilla"])      # Pacífico, 0.5°
LONd, LATd, DETALLE, _ = capa(siguiente, meta["detalle"])    # frente al Perú, 0.25°

# ---------- 2. LOS DOS EVENTOS ----------
desde = 0
for tramo in meta["tramos"]:
    tramo["desde"], desde = desde, desde + tramo["n"]
historico = next(t for t in meta["tramos"] if t["seg"] == "h")
reciente = next(t for t in meta["tramos"] if t["seg"] == "a")
EVENTO = historico.get("evento", "1997-98")
ANIO_HIST = historico.get("anio_inicio", 1997)
DESFASE = ANIO_ACTUAL - ANIO_HIST                # 2026 - 1997 = 29 años

def pendientes(x, y):
    """Pendientes para una curva redondeada que pasa por cada valor (PCHIP)."""
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros_like(y)
    for i in range(1, len(x) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], d[-1]
    return m

def curva(s, xq):
    x, y, m = s["x"], s["serie"], s["m"]
    xq = np.asarray(xq, float)
    if not SUAVIZAR or len(x) < 3:
        return np.interp(xq, x, y)
    h = np.diff(x)
    i = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    t = (xq - x[i]) / h[i]
    return ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * h[i] * m[i]
            + (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * h[i] * m[i + 1])

def preparar(tramo, inicio, fin, anio0):
    """Semanas de un evento. x = días desde el 1 de enero de su año."""
    idx = [i for i in range(tramo["desde"], tramo["desde"] + tramo["n"])
           if inicio <= pd.Timestamp(semanas[i][0]) <= fin]
    if len(idx) < 3:
        raise SystemExit("Hay muy pocas semanas en ese rango de fechas. Revisa los parámetros.")
    rangos = [(pd.Timestamp(semanas[i][0]), pd.Timestamp(semanas[i][1])) for i in idx]
    base = pd.Timestamp(anio0, 1, 1)
    x = np.array([(a + (b - a) / 2 - base) / pd.Timedelta(days=1) for a, b in rangos])
    serie = np.array([meta["zonas"]["nino12"][i] for i in idx], float)
    return dict(x=x, serie=serie, m=pendientes(x, serie), rangos=rangos, n=len(idx),
                grilla=GRILLA[idx], detalle=DETALLE[idx],
                denso=np.linspace(x[0], x[-1], 20 * (len(x) - 1) + 1))

inicio = pd.Timestamp(INICIO_1997)
H = preparar(historico, inicio, pd.Timestamp(FIN_1997), ANIO_HIST)
A = preparar(reciente, inicio + pd.DateOffset(years=DESFASE), pd.Timestamp.max, ANIO_ACTUAL)

PICO = H["serie"].max()
PICO_HOY = A["serie"].max()                      # pico del evento actual hasta hoy
V_HOY = A["serie"][-1]
X_HOY = A["x"][-1]
V_MISMA_FECHA = float(curva(H, X_HOY)) if H["x"][0] <= X_HOY <= H["x"][-1] else None

def fmt(v):
    s = f"{v:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "−")

# Titular del gancho y del cierre: sale de los datos, no está escrito a mano
if V_HOY > PICO:
    TITULAR = f"El mar ya se calentó más que en {EVENTO}"
elif V_MISMA_FECHA is not None and V_HOY > V_MISMA_FECHA:
    TITULAR = f"Más caliente que en esta fecha de {ANIO_HIST}"
else:
    TITULAR = f"{fmt(V_HOY)} °C frente a lo normal"
print(f"Datos al {meta['ultimo_dia']}. Niño 1+2 hoy: {fmt(V_HOY)} °C | "
      f"pico {ANIO_ACTUAL}: {fmt(PICO_HOY)} °C | pico de {EVENTO}: {fmt(PICO)} °C")

# ---------- 3. ESTILO ----------
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set",
         "Oct", "Nov", "Dic"]
plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
TEXTO, GRIS, GRIS_CLARO = "#1f2937", "#6b7280", "#9ca3af"
ROJO_TXT, AZUL_TXT = "#b91c1c", "#1d4ed8"

# Colores por bandas de 0.5 °C: azules (frío), blanco (normal), cálidos (calor)
frio = np.arange(ESCALA[0], -0.49, 0.5)
calor = np.arange(0.5, ESCALA[1] + 0.01, 0.5)
niveles = np.concatenate([frio, calor])
frios = plt.get_cmap("Blues_r")(np.linspace(0.1, 0.75, len(frio) - 1))
calidos = plt.get_cmap("YlOrRd")(np.linspace(0.12, 1.0, len(calor) - 1))
paleta = ListedColormap(np.vstack([frios, [[1, 1, 1, 1]], calidos]))
paleta.set_under("#08306b")
paleta.set_over("#4a0015")
norma = BoundaryNorm(niveles, paleta.N)
ROJO, AZUL = calidos[-3], frios[2]
# Relleno bajo la curva: solo dos colores, como en el GIF
RELLENO_CALIDO = "#fedd7e"                       # amarillo: sobre lo normal
RELLENO_FRIO = "#c7dbef"                         # celeste: bajo lo normal

def color_valor(v):
    return ROJO_TXT if v >= 0.5 else (AZUL_TXT if v <= -0.5 else TEXTO)

# ---------- 4. LIENZO (4:5 = 1080 x 1350 px | 9:16 = 1080 x 1920 px) ----------
ANCHO = 1080
ALTO = 1350 if FORMATO == "4:5" else 1920
ARRIBA, ABAJO = (40, 26) if FORMATO == "4:5" else (170, 230)   # 9:16 deja libre la zona de botones
IZQ, DER = 54, ANCHO - 54
PX = 120 / 72                                    # pixeles por punto tipográfico
fig = plt.figure(figsize=(ANCHO / 120, ALTO / 120), dpi=120, facecolor="white")
FigureCanvasAgg(fig)
render = fig.canvas.get_renderer()

def fx(px):
    return px / ANCHO

def fy(px):                                      # px medidos desde arriba
    return 1 - px / ALTO

def caja(x, y, ancho, alto):
    return [x / ANCHO, 1 - (y + alto) / ALTO, ancho / ANCHO, alto / ALTO]

def texto_justificado(x0, x1, y, texto, interlineado=1.45, **estilo):
    """Párrafo justificado (como Ctrl+J en Word). Devuelve dónde termina."""
    medidas = {}
    def ancho(t):
        if t not in medidas:
            tmp = fig.text(0, 0, t, **estilo)
            medidas[t] = tmp.get_window_extent(render).width
            tmp.remove()
        return medidas[t]
    total = x1 - x0
    espacio = ancho("a a") - ancho("aa")
    lineas, actual = [], []
    for palabra in texto.split(" "):
        prueba = actual + [palabra]
        if actual and sum(map(ancho, prueba)) + espacio * (len(prueba) - 1) > total:
            lineas.append(actual)
            actual = [palabra]
        else:
            actual = prueba
    lineas.append(actual)
    salto = estilo["fontsize"] * PX * interlineado
    for k, linea in enumerate(lineas):
        ultima = k == len(lineas) - 1
        hueco = espacio if (ultima or len(linea) == 1) else \
            (total - sum(map(ancho, linea))) / (len(linea) - 1)
        cursor = x0
        for palabra in linea:
            fig.text(fx(cursor), fy(y + k * salto), palabra, va="top", **estilo)
            cursor += ancho(palabra) + hueco
    return y + (len(lineas) - 1) * salto + estilo["fontsize"] * PX * 1.25

# --- Antetítulo, título (cambian en cada parte del video) y subtítulo ---
antetitulo = fig.text(fx(IZQ), fy(ARRIBA), "", va="top", fontsize=13,
                      weight="bold", color=ROJO_TXT)
titulo = fig.text(fx(IZQ), fy(ARRIBA + 27), "", va="top", fontsize=26,
                  weight="bold", color=TEXTO)

def poner_titulos(arriba, texto, color):
    antetitulo.set_text(arriba.upper())
    antetitulo.set_color(color)
    titulo.set_text(texto)
    titulo.set_fontsize(26)
    while titulo.get_window_extent(render).width > DER - IZQ and titulo.get_fontsize() > 16:
        titulo.set_fontsize(titulo.get_fontsize() - 0.5)   # se achica solo si no entra

y = texto_justificado(
    IZQ, DER, ARRIBA + 82,
    "Diferencia entre la temperatura del mar y su promedio de 1991-2020. "
    "Promedio semanal, en °C.",
    fontsize=12, color=GRIS)

# --- Mapa ---
ext = [LONg[0], LONg[-1], LATg[0], LATg[-1]]
ALTO_MAPA = (DER - IZQ) * (ext[3] - ext[2]) / (ext[1] - ext[0])
Y_MAPA = y + 12
ax = fig.add_axes(caja(IZQ, Y_MAPA, DER - IZQ, ALTO_MAPA),
                  projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent(ext, crs=ccrs.PlateCarree())
ax.spines["geo"].set_edgecolor(GRIS_CLARO)
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e5e7eb", zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="white",
               linewidth=0.5, zorder=3)
ax.coastlines(resolution="50m", color=GRIS, linewidth=0.5, zorder=3)
if LINEA_ECUADOR:
    ax.plot([ext[0], ext[1]], [0, 0], color=GRIS, lw=0.6, ls=(0, (4, 4)),
            transform=ccrs.PlateCarree(), zorder=3)
ax.text(284.5, -10, "PERÚ", color=GRIS, fontsize=10, weight="bold",
        transform=ccrs.PlateCarree(), zorder=4)
if REGIONES:
    for lon1, lon2, lat1, lat2, nombre in [(270, 280, -10, 0, "Niño 1+2"),
                                           (190, 240, -5, 5, "Niño 3.4")]:
        ax.add_patch(Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1,
                               fill=False, ec=TEXTO, lw=1.1, ls=(0, (3, 2)),
                               transform=ccrs.PlateCarree(), zorder=5))
        ax.text(lon1, lat1 - 5, nombre, fontsize=9.5, color=TEXTO,
                weight="bold", transform=ccrs.PlateCarree(), zorder=5)

# Coordenadas del borde (fijas: así cada cuadro se dibuja más rápido)
for lon in range(120, 301, 30):
    if ext[0] - 0.5 <= lon <= ext[1] + 0.5:
        nombre = "180°" if lon == 180 else (f"{lon}°E" if lon < 180 else f"{360 - lon}°O")
        xp = IZQ + (lon - ext[0]) / (ext[1] - ext[0]) * (DER - IZQ)
        fig.text(fx(xp), fy(Y_MAPA + ALTO_MAPA + 6), nombre, ha="center", va="top",
                 fontsize=9, color=GRIS)
for lat in (-20, 0, 20):
    yp = Y_MAPA + (ext[3] - lat) / (ext[3] - ext[2]) * ALTO_MAPA
    nombre = "0°" if lat == 0 else f"{abs(lat)}°{'N' if lat > 0 else 'S'}"
    fig.text(fx(IZQ - 6), fy(yp), nombre, ha="right", va="center", fontsize=9, color=GRIS)

fecha_txt = ax.text(0.015, 0.045, "", transform=ax.transAxes, fontsize=18,
                    weight="bold", color=TEXTO, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=5))
capitulo_txt = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center",
                       va="center", fontsize=96, weight="bold", color=TEXTO, zorder=7,
                       path_effects=[pe.withStroke(linewidth=12, foreground="white")])

# --- Escala de colores ---
y = Y_MAPA + ALTO_MAPA + 50
barra = fig.colorbar(plt.cm.ScalarMappable(norm=norma, cmap=paleta),
                     cax=fig.add_axes(caja(216, y, 648, 20)),
                     orientation="horizontal", extend="both",
                     ticks=range(ESCALA[0], ESCALA[1] + 1))
barra.outline.set_edgecolor("#d1d5db")
barra.outline.set_linewidth(0.6)
barra.ax.tick_params(labelsize=10.5, colors=TEXTO, length=0)
for texto, dx, lado in [("°C", 0, "center"),
                        ("Más frío que lo normal  ←", -14, "right"),
                        ("→  Más cálido que lo normal", 14, "left")]:
    barra.ax.annotate(texto, xy=(0, 0), xycoords=("data", "axes fraction"),
                      xytext=(dx, -20), textcoords="offset points",
                      ha=lado, va="top", fontsize=10, color=GRIS)

# --- Bloque inferior: costa norte del Perú ---
y += 92
fig.text(fx(IZQ), fy(y), "Frente a la costa norte del Perú", va="top",
         fontsize=15, weight="bold", color=TEXTO)
y = texto_justificado(
    IZQ, DER, y + 31,
    "Zona Niño 1+2: de 0° a 10°S y de 90°O a 80°O. Suele ser la primera en "
    "calentarse cuando llega El Niño.", fontsize=11, color=GRIS)
FILA = y + 14                                    # arriba del bloque
FONDO = ALTO - ABAJO - 70                        # abajo del gráfico
COLUMNA = 270                                    # ancho de la columna izquierda

# Cifra grande (columna izquierda)
cifra_etq = fig.text(fx(IZQ), fy(FILA), "", va="top", fontsize=12,
                     weight="bold", color=GRIS)
cifra = fig.text(fx(IZQ - 3), fy(FILA + 20), "−8.8 °C", va="top", fontsize=42,
                 weight="bold", color=TEXTO)
while cifra.get_window_extent(render).width > COLUMNA + 10:   # que no choque con el gráfico
    cifra.set_fontsize(cifra.get_fontsize() - 1)
cifra_sub = fig.text(fx(IZQ), fy(FILA + 100), "", va="top", fontsize=11.5,
                     color=GRIS, linespacing=1.35)

# Mini mapa de la zona Niño 1+2 (debajo de la cifra)
if MINIMAPA:
    lado = min(COLUMNA, FONDO - (FILA + 166))
    axm = fig.add_axes(caja(IZQ, FONDO - lado, lado, lado),
                       projection=ccrs.PlateCarree(central_longitude=180))
    axm.set_extent([268.5, 282.5, -12, 2], crs=ccrs.PlateCarree())
    axm.spines["geo"].set_visible(False)          # sin borde: efecto flotante
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e5e7eb", zorder=2)
    axm.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="white",
                    linewidth=0.8, zorder=3)
    axm.coastlines(resolution="50m", color=GRIS, linewidth=0.6, zorder=3)
    axm.add_patch(Rectangle((270, -10), 10, 10, fill=False, ec=TEXTO, lw=1.2,
                            ls=(0, (3, 2)), transform=ccrs.PlateCarree(), zorder=5))
    axm.text(270.2, 0.35, "Niño 1+2", fontsize=9, weight="bold", color=TEXTO,
             transform=ccrs.PlateCarree(), zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    sel_lat = (LATd >= -13) & (LATd <= 3)
    sel_lon = (LONd >= 267.5) & (LONd <= 283.5)
    LONm, LATm = np.meshgrid(LONd[sel_lon], LATd[sel_lat])

# Gráfico: 1997-98 y El Niño actual, alineados por mes
X0 = IZQ + COLUMNA + 64
axg = fig.add_axes(caja(X0, FILA + 6, DER - X0, FONDO - FILA - 6))
base = pd.Timestamp(ANIO_HIST, 1, 1)
x_ini = (inicio - base) / pd.Timedelta(days=1)
x_fin = (pd.Timestamp(FIN_1997) - base) / pd.Timedelta(days=1)
margen = (x_fin - x_ini) * 0.02 + 1
axg.set_xlim(x_ini - margen, x_fin + margen)
todos = np.concatenate([H["serie"], A["serie"]])
y_min = min(-1, np.floor(todos.min() - 0.4))
y_max = max(1, np.ceil(todos.max() + 0.6))
axg.set_ylim(y_min, y_max)
axg.set_autoscale_on(False)
axg.set_yticks(np.arange(y_min, y_max + 0.1, 1))
axg.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}".replace("-", "−") if v else "0")
for borde in ["top", "right"]:
    axg.spines[borde].set_visible(False)
for borde in ["left", "bottom"]:
    axg.spines[borde].set_color("#d1d5db")
axg.tick_params(colors=GRIS, labelsize=10.5, length=0)
axg.axhline(0, color=GRIS_CLARO, lw=0.8, zorder=2.6)   # encima del relleno

meses = pd.date_range(inicio.replace(day=1), pd.Timestamp(FIN_1997), freq="MS")
axg.set_xticks([(m - base) / pd.Timedelta(days=1) for m in meses])
etiquetas = []
for k, m in enumerate(meses):
    e = MESES[m.month - 1] if (m.month - meses[0].month) % 2 == 0 else ""
    if e and (m.month == 1 or k == 0):
        e += f"\n{m.year} / {m.year + DESFASE}"
    etiquetas.append(e)
axg.set_xticklabels(etiquetas)

# Referencia 1997-98 (se ve en la parte actual, en gris)
ref = axg.plot(H["denso"], curva(H, H["denso"]), color=GRIS_CLARO, lw=1.8, zorder=2.5)
ref.append(axg.axhline(PICO, color=GRIS_CLARO, lw=0.9, ls=(0, (4, 3)), zorder=2.4))
ref.append(axg.annotate(f"Pico de {EVENTO}: {fmt(PICO)} °C", (x_fin + margen, PICO),
                        xytext=(0, 4), textcoords="offset points", ha="right",
                        va="bottom", fontsize=10, color=GRIS))
ref.append(axg.annotate(EVENTO, (H["x"][-1], H["serie"][-1]), xytext=(0, -10),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=11, weight="bold", color=GRIS))

# Pico del evento actual (aparece al llegar al último dato)
pico_hoy = [axg.axhline(PICO_HOY, color=TEXTO, lw=1.0, ls=(0, (4, 3)), zorder=2.4),
            axg.annotate(f"Pico {ANIO_ACTUAL}: {fmt(PICO_HOY)} °C",
                         (x_ini - margen, PICO_HOY), xytext=(4, 4),
                         textcoords="offset points", ha="left", va="bottom",
                         fontsize=10, weight="bold", color=TEXTO, zorder=6,
                         bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.85, pad=1.5))]
for p in pico_hoy:
    p.set_visible(False)

fig.text(fx(IZQ), fy(ALTO - ABAJO), "Fuente: NOAA OISST v2.1", va="bottom",
         fontsize=10, color=GRIS)
fig.text(fx(DER), fy(ALTO - ABAJO), AUTOR, va="bottom", ha="right",
         fontsize=10, color=GRIS)

# ---------- 5. CUADROS ----------
capa_dinamica = []
LON2, LAT2 = np.meshgrid(LONg, LATg)

def etiqueta(ini, fin):
    m1, m2 = MESES[ini.month - 1], MESES[fin.month - 1]
    if ini.year != fin.year:
        return f"{ini.day} {m1} {ini.year} - {fin.day} {m2} {fin.year}"
    if ini.month != fin.month:
        return f"{ini.day} {m1} - {fin.day} {m2} {fin.year}"    # 27 Oct - 2 Nov 2025
    if ini.day == fin.day:
        return f"{ini.day} {m2} {fin.year}"
    return f"{ini.day}-{fin.day} {m2} {fin.year}"               # 6-12 Oct 2025

def cuadro(s, pos, capitulo=False, pulso=None, con_pico_hoy=False):
    """s = evento (H o A). pos = semana, puede ser fraccionaria (transición suave).
    capitulo = muestra el año grande sobre el mapa. pulso = 0 a 1 (anillo que late).
    con_pico_hoy = muestra la línea del pico del evento actual."""
    for c in capa_dinamica:
        c.remove()
    capa_dinamica.clear()
    hoy = s is A
    k = min(int(pos), s["n"] - 1)
    t = pos - k
    if t > 0:
        campo = (1 - t) * s["grilla"][k] + t * s["grilla"][k + 1]
        detalle = (1 - t) * s["detalle"][k] + t * s["detalle"][k + 1]
    else:
        campo, detalle = s["grilla"][k], s["detalle"][k]
    semana = k + (t >= 0.5)

    # Mapas
    capa_dinamica.append(ax.contourf(LON2, LAT2, campo, levels=niveles, cmap=paleta,
                                     norm=norma, extend="both", zorder=1,
                                     transform=ccrs.PlateCarree(), transform_first=True))
    if MINIMAPA:
        capa_dinamica.append(axm.contourf(LONm, LATm, detalle[sel_lat][:, sel_lon],
                                          levels=niveles, cmap=paleta, norm=norma,
                                          extend="both", zorder=1,
                                          transform=ccrs.PlateCarree(),
                                          transform_first=True))
    fecha_txt.set_text(etiqueta(*s["rangos"][semana]))
    capitulo_txt.set_text(str(ANIO_ACTUAL if hoy else ANIO_HIST) if capitulo else "")

    # Línea que se dibuja hasta la semana actual: amarillo arriba de 0, celeste abajo
    x_punto = s["x"][k] + t * (s["x"][k + 1] - s["x"][k]) if t > 0 else s["x"][k]
    xs = np.append(s["denso"][s["denso"] < x_punto], x_punto)
    ys = curva(s, xs)
    v = float(ys[-1])
    for arriba, color in [(True, RELLENO_CALIDO), (False, RELLENO_FRIO)]:
        capa_dinamica.append(axg.fill_between(
            xs, ys, 0, where=(ys >= 0) if arriba else (ys <= 0), interpolate=True,
            facecolor=color, edgecolor="none", lw=0, zorder=2))
    capa_dinamica.extend(axg.plot(xs, ys, color=TEXTO, lw=2.2, zorder=3))
    color_punto = ROJO if v >= 0 else AZUL
    capa_dinamica.extend(axg.plot(x_punto, v, "o", ms=9, color=color_punto,
                                  mec="white", mew=1.5, zorder=5))
    if pulso is not None:                         # anillo que late en el último dato
        capa_dinamica.extend(axg.plot(x_punto, v, "o", ms=10 + 26 * pulso, mfc="none",
                                      mec=color_punto, mew=2, alpha=0.8 * (1 - pulso),
                                      zorder=4))
    supera = hoy and v > PICO
    for r in ref:
        r.set_visible(hoy)
    ref[1].set_color(ROJO_TXT if supera else GRIS_CLARO)
    ref[2].set_color(ROJO_TXT if supera else GRIS)
    for p in pico_hoy:
        p.set_visible(hoy and con_pico_hoy)
    # Si el pico actual queda justo encima del de 1997-98, esa etiqueta pasa a la
    # izquierda, debajo de su línea (ahí el gráfico está despejado)
    cerca = hoy and con_pico_hoy and 0 < PICO_HOY - PICO < 0.5
    ref[2].xy = (x_ini - margen, PICO) if cerca else (x_fin + margen, PICO)
    ref[2].xyann = (4, -4) if cerca else (0, 4)
    ref[2].set_ha("left" if cerca else "right")
    ref[2].set_va("top" if cerca else "bottom")
    if hoy:
        a_la_derecha = x_punto < x_ini + 0.12 * (x_fin - x_ini)
        y_etq = max(v, PICO_HOY) if con_pico_hoy else v   # encima de la línea del pico
        capa_dinamica.append(axg.annotate(
            str(ANIO_ACTUAL), (x_punto, y_etq),
            xytext=(10 if a_la_derecha else -10, 5 if con_pico_hoy else 10),
            textcoords="offset points", ha="left" if a_la_derecha else "right",
            va="bottom", fontsize=12, weight="bold", color=TEXTO, zorder=6))

    # Cifra grande
    anio = s["rangos"][semana][1].year
    cifra_etq.set_text(f"Zona Niño 1+2 · {anio}")
    cifra.set_text(f"{fmt(v)} °C")
    cifra.set_color(color_valor(v))
    if hoy and supera:
        cifra_sub.set_text(f"{fmt(v - PICO)} °C sobre el pico\nde {EVENTO} "
                           f"({fmt(PICO)} °C)")
    elif hoy:
        cifra_sub.set_text(f"Pico de {EVENTO}:\n{fmt(PICO)} °C")
    else:
        cifra_sub.set_text("")
    cifra_sub.set_color(ROJO_TXT if supera else GRIS)
    cifra_sub.set_fontweight("bold" if supera else "normal")

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# ---------- 6. VIDEO ----------
video = imageio_ffmpeg.write_frames(f"{SALIDA}.mp4", (ANCHO, ALTO), fps=FPS,
                                    quality=8, macro_block_size=2,
                                    pix_fmt_out="yuv420p",
                                    output_params=["-movflags", "+faststart"])
video.send(None)
total = 0

def enviar(img, veces=1):
    global total
    for _ in range(veces):
        video.send(img)
        total += 1

def fundido(a, b, segundos=0.5):
    n = max(1, round(FPS * segundos))
    for p in range(1, n + 1):
        f = p / (n + 1)
        f = f * f * (3 - 2 * f)                   # suave al entrar y al salir
        enviar((a * (1 - f) + b * f).astype(np.uint8))

def recorrido(s, seg_semana, previo, frenar=0):
    """Entra desde la escena anterior con el año grande sobre el mapa.
    Luego avanza semana a semana, con transiciones suaves."""
    con_anio = cuadro(s, 0, capitulo=True)
    fundido(previo, con_anio)
    enviar(con_anio, round(FPS * 0.6))
    fundido(con_anio, cuadro(s, 0), 0.6)          # el año grande se desvanece
    duracion = np.full(s["n"] - 1, seg_semana)
    if frenar:                                    # cámara lenta en las últimas semanas
        n = min(frenar, len(duracion))
        duracion[-n:] *= 1 + 2.5 * (np.arange(1, n + 1) / n) ** 2
    marcas = np.round(np.concatenate([[0], np.cumsum(duracion)]) * FPS).astype(int)
    for i in range(s["n"] - 1):
        pasos = max(1, marcas[i + 1] - marcas[i])
        for p in range(1, pasos + 1):
            ultimo = cuadro(s, i + p / pasos)
            enviar(ultimo)
    return ultimo

# 1) Gancho: lo último, con un anillo que late sobre el dato de hoy
poner_titulos("Frente al norte del Perú", TITULAR, ROJO_TXT)
print("Dibujando el gancho...")
latido = [cuadro(A, A["n"] - 1, pulso=p / FPS, con_pico_hoy=True)
          for p in range(FPS)]                    # 1 segundo, se repite
n_gancho = round(FPS * GANCHO)
for i in range(n_gancho):
    enviar(latido[i % FPS])

# 2) 1997-98
poner_titulos(f"Volvamos a {ANIO_HIST}", f"Así se calentó el Pacífico en {EVENTO}", GRIS)
print(f"Dibujando {EVENTO} ({H['n']} semanas)...")
fin_h = recorrido(H, SEG_SEMANA_1997, latido[(n_gancho - 1) % FPS])
enviar(fin_h, round(FPS * 0.6))

# 3) El Niño actual, con 1997-98 de referencia
poner_titulos("Y ahora", f"Así se calienta el Pacífico en {ANIO_ACTUAL}", ROJO_TXT)
print(f"Dibujando {ANIO_ACTUAL} ({A['n']} semanas)...")
fin_a = recorrido(A, SEG_SEMANA_HOY, fin_h, frenar=6)
enviar(fin_a, round(FPS * 0.3))
con_pico = cuadro(A, A["n"] - 1, con_pico_hoy=True)   # aparece la línea del pico actual
fundido(fin_a, con_pico, 0.5)
enviar(con_pico, round(FPS * 1.2))

# 4) Cierre: vuelve al gancho (en bucle no se nota el corte)
fundido(con_pico, latido[0])
for i in range(1, max(1, round(PAUSA_FINAL)) * FPS):
    enviar(latido[i % FPS])
video.close()
plt.close(fig)
print(f"MP4 listo: {SALIDA}.mp4 ({ANCHO} x {ALTO} px, {total / FPS:.0f} segundos)")
