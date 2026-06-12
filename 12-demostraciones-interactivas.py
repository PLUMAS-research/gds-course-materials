# %%
"""Demostración interactiva: DBSCAN en vivo sobre reportes SOSAFE.

Este script es el ejemplo trabajado de la clase sobre visualizaciones
interactivas. Produce un archivo HTML autónomo que corre DBSCAN en el navegador:
el usuario mueve los sliders de eps y minPts, cambia de grupo de reportes y ve
los hotspots recalcularse en el mapa. La lección de la clase de clustering (no
todo fenómeno tiene hotspots separables) se vuelve manipulable: con Disturbios
aparecen concentraciones discretas, con Delitos un solo gradiente central que
ningún eps separa.

El patrón es general y sirve para los proyectos del curso:

1. El análisis (Python) produce los datos y los embebe en un template HTML.
2. El HTML carga las librerías desde un CDN (maplibre + deck.gl), sin build.
3. El cómputo corre en el navegador porque el dataset es chico (~22 mil puntos).
4. Se sirve por HTTP, no se abre como archivo local (los módulos ES y fetch no
   funcionan desde file:// por CORS).

Servir y abrir:

    uv run python -m http.server -d images 8000
    # http://localhost:8000/12-clustering-demo.html
"""

# %% Descargas
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/sosafe-clustering.tgz")

# %%
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

CRS_UTM_19S = "EPSG:32719"
BBOX = (-70.85, -33.65, -70.45, -33.30)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

COLORES_GRUPO = {
    "Ambiental": "#2a9d8f",
    "Disturbios": "#e9c46a",
    "Delitos": "#e76f51",
}

# %%
# ========================================================================
# PARTE 1. Cargar los datos del análisis
# ========================================================================
# Los mismos reportes de la clase de clustering, recortados a la ventana del
# Gran Santiago. Se proyectan a UTM 19S (metros) porque DBSCAN trabaja con
# distancias y eps se interpreta en metros: calcular distancias sobre
# longitud/latitud daría valores en grados, sin sentido físico.
reportes = gpd.read_parquet(
    Path("data") / "sosafe-clustering" / "reportes-quincena.parquet"
)
reportes = reportes.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].to_crs(CRS_UTM_19S)
reportes["x"] = reportes.geometry.x
reportes["y"] = reportes.geometry.y
print(f"Reportes en la ventana: {len(reportes):,}")
print(reportes["grupo"].value_counts())

# %%
# ========================================================================
# PARTE 2. Preparar el payload que se embebe en el HTML
# ========================================================================
# Dos consideraciones de tamaño y correctitud al pasar datos al navegador:
#
# - Tamaño: se embeben arrays paralelos (no una lista de objetos) y se redondean
#   las coordenadas. lon/lat a cinco decimales (~1 m) bastan para dibujar; las
#   coordenadas métricas se guardan como enteros con un offset restado, así los
#   números son cortos. La traslación no altera las distancias de DBSCAN.
# - Correctitud: el navegador necesita lon/lat para dibujar sobre el mapa y x/y
#   en metros para clusterizar. Se llevan ambos.

# Orden del panel: Disturbios primero por ser el caso con hotspots separables.
# Cada grupo trae los parámetros con que abre la demo (los de la clase 11).
GRUPOS_DEMO = [
    {"nombre": "Disturbios", "eps": 300, "minPts": 30},
    {"nombre": "Delitos", "eps": 300, "minPts": 30},
    {"nombre": "Ambiental", "eps": 400, "minPts": 15},
]
indice_grupo = {g["nombre"]: i for i, g in enumerate(GRUPOS_DEMO)}

reportes = reportes[reportes["grupo"].isin(indice_grupo)].copy()
lonlat = reportes.geometry.to_crs("EPSG:4326")
x0 = float(np.floor(reportes["x"].min()))
y0 = float(np.floor(reportes["y"].min()))

# Campos de texto para el tooltip al hacer clic en un reporte. Las descripciones
# ya vienen anonimizadas (correos y teléfonos removidos) desde la clase 11, así
# que es seguro embeberlas en el HTML público.
fecha = reportes["created_at"].dt.strftime("%d/%m/%Y %H:%M").tolist()

puntos = {
    "lon": lonlat.x.round(5).tolist(),
    "lat": lonlat.y.round(5).tolist(),
    "xi": (reportes["x"] - x0).round().astype(int).tolist(),
    "yi": (reportes["y"] - y0).round().astype(int).tolist(),
    "grupo": reportes["grupo"].map(indice_grupo).tolist(),
    "cat": reportes["categoria"].fillna("Sin categoría").tolist(),
    "desc": reportes["description"].fillna("").tolist(),
    "fecha": fecha,
    "likes": reportes["likes"].fillna(0).astype(int).tolist(),
    "comments": reportes["comments"].fillna(0).astype(int).tolist(),
}


def hex_a_rgb(color_hex):
    h = color_hex.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


grupos_js = [
    {**g, "color": hex_a_rgb(COLORES_GRUPO[g["nombre"]])} for g in GRUPOS_DEMO
]
print(f"Payload: {len(puntos['lon']):,} puntos en {len(grupos_js)} grupos")

# %%
# ========================================================================
# PARTE 3. Rellenar el template y escribir el HTML
# ========================================================================
# El template vive en gdsutils/templates/clustering.html y trae el panel de
# controles, el mapa y el código JavaScript que implementa DBSCAN con un índice
# de grilla. Aquí solo se inyectan los datos y la configuración de la cámara.
# Separar template (estructura) de datos (análisis) deja el HTML reutilizable
# para otro dataset sin tocar el JavaScript.
template = (Path("gdsutils") / "templates" / "clustering.html").read_text(
    encoding="utf-8"
)

descripcion = (
    "<hr class='sep'/>"
    "<small>Cada punto es un reporte SOSAFE. DBSCAN corre en el navegador: "
    "<b>eps</b> es el radio en metros que define vecindad y <b>minPts</b> los "
    "vecinos mínimos para formar un núcleo. El ruido va en gris. Haz clic en un "
    "reporte para ver su contenido. Cambia de grupo para ver que no todo fenómeno "
    "tiene hotspots separables.</small>"
)

html = (
    template.replace("__TITLE__", "DBSCAN interactivo (reportes SOSAFE)")
    .replace("__PANEL_TITULO__", "Clustering por densidad")
    .replace("__DESCRIPCION_HTML__", descripcion)
    .replace(
        "__BASEMAP_URL__",
        "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )
    .replace("__CENTER_LON__", "-70.65")
    .replace("__CENTER_LAT__", "-33.45")
    .replace("__ZOOM__", "10.5")
    .replace("__PUNTOS__", json.dumps(puntos))
    .replace("__GRUPOS__", json.dumps(grupos_js))
)

ruta = IMAGES / "12-clustering-demo.html"
ruta.write_text(html, encoding="utf-8")
print(f"Guardado: {ruta} ({len(html) / 1024:.0f} KB)")
print("Servir: uv run python -m http.server -d images 8000")
print("Abrir:  http://localhost:8000/12-clustering-demo.html")

# %%
# ========================================================================
# PARTE 4. Verificar que el cómputo del navegador coincide con el análisis
# ========================================================================
# Regla central de la clase: el código generado para el navegador no se confía,
# se verifica. La implementación de DBSCAN en JavaScript debe producir los
# mismos clusters que scikit-learn, o la demo estaría mostrando otra cosa que el
# análisis. Se reproduce aquí la lógica del índice de grilla del template y se
# compara contra scikit-learn en varios ajustes de parámetros.
from sklearn.cluster import DBSCAN


def dbscan_grilla(xs, ys, eps, min_pts):
    """Misma lógica que el JavaScript del template: DBSCAN con índice de grilla."""
    n = len(xs)
    lab = np.full(n, -2)
    ci = np.floor(xs / eps).astype(int)
    cj = np.floor(ys / eps).astype(int)
    grid = {}
    for p in range(n):
        grid.setdefault((ci[p], cj[p]), []).append(p)
    eps2 = eps * eps

    def vecinos(p):
        res = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for q in grid.get((ci[p] + di, cj[p] + dj), ()):
                    if (xs[p] - xs[q]) ** 2 + (ys[p] - ys[q]) ** 2 <= eps2:
                        res.append(q)
        return res

    c = -1
    for p in range(n):
        if lab[p] != -2:
            continue
        nb = vecinos(p)
        if len(nb) < min_pts:
            lab[p] = -1
            continue
        c += 1
        lab[p] = c
        cola = list(nb)
        s = 0
        while s < len(cola):
            q = cola[s]
            s += 1
            if lab[q] == -1:
                lab[q] = c
            if lab[q] != -2:
                continue
            lab[q] = c
            nbq = vecinos(q)
            if len(nbq) >= min_pts:
                cola.extend(nbq)
    return lab


foco = reportes[reportes["grupo"] == "Disturbios"]
xs = foco["x"].to_numpy()
ys = foco["y"].to_numpy()
print("Verificación DBSCAN (grilla propia vs scikit-learn):")
for eps, mp in [(300, 30), (200, 30), (500, 30), (300, 15)]:
    propio = dbscan_grilla(xs, ys, eps, mp)
    sklearn = DBSCAN(eps=eps, min_samples=mp).fit(np.c_[xs, ys]).labels_
    n_propio = len(set(propio)) - (1 if -1 in propio else 0)
    n_sklearn = len(set(sklearn)) - (1 if -1 in sklearn else 0)
    ok = "OK" if n_propio == n_sklearn else "DIFERENtE"
    print(
        f"  eps={eps} minPts={mp}: propio={n_propio} clusters, "
        f"sklearn={n_sklearn} clusters [{ok}]"
    )
