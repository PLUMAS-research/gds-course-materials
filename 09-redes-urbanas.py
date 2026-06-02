# %% [markdown]
# # Redes urbanas: topologia, centralidades y resiliencia
#
# La ciudad contiene varias redes superpuestas que el analisis tabular
# no captura: calles, transporte publico, ciclovias, viajes
# origen-destino. Modelarlas como grafos permite responder preguntas
# que las filas y columnas no permiten responder. Cuales son los
# corredores estructurantes? Que tan fragmentada esta la
# infraestructura ciclista? Que pasa si se cierra una avenida critica?
# Cuanta poblacion vive cerca del Transantiago?
#
# En esta clase trabajamos tres redes urbanas del Gran Santiago:
#   1. Calles vehiculares de Santiago Centro (OSM).
#   2. Ciclovias dedicadas del Gran Santiago (MINVU + OSM).
#   3. Red de paraderos del Transantiago multimodo (GTFS DTPM).
#
# Cada parte abre con una pregunta concreta y se cierra con una
# respuesta cuantitativa. La logica de procesamiento vive en
# `gdsutils.redes` para que el script se concentre en las preguntas.

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/redes-santiago.tgz")
descargar_datos(
    "https://dcc.uchile.cl/~egraells/gds-data/censo2024-asignacion-rm.tgz"
)

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from chiricoca.colors import categorical_color_legend
from chiricoca.config import setup_style
from chiricoca.geo.figures import (
    figure_from_geodataframe,
    small_multiples_from_geodataframe,
)
from chiricoca.geo.grid import h3_grid_from_bounds, h3_grid_from_ids
from chiricoca.maps import bubble_map, choropleth_map, heat_map
from chiricoca.networks.stats import summary as resumen_red
from shapely.geometry import LineString, Point, box

setup_style(dpi=128)

from gdsutils.redes import (
    construir_grafo_desde_lineas,
    geometrias_aristas_desde_shapes_gtfs,
    red_paraderos_desde_gtfs,
)

DIR_REDES = Path("data") / "redes-santiago"
DIR_ASIGNACION = Path("data") / "censo2024-asignacion-rm"
CRS_METRICO = "EPSG:32719"

# %% [markdown]
# ## PARTE 1. Como es la red de calles del centro de Santiago?
#
# Cargamos las calles vehiculares del Gran Santiago, recortamos al
# centro historico y construimos el grafo no dirigido. La funcion
# `construir_grafo_desde_lineas` parte cada LineString en sus
# intersecciones internas, no solo en sus endpoints. Sin este paso una
# avenida larga que cruza varias esquinas seguiria siendo una sola
# arista y la topologia quedaria mal modelada.

# %%
calles_completo = gpd.read_parquet(DIR_REDES / "calles.parquet").to_crs("EPSG:4326")
print(
    f"Calles del Gran Santiago: {len(calles_completo)} LineStrings, "
    f"CRS {calles_completo.crs.to_string()}"
)

# %% [markdown]
# ### Anatomia de los datos OSM
#
# OpenStreetMap modela las calles como **ways**: secuencias ordenadas
# de coordenadas que forman una linea, con atributos como `name` y
# `highway`. Al exportar a GeoDataFrame cada way queda como un
# LineString.
#
# Detalle clave para construir un grafo: cuando dos calles distintas
# se cruzan en una esquina, OSM guarda **dos LineStrings que pasan por
# el mismo punto**, pero ese punto **no aparece como un nodo separado**
# en ninguna de las dos ways. Por eso `construir_grafo_desde_lineas`
# detecta intersecciones internas antes de armar el grafo: si no, una
# avenida larga que cruza diez esquinas seguiria siendo una sola
# arista. Las columnas tipicas son `name`, `highway` (categoria vial),
# `geometry` (LineString); el resto son tags opcionales.

# %%
print("Columnas disponibles en el parquet:")
print(list(calles_completo.columns))
print("\nPrimeras filas (atributos no geometricos):")
cols_meta = [c for c in calles_completo.columns if c != "geometry"][:6]
print(calles_completo[cols_meta].head(5).to_string(index=False))

# Una way concreta: tipo de geometria y secuencia de vertices.
geom_ej = calles_completo.geometry.iloc[0]
# Para medir el largo en metros, proyectamos a UTM 19S antes (en
# EPSG:4326 los grados de lat y lon no miden lo mismo).
largo_m_ej = (
    gpd.GeoSeries([geom_ej], crs="EPSG:4326").to_crs(CRS_METRICO).length.iloc[0]
)
print(f"\nGeometria de la fila 0:")
print(f"  Tipo:      {geom_ej.geom_type}")
print(f"  Vertices:  {len(geom_ej.coords)}")
print(f"  Largo:     {largo_m_ej:.0f} m (proyectado a UTM 19S)")
print(f"  Primeros 3 vertices (lon, lat): {list(geom_ej.coords)[:3]}")

# %%
BBOX_SANTIAGO_CENTRO = (-70.68, -33.46, -70.62, -33.42)
# clip() recorta las LineStrings al bbox: las ways largas (avenidas)
# quedan partidas en el limite en vez de extenderse fuera. Sin esto,
# .cx[] conserva la geometria completa de cualquier feature que
# intersecte el bbox aunque la mayor parte quede fuera.
bbox_centro = box(*BBOX_SANTIAGO_CENTRO)
calles_centro = calles_completo.clip(bbox_centro)
calles_centro = calles_centro[
    calles_centro.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()
print(f"Calles en Santiago Centro: {len(calles_centro)}")

g_calles, nodos_calles, aristas_calles = construir_grafo_desde_lineas(
    calles_centro, tolerancia_metros=2.0
)

# %% [markdown]
# ### Anatomia del grafo
#
# `construir_grafo_desde_lineas` devuelve tres objetos sincronizados:
#
#   - `g_calles`: un `networkx.MultiGraph`. Es Multi porque dos
#     intersecciones pueden estar unidas por mas de una calle (e.g.,
#     una avenida y un pasaje). Cada nodo guarda `x`, `y` (sus
#     coordenadas); cada arista guarda `largo_m` (en metros, calculado
#     en UTM 19S) y `geometry` (el LineString del segmento).
#   - `nodos_calles`: GeoDataFrame con `node_id`, geometria `Point`.
#     Sirve para visualizar nodos y para hacer joins espaciales.
#   - `aristas_calles`: GeoDataFrame con `u`, `v` (los ids de los nodos
#     extremos), `largo_m` y la geometria del segmento.
#
# El truco de usar dos representaciones (NetworkX + GeoPandas) es
# estandar: NetworkX es eficiente para algoritmos de grafos pero no
# sabe nada de geometria; GeoPandas pinta mapas pero no calcula
# caminos cortos. Mantener ambas con los mismos ids permite ir y
# venir.

# %%
print(f"Tipo del grafo: {type(g_calles).__name__}")
print(
    f"Nodos: {g_calles.number_of_nodes()}, "
    f"Aristas: {g_calles.number_of_edges()}"
)
print("\nPrimeros 3 nodos con sus atributos (formato NetworkX):")
for n, datos in list(g_calles.nodes(data=True))[:3]:
    print(f"  node_id={n}: {datos}")

print("\nPrimeras 3 aristas con sus atributos (sin la geometria):")
for u, v, datos in list(g_calles.edges(data=True))[:3]:
    sin_geom = {k: round(v, 2) if isinstance(v, float) else v
                for k, v in datos.items() if k != "geometry"}
    print(f"  ({u} -- {v}): {sin_geom}")

print("\nGeoDataFrame de nodos (head):")
print(nodos_calles.head(3).to_string(index=False))

print("\nGeoDataFrame de aristas (head, sin geometry):")
print(
    aristas_calles.drop(columns="geometry").head(3).to_string(index=False)
)

# %%
# Resumen topologico con chiricoca.networks.stats.summary.
descriptivos = resumen_red(nx.Graph(g_calles))
print("Resumen de la red de calles (Santiago Centro):")
for k, v in descriptivos.items():
    if isinstance(v, float):
        print(f"  {k:30s} {v:.4f}")
    else:
        print(f"  {k:30s} {v}")

# %%
fig, ax = figure_from_geodataframe(aristas_calles, height=8)
aristas_calles.plot(ax=ax, color="black", linewidth=0.4)
nodos_calles.plot(ax=ax, color="firebrick", markersize=2, alpha=0.5)
ax.set_axis_off()
ax.set_title("Red de calles vehiculares: Santiago Centro")
fig.savefig("images/09-red-calles.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 2. Se parece a una grilla 4-regular o tiene asimetrias?
#
# La red de calles ocupa el espacio del damero: la teoria predice
# que el grado tipico es 3 (T) o 4 (cruz). Para contrastarlo
# comparamos la distribucion empirica con la de un grafo aleatorio
# Erdős-Rényi de igual tamaño (mismos nodos y aristas), que NO tiene
# ninguna restriccion espacial. Si las dos distribuciones son
# parecidas, la geografia no influye; si son distintas, la planaridad
# concentra el grado donde la teoria predice.

# %%
grados = np.array([d for _, d in g_calles.degree()])
n_nodos = g_calles.number_of_nodes()
n_aristas = g_calles.number_of_edges()

# Grafo aleatorio Erdős-Rényi con los mismos n y m.
rng = np.random.default_rng(42)
g_er = nx.gnm_random_graph(n_nodos, n_aristas, seed=42)
grados_er = np.array([d for _, d in g_er.degree()])

tabla_comp = pd.DataFrame(
    {
        "calles": pd.Series(grados).describe(),
        "aleatorio (ER)": pd.Series(grados_er).describe(),
    }
).round(2)
print("Distribucion del grado:")
print(tabla_comp)
print("\nFraccion de nodos con grado en {3, 4}:")
print(
    f"  calles:     {((grados >= 3) & (grados <= 4)).mean():.2%}"
    f"  (esperado para red planar tipo damero)"
)
print(f"  aleatorio:  {((grados_er >= 3) & (grados_er <= 4)).mean():.2%}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
bins = np.arange(1, max(grados.max(), grados_er.max()) + 2) - 0.5
axes[0].hist(grados, bins=bins, edgecolor="white", color="steelblue")
axes[0].set_title("Calles (planar)")
axes[0].set_xlabel("Grado")
axes[0].set_ylabel("Nodos")
axes[1].hist(grados_er, bins=bins, edgecolor="white", color="gray")
axes[1].set_title(f"Aleatorio Erdős-Rényi (n={n_nodos}, m={n_aristas})")
axes[1].set_xlabel("Grado")
fig.savefig("images/09-grado-comparacion.png", dpi=128, bbox_inches="tight")
plt.show()
print(
    "Respuesta: la red de calles concentra mas del 80% de sus nodos en grado"
)
print("3 o 4 (esquinas en T o en cruz). El grafo aleatorio del mismo tamaño")
print("dispersa el grado en una Poisson centrada en el grado medio, sin")
print("preferencia geometrica. La planaridad si deja huella en la topologia.")

# %% [markdown]
# ## PARTE 3. Cuanto se desvia el camino mas corto de la linea recta?
#
# Dijkstra encuentra el camino mas corto entre dos nodos en una red
# ponderada por largo. La razon entre el largo del camino y la
# distancia euclidiana entre origen y destino se llama indice de
# circuidad. Una grilla perfecta tiene circuidad cercana a 1.27
# (camino taxista en damero); valores mayores indican obstaculos o
# desvios.
#
# Para mostrar el rango de variacion comparamos cinco pares de
# puntos icónicos del centro:
#
#   - Damero ideal en eje (Alameda E-W): muy baja circuidad.
#   - Trayecto N-S a traves de la grilla.
#   - Diagonal larga NW-SE que cruza la trama.
#   - Recorrido que rodea el cerro Santa Lucia.
#   - Trayecto corto perpendicular a la avenida principal.

# %%
componentes = list(nx.connected_components(g_calles))
gcc = max(componentes, key=len)
g_gcc = g_calles.subgraph(gcc).copy()
nodos_gcc = nodos_calles[nodos_calles["node_id"].isin(gcc)].copy()


def nodo_mas_cercano_a(lon: float, lat: float) -> int:
    distancias = np.hypot(
        nodos_gcc.geometry.x - lon,
        nodos_gcc.geometry.y - lat,
    )
    return int(nodos_gcc["node_id"].iloc[distancias.values.argmin()])


# Lugares iconicos del centro (lon, lat). Verificados en el bbox de
# Santiago Centro (-70.68, -33.46, -70.62, -33.42).
LUGARES = {
    "Plaza de Armas":    (-70.6506, -33.4378),
    "Plaza Italia":      (-70.6336, -33.4368),
    "Plaza Brasil":      (-70.6678, -33.4421),
    "Estacion Mapocho":  (-70.6535, -33.4326),
    "Parque Almagro":    (-70.6480, -33.4540),
    "Bellas Artes":      (-70.6390, -33.4360),
    "Bellavista (PIo Nono)": (-70.6360, -33.4320),
}

# Pares O-D pensados para producir circuidades distintas.
PARES_OD = [
    ("Plaza de Armas",    "Plaza Italia",     "Alameda E-W"),
    ("Estacion Mapocho",  "Parque Almagro",   "Grilla N-S"),
    ("Plaza Brasil",      "Parque Almagro",   "Diagonal NW-SE"),
    ("Bellavista (PIo Nono)", "Bellas Artes", "Atravesando rio + cerro"),
    ("Plaza Brasil",      "Plaza Italia",     "Trayecto largo W-E"),
]

# Calculamos cada camino y reunimos los resultados en una tabla.
caminos_info = []
caminos_geom = []
for origen_nombre, destino_nombre, etiqueta in PARES_OD:
    o_lon, o_lat = LUGARES[origen_nombre]
    d_lon, d_lat = LUGARES[destino_nombre]
    o_id = nodo_mas_cercano_a(o_lon, o_lat)
    d_id = nodo_mas_cercano_a(d_lon, d_lat)
    if o_id == d_id or not nx.has_path(g_gcc, o_id, d_id):
        continue
    camino = nx.shortest_path(g_gcc, o_id, d_id, weight="largo_m")
    largo = nx.shortest_path_length(g_gcc, o_id, d_id, weight="largo_m")
    puntos = gpd.GeoSeries(
        [
            nodos_gcc[nodos_gcc["node_id"] == o_id].geometry.iloc[0],
            nodos_gcc[nodos_gcc["node_id"] == d_id].geometry.iloc[0],
        ],
        crs=nodos_gcc.crs,
    ).to_crs(CRS_METRICO)
    recta = puntos.iloc[0].distance(puntos.iloc[1])
    circuidad = largo / recta if recta > 0 else float("nan")

    aristas_geoms = []
    for u, v in zip(camino[:-1], camino[1:]):
        datos = g_gcc.get_edge_data(u, v)
        arista = min(datos.values(), key=lambda d: d["largo_m"])
        aristas_geoms.append(arista["geometry"])

    caminos_info.append(
        {
            "origen": origen_nombre,
            "destino": destino_nombre,
            "etiqueta": etiqueta,
            "largo_m": largo,
            "recta_m": recta,
            "circuidad": circuidad,
            "nodos": len(camino),
        }
    )
    caminos_geom.append(
        {
            "etiqueta": etiqueta,
            "segmentos": aristas_geoms,
            "o_id": o_id,
            "d_id": d_id,
        }
    )

tabla_caminos = pd.DataFrame(caminos_info)
print("Comparacion de caminos mas cortos en Santiago Centro:")
print(tabla_caminos.round(2).to_string(index=False))
print(
    "\nCircuidad: 1.00 = linea recta perfecta. ~1.27 = damero ideal. "
    "1.5+ = desvios fuertes."
)

# %%
# Mapa con todos los caminos superpuestos sobre la red. Cada camino
# en un color distinto, mas la linea recta correspondiente como
# guia.
colores_caminos = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
for fila, color in zip(caminos_geom, colores_caminos):
    fila["color"] = color
info_por_etiqueta = {f["etiqueta"]: f for f in caminos_info}

fig, ax = figure_from_geodataframe(aristas_calles, height=8)
aristas_calles.plot(ax=ax, color="lightgray", linewidth=0.3)
for fila in caminos_geom:
    geom_recta = LineString(
        [
            nodos_gcc[nodos_gcc["node_id"] == fila["o_id"]].geometry.iloc[0],
            nodos_gcc[nodos_gcc["node_id"] == fila["d_id"]].geometry.iloc[0],
        ]
    )
    gpd.GeoSeries([geom_recta], crs=aristas_calles.crs).plot(
        ax=ax, color=fila["color"], linewidth=0.6, linestyle=":", alpha=0.7
    )
    gpd.GeoSeries(fila["segmentos"], crs=aristas_calles.crs).plot(
        ax=ax, color=fila["color"], linewidth=2.2, alpha=0.9
    )

# Puntos extremos: union de origenes y destinos de los pares calculados.
puntos_clave = set()
for fila in caminos_info:
    puntos_clave.add(fila["origen"])
    puntos_clave.add(fila["destino"])
puntos_gdf = gpd.GeoDataFrame(
    {"nombre": list(puntos_clave)},
    geometry=[Point(*LUGARES[n]) for n in puntos_clave],
    crs="EPSG:4326",
).to_crs(aristas_calles.crs)
puntos_gdf.plot(ax=ax, color="black", markersize=60, zorder=5)

# Leyenda: etiqueta y circuidad por camino.
etiquetas = [
    f"{fila['etiqueta']} (circuidad {info_por_etiqueta[fila['etiqueta']]['circuidad']:.2f})"
    for fila in caminos_geom
]
categorical_color_legend(
    ax,
    dict(zip(etiquetas, [f["color"] for f in caminos_geom])),
    loc="lower left",
)
ax.set_axis_off()
ax.set_title(
    "Cinco caminos mas cortos en Santiago Centro "
    "(linea continua = camino, punteada = linea recta)"
)
fig.savefig("images/09-camino-mas-corto.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 4. Cuales son los corredores estructurantes?
#
# Cuatro centralidades capturan intuiciones distintas:
#
#   - Grado: cuantos vecinos tiene un nodo (hubs locales).
#   - Cercania: inverso de la distancia promedio (accesibilidad).
#   - Intermediacion (betweenness): fraccion de caminos cortos que
#     pasan por el nodo (cuellos de botella).
#   - PageRank: importancia recursiva.
#
# La intermediacion de aristas (edge betweenness) identifica los
# corredores que estructuran el flujo en la red. La intermediacion de
# nodos identifica intersecciones criticas; la agregamos a una grilla
# H3 para verla como continuo, mas legible que el scatter de nodos.

# %%
# Grafo simple para que las centralidades de NetworkX corran rapido.
g_simple = nx.Graph()
for u, v, datos in g_gcc.edges(data=True):
    if g_simple.has_edge(u, v):
        if datos["largo_m"] < g_simple[u][v]["largo_m"]:
            g_simple[u][v]["largo_m"] = datos["largo_m"]
    else:
        g_simple.add_edge(u, v, largo_m=datos["largo_m"])
for n, datos in g_gcc.nodes(data=True):
    g_simple.nodes[n].update(datos)

print(
    f"Grafo simple: {g_simple.number_of_nodes()} nodos, "
    f"{g_simple.number_of_edges()} aristas"
)
print("Calculando centralidades (puede tomar ~30s)")
cent_grado = nx.degree_centrality(g_simple)
cent_cercania = nx.closeness_centrality(g_simple, distance="largo_m")
cent_btw = nx.betweenness_centrality(g_simple, weight="largo_m", k=300, seed=42)
cent_pagerank = nx.pagerank(g_simple, weight="largo_m")

centralidades = pd.DataFrame(
    {
        "node_id": list(g_simple.nodes),
        "grado": [cent_grado[n] for n in g_simple.nodes],
        "cercania": [cent_cercania[n] for n in g_simple.nodes],
        "betweenness": [cent_btw[n] for n in g_simple.nodes],
        "pagerank": [cent_pagerank[n] for n in g_simple.nodes],
    }
)
nodos_cent = nodos_gcc.merge(centralidades, on="node_id")

# %%
# Para legibilidad, agregamos cada centralidad a una grilla H3-9
# (~150 m de lado) calculando el promedio de la centralidad de los
# nodos que caen dentro de cada hexagono.
hex_centro = h3_grid_from_bounds(BBOX_SANTIAGO_CENTRO, grid_level=9).reset_index()
join = gpd.sjoin(nodos_cent, hex_centro, how="inner", predicate="within")
agregadas = (
    join.groupby("h3_cell_id")[["grado", "cercania", "betweenness", "pagerank"]]
    .mean()
    .reset_index()
)
hex_centralidad = hex_centro.merge(agregadas, on="h3_cell_id").dropna()

cols = ["grado", "cercania", "betweenness", "pagerank"]
fig, axes = small_multiples_from_geodataframe(
    hex_centralidad, n_variables=4, col_wrap=2, height=4
)
for ax, col in zip(axes, cols):
    aristas_calles.plot(ax=ax, color="lightgray", linewidth=0.25, zorder=5)
    choropleth_map(
        hex_centralidad,
        col,
        ax=ax,
        k=6,
        palette="viridis",
        binning="fisher_jenks",
        edgecolor="none",
        alpha=0.85,
    )
    ax.set_title(col)
fig.savefig("images/09-centralidades.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Edge betweenness: las aristas con mas caminos cortos que pasan por
# ellas son los corredores estructurantes. Pintamos todas las aristas
# con color y ancho proporcional al valor.
print("Calculando edge betweenness (red de calles)")
eb_calles = nx.edge_betweenness_centrality(
    g_simple, weight="largo_m", k=300, seed=42
)


def aristas_con_centralidad(grafo, centralidad, crs, geom_lookup=None):
    """Convierte un dict {(u,v): valor} en GeoDataFrame de aristas."""
    nodos_xy = {n: (d["x"], d["y"]) for n, d in grafo.nodes(data=True)}
    filas = []
    geoms = []
    for (u, v), valor in centralidad.items():
        if u not in nodos_xy or v not in nodos_xy:
            continue
        geom = None
        if geom_lookup is not None:
            geom = geom_lookup.get((u, v)) or geom_lookup.get((v, u))
        if geom is None:
            geom = LineString([nodos_xy[u], nodos_xy[v]])
        filas.append({"u": u, "v": v, "centralidad": float(valor)})
        geoms.append(geom)
    return gpd.GeoDataFrame(filas, geometry=geoms, crs=crs)


def graficar_edge_centralidad(
    aristas_gdf, columna, titulo, ruta_salida, fondo=None, cmap="plasma"
):
    fig, ax = figure_from_geodataframe(aristas_gdf, height=8)
    if fondo is not None:
        fondo.plot(ax=ax, color="lightgray", linewidth=0.2)
    ordenadas = aristas_gdf.sort_values(columna)
    vmax = ordenadas[columna].max()
    anchos = 0.3 + 4.0 * ordenadas[columna] / (vmax if vmax > 0 else 1)
    ordenadas.plot(
        ax=ax,
        column=columna,
        cmap=cmap,
        linewidth=anchos.values,
        legend=True,
        legend_kwds={"shrink": 0.6, "label": columna},
    )
    ax.set_axis_off()
    ax.set_title(titulo)
    fig.savefig(ruta_salida, dpi=128, bbox_inches="tight")
    plt.show()


calles_centralidad = aristas_con_centralidad(
    g_simple, eb_calles, aristas_calles.crs
)
graficar_edge_centralidad(
    calles_centralidad,
    "centralidad",
    "Edge betweenness: red de calles (Santiago Centro)",
    "images/09-edge-betweenness-calles.png",
    fondo=aristas_calles,
)

# %%
# Top 5 corredores agrupados por nombre de calle (cuando el atributo
# existe en la cartografia OSM). Asi vemos que una avenida larga
# aparece como muchas aristas con valores altos, no como una sola
# entrada.
columna_nombre = None
for cand in ("name", "Name", "nombre"):
    if cand in aristas_calles.columns:
        columna_nombre = cand
        break

if columna_nombre is not None:
    aristas_eb = calles_centralidad.merge(
        aristas_calles[["u", "v", columna_nombre]], on=["u", "v"], how="left"
    )
    top_corredores = (
        aristas_eb.dropna(subset=[columna_nombre])
        .groupby(columna_nombre)["centralidad"]
        .agg(["mean", "max", "count"])
        .sort_values("mean", ascending=False)
        .head(10)
    )
    print("Top 10 corredores por edge betweenness promedio:")
    print(top_corredores.round(4))
else:
    print(
        "La cartografia no incluye nombres de calle: se reporta solo el "
        "mapa de edge betweenness."
    )

# %% [markdown]
# ## PARTE 5. Cuanto crece el camino promedio si removemos los nodos criticos?
#
# Comparamos cuatro estrategias de remocion. La asimetria entre
# remocion aleatoria y las dirigidas es el diagnostico clasico de
# resiliencia; el contraste entre las tres dirigidas (grado,
# cercania, betweenness) muestra que la nocion de "nodo critico"
# depende del criterio:
#
#   - **Aleatoria**: una falla al azar (accidente, semaforo caido).
#   - **Top grado**: ataca primero los hubs locales (intersecciones
#     mas conectadas).
#   - **Top cercania**: ataca primero los nodos mas accesibles.
#   - **Top betweenness**: ataca primero los cuellos de botella.
#
# Ademas de la curva clasica de tamano relativo de la GCC, medimos
# cuanto crece el camino promedio entre 200 pares al azar tras
# remover el 5% segun cada criterio.

# %%
def evolucion_gcc(grafo, orden_nodos, fracciones):
    tamanos = []
    n0 = grafo.number_of_nodes()
    for frac in fracciones:
        n_remover = int(frac * n0)
        g_temp = grafo.copy()
        g_temp.remove_nodes_from(orden_nodos[:n_remover])
        if g_temp.number_of_nodes() == 0:
            tamanos.append(0)
            continue
        gcc_temp = max(nx.connected_components(g_temp), key=len)
        tamanos.append(len(gcc_temp) / n0)
    return tamanos


fracciones = np.linspace(0, 0.30, 16)
orden_aleatorio = list(g_simple.nodes)
rng.shuffle(orden_aleatorio)

ordenes = {
    "Aleatoria":            (orden_aleatorio, "steelblue", "o"),
    "Top grado":            (
        sorted(g_simple.nodes, key=lambda n: cent_grado[n], reverse=True),
        "#2ca02c",
        "^",
    ),
    "Top cercania":         (
        sorted(g_simple.nodes, key=lambda n: cent_cercania[n], reverse=True),
        "#ff7f0e",
        "D",
    ),
    "Top betweenness":      (
        sorted(g_simple.nodes, key=lambda n: cent_btw[n], reverse=True),
        "firebrick",
        "s",
    ),
}
curvas_gcc = {
    nombre: evolucion_gcc(g_simple, orden, fracciones)
    for nombre, (orden, _, _) in ordenes.items()
}

fig, ax = plt.subplots(figsize=(8, 5.5))
for nombre, (orden, color, marcador) in ordenes.items():
    ax.plot(
        fracciones,
        curvas_gcc[nombre],
        marcador + "-",
        label=nombre,
        color=color,
    )
ax.set_xlabel("Fraccion de nodos removidos")
ax.set_ylabel("Tamano relativo de la GCC")
ax.set_title(
    "Resiliencia: tres criterios dirigidos vs falla aleatoria "
    "(red de calles Santiago Centro)"
)
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("images/09-resiliencia.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Cuanto crece el camino promedio entre 200 pares al azar tras
# remover el 5% segun cada criterio. La metrica complementa la curva
# de GCC: la GCC mide conectividad, el camino promedio mide costo de
# moverse cuando hay conectividad.
def camino_promedio_pares(grafo, pares):
    largos = []
    for s, t in pares:
        if s == t or not nx.has_path(grafo, s, t):
            continue
        largos.append(nx.shortest_path_length(grafo, s, t, weight="largo_m"))
    return np.mean(largos) if largos else float("nan")


nodos_lista = list(g_simple.nodes)
sample_idx = rng.choice(len(nodos_lista), size=400, replace=True)
pares = list(
    zip(
        [nodos_lista[i] for i in sample_idx[:200]],
        [nodos_lista[i] for i in sample_idx[200:]],
    )
)
prom_base = camino_promedio_pares(g_simple, pares)
n_top = int(0.05 * g_simple.number_of_nodes())

filas_camino = [{"criterio": "Sin remocion", "n_removidos": 0,
                 "camino_promedio_m": prom_base, "incremento": 0.0,
                 "pares_validos": len(pares)}]
for nombre, (orden, _, _) in ordenes.items():
    top_remover = orden[:n_top]
    g_temp = g_simple.copy()
    g_temp.remove_nodes_from(top_remover)
    pares_validos = [
        (s, t) for s, t in pares if s in g_temp and t in g_temp
    ]
    prom = camino_promedio_pares(g_temp, pares_validos)
    filas_camino.append(
        {
            "criterio": nombre,
            "n_removidos": n_top,
            "camino_promedio_m": prom,
            "incremento": prom / prom_base - 1,
            "pares_validos": len(pares_validos),
        }
    )

tabla_resiliencia = pd.DataFrame(filas_camino)
print(
    f"Camino promedio entre 200 pares al azar, removiendo el 5% segun "
    f"criterio ({n_top} nodos):"
)
print(tabla_resiliencia.round(2).to_string(index=False))

# %% [markdown]
# ## PARTE 6. Que tan fragmentada y cuanta poblacion cubre la red ciclista?
#
# La red de calles del Gran Santiago tiene una sola componente gigante
# que cubre toda la ciudad. La red de ciclovias no. Para
# diagnosticarlo:
#
#   1. Contrastamos dos fuentes (MINVU oficial vs OSM colaborativa).
#   2. Hacemos un sweep de tolerancia de snap para separar
#      fragmentacion real de ruido cartografico.
#   3. Medimos cobertura: que fraccion de la poblacion del Gran
#      Santiago vive dentro de 500 m de la componente gigante MINVU.
#
# Fuente MINVU: https://geoportal.cl/geoportal/catalog/36392/Ciclovias

# %%
BBOX_GRAN_STGO = (-70.85, -33.65, -70.45, -33.30)

ciclovias_minvu = gpd.read_parquet(DIR_REDES / "ciclovias-minvu.parquet").to_crs(
    "EPSG:4326"
)
ciclovias_osm = gpd.read_parquet(DIR_REDES / "ciclovias-osm.parquet").to_crs(
    "EPSG:4326"
)
bbox_gran = box(*BBOX_GRAN_STGO)
ciclovias_minvu = ciclovias_minvu.clip(bbox_gran)
ciclovias_minvu = ciclovias_minvu[
    ciclovias_minvu.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()
ciclovias_osm = ciclovias_osm.clip(bbox_gran)
ciclovias_osm = ciclovias_osm[
    ciclovias_osm.geometry.geom_type.isin(["LineString", "MultiLineString"])
].copy()
print(f"Ciclovias MINVU (existentes, Gran Santiago): {len(ciclovias_minvu)}")
print(f"Ciclovias OSM (highway=cycleway): {len(ciclovias_osm)}")

# %%
# Contraste lado a lado MINVU vs OSM con small_multiples (mismo aspect
# y figsize que el resto de mapas del curso).
ciclovias_fuentes = pd.concat(
    [
        ciclovias_minvu.assign(fuente="MINVU"),
        ciclovias_osm.assign(fuente="OSM"),
    ],
    ignore_index=True,
)
fig, axes = small_multiples_from_geodataframe(
    ciclovias_fuentes, n_variables=2, col_wrap=2, height=6
)
ciclovias_minvu.plot(ax=axes[0], color="firebrick", linewidth=1.0)
axes[0].set_title(f"MINVU oficial ({len(ciclovias_minvu)} segmentos)")
ciclovias_osm.plot(ax=axes[1], color="steelblue", linewidth=1.0)
axes[1].set_title(f"OSM colaborativo ({len(ciclovias_osm)} segmentos)")
fig.savefig("images/09-ciclovias-fuentes.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Sweep de tolerancia de snap (MINVU). Las cartografias oficiales
# suelen dibujar segmentos contiguos con endpoints separados por unos
# metros: si no compensamos, la red aparece mas fragmentada de lo que
# es. El salto entre tolerancias bajas y medias indica cuanta
# conexion se pierde por puro ruido.
print("Sensibilidad a la tolerancia de snap (MINVU):")
print(f"{'tol (m)':>8} | {'nodos':>6} | {'componentes':>11} | {'GCC %':>6}")
for tol in [2, 5, 10, 15, 20]:
    g_tmp, _, _ = construir_grafo_desde_lineas(
        ciclovias_minvu, tolerancia_metros=tol
    )
    comp = list(nx.connected_components(g_tmp))
    tam = sorted([len(c) for c in comp], reverse=True)
    print(
        f"{tol:>8} | {g_tmp.number_of_nodes():>6} | {len(comp):>11} | "
        f"{tam[0] / g_tmp.number_of_nodes() * 100:>5.1f}"
    )

TOL_CICLOVIAS = 10.0
g_cic, nodos_cic, aristas_cic = construir_grafo_desde_lineas(
    ciclovias_minvu, tolerancia_metros=TOL_CICLOVIAS
)
comp_cic = list(nx.connected_components(g_cic))
tamanos_cic = sorted([len(c) for c in comp_cic], reverse=True)
gcc_cic_set = max(comp_cic, key=len)
print(
    f"\nGrafo MINVU (tol={TOL_CICLOVIAS:.0f} m): "
    f"{g_cic.number_of_nodes()} nodos, {g_cic.number_of_edges()} aristas, "
    f"{len(comp_cic)} componentes"
)
print(f"GCC: {tamanos_cic[0]} nodos ({tamanos_cic[0] / g_cic.number_of_nodes():.1%})")

# %%
# Mapa coloreado por componente para ver la fragmentacion.
id_a_componente = {n: idx for idx, comp in enumerate(comp_cic) for n in comp}
aristas_cic["componente"] = aristas_cic["u"].map(id_a_componente)
aristas_cic["es_gcc"] = aristas_cic["u"].isin(gcc_cic_set)

fig, ax = figure_from_geodataframe(aristas_cic, height=7)
aristas_cic.plot(
    ax=ax, column="componente", cmap="tab20", linewidth=1.2, legend=False
)
ax.set_axis_off()
ax.set_title(
    f"Ciclovias MINVU (existentes, Gran Santiago) por componente "
    f"({len(comp_cic)} componentes)"
)
fig.savefig("images/09-ciclovias-componentes.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Cobertura poblacional: que fraccion de los habitantes del Gran
# Santiago vive a ≤ 500 m de la GCC ciclista? Usamos el resultado de
# la clase 08 (asignacion censal H3-8) como fuente de poblacion.
asig_df = pd.read_parquet(DIR_ASIGNACION / "asignaciones-h3-8.parquet")
asig_df["n_personas"] = asig_df["n_hombres"] + asig_df["n_mujeres"]
asignacion = h3_grid_from_ids(asig_df["h3_cell_id"].tolist()).reset_index()
asignacion = asignacion.merge(asig_df, on="h3_cell_id")
asignacion = asignacion.cx[
    BBOX_GRAN_STGO[0] : BBOX_GRAN_STGO[2],
    BBOX_GRAN_STGO[1] : BBOX_GRAN_STGO[3],
].copy()
print(f"Hexagonos H3-8 en el Gran Santiago: {len(asignacion)}")
print(f"Poblacion total: {asignacion['n_personas'].sum():,.0f}")

# Buffer 500 m en UTM 19S, alrededor de la GCC.
aristas_gcc = aristas_cic[aristas_cic["es_gcc"]].copy()
gcc_buffer = aristas_gcc.to_crs(CRS_METRICO).buffer(500).union_all()
asignacion_m = asignacion.to_crs(CRS_METRICO)
asignacion["dentro_buffer"] = (
    asignacion_m.geometry.representative_point().within(gcc_buffer)
)
pob_cubierta = asignacion.loc[asignacion["dentro_buffer"], "n_personas"].sum()
pob_total = asignacion["n_personas"].sum()
print(
    f"\nPoblacion a ≤ 500 m de la GCC ciclista MINVU: "
    f"{pob_cubierta:,.0f} de {pob_total:,.0f} ({pob_cubierta / pob_total:.1%})"
)

fig, ax = figure_from_geodataframe(asignacion, height=8)
asignacion.plot(
    ax=ax,
    column="dentro_buffer",
    categorical=True,
    cmap="Set2",
    linewidth=0,
    legend=True,
    legend_kwds={"title": "≤ 500 m GCC", "loc": "upper right"},
)
aristas_gcc.plot(ax=ax, color="black", linewidth=0.8)
ax.set_axis_off()
ax.set_title(
    f"Cobertura ciclista: {pob_cubierta / pob_total:.1%} de la poblacion "
    f"del Gran Santiago a ≤ 500 m de la GCC MINVU"
)
fig.savefig("images/09-ciclovias-cobertura.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Edge betweenness sobre la GCC ciclista para identificar tramos
# criticos dentro del subsistema mas grande.
g_cic_gcc = nx.Graph()
for u, v, d in g_cic.subgraph(gcc_cic_set).edges(data=True):
    if g_cic_gcc.has_edge(u, v):
        if d["largo_m"] < g_cic_gcc[u][v]["largo_m"]:
            g_cic_gcc[u][v]["largo_m"] = d["largo_m"]
    else:
        g_cic_gcc.add_edge(u, v, largo_m=d["largo_m"])
for n, d in g_cic.nodes(data=True):
    if n in gcc_cic_set:
        g_cic_gcc.nodes[n].update(d)

eb_cic = nx.edge_betweenness_centrality(g_cic_gcc, weight="largo_m")
cic_centralidad = aristas_con_centralidad(g_cic_gcc, eb_cic, ciclovias_minvu.crs)
graficar_edge_centralidad(
    cic_centralidad,
    "centralidad",
    "Edge betweenness: GCC de ciclovias MINVU",
    "images/09-edge-betweenness-ciclovias.png",
    fondo=ciclovias_minvu,
)

# %% [markdown]
# ## PARTE 7. Como es el sistema? Descriptivos del GTFS
#
# Antes de construir la red conviene mirar la estructura del GTFS:
# cuantos viajes hay por hora, cuantas paradas tiene una ruta tipica,
# cuantas rutas pasan por un paradero, y como se distribuye la oferta
# por comuna. Estos cuatro cortes resumen el sistema sin necesidad de
# grafos.

# %%
import urllib.request
import zipfile

from gdsutils.dtpm import RUTA_GTFS, URL_GTFS

if not RUTA_GTFS.exists():
    RUTA_GTFS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando GTFS DTPM: {URL_GTFS}")
    urllib.request.urlretrieve(URL_GTFS, RUTA_GTFS)

with zipfile.ZipFile(RUTA_GTFS) as z:
    stops = pd.read_csv(z.open("stops.txt"), dtype=str)
    stop_times = pd.read_csv(z.open("stop_times.txt"), dtype=str)
    trips = pd.read_csv(z.open("trips.txt"), dtype=str)
    routes = pd.read_csv(z.open("routes.txt"), dtype=str)
    shapes = pd.read_csv(z.open("shapes.txt"), dtype=str)
    frequencies = pd.read_csv(z.open("frequencies.txt"), dtype=str)

print(
    f"GTFS: {len(stops)} stops, {len(trips)} trips, {len(routes)} routes, "
    f"{shapes['shape_id'].nunique()} shapes"
)
print("\nTipos de servicio (route_type) en el GTFS DTPM:")
print(routes["route_type"].value_counts())

# %% [markdown]
# ### Anatomia del GTFS
#
# GTFS (General Transit Feed Specification) describe la oferta de
# transporte publico en un conjunto de archivos CSV. Las tablas que
# usamos en esta clase y como se conectan:
#
#   - `routes.txt`: una linea o servicio (e.g., "Linea 1" del metro, o
#     el recorrido 506 de bus). Campos: `route_id`, `route_short_name`,
#     `route_long_name`, `route_type` (en DTPM: 0=metrotren, 1=metro,
#     3=bus, sin tranvias).
#   - `trips.txt`: una corrida especifica de una ruta (la salida de las
#     06:15 de la 506 hacia el oeste). Campos: `trip_id`, `route_id`,
#     `shape_id`, `direction_id`.
#   - `shapes.txt`: secuencia de puntos lat/lon que dibuja el recorrido
#     fisico de un trip. Indexada por `shape_id` + `shape_pt_sequence`.
#   - `stops.txt`: paraderos (`stop_id`, `stop_lat`, `stop_lon`,
#     `stop_name`, `stop_code`).
#   - `stop_times.txt`: la tabla mas grande. Por cada trip dice que
#     paraderos toca y cuando: (`trip_id`, `stop_id`, `stop_sequence`,
#     `arrival_time`, `departure_time`). Es lo que une trips con stops.
#   - `frequencies.txt` (opcional): cuando un servicio tiene horario
#     repetido (cada N minutos entre HH:MM y HH:MM), aqui se registra
#     el headway en lugar de generar miles de filas de stop_times.
#
# **Tip DTPM (importante):** los `arrival_time` y `departure_time` de
# DTPM son **relativos al inicio del trip** (todos arrancan en
# 0:00:00), no tiempos del reloj. Las horas absolutas viven en
# `frequencies.txt`. Por eso para contar viajes por hora del dia
# usamos frequencies, no stop_times.
#
# Esquema de relaciones:
#
#     routes <--- trips <--- stop_times ---> stops
#                  |
#                  +--> shapes
#                  +--> frequencies

# %%
print("routes.txt (head):")
print(routes.head(3)[
    ["route_id", "route_short_name", "route_long_name", "route_type"]
].to_string(index=False))

print("\ntrips.txt (head):")
print(trips.head(3)[
    [c for c in ["route_id", "trip_id", "shape_id", "direction_id"]
     if c in trips.columns]
].to_string(index=False))

print("\nstops.txt (head):")
print(stops.head(3)[
    ["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"]
].to_string(index=False))

print("\nstop_times.txt (head, primeras paradas del primer trip):")
print(stop_times.head(5)[
    ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
].to_string(index=False))

print("\nshapes.txt (head):")
print(shapes.head(3).to_string(index=False))

print("\nfrequencies.txt (head):")
print(frequencies.head(3).to_string(index=False))

# %%
# Modos como categoria reutilizable en los descriptivos.
MAPEO_MODO = {"0": "metrotren", "1": "metro", "3": "bus"}
COLORES_MODO = {"bus": "#3776c1", "metro": "#e83a52", "metrotren": "#2e8b57"}

routes_validos = routes[routes["route_type"].isin(MAPEO_MODO)].copy()
routes_validos["modo"] = routes_validos["route_type"].map(MAPEO_MODO)
trips_modo = trips.merge(
    routes_validos[["route_id", "modo"]], on="route_id", how="inner"
)
st_modo = stop_times.merge(
    trips_modo[["trip_id", "route_id", "modo"]], on="trip_id"
)

# %% [markdown]
# ### 7a. Viajes por hora del dia
#
# El GTFS de DTPM publica las horas en `stop_times.txt` relativas al
# inicio de cada trip (todos los trips empiezan en 0:00:00). Los
# horarios reales viven en `frequencies.txt`, que define ventanas de
# operacion con su intervalo entre buses (headway). Para estimar el
# numero de viajes por hora distribuimos uniformemente los viajes de
# cada ventana sobre las horas que cubre.

# %%
def segundos_desde_medianoche(t: str) -> float:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


frequencies["start_s"] = frequencies["start_time"].apply(segundos_desde_medianoche)
frequencies["end_s"] = frequencies["end_time"].apply(segundos_desde_medianoche)
frequencies["headway"] = frequencies["headway_secs"].astype(int)
frequencies["n_viajes"] = (
    (frequencies["end_s"] - frequencies["start_s"]) / frequencies["headway"]
)
# Adjuntar modo del trip.
frequencies = frequencies.merge(
    trips_modo[["trip_id", "modo"]], on="trip_id", how="inner"
)
print(f"Ventanas en frequencies.txt: {len(frequencies)} "
      f"({frequencies['trip_id'].nunique()} trips, "
      f"{frequencies['modo'].nunique()} modos)")

# Distribuir viajes por hora del dia (0-23). Cada ventana asigna
# n_viajes / horas_cubiertas a cada hora que toca, sumando sobre
# todas las ventanas y modos.
viajes_hora = np.zeros((24, 3))
modos_orden = ["bus", "metro", "metrotren"]
idx_modo = {m: i for i, m in enumerate(modos_orden)}
for fila in frequencies.itertuples(index=False):
    h_ini = int(fila.start_s // 3600)
    h_fin = int(np.ceil(fila.end_s / 3600))
    horas = list(range(h_ini, h_fin))
    if not horas:
        continue
    por_hora = fila.n_viajes / len(horas)
    j = idx_modo.get(fila.modo)
    if j is None:
        continue
    for h in horas:
        viajes_hora[h % 24, j] += por_hora

print("Viajes por hora del dia (top 5 horas):")
total_hora = viajes_hora.sum(axis=1)
for h in np.argsort(total_hora)[::-1][:5]:
    print(f"  {h:02d}:00 -> {total_hora[h]:.0f} viajes/h")

fig, ax = plt.subplots(figsize=(9, 4.5))
horas_idx = np.arange(24)
bottom = np.zeros(24)
for j, modo in enumerate(modos_orden):
    valores = viajes_hora[:, j]
    ax.bar(
        horas_idx,
        valores,
        bottom=bottom,
        color=COLORES_MODO[modo],
        label=modo,
        edgecolor="white",
        linewidth=0.5,
    )
    bottom += valores
ax.set_xticks(horas_idx)
ax.set_xlabel("Hora del dia")
ax.set_ylabel("Viajes programados por hora")
ax.set_title("Viajes por hora del dia (frequencies.txt), apilado por modo")
ax.legend()
fig.savefig("images/09-gtfs-hora.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 7b. Distribucion de paradas por ruta
#
# Por cada ruta calculamos el numero de paradas unicas que toca en
# alguno de sus trips. Las rutas troncales tienen 60-80 paradas; los
# alimentadores 20-30. Las lineas de metro son cortas en numero de
# paradas pero largas en geografia.

# %%
paradas_por_ruta = (
    st_modo[["route_id", "stop_id", "modo"]]
    .drop_duplicates()
    .groupby(["route_id", "modo"])
    .size()
    .reset_index(name="n_paradas")
)
# Adjuntar nombre de ruta para el top.
paradas_por_ruta = paradas_por_ruta.merge(
    routes[["route_id", "route_short_name", "route_long_name"]], on="route_id"
)
print("Paradas por ruta, resumen por modo:")
print(paradas_por_ruta.groupby("modo")["n_paradas"].describe().round(1))
print("\nTop 10 rutas por numero de paradas:")
print(
    paradas_por_ruta.nlargest(10, "n_paradas")[
        ["modo", "route_short_name", "route_long_name", "n_paradas"]
    ].to_string(index=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(0, paradas_por_ruta["n_paradas"].max() + 5, 5)
for modo, color in COLORES_MODO.items():
    sub = paradas_por_ruta[paradas_por_ruta["modo"] == modo]["n_paradas"]
    if sub.empty:
        continue
    ax.hist(
        sub, bins=bins, color=color, alpha=0.7, label=f"{modo} (n={len(sub)})"
    )
ax.set_xlabel("Paradas unicas por ruta")
ax.set_ylabel("Rutas")
ax.set_title("Distribucion del numero de paradas por ruta")
ax.legend()
fig.savefig("images/09-gtfs-paradas-ruta.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 7c. Distribucion de rutas por paradero
#
# Para cada paradero, cuantas rutas distintas pasan por el. La cola
# larga muestra que pocos paraderos concentran muchas rutas (hubs
# multi-linea) mientras la mayoria sirve a una o dos.

# %%
rutas_por_paradero = (
    st_modo[["stop_id", "route_id", "modo"]]
    .drop_duplicates()
    .groupby(["stop_id", "modo"])
    .size()
    .reset_index(name="n_rutas")
)
# Si un paradero sirve varios modos, se queda con el de mayor
# prioridad para el descriptivo.
prioridad_modo = {"metro": 0, "metrotren": 1, "bus": 2}
rutas_por_paradero["__prio__"] = rutas_por_paradero["modo"].map(prioridad_modo)
modo_principal = (
    rutas_por_paradero.sort_values("__prio__")
    .groupby("stop_id")
    .agg({"modo": "first", "n_rutas": "sum"})
    .reset_index()
)
print("Rutas por paradero, resumen por modo:")
print(modo_principal.groupby("modo")["n_rutas"].describe().round(1))

fig, ax = plt.subplots(figsize=(8, 5))
bins_rutas = np.arange(1, modo_principal["n_rutas"].max() + 2)
for modo, color in COLORES_MODO.items():
    sub = modo_principal[modo_principal["modo"] == modo]["n_rutas"]
    if sub.empty:
        continue
    ax.hist(
        sub, bins=bins_rutas, color=color, alpha=0.7,
        label=f"{modo} (n={len(sub)})"
    )
ax.set_yscale("log")
ax.set_xlabel("Rutas distintas que pasan por el paradero")
ax.set_ylabel("Paraderos (escala log)")
ax.set_title("Distribucion de rutas por paradero")
ax.legend()
fig.savefig("images/09-gtfs-rutas-paradero.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 7d. Cobertura de paraderos por comuna
#
# Densidad de paraderos por km^2 por comuna del Gran Santiago.
# Muestra la desigualdad de oferta entre el centro y la periferia.

# %%
comunas_pais = gpd.read_parquet(
    "data/censo2024-cartografia/Cartografia_censo2024_Pais_Comunal.parquet"
)
# La cartografia censal usa columna SHAPE como geometry y CRS 4674.
if "SHAPE" in comunas_pais.columns and "geometry" not in comunas_pais.columns:
    comunas_pais = comunas_pais.set_geometry("SHAPE").rename_geometry("geometry")
comunas_pais = comunas_pais.to_crs("EPSG:4326")
# Filtramos a la RM y al bbox del Gran Santiago para excluir comunas
# rurales (Til Til, Lampa norte, San Pedro, etc.).
comunas_rm = comunas_pais[comunas_pais["COD_REGION"] == 13].copy()
comunas_rm = comunas_rm.clip(bbox_gran)
comunas_rm = comunas_rm[
    comunas_rm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()
comunas_rm["area_km2"] = comunas_rm.to_crs(CRS_METRICO).area / 1e6

paraderos_gdf = gpd.GeoDataFrame(
    stops[["stop_id"]],
    geometry=gpd.points_from_xy(
        pd.to_numeric(stops["stop_lon"], errors="coerce"),
        pd.to_numeric(stops["stop_lat"], errors="coerce"),
    ),
    crs="EPSG:4326",
).dropna(subset=["geometry"])
join_paraderos = gpd.sjoin(
    paraderos_gdf, comunas_rm[["CUT", "geometry"]], how="inner", predicate="within"
)
conteo_paraderos = (
    join_paraderos.groupby("CUT").size().reset_index(name="n_paraderos")
)
comunas_cob = comunas_rm.merge(conteo_paraderos, on="CUT", how="left")
comunas_cob["n_paraderos"] = comunas_cob["n_paraderos"].fillna(0)
comunas_cob["paraderos_km2"] = (
    comunas_cob["n_paraderos"] / comunas_cob["area_km2"]
)
print("Top 10 comunas por densidad de paraderos:")
print(
    comunas_cob.nlargest(10, "paraderos_km2")[
        ["COMUNA", "n_paraderos", "area_km2", "paraderos_km2"]
    ].round(2).to_string(index=False)
)

fig, ax = figure_from_geodataframe(comunas_cob, height=7)
choropleth_map(
    comunas_cob[comunas_cob["paraderos_km2"] > 0],
    "paraderos_km2",
    ax=ax,
    k=6,
    palette="viridis",
    binning="fisher_jenks",
    edgecolor="white",
    linewidth=0.5,
)
ax.set_axis_off()
ax.set_title("Densidad de paraderos por km$^2$ por comuna (Gran Santiago)")
fig.savefig("images/09-gtfs-cobertura-comuna.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 8. Que paraderos concentran el servicio del Transantiago?
#
# Pasamos del descriptivo al analisis de red. Construimos un grafo
# multimodal (bus, metro, metrotren) y respondemos tres preguntas:
#
#   1. Cuales paraderos concentran mas servicios (strength)?
#   2. Donde esta la cobertura mas densa?
#   3. A cuantos transbordos de Plaza de Armas esta cada paradero?

# %%
# Construimos la red multimodal. En el GTFS DTPM:
#   0 = metrotren, 1 = metro, 3 = bus.
modos_int = tuple(int(k) for k in MAPEO_MODO)
g_dtpm, nodos_dtpm, aristas_dtpm = red_paraderos_desde_gtfs(
    stops, stop_times, trips, routes, modos=modos_int
)
# Recortamos al Gran Santiago para evitar paraderos suburbanos
# distantes que distorsionan el aspect ratio de los mapas.
nodos_dtpm = nodos_dtpm.cx[
    BBOX_GRAN_STGO[0] : BBOX_GRAN_STGO[2],
    BBOX_GRAN_STGO[1] : BBOX_GRAN_STGO[3],
].copy()
aristas_dtpm = aristas_dtpm[
    aristas_dtpm["u_id"].isin(nodos_dtpm["stop_id"])
    & aristas_dtpm["v_id"].isin(nodos_dtpm["stop_id"])
].copy()
print(
    f"Red multimodal (recortada al Gran Santiago): "
    f"{len(nodos_dtpm)} paraderos, {len(aristas_dtpm)} aristas"
)

# Modo principal por paradero: prioridad metro > metrotren > bus.
routes_validos = routes[routes["route_type"].isin(MAPEO_MODO)]
trips_modo = trips.merge(routes_validos[["route_id", "route_type"]], on="route_id")
pares_paradero_modo = (
    stop_times.merge(trips_modo[["trip_id", "route_type"]], on="trip_id")[
        ["stop_id", "route_type"]
    ]
    .drop_duplicates()
)
prioridad = {"1": 0, "0": 1, "3": 2}
pares_paradero_modo["__prio__"] = pares_paradero_modo["route_type"].map(prioridad)
modo_por_paradero = (
    pares_paradero_modo.sort_values("__prio__")
    .groupby("stop_id")["route_type"]
    .first()
    .map(MAPEO_MODO)
)
nodos_dtpm["modo"] = nodos_dtpm["stop_id"].map(modo_por_paradero).fillna("bus")

# Strength por paradero: suma de servicios de sus aristas incidentes.
strength = {n: 0 for n in g_dtpm.nodes}
for u, v, d in g_dtpm.edges(data=True):
    strength[u] += d["servicios"]
    strength[v] += d["servicios"]
nodos_dtpm["strength"] = nodos_dtpm["stop_id"].map(strength).fillna(0).astype(int)

print("\nDistribucion de paraderos por modo:")
print(nodos_dtpm["modo"].value_counts())
print("\nTop 10 paraderos por strength:")
print(
    nodos_dtpm.nlargest(10, "strength")[["stop_id", "stop_name", "modo", "strength"]]
    .to_string(index=False)
)

# %%
# Bubble map de paraderos: tamaño = strength, color = modo. Pintamos
# primero los buses (densos pero homogeneos) y luego metrotren y metro
# para que los hubs estructurantes queden encima del fondo de buses.
fig, ax = figure_from_geodataframe(nodos_dtpm, height=8)
escalas = {"bus": 0.8, "metrotren": 6.0, "metro": 6.0}
orden_modos = ["bus", "metrotren", "metro"]
for modo in orden_modos:
    sub = nodos_dtpm[nodos_dtpm["modo"] == modo]
    if sub.empty:
        continue
    bubble_map(
        sub,
        size="strength",
        scale=escalas[modo],
        color=COLORES_MODO[modo],
        ax=ax,
        alpha=0.75 if modo == "bus" else 0.95,
        edgecolor="white",
        linewidth=0.2 if modo == "bus" else 0.4,
        add_legend=False,
    )
categorical_color_legend(ax, COLORES_MODO, loc="upper right")
ax.set_axis_off()
ax.set_title("Paraderos DTPM: tamaño = servicios incidentes, color = modo")
fig.savefig("images/09-paraderos-modo.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Heat map de cobertura: KDE sobre los paraderos pesados por
# strength. Las zonas calientes son las que concentran mas oferta de
# transporte publico.
fig, ax = figure_from_geodataframe(nodos_dtpm, height=8)
heat_map(
    nodos_dtpm,
    weight="strength",
    bandwidth=0.015,
    n_levels=8,
    palette="magma",
    ax=ax,
    alpha=0.85,
)
ax.set_axis_off()
ax.set_title(
    "Cobertura DTPM: KDE de paraderos ponderada por servicios incidentes"
)
fig.savefig("images/09-cobertura-dtpm.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Geometrias shape-based: en lugar de la linea recta entre dos
# paraderos consecutivos, proyectamos cada paradero sobre la shape de
# su trip y extraemos el substring entre paraderos. Asi las aristas
# siguen las calles reales.
shape_ids_bus = trips[
    trips["route_id"].isin(routes[routes["route_type"] == "3"]["route_id"])
]
stop_times_bus = stop_times.merge(shape_ids_bus[["trip_id"]], on="trip_id")
print(
    f"Calculando geometrias shape-based ({len(shape_ids_bus)} trips de bus, "
    f"~1-2 min)"
)
geoms_por_arista = geometrias_aristas_desde_shapes_gtfs(
    stops, stop_times_bus, shape_ids_bus, shapes
)
print(f"Aristas con geometria real: {len(geoms_por_arista)}")

geometrias_nuevas = [
    geoms_por_arista.get((u, v))
    for u, v in zip(aristas_dtpm["u_id"], aristas_dtpm["v_id"])
]
aristas_dtpm["geometry"] = geometrias_nuevas
aristas_dtpm = aristas_dtpm[~aristas_dtpm.geometry.isna()].copy()
aristas_dtpm = aristas_dtpm.set_geometry("geometry")
aristas_dtpm.crs = "EPSG:4326"
aristas_dtpm["log_servicios"] = np.log1p(aristas_dtpm["servicios"])

fig, ax = figure_from_geodataframe(aristas_dtpm, height=8)
aristas_dtpm.plot(
    ax=ax,
    column="log_servicios",
    cmap="viridis",
    linewidth=0.5,
    legend=True,
    legend_kwds={"label": "log(servicios + 1)", "shrink": 0.6},
)
ax.set_axis_off()
ax.set_title(
    "Red DTPM (geometria shape-based): aristas pesadas por servicios"
)
fig.savefig("images/09-red-dtpm.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Saltos en la red de buses desde el paradero mas cercano a Plaza de
# Armas. Las redes de metro y bus quedan desconectadas en este grafo:
# solo conectamos paraderos consecutivos del mismo trip y un trip de
# metro nunca pasa por una parada de bus. Para tener una respuesta
# significativa filtramos el ancla a un paradero de bus, donde la
# componente conexa cubre la mayor parte de la ciudad.
# Solo paraderos de bus con aristas reales (filtramos huerfanos del
# GTFS que aparecen en stops.txt pero no en stop_times.txt).
grados_dtpm = pd.Series(dict(g_dtpm.degree()))
con_aristas = grados_dtpm[grados_dtpm > 0].index
nodos_bus = nodos_dtpm[
    (nodos_dtpm["modo"] == "bus") & nodos_dtpm["stop_id"].isin(con_aristas)
].copy()
nodos_bus_m = nodos_bus.to_crs(CRS_METRICO)
plaza_armas = (
    gpd.GeoDataFrame(
        geometry=[Point(-70.6506, -33.4378)], crs="EPSG:4326"
    ).to_crs(CRS_METRICO).geometry.iloc[0]
)
idx_ancla = nodos_bus_m.geometry.distance(plaza_armas).idxmin()
stop_ancla = nodos_bus_m.loc[idx_ancla, "stop_id"]
print(f"Paradero ancla mas cercano a Plaza de Armas: {stop_ancla}")

g_dtpm_undirected = g_dtpm.to_undirected()
saltos = nx.single_source_shortest_path_length(g_dtpm_undirected, stop_ancla)
nodos_dtpm["saltos"] = nodos_dtpm["stop_id"].map(saltos)
accesibles = nodos_dtpm.dropna(subset=["saltos"]).copy()
print(
    f"Paraderos alcanzables desde {stop_ancla}: {len(accesibles)} de "
    f"{len(nodos_dtpm)} ({len(accesibles) / len(nodos_dtpm):.1%}) "
    f"(la red de metro queda excluida por construccion del grafo)"
)

hex_gran = h3_grid_from_bounds(BBOX_GRAN_STGO, grid_level=8).reset_index()
join_dtpm = gpd.sjoin(accesibles, hex_gran, how="inner", predicate="within")
saltos_por_hex = (
    join_dtpm.groupby("h3_cell_id")["saltos"].median().reset_index()
)
hex_saltos = hex_gran.merge(saltos_por_hex, on="h3_cell_id").dropna()

fig, ax = figure_from_geodataframe(hex_saltos, height=8)
choropleth_map(
    hex_saltos,
    "saltos",
    ax=ax,
    k=6,
    palette="viridis",
    binning="fisher_jenks",
    edgecolor="none",
    alpha=0.85,
)
accesibles[accesibles["saltos"] == 0].plot(
    ax=ax, color="white", edgecolor="black", markersize=80, zorder=5
)
ax.set_axis_off()
ax.set_title(
    "Saltos desde el paradero mas cercano a Plaza de Armas "
    "(mediana por hexagono H3-8)"
)
fig.savefig("images/09-isocronas-dtpm.png", dpi=128, bbox_inches="tight")
plt.show()

# %%
# Edge betweenness sobre la red de paraderos. Combina con strength:
# strength alta + edge betweenness alta = corredor estructurante.
# Strength alta + edge betweenness baja = paradero servido pero
# periferico.
g_dtpm_simple = nx.Graph()
for u, v, d in g_dtpm.edges(data=True):
    if g_dtpm_simple.has_edge(u, v):
        g_dtpm_simple[u][v]["servicios"] = max(
            g_dtpm_simple[u][v]["servicios"], d["servicios"]
        )
    else:
        g_dtpm_simple.add_edge(u, v, servicios=d["servicios"])
for n, d in g_dtpm.nodes(data=True):
    if n in g_dtpm_simple.nodes:
        g_dtpm_simple.nodes[n].update(d)

gcc_dtpm = max(nx.connected_components(g_dtpm_simple), key=len)
g_dtpm_gcc = g_dtpm_simple.subgraph(gcc_dtpm).copy()
print(
    f"Edge betweenness sobre GCC DTPM: {g_dtpm_gcc.number_of_nodes()} nodos, "
    f"{g_dtpm_gcc.number_of_edges()} aristas (k=200 pivotes)"
)
eb_dtpm = nx.edge_betweenness_centrality(g_dtpm_gcc, k=200, seed=42)
dtpm_centralidad = aristas_con_centralidad(
    g_dtpm_gcc, eb_dtpm, aristas_dtpm.crs, geom_lookup=geoms_por_arista
)
graficar_edge_centralidad(
    dtpm_centralidad,
    "centralidad",
    "Edge betweenness: red de paraderos DTPM (GCC)",
    "images/09-edge-betweenness-dtpm.png",
    cmap="magma",
)

# %% [markdown]
# ## Cierre
#
# Las tres redes que vimos comparten el lenguaje de la ciencia de
# redes pero contestan preguntas distintas:
#
#   - Calles (OSM): la red es planar y concentra el grado en 3 o 4
#     (esquinas en T o en cruz), bien distinta de un grafo aleatorio.
#     Los corredores estructurantes concentran los caminos cortos y
#     remover el 5% mas central aumenta el camino promedio en ~50%.
#   - Ciclovias (MINVU oficial + OSM colaborativo): la red esta muy
#     fragmentada y la cobertura poblacional a 500 m es modesta. El
#     contraste entre fuentes muestra que la eleccion de fuente afecta
#     conclusiones.
#   - Paraderos (GTFS DTPM): la cobertura concentra servicio en el
#     corredor Alameda-Vicuña Mackenna y Vespucio, y los paraderos
#     metro son los hubs estructurantes pese a su menor numero.
#
# El mismo grafo, con distintos atributos en nodos y aristas, sirve
# como punto de partida para las clases siguientes: en la 10
# trabajaremos flujos OD y accesibilidad multimodal, y en la 11
# clusters espaciales.
