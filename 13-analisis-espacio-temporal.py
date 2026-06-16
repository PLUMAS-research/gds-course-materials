# %% [markdown]
# # Análisis espacio-temporal: el caso del Paseo Bandera
#
# En diciembre de 2017 se peatonalizó la calle Bandera, en el centro de
# Santiago. Bandera no es un tramo cualquiera: es la continuación norte de
# San Diego, y juntas forman un eje que cruza la ciudad de sur a norte. Por
# ese eje circulan decenas de líneas de bus que conectan los sectores del
# sur (Sur, Poniente) con el Centro y el Norte. Al cerrar Bandera, esos
# recorridos tuvieron que tomar calles paralelas.
#
# La pregunta de la clase no es la obvia (¿se movieron las subidas?, que
# por definición sí), sino la sustantiva: ¿qué le costó a la red de buses
# perder ese eje, y fue beneficiosa la intervención para la ciudad? La
# respondemos comparando la distribución de un fenómeno en distintos
# momentos, en dos planos y a la escala de la ciudad:
#
#   1. Estructural: la centralidad del eje San Diego-Bandera en la red de
#      buses (filtrada con el GTFS), con y sin Bandera.
#   2. Empírico: la actividad de transporte por sector de la ciudad y por
#      régimen, con el resto de Santiago como control de tendencias
#      paralelas.
#
# Cronología (fuentes en el apunte):
#   - 2013-2017: Bandera cerrada por obras de la Línea 3 del Metro.
#   - dic 2017: peatonalización (táctica y luego permanente).
#   - ene 2019: apertura de la Línea 3 del Metro.
#   - 18 oct 2019: estallido social (disrupción mayor del transporte, centro
#     como epicentro). La encuesta DTPM de 2019 es de agosto, anterior al
#     estallido: el régimen peatonal queda limpio, el estallido entra en 2020.
#   - 2020-2021: pandemia (quiebre estructural en la demanda).
#   - 2026: retorno de buses Red (fuera de la ventana de datos).

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/bandera.tgz")

# %%
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from chiricoca.config import setup_style
from chiricoca.geo.figures import (
    figure_from_geodataframe,
    small_multiples_from_geodataframe,
)
from chiricoca.geo.grid import h3_grid_from_ids
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from shapely.geometry import LineString, box

setup_style(dpi=128)

from gdsutils.dtpm import RUTA_GTFS, URL_GTFS
from gdsutils.redes import construir_grafo_desde_lineas

DIR = Path("data") / "bandera"
CRS_METRICO = "EPSG:32719"
BBOX_CENTRO = (-70.662, -33.448, -70.642, -33.430)
# Banda N-S de La Cisterna a Independencia, por el eje San Diego-Bandera.
BANDA = (-70.71, -33.55, -70.61, -33.41)
# Bounding box del área urbana del Gran Santiago, para recortar los mapas.
BBOX_URBANO = (-70.85, -33.65, -70.45, -33.30)

REGIMENES = {
    "Construcción (2014-2017)": [2014, 2015, 2016, 2017],
    "Peatonal (2018-2019)": [2018, 2019],
    "Pandemia y recuperación (2020-2025)": [2020, 2021, 2022, 2023, 2024, 2025],
}
cols_regimen = list(REGIMENES.keys())
EVENTOS = {"Peatonalización": 2017.97, "Apertura Metro L3": 2019.0,
           "Estallido social": 2019.80, "Pandemia": 2020.2}
anio_a_regimen = {a: r for r, anios in REGIMENES.items() for a in anios}
# Sectores que conecta el eje San Diego-Bandera (para resaltarlos).
SECTORES_EJE = ["Sur", "Poniente", "Centro", "Norte"]

# Bordes comunales del Gran Santiago, para orientarse en los mapas.
comunas = gpd.read_parquet(DIR / "comunas.parquet").to_crs("EPSG:4326")


def dibujar_comunas(ax, color="0.55", lw=0.5):
    comunas.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=4)

# %% [markdown]
# ## PARTE 1. Anatomía de los datos
#
# Antes de analizar conviene mirar los datos. Tres cortes exploratorios:
# cuántos viajes hay por año (el tamaño de cada encuesta DTPM), cuántas
# líneas de bus pasan por cada corredor del centro (la carga del eje) y
# cómo se reparten los viajes entre los sectores de la ciudad que define
# la EOD.

# %%
# 1a. Volumen de viajes por año: cada año es una semana tipo, de tamaño
# distinto. Esto importa porque impide comparar niveles crudos entre años.
totales = pd.read_parquet(DIR / "totales-anuales.parquet").sort_values("anio")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(totales["anio"], totales["viajes_total"] / 1e6, color="steelblue")
ax.set_xlabel("Año")
ax.set_ylabel("Viajes en la semana tipo (millones, ponderado)")
ax.set_title("Tamaño de la encuesta DTPM por año (días laborales)")
fig.savefig("images/13-eda-viajes-anuales.png", dpi=128, bbox_inches="tight")
print("Viajes totales por año (millones):")
print((totales.set_index("anio")["viajes_total"] / 1e6).round(1).to_string())

# %%
# 1b. Líneas de bus por sector de origen (EOD): qué sectores generan más
# viajes en transporte público. Sur y Poniente, los que alimentan el eje
# San Diego-Bandera, están entre los mayores.
vol_sector = pd.read_parquet(DIR / "volumen-sector.parquet")
sector_2016 = (vol_sector[vol_sector["anio"] == 2016].groupby("sector")["viajes"].sum()
               .sort_values())
fig, ax = plt.subplots(figsize=(8, 4))
colores = ["firebrick" if s in SECTORES_EJE else "steelblue" for s in sector_2016.index]
ax.barh(sector_2016.index, sector_2016.values / 1e6, color=colores)
ax.set_xlabel("Viajes de bus con origen en el sector (millones, 2016)")
ax.set_title("Viajes por sector de origen (EOD); en rojo, los del eje")
fig.savefig("images/13-eda-sector-origen.png", dpi=128, bbox_inches="tight")

# %%
# 1c. GTFS: shapes de las líneas de bus, para filtrar la red y contar
# cuántas líneas pasan por cada corredor.
if not RUTA_GTFS.exists():
    RUTA_GTFS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando GTFS DTPM: {URL_GTFS}")
    urllib.request.urlretrieve(URL_GTFS, RUTA_GTFS)
with zipfile.ZipFile(RUTA_GTFS) as z:
    routes = pd.read_csv(z.open("routes.txt"), dtype=str)
    trips = pd.read_csv(z.open("trips.txt"), dtype=str)
    shapes = pd.read_csv(z.open("shapes.txt"), dtype=str)
shapes["lat"] = pd.to_numeric(shapes["shape_pt_lat"])
shapes["lon"] = pd.to_numeric(shapes["shape_pt_lon"])
shapes["seq"] = pd.to_numeric(shapes["shape_pt_sequence"])
rutas_bus = set(routes.loc[routes["route_type"] == "3", "route_id"])
shape2route = dict(zip(trips["shape_id"], trips["route_id"]))
geoms, rids = [], []
for sid, grp in shapes.sort_values("seq").groupby("shape_id"):
    rid = shape2route.get(sid)
    if rid in rutas_bus and len(grp) > 1:
        geoms.append(LineString(list(zip(grp["lon"], grp["lat"]))))
        rids.append(rid)
bus_shapes = gpd.GeoDataFrame({"route_id": rids}, geometry=geoms, crs="EPSG:4326").to_crs(CRS_METRICO)
print(f"Shapes de bus: {len(bus_shapes)} de {bus_shapes['route_id'].nunique()} líneas")

calles = gpd.read_parquet(DIR / "calles-antes.parquet").to_crs("EPSG:4326")
calles = calles[calles.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()


def rutas_en_calle(nombre):
    sub = calles[calles["name"] == nombre]
    if sub.empty:
        return 0
    buf = sub.to_crs(CRS_METRICO).buffer(25).union_all()
    return bus_shapes[bus_shapes.intersects(buf)]["route_id"].nunique()


corredores = ["San Diego", "Bandera", "San Antonio", "Enrique Mac Iver",
              "Morandé", "Amunátegui", "Teatinos", "Nataniel Cox"]
conteo = pd.Series({c: rutas_en_calle(c) for c in corredores}).sort_values(ascending=False)
print("Líneas de bus por corredor N-S del centro:")
print(conteo.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
colores = ["firebrick" if c in ("Bandera", "San Diego") else "steelblue" for c in conteo.index]
ax.barh(conteo.index[::-1], conteo.values[::-1], color=colores[::-1])
ax.set_xlabel("Líneas de bus que pasan por el corredor (GTFS)")
ax.set_title("Carga de líneas por corredor N-S del centro")
fig.savefig("images/13-lineas-corredor.png", dpi=128, bbox_inches="tight")

# %%
# 1d. Mapa de los sectores de la EOD con el eje San Diego-Bandera. El eje
# es un conector entre el sur de la ciudad y el centro-norte.
sectores = gpd.read_parquet(DIR / "sectores.parquet").to_crs("EPSG:4326")
sectores_urb = sectores.clip(box(*BBOX_URBANO))
eje = calles[calles["name"].isin(["San Diego", "Bandera"])]
fig, ax = figure_from_geodataframe(sectores_urb, height=8)
sectores_urb.plot(ax=ax, column="sector", cmap="Pastel2", edgecolor="none", legend=True,
                  legend_kwds={"loc": "lower left", "fontsize": 7})
dibujar_comunas(ax, color="0.5", lw=0.4)
eje.to_crs("EPSG:4326").plot(ax=ax, color="black", linewidth=2.5, zorder=5)
ax.set_axis_off()
ax.set_title("Sectores de la ciudad (EOD), comunas y el eje San Diego-Bandera (negro)")
fig.savefig("images/13-sectores-mapa.png", dpi=128, bbox_inches="tight")

# %% [markdown]
# ## PARTE 2. El eje San Diego-Bandera en la red de buses
#
# El error de mirar solo el centro es que en el damero sobran calles
# paralelas. Pero los buses no usan cualquier calle: siguen corredores.
# Filtramos las calles de 2015 a las que llevan al menos una línea según
# el GTFS, y construimos un grafo dirigido que respeta los sentidos únicos
# (un bus no sube por una calle de bajada).

# %%
buffer_bus = bus_shapes.buffer(25).union_all()
calles_m = calles.to_crs(CRS_METRICO)
calles_bus = calles[calles_m.intersects(buffer_bus)].copy()
print(f"Calles que llevan buses: {len(calles_bus)} de {len(calles)} candidatas")

g_multi, nodos, aristas = construir_grafo_desde_lineas(
    calles_bus[["name", "oneway", "geometry"]], tolerancia_metros=3.0, dirigido=True
)
g = nx.DiGraph()
for u, v, d in g_multi.edges(data=True):
    ow = str(d.get("oneway", "")).lower()
    pares = [(v, u)] if ow == "-1" else [(u, v)] + ([] if ow in ("yes", "true", "1") else [(v, u)])
    for a, b in pares:
        if not g.has_edge(a, b) or d["largo_m"] < g[a][b]["largo_m"]:
            g.add_edge(a, b, largo_m=d["largo_m"])
for n, d in g_multi.nodes(data=True):
    if n in g:
        g.nodes[n].update(d)
g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
print(f"Red de buses (componente fuerte): {g.number_of_nodes()} nodos, {g.number_of_edges()} aristas")

aristas["clave"] = aristas.apply(lambda r: (min(r.u, r.v), max(r.u, r.v)), axis=1)
nombre_por_arista = dict(zip(aristas["clave"], aristas["name"]))
nod4 = nodos.set_index("node_id").to_crs("EPSG:4326")


def aristas_de_calle(grafo, nombre):
    return [(u, v) for u, v in grafo.edges()
            if nombre_por_arista.get((min(u, v), max(u, v))) == nombre]


def geom_aristas(edges):
    return gpd.GeoSeries(
        [LineString([(nod4.geometry.loc[u].x, nod4.geometry.loc[u].y),
                     (nod4.geometry.loc[v].x, nod4.geometry.loc[v].y)]) for u, v in edges],
        crs="EPSG:4326",
    )


aristas_bandera = aristas_de_calle(g, "Bandera")
aristas_sandiego = aristas_de_calle(g, "San Diego")
print(f"Aristas en el grafo: Bandera {len(aristas_bandera)}, San Diego {len(aristas_sandiego)}")

# %%
fig, ax = figure_from_geodataframe(aristas, height=9)
aristas.plot(ax=ax, color="0.82", linewidth=0.4)
geom_aristas(aristas_sandiego).plot(ax=ax, color="#1f77b4", linewidth=2.6)
geom_aristas(aristas_bandera).plot(ax=ax, color="firebrick", linewidth=2.6)
dibujar_comunas(ax)
ax.set_axis_off()
ax.set_title("Eje San Diego (azul) - Bandera (rojo) en la red de buses, 2015")
fig.savefig("images/13-red-bus-eje.png", dpi=128, bbox_inches="tight")

# %% [markdown]
# ## PARTE 3. ¿Es San Diego-Bandera un corredor estructurante?
#
# La intermediación de arista cuenta la fracción de caminos más cortos,
# sobre todos los pares de nodos de la red de buses, que pasan por cada
# arista. Si el eje estructura el flujo norte-sur, sus aristas deberían
# estar en la cola alta de la distribución.

# %%
print("Calculando edge betweenness de la red de buses")
eb = nx.edge_betweenness_centrality(g, weight="largo_m", k=300, seed=42)
eb = {(min(u, v), max(u, v)): val for (u, v), val in eb.items()}
eb_serie = pd.Series(eb).sort_values(ascending=False)
n_aristas = len(eb_serie)
ranks_axis = sorted(int((eb_serie > eb[(min(u, v), max(u, v))]).sum()) + 1
                    for u, v in aristas_bandera + aristas_sandiego if (min(u, v), max(u, v)) in eb)
print(f"Ranks de intermediación del eje San Diego-Bandera (de {n_aristas}): {ranks_axis[:12]}")

top_corredores = (
    pd.DataFrame([{"calle": nombre_por_arista.get(k), "betweenness": v} for k, v in eb.items()])
    .dropna().groupby("calle")["betweenness"].mean().sort_values(ascending=False).head(10)
)
print("\nTop corredores de la red de buses por intermediación media:")
print(top_corredores.round(4).to_string())

# %%
aristas_eb = aristas.copy()
aristas_eb["betweenness"] = aristas_eb["clave"].map(eb)
aristas_eb = aristas_eb.dropna(subset=["betweenness"]).sort_values("betweenness")
fig, ax = figure_from_geodataframe(aristas_eb, height=9)
vmax = aristas_eb["betweenness"].max()
aristas_eb.plot(ax=ax, column="betweenness", cmap="plasma",
                linewidth=(0.3 + 4.0 * aristas_eb["betweenness"] / vmax).values,
                legend=True, legend_kwds={"shrink": 0.6, "label": "intermediación de arista"})
dibujar_comunas(ax)
ax.set_axis_off()
ax.set_title("Intermediación en la red de buses: corredores estructurantes")
fig.savefig("images/13-betweenness-bus.png", dpi=128, bbox_inches="tight")
print("El eje San Diego-Bandera aparece en la cola alta: es un corredor estructurante.")

# %% [markdown]
# ## PARTE 4. El contrafactual: remover Bandera (peatonalización)
#
# Removemos las aristas de Bandera (San Diego sigue vehicular) y medimos
# dos cosas para viajes de buses entre el sur y el norte: cuánto se alarga
# el recorrido, y a qué corredores se traslada la carga.

# %%
y = pd.Series({n: nod4.geometry.loc[n].y for n in g.nodes if n in nod4.index})
sur_nodos = y[y < -33.49].index.tolist()
norte_nodos = y[y > -33.43].index.tolist()
g_sin = g.copy()
g_sin.remove_edges_from(aristas_bandera)

rng = np.random.default_rng(0)
desvios = []
for _ in range(400):
    s = sur_nodos[rng.integers(len(sur_nodos))]
    t = norte_nodos[rng.integers(len(norte_nodos))]
    if nx.has_path(g, s, t) and nx.has_path(g_sin, s, t):
        d0 = nx.shortest_path_length(g, s, t, weight="largo_m")
        d1 = nx.shortest_path_length(g_sin, s, t, weight="largo_m")
        if d0 > 0:
            desvios.append(d1 / d0 - 1)
desvios = np.array(desvios)
print("Desvío de viajes de buses sur -> norte al peatonalizar Bandera "
      f"({len(desvios)} pares):")
print(f"  medio {desvios.mean() * 100:.2f}% | p90 {np.percentile(desvios, 90) * 100:.2f}% | "
      f"máx {desvios.max() * 100:.1f}%")
print(f"  pares que usaban Bandera (desvío > 0): {(desvios > 1e-4).mean() * 100:.0f}%")
print("\nEl desvío es chico aun para los viajes que sí pasaban por Bandera: en la red")
print("de buses quedan paralelas (San Antonio, Mac Iver, Morandé) que la sustituyen.")

# %%
# El costo real es congestión: las líneas de Bandera se concentran en
# corredores ya cargados. Recalculamos la intermediación sin Bandera.
print("Recalculando intermediación sin Bandera")
eb_sin = nx.edge_betweenness_centrality(g_sin, weight="largo_m", k=300, seed=42)
eb_sin = {(min(u, v), max(u, v)): val for (u, v), val in eb_sin.items()}
aristas_delta = aristas.copy()
aristas_delta["delta"] = aristas_delta["clave"].map(lambda k: eb_sin.get(k, 0.0) - eb.get(k, 0.0))
aristas_delta = aristas_delta.dropna(subset=["delta"])
ganancia = (
    aristas_delta.assign(calle=aristas_delta["clave"].map(nombre_por_arista))
    .dropna(subset=["calle"]).groupby("calle")["delta"].sum().sort_values(ascending=False)
)
print(f"\nBandera lleva {conteo['Bandera']} líneas. Corredores que absorben su carga (top 6),")
print("con las líneas que YA llevaban (entre paréntesis):")
for calle, d in ganancia.head(6).items():
    print(f"  {calle:18s}: +{d:.4f} intermediación  (ya lleva {rutas_en_calle(calle)} líneas)")

# %%
# Mapa de redistribución sobre toda la banda: el flujo que llevaba Bandera
# (azul) se traslada a los corredores paralelos (rojo) del eje N-S.
fig, ax = figure_from_geodataframe(aristas_delta, height=9)
lim = np.abs(aristas_delta["delta"]).quantile(0.98) or aristas_delta["delta"].abs().max()
aristas_delta.plot(ax=ax, color="0.88", linewidth=0.3)
orden = aristas_delta.reindex(aristas_delta["delta"].abs().sort_values().index)
orden.plot(ax=ax, column="delta", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
           linewidth=(0.5 + 3.5 * orden["delta"].abs() / lim).clip(upper=4).values,
           legend=True, legend_kwds={"shrink": 0.6, "label": "cambio de intermediación (sin - con Bandera)"})
geom_aristas(aristas_bandera).plot(ax=ax, color="black", linewidth=1.2, linestyle=":")
dibujar_comunas(ax)
ax.set_axis_off()
ax.set_title("Redistribución de la carga de buses al peatonalizar Bandera (punteado)")
fig.savefig("images/13-redistribucion-betweenness.png", dpi=128, bbox_inches="tight")
print("\nEl desvío en distancia es chico, pero las líneas de Bandera se suman a")
print("corredores que ya iban cargados: el costo es congestión, no kilómetros.")

# %% [markdown]
# ## PARTE 5. Actividad por sector de la ciudad y régimen
#
# Pasamos al plano empírico a escala de ciudad. Comparamos cómo se reparte
# la actividad de transporte entre los sectores de la EOD en cada régimen,
# normalizando a participación dentro de cada régimen (cada momento suma
# uno) para cancelar el cambio de volumen total entre años. Es la versión
# espacio-temporal de la pregunta: ¿cambió la distribución de la actividad
# entre sectores antes y después de la intervención?

# %%
vs = vol_sector.copy()
vs["regimen"] = vs["anio"].map(anio_a_regimen)
piv_sec = vs.groupby(["regimen", "sector"])["viajes"].mean().reset_index()
piv_sec["share"] = piv_sec.groupby("regimen")["viajes"].transform(lambda s: s / s.sum())
sec_share = piv_sec.pivot(index="sector", columns="regimen", values="share").fillna(0.0)
sectores_share = sectores.merge(sec_share, on="sector")

sectores_share_urb = sectores_share.clip(box(*BBOX_URBANO))
vmax = sectores_share[cols_regimen].to_numpy().max()
fig, axes = small_multiples_from_geodataframe(sectores_share_urb, n_variables=3, col_wrap=3, height=5)
for ax, col in zip(axes, cols_regimen):
    sectores_share_urb.plot(ax=ax, column=col, cmap="magma_r", vmin=0, vmax=vmax, edgecolor="none")
    dibujar_comunas(ax, color="0.5", lw=0.3)
    ax.set_title(col.split(" (")[0], fontsize=10)
    ax.set_axis_off()
cax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="magma_r"), cax=cax,
             orientation="horizontal", label="participación del sector en los viajes, por régimen")
fig.savefig("images/13-sector-regimen.png", dpi=128, bbox_inches="tight")

# %%
# Tendencia de la participación de cada sector en el tiempo. Los sectores
# del eje se resaltan; los demás van en gris.
share_anio = vs.copy()
share_anio["share"] = share_anio.groupby("anio")["viajes"].transform(lambda s: s / s.sum())
piv_t = share_anio.pivot(index="anio", columns="sector", values="share")
fig, ax = plt.subplots(figsize=(9, 5))
for sector in piv_t.columns:
    es_eje = sector in SECTORES_EJE
    ax.plot(piv_t.index, piv_t[sector], "o-" if es_eje else "-",
            color=None if es_eje else "0.75", linewidth=2 if es_eje else 1,
            label=sector if es_eje else None, zorder=3 if es_eje else 1)
for x in EVENTOS.values():
    ax.axvline(x, color="gray", linestyle="--", alpha=0.4)
ax.set_xlabel("Año")
ax.set_ylabel("Participación del sector en los viajes")
ax.set_title("Participación de cada sector en el tiempo (en color, los del eje)")
ax.legend(fontsize=8, title="Sectores del eje")
fig.savefig("images/13-sector-tendencia.png", dpi=128, bbox_inches="tight")
print("Participación media por sector y régimen:")
print((sec_share * 100).round(1).to_string())

# %%
# Las tres distribuciones se parecen: para ver el cambio hay que restarlas.
# Cambio de participación de cada sector entre el régimen peatonal y el de
# construcción (en puntos porcentuales).
cambio_sector = ((sec_share[cols_regimen[1]] - sec_share[cols_regimen[0]]) * 100).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(cambio_sector.index, cambio_sector.values,
        color=["firebrick" if v > 0 else "steelblue" for v in cambio_sector.values])
ax.axvline(0, color="0.4", linewidth=0.8)
ax.set_xlabel("Cambio de participación, peatonal - construcción (puntos %)")
ax.set_title("Cambio de participación por sector (peatonal vs construcción)")
fig.savefig("images/13-sector-cambio.png", dpi=128, bbox_inches="tight")
print("\nLos cambios por sector son chicos (pocos puntos): a nivel de distribución")
print("el efecto de una intervención local como Bandera casi no se nota en la ciudad.")

# %% [markdown]
# ## PARTE 6. Actividad en el corredor por régimen (hexágonos H3-8)
#
# Acercamos la mirada al corredor con una grilla H3-8 (~460 m de arista)
# sobre la banda completa, no solo el centro. Misma normalización a
# participación y misma escala de color entre paneles, para que la
# comparación sea legítima.

# %%
validaciones = pd.read_parquet(DIR / "validaciones-banda.parquet")
validaciones["actividad"] = validaciones["subidas"] + validaciones["bajadas"]
validaciones["h3"] = [h3.latlng_to_cell(lat, lon, 8)
                      for lat, lon in zip(validaciones["lat"], validaciones["lon"])]
validaciones["regimen"] = validaciones["anio"].map(anio_a_regimen)
print(f"Validaciones: {len(validaciones):,} filas, {validaciones['h3'].nunique()} hexágonos H3-8")

piv = validaciones.groupby(["regimen", "h3"])["actividad"].mean().reset_index()
piv["share"] = piv.groupby("regimen")["actividad"].transform(lambda s: s / s.sum())
hex_share = piv.pivot(index="h3", columns="regimen", values="share").fillna(0.0)
hexes = h3_grid_from_ids(hex_share.index.tolist()).reset_index().merge(
    hex_share, left_on="h3_cell_id", right_index=True)

vmax = hexes[cols_regimen].to_numpy().max() * 0.9
fig, axes = small_multiples_from_geodataframe(hexes, n_variables=3, col_wrap=3, height=5.5)
for ax, col in zip(axes, cols_regimen):
    aristas.plot(ax=ax, color="0.85", linewidth=0.2, zorder=1)
    hexes.plot(ax=ax, column=col, cmap="magma_r", vmin=0, vmax=vmax,
               edgecolor="none", alpha=0.85, zorder=2)
    dibujar_comunas(ax)
    ax.set_title(col.split(" (")[0], fontsize=10)
    ax.set_axis_off()
cax = fig.add_axes([0.30, 0.04, 0.40, 0.015])
fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="magma_r"), cax=cax,
             orientation="horizontal",
             label="participación de la actividad del corredor por régimen (H3-8)")
fig.savefig("images/13-actividad-regimen.png", dpi=128, bbox_inches="tight")

# %% [markdown]
# ## PARTE 7. Mapa de diferencia: peatonal menos construcción
#
# El mapa de diferencia resta la participación de cada hexágono entre el
# régimen peatonal y el de construcción. Rampa divergente centrada en cero:
# rojo donde subió, azul donde bajó. Como es participación, mide
# redistribución, no cambio de volumen.

# %%
hexes["diferencia"] = hexes[cols_regimen[1]] - hexes[cols_regimen[0]]
lim = np.abs(hexes["diferencia"]).quantile(0.98) or hexes["diferencia"].abs().max()
fig, ax = figure_from_geodataframe(hexes, height=9)
aristas.plot(ax=ax, color="0.85", linewidth=0.2, zorder=1)
hexes.plot(ax=ax, column="diferencia", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
           edgecolor="none", alpha=0.9, legend=True,
           legend_kwds={"shrink": 0.6, "label": "cambio de participación (peatonal - construcción)"},
           zorder=2)
geom_aristas(aristas_bandera).plot(ax=ax, color="black", linewidth=1.5, linestyle=":", zorder=3)
dibujar_comunas(ax)
ax.set_axis_off()
ax.set_title("Cambio en la distribución de la actividad (peatonal vs construcción)")
fig.savefig("images/13-diferencia-actividad.png", dpi=128, bbox_inches="tight")

# %% [markdown]
# ## PARTE 8. Serie de tiempo con quiebre: ¿qué evento domina?
#
# La actividad total del centro por año, con los eventos marcados. Una
# regresión segmentada (ITS) en torno a la peatonalización separa la
# tendencia previa del salto. La lección es cuál quiebre domina.

# %%
serie = validaciones.groupby("anio")["actividad"].sum().reset_index().sort_values("anio")
serie["actividad_mm"] = serie["actividad"] / 1e6
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(serie["anio"], serie["actividad_mm"], "o-", color="steelblue")
colores_ev = {"Peatonalización": "firebrick", "Apertura Metro L3": "darkgreen",
              "Estallido social": "darkorange", "Pandemia": "gray"}
for nombre, x in EVENTOS.items():
    ax.axvline(x, color=colores_ev[nombre], linestyle="--", alpha=0.8)
    ax.text(x, ax.get_ylim()[1] * 0.97, nombre, rotation=90, va="top", ha="right",
            fontsize=8, color=colores_ev[nombre])
ax.set_xlabel("Año")
ax.set_ylabel("Actividad del corredor (millones, ponderada)")
ax.set_title("Actividad de transporte y eventos de la intervención")
fig.savefig("images/13-serie-eventos.png", dpi=128, bbox_inches="tight")

pre = serie[serie["anio"] <= 2019].copy()
t_estrella = 2018
pre["t"] = pre["anio"] - pre["anio"].min()
pre["D"] = (pre["anio"] >= t_estrella).astype(int)
pre["tD"] = (pre["anio"] - t_estrella) * pre["D"]
X = np.column_stack([np.ones(len(pre)), pre["t"], pre["D"], pre["tD"]])
beta, *_ = np.linalg.lstsq(X, pre["actividad_mm"].to_numpy(), rcond=None)
print("Regresión segmentada (ITS) en la peatonalización, 2014-2019:")
print(f"  beta1 (tendencia previa):    {beta[1]:+.3f} por año")
print(f"  beta2 (salto en 2018):       {beta[2]:+.3f}")
print(f"  beta3 (cambio de pendiente): {beta[3]:+.3f} por año")
caida = serie.loc[serie.anio == 2021, "actividad_mm"].iloc[0] / serie.loc[serie.anio == 2019, "actividad_mm"].iloc[0] - 1
print(f"\nEl salto de la peatonalización es chico frente a la caída de la pandemia")
print(f"(2021 vs 2019: {caida * 100:.0f}%). La pandemia domina la serie.")

# %% [markdown]
# ## PARTE 9. Tendencias paralelas: el centro frente al resto de Santiago
#
# Para acercarse a la atribución necesitamos un control. Comparamos el
# volumen de viajes que tocan el centro contra el del resto de la ciudad.
# El cociente centro/resto cancela el factor común (el tamaño de cada
# muestra anual, la pandemia). Si las dos series corrían paralelas antes de
# la intervención y divergen después, hay evidencia de un efecto.

# %%
totales["ratio"] = totales["viajes_centro"] / totales["viajes_resto"]
base = totales.loc[totales["anio"].between(2014, 2017), ["viajes_centro", "viajes_resto"]].mean()
totales["centro_idx"] = totales["viajes_centro"] / base["viajes_centro"]
totales["resto_idx"] = totales["viajes_resto"] / base["viajes_resto"]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(totales["anio"], totales["centro_idx"], "o-", color="firebrick", label="Centro (Bandera)")
ax.plot(totales["anio"], totales["resto_idx"], "o-", color="steelblue", label="Resto de Santiago")
ax.axhline(1, color="0.7", linewidth=0.8)
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="gray", linestyle="--", alpha=0.5)
    ax.text(x, ax.get_ylim()[1], nombre, rotation=90, va="top", ha="right", fontsize=8)
ax.set_xlabel("Año")
ax.set_ylabel("Volumen relativo a la media 2014-2017")
ax.set_title("Volumen de viajes: centro vs resto de Santiago")
ax.legend(fontsize=8)
fig.savefig("images/13-volumen-centro-resto.png", dpi=128, bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(totales["anio"], totales["ratio"], "o-", color="purple")
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="gray", linestyle="--", alpha=0.5)
    ax.text(x, ax.get_ylim()[1], nombre, rotation=90, va="top", ha="right", fontsize=8)
ax.set_xlabel("Año")
ax.set_ylabel("Cociente volumen centro / resto")
ax.set_title("Peso relativo del centro en el sistema")
fig.savefig("images/13-cociente-centro-resto.png", dpi=128, bbox_inches="tight")

# %%
ratio_pre = totales.loc[totales["anio"].between(2014, 2017), "ratio"].mean()
ratio_peatonal = totales.loc[totales["anio"].isin([2018, 2019]), "ratio"].mean()
print(f"Cociente centro/resto, construcción (2014-2017): {ratio_pre:.4f}")
print(f"Cociente centro/resto, peatonal (2018-2019):     {ratio_peatonal:.4f}")
print(f"Caída relativa del peso del centro: {(ratio_peatonal / ratio_pre - 1) * 100:.0f}%")
print("\nEl cociente es plano hasta 2017 (tendencias paralelas) y cae desde 2018.")
print("La caída 2018-2019 (la encuesta de 2019 es de agosto, anterior al estallido)")
print("ya mezcla peatonalización y apertura del Metro L3. Desde 2020 se suma el")
print("estallido social (con el centro como epicentro) y la pandemia. Cuatro shocks")
print("sobre el mismo lugar: la atribución a una sola causa es imposible.")

# %% [markdown]
# ## PARTE 10. Tiempo de viaje: ¿se frenó el centro más que la ciudad?
#
# El tiempo de viaje está en minutos, así que no depende del tamaño de la
# muestra anual. Pero hay una trampa: la ciudad entera se congestiona con
# los años, así que comparar el tiempo del centro entre años mezcla el
# efecto de Bandera con esa tendencia general. La solución es la misma de
# siempre: comparar el centro contra el resto de Santiago (el control que
# absorbe la congestión común) y mirar la diferencia, no el nivel.

# %%
tiempos = pd.read_parquet(DIR / "tiempos.parquet")
tiempos["regimen"] = tiempos["anio"].map(anio_a_regimen)
mediana = tiempos.groupby(["anio", "zona"])["tviaje_min"].median().unstack()

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(mediana.index, mediana["centro"], "o-", color="firebrick", label="Viajes que tocan el centro")
ax.plot(mediana.index, mediana["resto"], "o-", color="steelblue", label="Resto de Santiago")
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="gray", linestyle="--", alpha=0.5)
    ax.text(x, ax.get_ylim()[1], nombre, rotation=90, va="top", ha="right", fontsize=8)
ax.set_xlabel("Año")
ax.set_ylabel("Tiempo de viaje mediano (min)")
ax.set_title("Tiempo de viaje mediano: centro vs resto (la ciudad entera se frena)")
ax.legend(fontsize=8)
fig.savefig("images/13-tiempos-tendencia.png", dpi=128, bbox_inches="tight")

# %%
# La diferencia centro - resto descuenta la congestión común a toda la
# ciudad: es el tiempo "extra" del centro. Si Bandera frenó al centro, esta
# diferencia debería crecer tras la peatonalización. No crece: se reduce.
mediana["diff"] = mediana["centro"] - mediana["resto"]
base_diff = mediana.loc[mediana.index <= 2017, "diff"].mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(mediana.index, mediana["diff"], "o-", color="purple")
ax.axhline(base_diff, color="0.6", linestyle=":", label=f"media 2014-2017 ({base_diff:.1f} min)")
ax.axhline(0, color="0.4", linewidth=0.8)
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="gray", linestyle="--", alpha=0.5)
    ax.text(x, ax.get_ylim()[1], nombre, rotation=90, va="top", ha="right", fontsize=8)
ax.set_xlabel("Año")
ax.set_ylabel("Tiempo extra del centro (centro - resto, min)")
ax.set_title("Diferencia de tiempo de viaje centro - resto (descuenta la congestión de la ciudad)")
ax.legend(fontsize=8)
fig.savefig("images/13-tiempos-diferencia.png", dpi=128, bbox_inches="tight")
pre = [2014, 2015, 2016, 2017]
peat = [2018, 2019]
did = ((mediana.loc[peat, "centro"].mean() - mediana.loc[pre, "centro"].mean())
       - (mediana.loc[peat, "resto"].mean() - mediana.loc[pre, "resto"].mean()))
print(f"Congestión de la ciudad (resto): {mediana.loc[pre, 'resto'].mean():.1f} min (2014-17) "
      f"-> {mediana.loc[2025, 'resto']:.1f} min (2025), +{mediana.loc[2025, 'resto'] - mediana.loc[pre, 'resto'].mean():.1f}")
print(f"Centro: {mediana.loc[pre, 'centro'].mean():.1f} -> {mediana.loc[2025, 'centro']:.1f} min (no empeora)")
print(f"DiD del tiempo (centro - resto, pre -> peatonal): {did:+.2f} min")
print("El centro no se frenó más que la ciudad: controlando la congestión común,")
print("no hay penalización de tiempo atribuible a la peatonalización (más bien al revés).")

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for regimen, color in zip(cols_regimen, ["#4c72b0", "#dd8452", "#55a868"]):
    sub = tiempos[(tiempos["regimen"] == regimen) & (tiempos["zona"] == "centro")]["tviaje_min"]
    ax.hist(sub, bins=np.arange(0, 120, 5), density=True, histtype="step",
            linewidth=2, color=color, label=regimen.split(" (")[0])
ax.set_xlabel("Tiempo de viaje (min)")
ax.set_ylabel("Densidad")
ax.set_title("Distribución del tiempo de viaje en el centro por régimen")
ax.legend(fontsize=8)
fig.savefig("images/13-tiempos-distribucion.png", dpi=128, bbox_inches="tight")
print("La distribución del tiempo de viaje del centro casi no cambia entre regímenes:")
print("a nivel de distribución, el efecto de Bandera no se ve. El cambio aparece solo")
print("al descontar la congestión de la ciudad, y va en contra de la penalización.")

# %% [markdown]
# ## PARTE 11. Síntesis: ¿fue beneficiosa la peatonalización?
#
# La pregunta no la resuelven los datos por sí solos, porque la
# intervención produce ganadores y perdedores que estos datos miden de
# forma desigual.
#
#   - El eje San Diego-Bandera SÍ era un corredor estructurante de la red
#     de buses (intermediación alta, decenas de líneas, conectando los
#     sectores Sur y Poniente con el Centro y el Norte). La intuición de
#     que importaba es correcta.
#   - Pero el costo de la peatonalización no es desvío en distancia: aun en
#     la red de buses quedan corredores paralelos (San Antonio, Mac Iver,
#     Morandé) que sustituyen el eje con muy pocos metros extra. El desvío
#     de los viajes sur-norte es de orden 0%.
#   - El costo real es congestión: las líneas que iban por Bandera se
#     concentran en corredores que ya iban cargados. La carga del sistema
#     no baja, se apila en menos calles. Medir cuánto empeora la velocidad
#     exige datos de capacidad y de tiempos por tramo que no tenemos.
#   - El beneficio peatonal es real pero ausente de los datos. El espacio
#     público, el flujo peatonal y la actividad comercial no aparecen en
#     las validaciones ni en el grafo de calles.
#   - A nivel de distribución no se ve nada: los mapas de participación por
#     régimen y las distribuciones de tiempo se parecen entre sí. El cambio
#     solo aparece al restar (mapas de diferencia) y al descontar las
#     tendencias comunes a toda la ciudad.
#   - La ciudad entera se frena con los años (el tiempo de viaje del resto
#     de Santiago sube varios minutos por la congestión). Comparado con esa
#     tendencia, el centro NO se frenó más: la diferencia centro - resto se
#     reduce tras la intervención. Controlando la congestión común, no hay
#     penalización de tiempo atribuible a Bandera.
#   - El plano empírico no permite atribuir nada de todos modos: sobre el
#     centro caen cuatro shocks casi juntos (peatonalización dic 2017, Metro
#     L3 ene 2019, estallido social oct 2019 con el centro como epicentro,
#     pandemia 2020).
#
# La conclusión defendible es que la peatonalización movió un eje de buses
# estructurante a corredores paralelos sin alargar mucho los recorridos, a
# cambio de más congestión sobre esos corredores y de un beneficio peatonal
# real pero no medido. "Beneficiosa para la ciudad" depende de a quién se
# pondera y cómo, no de un único número. Dos lecciones metodológicas: la
# red que se analiza debe corresponder al modo (la red de buses filtrada
# con GTFS, no todas las calles) y a la escala (el eje que cruza la ciudad,
# no un tramo del centro); y cuando varios shocks caen sobre el mismo lugar
# y momento, el análisis estructural (contrafactual sobre la red) es más
# limpio que el empírico (la serie), que queda confundido sin remedio.
print("Fin del análisis. Figuras en images/13-*.png")
