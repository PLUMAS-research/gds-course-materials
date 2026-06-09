# %%
"""Clustering espacial sobre reportes SOSAFE y zonas censales de Santiago.

El script recorre tres preguntas distintas que comparten el nombre "clustering":

1. Detectar concentraciones de puntos (hotspots de reportes ciudadanos). Se
   comparan k-means (ciego a la densidad), DBSCAN (un umbral de densidad) y
   HDBSCAN (densidades múltiples).
2. Regionalizar el territorio en zonas contiguas y homogéneas, combinando el
   perfil censal y la densidad de reportes por hexágono. Se comparan SKATER y
   Ward con restricción de contigüidad.
3. Cuantificar el precio de la contigüidad: k-means produce particiones
   homogéneas pero geográficamente fragmentadas; la regionalización las evita.

Datos:
- Puntos: dos semanas de reportes SOSAFE 2024 (anonimizados), tres grupos
  (Delitos, Disturbios, Ambiental).
- Zonas: hexágonos H3-8 del Gran Santiago con perfil demográfico del Censo 2024
  (asignación de la clase 08), más la densidad de reportes SOSAFE por grupo.
"""

# %% Descargas
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/sosafe-clustering.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-asignacion-rm.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.maps import choropleth_map
from libpysal.weights import Queen
from shapely import concave_hull
from shapely.geometry import MultiPoint, Polygon, box
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from spopt.region import Skater, WardSpatial

setup_style(dpi=96)

# %% Configuración
CRS_UTM_19S = "EPSG:32719"
BBOX = (-70.85, -33.65, -70.45, -33.30)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

COLORES_GRUPO = {
    "Ambiental": "#2a9d8f",
    "Disturbios": "#e9c46a",
    "Delitos": "#e76f51",
}
GRIS_RUIDO = "#cccccc"

# %%
# ========================================================================
# PARTE 1. Carga y preparación
# ========================================================================

# Puntos: reportes SOSAFE de la quincena, recortados a la ventana del Gran
# Santiago (la misma bbox de los demás scripts) y proyectados a metros (UTM 19S)
# para que las distancias de DBSCAN/HDBSCAN sean isotrópicas y en metros.
reportes = gpd.read_parquet(
    Path("data") / "sosafe-clustering" / "reportes-quincena.parquet"
)
reportes = reportes.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].to_crs(CRS_UTM_19S)
reportes["x"] = reportes.geometry.x
reportes["y"] = reportes.geometry.y
print(f"Reportes en la ventana: {len(reportes):,}")
print(reportes["grupo"].value_counts())

# Comunas como contexto visual de fondo.
comunas = gpd.read_parquet(
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
comunas = comunas.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]].to_crs(CRS_UTM_19S)
print(f"Comunas en la ventana: {len(comunas)}")

# La bbox en UTM fija los límites de los mapas de puntos: sin esto, las comunas
# de la periferia (que se dibujan completas) expanden la vista a zonas rurales.
BBOX_UTM = gpd.GeoSeries([box(*BBOX)], crs="EPSG:4326").to_crs(CRS_UTM_19S).total_bounds


def marco_santiago(ax):
    """Fondo de comunas y límites fijos del Gran Santiago para mapas de puntos."""
    comunas.boundary.plot(ax=ax, color="lightgray", linewidth=0.5, zorder=1)
    ax.set_xlim(BBOX_UTM[0], BBOX_UTM[2])
    ax.set_ylim(BBOX_UTM[1], BBOX_UTM[3])
    # Evita que los scatter posteriores reajusten la vista a las comunas, que se
    # dibujan completas y exceden la bbox.
    ax.set_autoscale_on(False)
    ax.set_axis_off()


def dibujar_alpha_shapes(ax, df, columna, ratio=0.4):
    """Dibuja el alpha shape (concave hull) de cada cluster con su etiqueta.

    El alpha shape envuelve los puntos del cluster con un contorno cóncavo, más
    fiel a la forma real que el convex hull. `ratio` controla cuán cóncavo es
    (0 = muy cóncavo, 1 = convex hull). La etiqueta es el id del cluster.
    """
    etiquetas = sorted(e for e in df[columna].unique() if e != -1)
    colores = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    for i, etiqueta in enumerate(etiquetas):
        grupo = df[df[columna] == etiqueta]
        if len(grupo) < 3:
            continue
        color = colores[i % 20]
        puntos = MultiPoint(list(zip(grupo["x"], grupo["y"])))
        forma = concave_hull(puntos, ratio=ratio)
        gpd.GeoSeries([forma]).boundary.plot(
            ax=ax, color=color, linewidth=1.5, zorder=4
        )
        cx, cy = forma.centroid.x, forma.centroid.y
        ax.annotate(
            str(etiqueta),
            (cx, cy),
            ha="center",
            va="center",
            zorder=5,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, lw=1.5),
        )


# %%
# Zonas: hexágonos H3-8 con perfil censal. El parquet trae los conteos pero no
# la geometría; se reconstruye el polígono de cada hexágono desde su id H3.
zonas = pd.read_parquet(
    Path("data") / "censo2024-asignacion-rm" / "asignaciones-h3-8.parquet"
)


def poligono_h3(cell_id):
    # cell_to_boundary entrega (lat, lng); Polygon espera (x=lng, y=lat).
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell_id)])


zonas = gpd.GeoDataFrame(
    zonas,
    geometry=zonas["h3_cell_id"].map(poligono_h3),
    crs="EPSG:4326",
).to_crs(CRS_UTM_19S)
print(f"Hexágonos H3-8: {len(zonas)}")

# %%
# ========================================================================
# PARTE 2. K-means sobre coordenadas: el contraste
# ========================================================================
# K-means aplicado a las coordenadas de los puntos particiona el plano en
# celdas compactas (tipo Voronoi). No detecta densidad: asigna todos los
# puntos a algún grupo, incluso los aislados, y las fronteras son arbitrarias.
# El grupo Disturbios es el caso de estudio de las partes 2 a 4: a diferencia
# de Delitos (un solo gradiente central), tiene concentraciones discretas en
# torno a mercados, comercio y zonas de vida nocturna.

GRUPO_FOCO = "Disturbios"
foco = reportes[reportes["grupo"] == GRUPO_FOCO].copy()
coords_foco = foco[["x", "y"]].to_numpy()
print(f"Reportes de {GRUPO_FOCO}: {len(foco):,}")

K_KMEANS_PUNTOS = 10
km_puntos = KMeans(n_clusters=K_KMEANS_PUNTOS, random_state=0, n_init=10)
foco["km_label"] = km_puntos.fit_predict(coords_foco)

fig, ax = plt.subplots(figsize=(8, 8))
marco_santiago(ax)
ax.scatter(
    foco["x"], foco["y"], c=foco["km_label"], cmap="tab10", s=4, alpha=0.6, zorder=2
)
ax.scatter(
    km_puntos.cluster_centers_[:, 0],
    km_puntos.cluster_centers_[:, 1],
    c="black",
    marker="x",
    s=120,
    zorder=3,
)
ax.set_title(
    f"K-means sobre coordenadas (k={K_KMEANS_PUNTOS})\nparticiona el plano, no detecta densidad"
)
fig.savefig(IMAGES / "11-kmeans-puntos.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-kmeans-puntos.png")

# %%
# ========================================================================
# PARTE 3. DBSCAN: hotspots por densidad con un umbral
# ========================================================================
# minPts fija cuántos vecinos definen un núcleo. eps se calibra con el gráfico
# de k-distancias: para cada punto, distancia a su minPts-ésimo vecino, ordenada
# de menor a mayor. El codo marca la transición entre densidad de cluster y
# densidad de fondo.

MIN_PTS = 30
vecinos = NearestNeighbors(n_neighbors=MIN_PTS).fit(coords_foco)
dist_k, _ = vecinos.kneighbors(coords_foco)
dist_ordenada = np.sort(dist_k[:, -1])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dist_ordenada)
ax.set_xlabel("Puntos ordenados")
ax.set_ylabel(f"Distancia al vecino {MIN_PTS} (m)")
ax.set_title("Gráfico de k-distancias: el codo sugiere eps")
ax.set_ylim(0, np.percentile(dist_ordenada, 99))
ax.grid(alpha=0.3)
fig.savefig(IMAGES / "11-dbscan-kdistancia.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-dbscan-kdistancia.png")

# %%
# El codo del gráfico cae alrededor de 300 m para esta densidad de reportes.
EPS_DBSCAN = 300
db = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_PTS).fit(coords_foco)
foco["db_label"] = db.labels_
n_clusters_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_ruido_db = int((db.labels_ == -1).sum())
print(
    f"DBSCAN (eps={EPS_DBSCAN} m, minPts={MIN_PTS}): "
    f"{n_clusters_db} clusters, {n_ruido_db:,} puntos de ruido "
    f"({100 * n_ruido_db / len(foco):.0f}%)"
)


def graficar_clusters(ax, df, columna, titulo):
    """Dibuja clusters de puntos: ruido en gris, cada cluster en un color."""
    marco_santiago(ax)
    ruido = df[df[columna] == -1]
    ax.scatter(ruido["x"], ruido["y"], c=GRIS_RUIDO, s=3, alpha=0.4, zorder=2)
    senal = df[df[columna] != -1]
    ax.scatter(
        senal["x"], senal["y"], c=senal[columna], cmap="tab20", s=6, alpha=0.8, zorder=3
    )
    ax.set_title(titulo)


fig, ax = plt.subplots(figsize=(8, 8))
graficar_clusters(
    ax,
    foco,
    "db_label",
    f"DBSCAN: {n_clusters_db} hotspots de {GRUPO_FOCO}\n{100 * n_ruido_db / len(foco):.0f}% etiquetado como ruido",
)
fig.savefig(IMAGES / "11-dbscan-foco.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-dbscan-foco.png")

# %%
# ========================================================================
# PARTE 4. HDBSCAN: densidades múltiples sin elegir eps
# ========================================================================
# HDBSCAN no recibe eps. min_cluster_size es el mínimo de puntos que constituye
# un hotspot. A diferencia de DBSCAN, puede reportar a la vez un hotspot
# compacto en el centro y una concentración difusa en la periferia.

MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 10
hdb = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit(
    coords_foco
)
foco["hdb_label"] = hdb.labels_
n_clusters_hdb = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
n_ruido_hdb = int((hdb.labels_ == -1).sum())
print(
    f"HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}): "
    f"{n_clusters_hdb} clusters, {n_ruido_hdb:,} puntos de ruido "
    f"({100 * n_ruido_hdb / len(foco):.0f}%)"
)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
graficar_clusters(axes[0], foco, "db_label", f"DBSCAN: {n_clusters_db} clusters")
graficar_clusters(axes[1], foco, "hdb_label", f"HDBSCAN: {n_clusters_hdb} clusters")
fig.suptitle(
    f"Mismo dataset ({GRUPO_FOCO}): un umbral de densidad vs densidades múltiples"
)
fig.savefig(IMAGES / "11-dbscan-vs-hdbscan.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-dbscan-vs-hdbscan.png")

# %%
# ========================================================================
# PARTE 5. HDBSCAN por grupo: no todo fenómeno tiene hotspots separables
# ========================================================================
# Con los mismos parámetros para los tres grupos, HDBSCAN refleja la estructura
# real de cada uno: Disturbios se concentra en decenas de puntos discretos
# (mercados, comercio, vida nocturna), mientras Delitos y reportes ambientales
# forman un único gradiente central denso que el método reporta como un solo
# cluster dominante. La detección de hotspots solo tiene sentido cuando existen
# valles de densidad entre concentraciones.

MCS_GRUPOS = 40
grupos = ["Delitos", "Disturbios", "Ambiental"]
fig, axes = small_multiples_from_geodataframe(
    reportes, len(grupos), col_wrap=3, height=6, bbox=tuple(BBOX_UTM)
)
for ax, grupo in zip(axes, grupos):
    sub = reportes[reportes["grupo"] == grupo].copy()
    etiquetas = HDBSCAN(
        min_cluster_size=MCS_GRUPOS, min_samples=MIN_SAMPLES
    ).fit_predict(sub[["x", "y"]].to_numpy())
    sub["label"] = etiquetas
    n_cl = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    marco_santiago(ax)
    ruido = sub[sub["label"] == -1]
    ax.scatter(ruido["x"], ruido["y"], c=GRIS_RUIDO, s=2, alpha=0.3, zorder=2)
    senal = sub[sub["label"] != -1]
    ax.scatter(
        senal["x"], senal["y"], c=senal["label"], cmap="tab20", s=5, alpha=0.8, zorder=3
    )
    ax.set_title(f"{grupo}: {n_cl} hotspots")
fig.savefig(IMAGES / "11-hdbscan-grupos.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-hdbscan-grupos.png")

# %%
# ========================================================================
# PARTE 6. Caracterización de los hotspots: contorno y ritmo temporal
# ========================================================================
# Un cluster no es solo una mancha de puntos: tiene una forma y un horario. El
# alpha shape (concave hull) delimita su extensión y se etiqueta con su id. El
# perfil por hora del día y por día de semana distingue ritmos: hotspots de
# vida nocturna (madrugada, fin de semana) frente a comercio o mercados (diurno,
# días hábiles). Se caracterizan los hotspots de Disturbios más grandes.

TOP_N = 12
tam = foco.loc[foco["hdb_label"] != -1, "hdb_label"].value_counts()
top_ids = tam.head(TOP_N).index.tolist()
print(f"Caracterizando los {len(top_ids)} hotspots más grandes de {GRUPO_FOCO}")

fig, ax = plt.subplots(figsize=(9, 9))
marco_santiago(ax)
fondo = foco[~foco["hdb_label"].isin(top_ids)]
ax.scatter(fondo["x"], fondo["y"], c=GRIS_RUIDO, s=3, alpha=0.3, zorder=2)
# Color discreto por hotspot, consistente entre puntos y contorno (mismo orden
# sorted y misma paleta que dibujar_alpha_shapes).
colores20 = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
mapa_color = {e: colores20[i % 20] for i, e in enumerate(sorted(top_ids))}
destacados = foco[foco["hdb_label"].isin(top_ids)]
ax.scatter(
    destacados["x"],
    destacados["y"],
    c=[mapa_color[e] for e in destacados["hdb_label"]],
    s=10,
    alpha=0.8,
    zorder=3,
)
dibujar_alpha_shapes(ax, destacados, "hdb_label")
ax.set_title(f"Hotspots de {GRUPO_FOCO}: contorno (alpha shape) y etiqueta")
fig.savefig(IMAGES / "11-hotspots-shapes.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-hotspots-shapes.png")

# %%
# Perfiles temporales: cada fila es un hotspot, normalizada para comparar la
# forma del ritmo (no el volumen). Columnas: hora del día (izq.) y día de
# semana (der.). Los hotspots se ordenan por tamaño.
sub = foco[foco["hdb_label"].isin(top_ids)]
perfil_hora = (
    sub.groupby(["hdb_label", "hora"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(24), fill_value=0)
    .loc[top_ids]
)
perfil_dia = (
    sub.groupby(["hdb_label", "dia_semana"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(7), fill_value=0)
    .loc[top_ids]
)
perfil_hora_n = perfil_hora.div(perfil_hora.sum(axis=1), axis=0)
perfil_dia_n = perfil_dia.div(perfil_dia.sum(axis=1), axis=0)

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
vmax = max(perfil_hora_n.to_numpy().max(), perfil_dia_n.to_numpy().max())
fig, axes = plt.subplots(
    1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [24, 7, 1]}
)
sns.heatmap(
    perfil_hora_n,
    ax=axes[0],
    cmap="magma",
    vmin=0,
    vmax=vmax,
    cbar=False,
    xticklabels=2,
)
axes[0].set_xlabel("Hora del día")
axes[0].set_ylabel("Hotspot (id HDBSCAN)")
axes[0].set_title("Ritmo horario")
axes[0].tick_params(axis="y", rotation=0)

sns.heatmap(
    perfil_dia_n,
    ax=axes[1],
    cmap="magma",
    vmin=0,
    vmax=vmax,
    cbar_ax=axes[2],
    cbar_kws={"label": "Fracción de reportes del hotspot"},
    xticklabels=DIAS,
    yticklabels=False,
)
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_title("Ritmo semanal")

fig.suptitle(f"Caracterización temporal de los hotspots de {GRUPO_FOCO}")
fig.savefig(IMAGES / "11-hotspots-temporal.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-hotspots-temporal.png")

# %%
# ========================================================================
# PARTE 7. Zonas: perfil demográfico, reportes y grafo de contigüidad
# ========================================================================
# Para regionalizar se combinan dos fuentes por hexágono: el perfil demográfico
# del censo (proporciones, acotadas a [0, 1]) y la densidad de reportes SOSAFE
# por grupo. Así las regiones reflejan a la vez quién vive en cada zona y qué se
# reporta en ella. Todo se estandariza para que cada variable pese igual.

zonas["poblacion"] = zonas[["n_hombres", "n_mujeres"]].sum(axis=1)
zonas = zonas[zonas["poblacion"] > 0].copy().reset_index(drop=True)

# --- Atributos demográficos (proporciones) ---
zonas["frac_mujeres"] = zonas["n_mujeres"] / zonas["poblacion"]
zonas["frac_mayores"] = zonas["n_edad_60_mas"] / zonas["poblacion"]
zonas["frac_ninos"] = (
    zonas[["n_edad_0_5", "n_edad_6_13"]].sum(axis=1) / zonas["poblacion"]
)
zonas["frac_inmigrantes"] = zonas["n_inmigrantes"] / zonas["poblacion"]
zonas["frac_terciaria"] = (
    zonas["n_cine_terciaria_maestria_doctorado"] / zonas["poblacion"]
)
zonas["frac_auto"] = zonas["n_transporte_auto"] / zonas[
    [c for c in zonas.columns if c.startswith("n_transporte_")]
].sum(axis=1)
zonas["frac_depto"] = zonas["n_tipo_viv_depto"] / zonas[
    [c for c in zonas.columns if c.startswith("n_tipo_viv_")]
].sum(axis=1)

ATRIBUTOS_DEMO = [
    "frac_mujeres",
    "frac_mayores",
    "frac_ninos",
    "frac_inmigrantes",
    "frac_terciaria",
    "frac_auto",
    "frac_depto",
]

# --- Atributos SOSAFE (densidad de reportes por grupo) ---
# Spatial join de los puntos a los hexágonos y conteo por grupo. La densidad
# (reportes/km^2) no depende de la población residente, relevante en zonas
# comerciales con muchos reportes y pocos habitantes. Para el clustering se usa
# log1p de la densidad: sin esa transformación, la cola del centro domina la
# estandarización y la regionalización ignoraría la demografía.
area_km2 = zonas.geometry.area / 1e6
r_hex = gpd.sjoin(
    reportes[["geometry", "grupo"]],
    zonas[["geometry"]],
    how="inner",
    predicate="within",
)
conteos_hex = (
    r_hex.groupby(["index_right", "grupo"])
    .size()
    .unstack(fill_value=0)
    .reindex(zonas.index, fill_value=0)
)
ATRIBUTOS_SOSAFE = []
for grupo in ["Delitos", "Disturbios", "Ambiental"]:
    g = grupo.lower()
    zonas[f"dens_{g}"] = conteos_hex.get(grupo, 0) / area_km2
    zonas[f"log_dens_{g}"] = np.log1p(zonas[f"dens_{g}"])
    ATRIBUTOS_SOSAFE.append(f"log_dens_{g}")

ATRIBUTOS = ATRIBUTOS_DEMO + ATRIBUTOS_SOSAFE
zonas[ATRIBUTOS] = zonas[ATRIBUTOS].fillna(0.0)
print(
    f"Atributos para clustering: {len(ATRIBUTOS_DEMO)} demográficos + "
    f"{len(ATRIBUTOS_SOSAFE)} de reportes"
)

# Grafo de contigüidad Queen. Para hexágonos coincide con las seis vecinas.
# La regionalización exige un grafo conexo: se restringe al componente mayor.
w = Queen.from_dataframe(zonas, use_index=True)
grafo = nx.Graph(w.neighbors)
componentes = list(nx.connected_components(grafo))
mayor = max(componentes, key=len)
print(f"Componentes conexas: {len(componentes)}; mayor con {len(mayor)} hexágonos")
zonas = zonas.iloc[sorted(mayor)].reset_index(drop=True)

# Estandarizar después de fijar el subconjunto final.
escalador = StandardScaler()
X = escalador.fit_transform(zonas[ATRIBUTOS].to_numpy())
zonas_z = zonas.copy()
zonas_z[ATRIBUTOS] = X
w = Queen.from_dataframe(zonas_z, use_index=True)
print(f"Zonas para regionalización: {len(zonas_z)}")

# %%
# Variación espacial de los atributos que alimentan el clustering. Cada uno
# tiene su propia estructura: la regionalización busca cortes que respeten las
# diez capas a la vez. Las demográficas se muestran como proporciones crudas y
# las de reportes como densidad cruda (reportes/km$^2$), por ser más
# interpretables que sus versiones estandarizadas.
config_atributos = [
    ("frac_mujeres", "Proporción de mujeres", "magma"),
    ("frac_mayores", "Proporción de 60+ años", "magma"),
    ("frac_ninos", "Proporción de niños (0-13)", "magma"),
    ("frac_inmigrantes", "Proporción de inmigrantes", "magma"),
    ("frac_terciaria", "Proporción con educación terciaria", "magma"),
    ("frac_auto", "Proporción que viaja en auto", "magma"),
    ("frac_depto", "Proporción en departamento", "magma"),
    ("dens_delitos", "Densidad de Delitos (rep/km$^2$)", "rocket"),
    ("dens_disturbios", "Densidad de Disturbios (rep/km$^2$)", "rocket"),
    ("dens_ambiental", "Densidad de Ambiental (rep/km$^2$)", "rocket"),
]
fig, axes = small_multiples_from_geodataframe(
    zonas, len(config_atributos), col_wrap=4, height=4
)
for ax, (col, titulo, paleta) in zip(axes, config_atributos):
    choropleth_map(
        zonas,
        col,
        k=5,
        binning="fisher_jenks",
        palette=paleta,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4)
    ax.set_title(titulo)
fig.suptitle("Atributos por hexágono (H3-8): demografía (censo) y reportes (SOSAFE)")
fig.savefig(IMAGES / "11-atributos-zonas.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-atributos-zonas.png")

# %%
# ========================================================================
# PARTE 8. K-means sobre atributos: homogéneo pero fragmentado
# ========================================================================
# K-means sobre los atributos (sin espacio) produce una tipología coherente,
# pero al mapearla aparece "sal y pimienta": zonas del mismo grupo dispersas
# por toda la ciudad. Agregar las coordenadas como features atenúa el efecto
# pero no garantiza contigüidad y depende del escalado relativo.

K_REGIONES = 7

km_attr = KMeans(n_clusters=K_REGIONES, random_state=0, n_init=10)
zonas_z["kmeans_attr"] = km_attr.fit_predict(X)

xy = StandardScaler().fit_transform(
    np.column_stack([zonas_z.geometry.centroid.x, zonas_z.geometry.centroid.y])
)
X_coords = np.column_stack([X, xy])
km_coords = KMeans(n_clusters=K_REGIONES, random_state=0, n_init=10)
zonas_z["kmeans_coords"] = km_coords.fit_predict(X_coords)

# %%
# ========================================================================
# PARTE 9. SKATER y Ward: regionalización con contigüidad garantizada
# ========================================================================
# SKATER poda el árbol de expansión mínima del grafo de contigüidad. Ward con
# restricción de contigüidad fusiona solo vecinos. Ambos minimizan la
# heterogeneidad intra-región y producen exactamente K regiones contiguas.

skater = Skater(zonas_z, w, ATRIBUTOS, n_clusters=K_REGIONES)
skater.solve()
zonas_z["skater"] = skater.labels_

ward = WardSpatial(zonas_z, w, ATRIBUTOS, n_clusters=K_REGIONES)
ward.solve()
zonas_z["ward"] = ward.labels_

print("Regionalización resuelta (SKATER y Ward)")
for m in ["skater", "ward"]:
    tam = sorted(zonas_z.groupby(m).size().tolist(), reverse=True)
    print(f"  Tamaños de región {m}: {tam}")

# %%
metodos = [
    ("kmeans_attr", "K-means (atributos)"),
    ("kmeans_coords", "K-means (atributos + coordenadas)"),
    ("skater", "SKATER"),
    ("ward", "Ward + contigüidad"),
]
fig, axes = small_multiples_from_geodataframe(
    zonas_z, len(metodos), col_wrap=2, height=6
)
for ax, (col, titulo) in zip(axes, metodos):
    zonas_z.plot(column=col, categorical=True, cmap="tab10", ax=ax, linewidth=0)
    ax.set_title(titulo)
    ax.set_axis_off()
fig.savefig(IMAGES / "11-regionalizacion.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-regionalizacion.png")

# %%
# ========================================================================
# PARTE 10. El precio de la contigüidad: homogeneidad vs fragmentación
# ========================================================================
# Dos métricas a igual K:
# - SSD intra-región: heterogeneidad de atributos (menor = más homogéneo).
# - Fragmentos espaciales: componentes conexas por etiqueta en el grafo de
#   contigüidad. SKATER y Ward dan K por construcción; k-means da muchos más.


def ssd_intra(labels):
    total = 0.0
    for etiqueta in np.unique(labels):
        bloque = X[labels == etiqueta]
        total += ((bloque - bloque.mean(axis=0)) ** 2).sum()
    return total


def fragmentos_espaciales(labels):
    n = 0
    for etiqueta in np.unique(labels):
        idx = set(np.where(labels == etiqueta)[0])
        sub = grafo_z.subgraph(idx)
        n += nx.number_connected_components(sub)
    return n


grafo_z = nx.Graph(w.neighbors)
resumen = pd.DataFrame(
    [
        {
            "metodo": titulo,
            "ssd_intra": ssd_intra(zonas_z[col].to_numpy()),
            "fragmentos": fragmentos_espaciales(zonas_z[col].to_numpy()),
        }
        for col, titulo in metodos
    ]
)
resumen["ssd_intra"] = resumen["ssd_intra"].round(0)
print(
    f"\nHomogeneidad (SSD intra) vs coherencia espacial (fragmentos), K={K_REGIONES}:"
)
print(resumen.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(resumen["ssd_intra"], resumen["fragmentos"], s=80, color="#264653")
for _, fila in resumen.iterrows():
    ax.annotate(
        fila["metodo"],
        (fila["ssd_intra"], fila["fragmentos"]),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=9,
    )
ax.set_xlabel("SSD intra-región (heterogeneidad de atributos)")
ax.set_ylabel("Fragmentos espaciales")
ax.set_title(
    f"El precio de la contigüidad (K={K_REGIONES})\nregionalizar sube algo la SSD pero elimina la fragmentación"
)
ax.grid(alpha=0.3)
fig.savefig(IMAGES / "11-precio-contiguidad.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-precio-contiguidad.png")

# %%
# ========================================================================
# PARTE 11. Caracterización de las regiones: qué representa cada una
# ========================================================================
# Una partición no se interpreta sola: hay que leer el perfil de atributos de
# cada región. El heatmap muestra la media estandarizada (z-score) de cada
# atributo por región. Rojo = sobre el promedio de la ciudad, azul = bajo. Cada
# fila es la "firma" de una región (por ejemplo, alta educación terciaria y uso
# de auto = sector oriente; alta densidad de reportes = centro).
#
# Se caracteriza la partición de Ward: con el mismo objetivo y K, SKATER dejó
# una región dominante (~800 de 1220 hexágonos), mientras Ward reparte de forma
# más pareja y produce un mapa más legible. El balance no es parte del objetivo
# que ambos optimizan, así que difiere entre las dos heurísticas.

METODO_FINAL = "ward"
ETIQUETAS_ATRIBUTOS = {
    "frac_mujeres": "Mujeres",
    "frac_mayores": "60+ años",
    "frac_ninos": "Niños",
    "frac_inmigrantes": "Inmigrantes",
    "frac_terciaria": "Ed. terciaria",
    "frac_auto": "Auto",
    "frac_depto": "Departamento",
    "log_dens_delitos": "Delitos",
    "log_dens_disturbios": "Disturbios",
    "log_dens_ambiental": "Ambiental",
}
perfil = zonas_z.groupby(METODO_FINAL)[ATRIBUTOS].mean()
perfil.columns = [ETIQUETAS_ATRIBUTOS[c] for c in perfil.columns]
tam_region = zonas_z.groupby(METODO_FINAL).size()
perfil.index = [f"R{i} (n={tam_region[i]})" for i in perfil.index]

# Mapa de las regiones etiquetadas (izquierda) y perfil de atributos (derecha).
fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.3]})
zonas_z.plot(
    column=METODO_FINAL, categorical=True, cmap="tab10", ax=axes[0], linewidth=0
)
comunas.boundary.plot(ax=axes[0], color="black", linewidth=0.3, alpha=0.4)
disuelto = zonas_z.dissolve(by=METODO_FINAL)
for i, geom in disuelto.geometry.items():
    axes[0].annotate(
        f"R{i}",
        (geom.centroid.x, geom.centroid.y),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=1),
    )
# Límites al extent de los hexágonos: las comunas se dibujan completas y, sin
# fijar los límites, expandirían la vista a toda la RM.
bz = zonas_z.total_bounds
axes[0].set_xlim(bz[0], bz[2])
axes[0].set_ylim(bz[1], bz[3])
axes[0].set_title(f"Regiones de {METODO_FINAL.upper()}")
axes[0].set_axis_off()

# Escala acotada al percentil 95 (no al máximo): una región puede ser un solo
# hexágono atípico con z-scores extremos que, sin acotar, aplastan el contraste
# de las regiones grandes. Los valores reales quedan en las anotaciones.
lim = np.percentile(np.abs(perfil.to_numpy()), 95)
sns.heatmap(
    perfil,
    ax=axes[1],
    cmap="RdBu_r",
    center=0,
    vmin=-lim,
    vmax=lim,
    annot=True,
    fmt=".1f",
    annot_kws={"size": 8},
    cbar_kws={"label": "Media estandarizada (z-score)"},
)
axes[1].set_title("Perfil de atributos por región")
axes[1].set_ylabel("")
axes[1].tick_params(axis="y", rotation=0)
axes[1].tick_params(axis="x", rotation=45)
fig.suptitle(f"Caracterización de las regiones (K={K_REGIONES})")
fig.savefig(IMAGES / "11-regiones-perfil.png", bbox_inches="tight", dpi=150)
print("Guardado: 11-regiones-perfil.png")
