# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from chiricoca.config import setup_style
from chiricoca.geo.figures import figure_from_geodataframe
from chiricoca.maps import bubble_map
from libpysal.weights import Queen

from gdsutils.censo2024 import regiones
from gdsutils.geo import clip_geodataframe
from gdsutils.legends import bubble_size_legend, categorical_color_legend

setup_style(dpi=96)

# %%
carto = gpd.read_parquet(
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)

# %%


# %% ========================================
# PARTE 1: Proyecciones y proporciones
# =========================================
# EPSG:4326 usa grados de latitud y longitud.
# Al graficar directamente, 1° de longitud = 1° de latitud en pantalla,
# pero en la realidad un grado de longitud se acorta hacia los polos.
# ¿Cuánto importa esta distorsión? Depende de la escala geográfica.

carto_rm = carto[carto["CUT"] // 1000 == 13].copy()

bbox_stgo = regiones["RM"]["capital_bbox"]


carto_stgo = clip_geodataframe(carto_rm, bbox_stgo)

escalas = [
    ("País (Chile)", carto),
    ("Región (RM)", carto_rm),
    ("Ciudad (Gran Santiago)", carto_stgo),
]

# %%
fig, axes = plt.subplots(3, 2, figsize=(8, 14))

plot_args = dict(edgecolor="white", facecolor="steelblue", linewidth=0.2)

for i, (nombre, geodf) in enumerate(escalas):
    bounds = geodf.total_bounds
    y_mean = np.mean(bounds[[1, 3]])

    for j, ax in enumerate(axes[i]):
        geodf.plot(ax=ax, **plot_args)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_axis_off()

    axes[i, 0].set_aspect("equal")
    axes[i, 0].set_title(f"{nombre}\nAspecto 1:1")

    axes[i, 1].set_aspect(1 / np.cos(np.radians(y_mean)))
    axes[i, 1].set_title(f"{nombre}\nCorrección cos({y_mean:.0f}°)")


# %%
# A escala de país, la distorsión es evidente: el sur aparece comprimido.
# A escala de ciudad, la corrección por cos(lat) es prácticamente perfecta
# porque el rango de latitudes es muy pequeño.
# Una proyección métrica resuelve esto para cualquier escala.

carto_proj = carto.to_crs(epsg=5361)

fig, ax = figure_from_geodataframe(carto_proj, height=8)
carto_proj.plot(ax=ax, edgecolor="white", facecolor="steelblue", linewidth=0.2)
ax.set_title("EPSG:5361 (SIRGAS-Chile 2002 / UTM 19S)")

# %% ========================================
# PARTE 2: Mapa de burbujas (Santiago)
# =========================================

# Cargar manzanas de la RM y agregar población migrante por distrito censal
manzanas_rm = gpd.read_parquet(
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Manzanas.parquet",
    filters=[("COD_REGION", "=", 13)],
)

datos_por_distrito = manzanas_rm.groupby("ID_DISTRITO")[
    ["n_inmigrantes", "n_per", "n_hog"]
].sum()

datos_por_distrito["tam_hogar"] = (
    datos_por_distrito["n_per"] / datos_por_distrito["n_hog"]
)

# %%
# Cargar cartografía distrital y recortar al Gran Santiago
distritos_rm = gpd.read_parquet(
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Distrital.parquet",
    filters=[("COD_REGION", "=", 13)],
)

distritos_stgo = clip_geodataframe(distritos_rm, bbox_stgo)

# Unir con datos agregados
distritos_stgo = distritos_stgo.set_index("ID_DISTRITO").join(datos_por_distrito)

# %%
# Discretizar tamaño del hogar en terciles
import pandas as pd

distritos_stgo["cat_tam_hogar"], bins_tam_hogar = pd.qcut(
    distritos_stgo["tam_hogar"],
    q=3,
    labels=["Pequeño", "Mediano", "Grande"],
    retbins=True,
)

print("Terciles tamaño del hogar (personas/hogar):")
for i, label in enumerate(["Pequeño", "Mediano", "Grande"]):
    print(f"  {label}: [{bins_tam_hogar[i]:.2f}, {bins_tam_hogar[i + 1]:.2f}]")

# Centroides para el mapa de burbujas
centroides = distritos_stgo.copy()
centroides = centroides.set_geometry(centroides.centroid)

# %%
palette = dict(
    zip(["Pequeño", "Mediano", "Grande"], sns.color_palette("plasma", n_colors=3))
)

fig, ax = figure_from_geodataframe(distritos_stgo, height=8)
distritos_stgo.plot(ax=ax, edgecolor="grey", facecolor="whitesmoke", linewidth=0.3)

scale = 0.05

for cat, grupo in centroides.groupby("cat_tam_hogar"):
    bubble_map(
        grupo,
        "n_inmigrantes",
        scale=scale,
        ax=ax,
        color=palette[cat],
        edgecolor="white",
        alpha=0.7,
        add_legend=False,
    )

categorical_color_legend(ax, palette, title="Tamaño del hogar\n(personas/hogar)")
bubble_size_legend(ax, scale, centroides["n_inmigrantes"], title="N° migrantes")

# Anotar las burbujas más grandes para verificar
top = centroides.nlargest(3, "n_inmigrantes")
for _, row in top.iterrows():
    ax.annotate(
        f"{int(row['n_inmigrantes']):,}",
        xy=(row.geometry.x, row.geometry.y),
        fontsize=6,
        ha="center",
        va="bottom",
        xytext=(0, 8),
        textcoords="offset points",
        color="black",
    )

ax.set_title(
    "Población migrante por distrito censal (Gran Santiago)\nTamaño: n° migrantes; Color: tamaño del hogar"
)

# %% ========================================
# PARTE 3: Red de vecindad comunal
# =========================================
# Dos comunas son vecinas si sus geometrías se tocan (comparten borde).
# Esto define un grafo de vecindad (contigüidad).


w = Queen.from_dataframe(carto_rm, ids="CUT")

G = nx.Graph(w.neighbors)

# %%
# Posiciones de los nodos = centroides de cada comuna
centroides_rm = carto_rm.set_index("CUT").geometry.centroid
pos = {node: (centroides_rm[node].x, centroides_rm[node].y) for node in G.nodes}

# %%
fig, ax = figure_from_geodataframe(carto_rm, height=8)
carto_rm.plot(ax=ax, edgecolor="grey", facecolor="whitesmoke", linewidth=0.5)

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="steelblue", width=1.5, alpha=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color="steelblue")

ax.set_title("Red de vecindad comunal (RM)")

# %%
# Estadísticas básicas del grafo
print(f"Nodos (comunas): {G.number_of_nodes()}")
print(f"Aristas (vecindades): {G.number_of_edges()}")
print(f"Grado promedio: {np.mean([d for _, d in G.degree()]):.1f}")
print(f"Componentes conexas: {nx.number_connected_components(G)}")
