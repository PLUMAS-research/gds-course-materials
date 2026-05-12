# %% [markdown]
# # Asignacion censal y desagregacion a una grilla regular
#
# Los datos del Censo 2024 se publican como conteos por manzana. Para
# cruzarlos con datasets de otras escalas (eBird en H3-8, NDVI en pixel,
# SINCA en estaciones puntuales) necesitamos llevarlos a una grilla
# comun. En esta clase pasamos de manzanas a celdas H3 con dos
# enfoques: areal weighting (reparto proporcional al area) y asignacion
# con microdatos + simulated annealing (preserva correlaciones
# intra-vivienda).

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-microdatos.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-asignacion-rm.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chiricoca.base.weights import normalize_rows
from chiricoca.config import setup_style
from chiricoca.geo.figures import (
    figure_from_geodataframe,
    small_multiples_from_geodataframe,
    small_multiples_from_geodataframes,
)
from chiricoca.geo.grid import h3_grid_from_bounds, h3_grid_from_ids
from chiricoca.maps import choropleth_map
from h3 import cell_to_parent

setup_style(dpi=128)

# %%
from gdsutils.asignacion import (
    asignar_viviendas,
    diagnosticar_atributos,
    reportar_errores,
    visualizar_errores,
)
from gdsutils.censo2024 import regiones, tabla_viviendas_desde_microdatos
from gdsutils.ndvi import descargar_ndvi_santiago_precomputado, ndvi_por_zona

# %% [markdown]
# ## PARTE 1. El problema del cambio de soporte
#
# Las manzanas son poligonos irregulares (privados por confidencialidad,
# tamanho variable). Las celdas H3 son hexagonos regulares (resolucion
# uniforme, jerarquicas). Cambiar de un soporte al otro implica
# redistribuir conteos. Comparamos visualmente el contraste entre dos
# comunas: Santiago (alta densidad, dominada por departamentos) y La
# Pintana (baja densidad, casi todo casa).

# %%
BBOX_STGO = regiones["RM"]["capital_bbox"]

manzanas_rm = (
    gpd.read_parquet(
        Path("data")
        / "censo2024-cartografia"
        / "Cartografia_censo2024_Pais_Manzanas.parquet",
        filters=[("COD_REGION", "=", 13)],
    )
    .assign(
        geometry=lambda x: x.clip_by_rect(*BBOX_STGO),
        MANZENT=lambda x: x["MANZENT"].astype(int),
        CUT=lambda x: x["CUT"].astype(int),
    )
    .set_geometry("geometry")
    .pipe(lambda x: x[~x.is_empty])
    .drop("SHAPE", axis=1, errors="ignore")
    .to_crs("EPSG:4326")
)

print(f"Manzanas RM en bbox Gran Santiago: {len(manzanas_rm):,}")

# %%
COD_SANTIAGO = 13101
COD_PINTANA = 13112

manzanas_santiago = manzanas_rm[manzanas_rm["CUT"] == COD_SANTIAGO]
manzanas_pintana = manzanas_rm[manzanas_rm["CUT"] == COD_PINTANA]

print(
    f"Santiago Centro: {len(manzanas_santiago):,} manzanas, "
    f"{manzanas_santiago['n_vp_ocupada'].sum():,.0f} viviendas"
)
print(
    f"La Pintana:      {len(manzanas_pintana):,} manzanas, "
    f"{manzanas_pintana['n_vp_ocupada'].sum():,.0f} viviendas"
)

# %%
fig, axes = small_multiples_from_geodataframes(
    [manzanas_santiago, manzanas_pintana],
    col_wrap=2,
    height=5,
)

choropleth_map(
    manzanas_santiago,
    "n_vp_ocupada",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title(f"Santiago Centro\n{len(manzanas_santiago):,} manzanas")

choropleth_map(
    manzanas_pintana,
    "n_vp_ocupada",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[1].set_title(f"La Pintana\n{len(manzanas_pintana):,} manzanas")

fig.suptitle("Viviendas particulares ocupadas por manzana")
plt.savefig("images/08-manzanas-comparacion.png", bbox_inches="tight")

# %% [markdown]
# Los dos paneles tienen aspect ratio comparable pero las densidades
# son distintas. Santiago Centro concentra mas viviendas en menos
# manzanas (predominio de edificios). La Pintana tiene mas manzanas
# con menos viviendas cada una (predominio de casas).

# %% [markdown]
# ## PARTE 2. Migracion: dos niveles de detalle
#
# El Censo publica datos a dos escalas con detalle asimetrico. Las
# manzanas vienen con `n_inmigrantes` total (espacial fino pero sin
# desglose). Los microdatos vienen con `p26_llegada_periodo` (detalle
# temporal: reciente >= 2010 vs. establecida < 2010), pero solo
# identificables a escala comunal. Para mapear migracion reciente y
# establecida al nivel de celda hay que combinar las dos fuentes, lo
# que motiva la asignacion de microdatos a celdas H3.

# %%
ruta_personas = Path("data") / "censo2024-microdatos" / "personas"

personas_pintana = pd.read_parquet(
    ruta_personas,
    filters=[("region", "==", 13), ("comuna", "==", COD_PINTANA)],
)
personas_santiago = pd.read_parquet(
    ruta_personas,
    filters=[("region", "==", 13), ("comuna", "==", COD_SANTIAGO)],
)


def desglose_periodo(personas):
    p26 = personas["p26_llegada_periodo"]
    return pd.Series(
        {
            "Reciente (>= 2010)": int(p26.between(1, 5, inclusive="both").sum()),
            "Establecida (< 2010)": int(p26.between(6, 8, inclusive="both").sum()),
        }
    )


desglose_comunas = pd.DataFrame(
    {
        "La Pintana": desglose_periodo(personas_pintana),
        "Santiago Centro": desglose_periodo(personas_santiago),
    }
).T
print("Inmigrantes por periodo de llegada (microdatos por comuna):")
print(desglose_comunas)

# %%
fig, axes = small_multiples_from_geodataframes(
    [manzanas_santiago, manzanas_pintana],
    col_wrap=2,
    height=5,
)
choropleth_map(
    manzanas_santiago,
    "n_inmigrantes",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title(
    f"Santiago Centro - {manzanas_santiago['n_inmigrantes'].sum():,.0f} inmigrantes"
)
choropleth_map(
    manzanas_pintana,
    "n_inmigrantes",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[1].set_title(
    f"La Pintana - {manzanas_pintana['n_inmigrantes'].sum():,.0f} inmigrantes"
)
fig.suptitle("Manzanas: inmigrantes totales (sin desglose temporal)")
plt.savefig("images/08-migracion-manzanas.png", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(7, 2.5))
desglose_comunas.plot.barh(
    stacked=True, ax=ax, color=["#c44e52", "#4c72b0"], edgecolor="white"
)
ax.set_xlabel("Personas inmigrantes")
ax.set_ylabel("")
ax.set_title("Comunas: desglose por periodo de llegada (microdatos)")
ax.legend(loc="lower right", frameon=False)
plt.savefig("images/08-migracion-comuna.png", bbox_inches="tight")

# %% [markdown]
# El mapa por manzana tiene resolucion espacial fina pero un solo
# numero por manzana. Las barras por comuna muestran el desglose
# temporal pero sin diferenciar zonas dentro de la comuna. El SA
# combina ambas: usa el total por manzana como objetivo espacial y
# preserva el desglose temporal del microdato al asignar viviendas
# completas a celdas.

# %%
# Heatmap pais x periodo de llegada (Pintana + Santiago Centro). El
# objetivo es mostrar que cada colectivo migrante tiene su propio
# perfil temporal, lo que justifica separar migracion reciente de
# establecida: tratarlas como una sola variable borraria estructura
# demografica relevante.
codigos_territorio = pd.read_csv(
    Path("data") / "censo2024-microdatos" / "cod_territoriales_especificos.csv"
)
nombre_pais = dict(
    zip(
        codigos_territorio["Código específico"],
        codigos_territorio["Territorio específico"],
    )
)

PERIODOS_ORDEN = {
    8: "Antes 1990",
    7: "1990-1999",
    6: "2000-2009",
    5: "2010-2013",
    4: "2014-2016",
    3: "2017-2019",
    2: "2020-2022",
    1: "2023-2024",
}


def matriz_pais_periodo(personas, top_n=10):
    inmig = personas[
        (personas["p25_lug_nacimiento"] == 3)
        & (personas["p27_nacionalidad"] == 3)
        & personas["p26_llegada_periodo"].between(1, 8, inclusive="both")
        & (personas["p25_lug_nacimiento_esp"] != -99)
    ].copy()
    inmig["pais"] = inmig["p25_lug_nacimiento_esp"].map(nombre_pais)
    top = inmig["pais"].value_counts().head(top_n).index.tolist()
    inmig["pais"] = inmig["pais"].where(inmig["pais"].isin(top), "Otros")
    inmig["periodo"] = inmig["p26_llegada_periodo"].map(PERIODOS_ORDEN)
    matriz = inmig.pivot_table(
        index="pais",
        columns="periodo",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=list(PERIODOS_ORDEN.values()), fill_value=0)
    # paises ordenados por total descendente, Otros al final
    orden = (
        matriz.drop("Otros", errors="ignore")
        .sum(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Otros" in matriz.index:
        orden = orden + ["Otros"]
    return matriz.loc[orden]


personas_pooled = pd.concat([personas_pintana, personas_santiago], ignore_index=True)
heat = matriz_pais_periodo(personas_pooled, top_n=10)
print(f"Inmigrantes en el heatmap: {heat.values.sum():,}")

# %%
from matplotlib.colors import LogNorm

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    heat,
    cmap="flare",
    norm=LogNorm(vmin=max(1, heat.values.min()), vmax=heat.values.max()),
    annot=True,
    fmt=",",
    annot_kws={"fontsize": 8},
    cbar_kws={"label": "Personas (log)"},
    ax=ax,
)
# Linea vertical separando establecida (Antes 1990 a 2000-2009) de
# reciente (2010-2013 en adelante). Cae entre las columnas 3 y 4.
ax.axvline(x=3, color="black", linewidth=1.5)
ax.text(3.0, -0.3, "  Reciente (>= 2010)", fontsize=9, va="bottom", ha="left")
ax.text(3.0, -0.3, "Establecida  ", fontsize=9, va="bottom", ha="right")
ax.set_xlabel("Periodo de llegada")
ax.set_ylabel("Pais de origen")
ax.set_title("Pintana + Santiago Centro: inmigrantes por pais y periodo")
plt.savefig("images/08-migracion-heatmap.png", bbox_inches="tight")

# %% [markdown]
# Cada colectivo migrante tiene su propia distribucion temporal.
# Venezuela y Haiti son casi enteramente recientes (post-2017).
# Peru tiene una historia mas larga con flujos importantes desde
# los 2000. Argentina y los flujos europeos concentran su volumen
# antes de 2010. Esa heterogeneidad en cantidad y temporalidad
# justifica separar migracion reciente de establecida: el SA puede
# preservar la composicion de origenes nacionales en cada periodo
# al asignar viviendas completas.

# %% [markdown]
# ## PARTE 3. Areal weighting proporcional
#
# El primer enfoque para mover conteos de manzanas a celdas H3 es el
# **areal weighting**. Asume densidad uniforme dentro de cada manzana:
# si una manzana tiene $n$ viviendas y se intersecta con dos celdas
# en proporciones 0.7 y 0.3, asignamos $0.7 n$ viviendas a la primera
# celda y $0.3 n$ a la segunda.
#
# Lo aplicamos a las dos comunas para comparar el efecto del cambio
# de soporte en geografias urbanas distintas. Usamos H3-9 (resolucion
# del orden de 100 m de borde, parecido al tamanho medio de una
# manzana urbana).

# %%
grid_pintana = h3_grid_from_bounds(
    manzanas_pintana.total_bounds, extra_margin=0.05, grid_level=9
)
print(f"Grid H3-9 sobre La Pintana: {len(grid_pintana):,} celdas")

# %%
overlay = gpd.overlay(manzanas_pintana, grid_pintana.reset_index(), how="intersection")
overlay["area_int"] = overlay.area

# Areal weighting proporcional: peso = area_intersección / area_manzana.
# Lo computamos por manzana para que las filas sumen 1.
overlay["area_manzana"] = overlay.groupby("MANZENT")["area_int"].transform("sum")
overlay["peso"] = overlay["area_int"] / overlay["area_manzana"]

count_cols = manzanas_pintana.filter(like="n_").columns

# Distribuir los conteos: cada celda recibe sum(peso * conteo_manzana)
conteos_celda = (
    overlay.assign(**{c: lambda x, c=c: x[c] * x["peso"] for c in count_cols})
    .groupby("h3_cell_id")[count_cols]
    .sum()
)

# Solo conservamos celdas que efectivamente intersectan con manzanas
# de La Pintana. La grilla H3 cubre el bbox rectangular completo y
# las celdas fuera del polígono comunal quedarían como ceros artificiales.
grid_areal = grid_pintana.loc[conteos_celda.index].join(conteos_celda)
print(f"Celdas H3-9 con intersección: {len(grid_areal):,} de {len(grid_pintana):,}")
print(
    f"Total viviendas en grid (areal weighting): {grid_areal['n_vp_ocupada'].sum():,.0f}"
)
print(
    f"Total viviendas en manzanas:               {manzanas_pintana['n_vp_ocupada'].sum():,.0f}"
)

# %%
fig, axes = small_multiples_from_geodataframe(grid_areal, 3, col_wrap=3, height=5)

choropleth_map(
    manzanas_pintana,
    "n_vp_ocupada",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title("Manzanas (publicado)")

choropleth_map(
    manzanas_pintana,
    "n_vp_ocupada",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
grid_areal.boundary.plot(ax=axes[1], color="black", linewidth=0.3)
axes[1].set_title("Manzanas + grilla H3-9")

choropleth_map(
    grid_areal,
    "n_vp_ocupada",
    ax=axes[2],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[2].set_title("Grid H3-9 (areal weighting)")

fig.suptitle(f"La Pintana: viviendas particulares ocupadas")
plt.savefig("images/08-areal-weighting.png", bbox_inches="tight")

# %%
grid_santiago = h3_grid_from_bounds(
    manzanas_santiago.total_bounds, extra_margin=0.05, grid_level=9
)
print(f"Grid H3-9 sobre Santiago Centro: {len(grid_santiago):,} celdas")

overlay_santiago = gpd.overlay(
    manzanas_santiago, grid_santiago.reset_index(), how="intersection"
)
overlay_santiago["area_int"] = overlay_santiago.area
overlay_santiago["area_manzana"] = overlay_santiago.groupby("MANZENT")[
    "area_int"
].transform("sum")
overlay_santiago["peso"] = (
    overlay_santiago["area_int"] / overlay_santiago["area_manzana"]
)

conteos_celda_santiago = (
    overlay_santiago.assign(**{c: lambda x, c=c: x[c] * x["peso"] for c in count_cols})
    .groupby("h3_cell_id")[count_cols]
    .sum()
)

grid_areal_santiago = grid_santiago.loc[conteos_celda_santiago.index].join(
    conteos_celda_santiago
)
print(
    f"Celdas H3-9 con intersección: {len(grid_areal_santiago):,} de {len(grid_santiago):,}"
)
print(
    f"Total viviendas en grid (areal weighting): {grid_areal_santiago['n_vp_ocupada'].sum():,.0f}"
)
print(
    f"Total viviendas en manzanas:               {manzanas_santiago['n_vp_ocupada'].sum():,.0f}"
)

# %%
fig, axes = small_multiples_from_geodataframe(
    grid_areal_santiago, 3, col_wrap=3, height=5
)

choropleth_map(
    manzanas_santiago,
    "n_vp_ocupada",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title("Manzanas (publicado)")

choropleth_map(
    manzanas_santiago,
    "n_vp_ocupada",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
grid_areal_santiago.boundary.plot(ax=axes[1], color="black", linewidth=0.3)
axes[1].set_title("Manzanas + grilla H3-9")

choropleth_map(
    grid_areal_santiago,
    "n_vp_ocupada",
    ax=axes[2],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[2].set_title("Grid H3-9 (areal weighting)")

fig.suptitle(f"Santiago Centro: viviendas particulares ocupadas")
plt.savefig("images/08-areal-weighting-santiago.png", bbox_inches="tight")

# %% [markdown]
# Los totales coinciden por construccion (sum(peso) = 1 sobre cada
# manzana). Lo que cambia es la geometria del soporte. Notar que en
# zonas donde una manzana grande cae en una sola celda H3-9, ambos
# mapas se ven casi iguales. Donde hay manzanas mas chicas que la
# celda, el grid H3 las agrega. En Santiago Centro las manzanas
# tienden a ser mas chicas que la celda y varias se agregan en un
# mismo hexagono; en La Pintana es mas comun encontrar manzanas
# grandes que dominan una celda entera.

# %% [markdown]
# ## PARTE 4. Limitacion del areal weighting
#
# El areal weighting reparte cada **conteo marginal** por separado:
# 50 viviendas con jefatura femenina, 50 con tres dormitorios, 50
# inmigrantes. Pero la coherencia intra-vivienda se pierde. No
# sabemos si los 50 hogares con jefatura femenina son los mismos que
# los 50 con tres dormitorios o son grupos disjuntos.
#
# Para responder preguntas como "cual es el ingreso medio de hogares
# con jefatura femenina en este hexagono", necesitamos viviendas
# individuales con sus atributos correlacionados. Esto requiere
# microdatos.

# %% [markdown]
# ## PARTE 5. Pre-imputacion: microdatos a tabla wide
#
# Los microdatos del censo tienen una fila por persona. Para asignar
# **viviendas** a celdas, transformamos personas en filas-vivienda con
# columnas que cuentan, por hogar, atributos demograficos (sexo, edad,
# nacionalidad, educacion, ocupacion CIUO) y atributos de la vivienda
# (tipo, dormitorios, transporte). La funcion
# `tabla_viviendas_desde_microdatos` encapsula esa transformacion.

# %%
viv_pintana, per_pintana = tabla_viviendas_desde_microdatos(COD_PINTANA)
print(
    f"La Pintana: {len(viv_pintana):,} viviendas, "
    f"{len(per_pintana):,} personas, "
    f"{viv_pintana.shape[1]} columnas"
)
print(viv_pintana.head(3))

# %% [markdown]
# Cada fila es una vivienda con conteos de sus integrantes (cuantos
# hombres, cuantos en cada rango de edad, cuantos en cada ocupacion).
# Esto es el dato que el optimizador va a asignar a celdas H3.

# %% [markdown]
# ## PARTE 6. Asignacion con simulated annealing
#
# El problema: dado un conjunto de viviendas y un grid con totales
# objetivo por celda, asignar cada vivienda a una celda tal que los
# totales agregados coincidan (o se acerquen) con los objetivos.
# Restriccion dura: una casa no puede asignarse a una celda como
# departamento. Restriccion blanda: respetar la composicion
# demografica y de hogares de cada celda.
#
# El algoritmo en `gdsutils.asignacion` opera en tres fases:
#
# - **Fase 0**: escala los totales del grid a los totales reales,
#   construye una matriz de distancia vivienda x celda con pesos por
#   prioridad y aplica softmax con la capacidad como prior.
# - **Fase 1**: asignacion estratificada por tipo (casa/depto/otro)
#   seguida de un equilibrado iterativo de capacidades.
# - **Fase 2**: simulated annealing con swaps entre viviendas del
#   mismo tipo, dirigidos por el residuo del atributo mas critico.

# %%
# Construir grid H3-9 con totales objetivo por celda. Usamos
# areal weighting con idxmax (cada manzana se asigna entera a la celda
# de mayor solapamiento). Es equivalente al areal weighting de PARTE
# 3 cuando una manzana cae mayoritariamente en una celda.
cell_to_manzana = (
    overlay.set_index(["MANZENT", "h3_cell_id"])["area_int"]
    .unstack(fill_value=0)
    .pipe(normalize_rows)
    .idxmax(axis=1)
    .rename("h3_cell_id")
)

grid_variables_pintana = grid_pintana.join(
    manzanas_pintana.join(cell_to_manzana, on="MANZENT")
    .groupby("h3_cell_id")[count_cols]
    .sum()
).dropna()
print(f"Celdas con datos: {len(grid_variables_pintana):,}")

# %%
config = {
    "persona": {
        "prioridad_alta": [
            "n_mujeres",
            "n_inmigrantes",
            "n_ciuo_0",
            "n_ciuo_1",
            "n_ciuo_2",
            "n_ciuo_3",
            "n_ciuo_4",
            "n_ciuo_5",
            "n_ciuo_6",
            "n_ciuo_7",
            "n_ciuo_8",
            "n_ciuo_9",
        ],
        "prioridad_media": [
            "n_edad_0_5",
            "n_edad_6_13",
            "n_edad_14_17",
            "n_edad_18_24",
            "n_edad_25_44",
            "n_edad_45_59",
            "n_edad_60_mas",
            "n_cine_nunca_curso_primera_infancia",
            "n_cine_primaria",
            "n_cine_secundaria",
            "n_cine_terciaria_maestria_doctorado",
            "n_cine_especial_diferencial",
        ],
        "prioridad_baja": [
            "n_transporte_auto",
            "n_transporte_publico",
            "n_transporte_camina",
            "n_transporte_bicicleta",
            "n_transporte_motocicleta",
            "n_transporte_cab_lan_bote",
            "n_transporte_otros",
        ],
    },
    "vivienda": {
        "prioridad_alta": [
            "n_dormitorios_1",
            "n_dormitorios_2",
            "n_dormitorios_3",
            "n_dormitorios_4",
            "n_dormitorios_5",
            "n_dormitorios_6_o_mas",
        ],
        "prioridad_media": [],
        "prioridad_baja": [],
    },
    "tipo_vivienda": ["n_tipo_viv_casa", "n_tipo_viv_depto"],
}

diagnostico, config_filtrada = diagnosticar_atributos(viv_pintana, config)
print(diagnostico[["tipo", "total", "freq", "apto"]].to_string())

# %%
# La asignacion con SA tarda alrededor de 1 a 2 minutos para una comuna
# de este tamanho. Usamos sa_iteraciones=200_000 para clase; el
# precomputo del Gran Santiago usa 500_000.
resultado_pintana, errores_pintana = asignar_viviendas(
    viv_pintana,
    grid_variables_pintana.reset_index(),
    config_filtrada,
    peso_alta=15.0,
    peso_media=1.0,
    peso_baja=0.1,
    peso_tipo=20.0,
    sa_iteraciones=200_000,
    verbose=True,
)
print(f"Asignadas {len(resultado_pintana):,} viviendas")

# %%
df_errores = reportar_errores(errores_pintana, viv_pintana, config_filtrada)
print(
    df_errores.to_string(
        formatters={
            "total": "{:,.0f}".format,
            "error_abs": "{:,.0f}".format,
            "error_rel": "{:.1%}".format,
        }
    )
)

# %% [markdown]
# El barplot de errores colorea cada atributo segun su grupo de
# prioridad en el `config`. Los grupos `prioridad_alta`, `media` y
# `baja` se traducen a los pesos `w_k` de la energia
# (peso_alta=15, peso_media=1, peso_baja=0.1). Los atributos rojos
# pesan 150 veces mas que los verdes en la energia; el SA sacrifica
# primero los marginales de menor peso cuando hay conflicto entre
# restricciones. Las lineas verticales en 5% y 10% son referencias:
# rojos esperamos < 5%, verdes pueden superar el 10% porque fueron
# sacrificados a proposito.

# %%
fig, ax = visualizar_errores(
    errores_pintana, viv_pintana, config_filtrada, figsize=(8, 10)
)
plt.savefig("images/08-errores-pintana.png", bbox_inches="tight")

# %%
# Repetimos el procedimiento para Santiago Centro. La comuna tiene
# muchas menos manzanas pero mas viviendas por celda (edificios), lo
# que cambia el balance del optimizador.
viv_santiago, per_santiago = tabla_viviendas_desde_microdatos(COD_SANTIAGO)
print(
    f"Santiago Centro: {len(viv_santiago):,} viviendas, {len(per_santiago):,} personas"
)

cell_to_manzana_santiago = (
    overlay_santiago.set_index(["MANZENT", "h3_cell_id"])["area_int"]
    .unstack(fill_value=0)
    .pipe(normalize_rows)
    .idxmax(axis=1)
    .rename("h3_cell_id")
)

grid_variables_santiago = grid_santiago.join(
    manzanas_santiago.join(cell_to_manzana_santiago, on="MANZENT")
    .groupby("h3_cell_id")[count_cols]
    .sum()
).dropna()
print(f"Celdas con datos: {len(grid_variables_santiago):,}")

diagnostico_santiago, config_filtrada_santiago = diagnosticar_atributos(
    viv_santiago, config
)

resultado_santiago, errores_santiago = asignar_viviendas(
    viv_santiago,
    grid_variables_santiago.reset_index(),
    config_filtrada_santiago,
    peso_alta=15.0,
    peso_media=1.0,
    peso_baja=0.1,
    peso_tipo=20.0,
    sa_iteraciones=200_000,
    verbose=True,
)
print(f"Asignadas {len(resultado_santiago):,} viviendas")

df_errores_santiago = reportar_errores(
    errores_santiago, viv_santiago, config_filtrada_santiago
)
print(
    df_errores_santiago.to_string(
        formatters={
            "total": "{:,.0f}".format,
            "error_abs": "{:,.0f}".format,
            "error_rel": "{:.1%}".format,
        }
    )
)

# %%
fig, ax = visualizar_errores(
    errores_santiago, viv_santiago, config_filtrada_santiago, figsize=(8, 10)
)
plt.savefig("images/08-errores-santiago.png", bbox_inches="tight")

# %% [markdown]
# Los atributos de prioridad alta (sexo, inmigrantes, CIUO) quedan con
# errores relativos bajos (en general menos del 5%). Los atributos de
# prioridad baja (modo de transporte) pueden quedar con errores mas
# altos: el optimizador los sacrifica si entran en conflicto con los
# de prioridad alta. Los errores agregados son una cifra unica y no
# dicen donde estan los peores desajustes. Para verlo, vamos a
# observar una variable que las manzanas no publican: la migracion
# reciente (personas inmigrantes que llegaron a Chile desde 2010 en
# adelante). El SA no la usa como objetivo, pero al asignar viviendas
# completas arrastra esta caracteristica desde el microdato.


# %%
# Construimos columnas por vivienda para migracion reciente (>= 2010)
# y migracion establecida (< 2010). p26_llegada_periodo codifica el
# periodo: 1..5 son periodos desde 2010 en adelante, 6..8 son llegadas
# anteriores. NaN para no migrantes.
def agregar_migracion_por_periodo(viviendas, personas):
    p26 = personas.set_index("id_vivienda")["p26_llegada_periodo"]
    reciente = (
        p26.between(1, 5, inclusive="both")
        .astype(int)
        .groupby("id_vivienda")
        .sum()
        .rename("n_inmigrantes_reciente")
    )
    establecida = (
        p26.between(6, 8, inclusive="both")
        .astype(int)
        .groupby("id_vivienda")
        .sum()
        .rename("n_inmigrantes_establecida")
    )
    return (
        viviendas.merge(reciente, on="id_vivienda", how="left")
        .merge(establecida, on="id_vivienda", how="left")
        .fillna({"n_inmigrantes_reciente": 0, "n_inmigrantes_establecida": 0})
    )


viv_pintana = agregar_migracion_por_periodo(viv_pintana, per_pintana)
viv_santiago = agregar_migracion_por_periodo(viv_santiago, per_santiago)
print(
    f"Pintana:  {int(viv_pintana['n_inmigrantes_reciente'].sum()):,} migrantes desde 2010, "
    f"{int(viv_pintana['n_inmigrantes_establecida'].sum()):,} establecidos antes de 2010, "
    f"{int(viv_pintana['n_inmigrantes'].sum()):,} inmigrantes en total"
)
print(
    f"Santiago: {int(viv_santiago['n_inmigrantes_reciente'].sum()):,} migrantes desde 2010, "
    f"{int(viv_santiago['n_inmigrantes_establecida'].sum()):,} establecidos antes de 2010, "
    f"{int(viv_santiago['n_inmigrantes'].sum()):,} inmigrantes en total"
)


# %%
# Mapa de tres capas por comuna: reciente (asignacion SA), establecida
# (asignacion SA) y total (objetivo de las manzanas agregado a H3-9).
# La suma de reciente y establecida no es exactamente el total porque
# hay inmigrantes sin periodo de llegada declarado.
def conteos_por_celda(resultado, viviendas, grid_variables, columnas):
    asignaciones = resultado.merge(
        viviendas[["id_vivienda"] + list(columnas)], on="id_vivienda"
    )
    sumas = asignaciones.groupby("h3_cell_id")[list(columnas)].sum()
    return grid_variables[["geometry", "n_inmigrantes"]].join(sumas).fillna(0)


capas_pintana = conteos_por_celda(
    resultado_pintana,
    viv_pintana,
    grid_variables_pintana,
    ["n_inmigrantes_reciente", "n_inmigrantes_establecida"],
)
capas_santiago = conteos_por_celda(
    resultado_santiago,
    viv_santiago,
    grid_variables_santiago,
    ["n_inmigrantes_reciente", "n_inmigrantes_establecida"],
)

# %%
fig, axes = small_multiples_from_geodataframes(
    [
        capas_pintana,
        capas_pintana,
        capas_pintana,
        capas_santiago,
        capas_santiago,
        capas_santiago,
    ],
    col_wrap=3,
    height=4,
)

for ax, (col, titulo) in zip(
    axes[:3],
    [
        ("n_inmigrantes_reciente", "Reciente (>= 2010)"),
        ("n_inmigrantes_establecida", "Establecida (< 2010)"),
        ("n_inmigrantes", "Total (manzanas)"),
    ],
):
    choropleth_map(
        capas_pintana,
        col,
        ax=ax,
        edgecolor="none",
        binning="fisher_jenks",
        palette="flare_r",
    )
    ax.set_title(f"La Pintana - {titulo}")

for ax, (col, titulo) in zip(
    axes[3:],
    [
        ("n_inmigrantes_reciente", "Reciente (>= 2010)"),
        ("n_inmigrantes_establecida", "Establecida (< 2010)"),
        ("n_inmigrantes", "Total (manzanas)"),
    ],
):
    choropleth_map(
        capas_santiago,
        col,
        ax=ax,
        edgecolor="none",
        binning="fisher_jenks",
        palette="flare_r",
    )
    ax.set_title(f"Santiago Centro - {titulo}")

fig.suptitle(
    "Inmigrantes por celda H3-9: reciente y establecida (SA) vs. total (manzanas)"
)
plt.savefig("images/08-mapas-migracion-periodo.png", bbox_inches="tight")

# %% [markdown]
# Los dos primeros paneles por comuna son producto del microdato:
# muestran donde el SA ubico viviendas con migracion reciente y
# establecida. El tercero muestra el total de inmigrantes proveniente
# de las manzanas. La comparacion visual deja ver que el patron de
# reciente y establecida no es necesariamente proporcional al total:
# hay celdas con concentracion alta de migracion reciente pero baja
# de establecida y viceversa. Esa heterogeneidad es la que cuantifica
# el mapa de discrepancia que viene a continuacion.


# %%
# Discrepancia respecto a la prediccion uniforme (lo que daria areal
# weighting): para cada celda, el SA asignado se compara con
# celda_total_inmigrantes * proporcion_comunal_reciente. La diferencia
# muestra donde el SA ubico viviendas con migracion reciente en una
# proporcion distinta a la del promedio comunal, gracias al microdato.
def discrepancia_uniforme(resultado, viviendas, grid_variables):
    p_reciente = viviendas["n_inmigrantes_reciente"].sum() / max(
        viviendas["n_inmigrantes"].sum(), 1
    )
    asignado = (
        resultado.merge(
            viviendas[["id_vivienda", "n_inmigrantes_reciente"]], on="id_vivienda"
        )
        .groupby("h3_cell_id")["n_inmigrantes_reciente"]
        .sum()
        .rename("asignado")
    )
    factor = viviendas["n_inmigrantes"].sum() / max(
        grid_variables["n_inmigrantes"].sum(), 1
    )
    objetivo = (grid_variables["n_inmigrantes"] * factor * p_reciente).rename(
        "uniforme"
    )
    df = grid_variables[["geometry"]].join(objetivo).join(asignado).fillna(0)
    df["discrepancia"] = df["asignado"] - df["uniforme"]
    df["abs_discrepancia"] = df["discrepancia"].abs()
    return df, p_reciente


discrep_pintana, p_rec_pintana = discrepancia_uniforme(
    resultado_pintana,
    viv_pintana,
    grid_variables_pintana,
)
discrep_santiago, p_rec_santiago = discrepancia_uniforme(
    resultado_santiago,
    viv_santiago,
    grid_variables_santiago,
)
print(
    f"Pintana  - proporcion reciente comunal: {p_rec_pintana:.1%}, "
    f"discrepancia media={discrep_pintana['abs_discrepancia'].mean():.2f}, "
    f"max={discrep_pintana['abs_discrepancia'].max():.2f}"
)
print(
    f"Santiago - proporcion reciente comunal: {p_rec_santiago:.1%}, "
    f"discrepancia media={discrep_santiago['abs_discrepancia'].mean():.2f}, "
    f"max={discrep_santiago['abs_discrepancia'].max():.2f}"
)

# %%
fig, axes = small_multiples_from_geodataframes(
    [discrep_pintana, discrep_santiago],
    col_wrap=2,
    height=5,
)

choropleth_map(
    discrep_pintana,
    "abs_discrepancia",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="rocket_r",
)
axes[0].set_title("La Pintana")

choropleth_map(
    discrep_santiago,
    "abs_discrepancia",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="rocket_r",
)
axes[1].set_title("Santiago Centro")

fig.suptitle("Discrepancia SA vs. prediccion uniforme en migracion reciente (>= 2010)")
plt.savefig("images/08-discrepancia-migracion-reciente.png", bbox_inches="tight")

# %% [markdown]
# Las celdas con discrepancia alta son aquellas donde el SA puso una
# concentracion de migracion reciente distinta a la que daria un
# reparto uniforme proporcional al total de inmigrantes. Esta es la
# senhal de sorting espacial que el microdato captura y el areal
# weighting no: las viviendas con migracion reciente tienden a
# concentrarse en zonas especificas dentro de cada comuna, no a
# distribuirse de forma homogenea.

# %% [markdown]
# ## PARTE 7. Comparacion: areal weighting vs. asignacion con SA
#
# Visualizamos la distribucion de inmigrantes en las dos comunas segun
# los dos metodos. El areal weighting reparte conteos manzana a celda.
# El SA toma los conteos por celda como objetivo y reasigna viviendas
# individuales para que los marginales coincidan respetando la
# coherencia intra-vivienda.


# %%
def construir_comparacion(resultado, viviendas, grid_geo, grid_areal_comuna, variable):
    asignado_sa = (
        resultado.merge(viviendas[["id_vivienda", variable]], on="id_vivienda")
        .groupby("h3_cell_id")[variable]
        .sum()
        .rename(f"{variable}_sa")
    )
    return (
        grid_areal_comuna[[variable]]
        .rename(columns={variable: f"{variable}_areal"})
        .join(asignado_sa)
        .fillna(0)
        .join(grid_geo[["geometry"]])
        .pipe(gpd.GeoDataFrame, geometry="geometry", crs=grid_geo.crs)
    )


comparacion_pintana = construir_comparacion(
    resultado_pintana,
    viv_pintana,
    grid_pintana,
    grid_areal,
    "n_inmigrantes",
)
comparacion_santiago = construir_comparacion(
    resultado_santiago,
    viv_santiago,
    grid_santiago,
    grid_areal_santiago,
    "n_inmigrantes",
)

fig, axes = small_multiples_from_geodataframes(
    [
        comparacion_pintana,
        comparacion_pintana,
        comparacion_santiago,
        comparacion_santiago,
    ],
    col_wrap=2,
    height=5,
)

choropleth_map(
    comparacion_pintana,
    "n_inmigrantes_areal",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title("La Pintana: areal weighting")

choropleth_map(
    comparacion_pintana,
    "n_inmigrantes_sa",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[1].set_title("La Pintana: asignacion con SA")

choropleth_map(
    comparacion_santiago,
    "n_inmigrantes_areal",
    ax=axes[2],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[2].set_title("Santiago Centro: areal weighting")

choropleth_map(
    comparacion_santiago,
    "n_inmigrantes_sa",
    ax=axes[3],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[3].set_title("Santiago Centro: asignacion con SA")

fig.suptitle("Numero de inmigrantes por celda H3-9")
plt.savefig("images/08-comparacion-areal-sa.png", bbox_inches="tight")

# %% [markdown]
# A escala visual los dos resultados son muy parecidos para variables
# bien distribuidas. La diferencia esta en que el resultado del SA
# tiene viviendas individuales detras: para cada hexagono podemos
# preguntar "cuantos inmigrantes con jefatura femenina viven aqui",
# cosa que el areal weighting no responde.

# %% [markdown]
# ## PARTE 8. Resultado a escala de Gran Santiago
#
# El profe-script `08-asignacion-rm.py` ejecuta el mismo procedimiento
# para las 43 comunas con manzanas dentro del bbox del Gran Santiago.
# La salida final viene reagregada a H3-8 (~1.4k hexagonos sobre la
# ciudad). Aprovechamos esa resolucion porque coincide con la de las
# clases 04, 05 y 06.

# %%
h8 = pd.read_parquet("data/censo2024-asignacion-rm/asignaciones-h3-8.parquet")
print(f"Hexagonos H3-8 sobre el Gran Santiago: {len(h8):,}")
print(f"Total viviendas: {h8['n_vp'].sum():,}")
print(f"Total personas:  {(h8['n_hombres'] + h8['n_mujeres']).sum():,}")

# %%
hex_geo = h3_grid_from_ids(h8["h3_cell_id"]).merge(h8, on="h3_cell_id")

hex_geo["n_per"] = hex_geo["n_hombres"] + hex_geo["n_mujeres"]
hex_geo["pct_inmigrantes"] = (
    100 * hex_geo["n_inmigrantes"] / hex_geo["n_per"].replace(0, np.nan)
)
hex_geo["pct_profesionales"] = (
    100 * hex_geo["n_ciuo_2"] / hex_geo["n_per"].replace(0, np.nan)
)
hex_geo["pct_servicios"] = (
    100 * hex_geo["n_ciuo_5"] / hex_geo["n_per"].replace(0, np.nan)
)
hex_geo["pct_terciaria"] = (
    100
    * hex_geo["n_cine_terciaria_maestria_doctorado"]
    / hex_geo["n_per"].replace(0, np.nan)
)

# %%
fig, axes = small_multiples_from_geodataframe(hex_geo, 4, col_wrap=2, height=6)

choropleth_map(
    hex_geo,
    "pct_inmigrantes",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title("Inmigrantes (% poblacion)")

choropleth_map(
    hex_geo,
    "pct_profesionales",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[1].set_title("CIUO 2 profesionales (% poblacion)")

choropleth_map(
    hex_geo,
    "pct_servicios",
    ax=axes[2],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[2].set_title("CIUO 5 servicios (% poblacion)")

choropleth_map(
    hex_geo,
    "pct_terciaria",
    ax=axes[3],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[3].set_title("Educacion terciaria (% poblacion)")

fig.suptitle("Gran Santiago: indicadores demograficos por H3-8")
plt.savefig("images/08-resultado-h3-8.png", bbox_inches="tight")

# %% [markdown]
# Cada hexagono tiene ahora todos los conteos del censo a escala
# comparable con eBird, NDVI o cualquier otra capa de la ciudad.
# Como cierre, cruzamos un indicador socioeconomico (porcentaje con
# educacion terciaria) con NDVI medio por hexagono, descargando el
# raster precomputado de la clase 05.

# %%
ruta_ndvi = descargar_ndvi_santiago_precomputado()
hex_geo = ndvi_por_zona(hex_geo, ruta_ndvi, columna="ndvi", estadistico="mean")

# %%
# Primero visualizamos ambas variables sobre la misma grilla H3-8 para
# inspeccionar el patron espacial antes del scatter.
fig, axes = small_multiples_from_geodataframe(hex_geo, 2, col_wrap=2, height=6)

choropleth_map(
    hex_geo,
    "pct_terciaria",
    ax=axes[0],
    edgecolor="none",
    binning="fisher_jenks",
    palette="flare_r",
)
axes[0].set_title("Educacion terciaria (% poblacion)")

choropleth_map(
    hex_geo.dropna(subset=["ndvi"]),
    "ndvi",
    ax=axes[1],
    edgecolor="none",
    binning="fisher_jenks",
    palette="YlGn",
)
axes[1].set_title("NDVI medio (Sentinel-2, verano 2023-2024)")

fig.suptitle("Gran Santiago: educacion terciaria y NDVI por hexagono H3-8")
plt.savefig("images/08-mapas-ndvi-terciaria.png", bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=hex_geo.dropna(subset=["pct_terciaria", "ndvi"]),
    x="ndvi",
    y="pct_terciaria",
    s=10,
    alpha=0.5,
    ax=ax,
)
sns.regplot(
    data=hex_geo.dropna(subset=["pct_terciaria", "ndvi"]),
    x="ndvi",
    y="pct_terciaria",
    scatter=False,
    color="black",
    ax=ax,
)
ax.set_xlabel("NDVI medio (Sentinel-2, verano 2023-2024)")
ax.set_ylabel("Educacion terciaria (% poblacion)")
ax.set_title("Educacion terciaria vs. NDVI por hexagono H3-8")
plt.savefig("images/08-cruce-ndvi.png", bbox_inches="tight")
