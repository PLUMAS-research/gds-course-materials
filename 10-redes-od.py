# %% [markdown]
# # Redes origen-destino y diferencias de genero en viajes de cuidado
#
# La movilidad urbana tiene patrones de genero que se invisibilizan
# cuando agregamos viajes sin desglose. Los viajes asociados al trabajo
# de cuidado (compras, salud, llevar y traer ninos, tramites) son
# mayoritariamente realizados por mujeres, son cortos y encadenados, y
# usan la red de calles de forma distinta a los viajes laborales.
#
# Esta clase replica el analisis de Viajes_Encadenados sobre la EOD
# Santiago 2012. Construimos una red origen-destino en H3-7 sobre Gran
# Santiago, desglosada por sexo x proposito (Mujer/Hombre x Cuidado /
# Empleo+Estudio / Personal), y respondemos:
#
#   1. Que pares O-D estan sobre-representados por categoria? (PMI)
#   2. La estructura topologica de las redes femeninas y masculinas es
#      distinta?
#   3. Las diferencias son distinguibles de la variacion al azar?
#      (modelo nulo por permutacion de sexo)

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import h3
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
from chiricoca.geo.grid import h3_grid_from_bounds
from chiricoca.maps import choropleth_map
from shapely.geometry import LineString, Point, box

from gdsutils import eodscl as eod
from gdsutils.redes import (
    aristas_en_triangulos,
    aristas_pmi_a_geodataframe,
    generar_html_flowmap,
    matriz_od_a_aristas_geodataframe,
    matriz_od_por_categoria,
    metricas_subgrafo,
    modelo_nulo_permutacion,
    pmi_od,
    strength_por_nodo,
    subgrafo_pmi,
    top_aristas,
)

setup_style(dpi=128)

EOD_PATH = Path("data") / "eod2012" / "EOD_STGO"
CARTO_PATH = (
    Path("data") / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)

# Bounding box del Gran Santiago (mismo que Viajes_Encadenados).
BBOX_GRAN_STGO = (-70.85, -33.65, -70.45, -33.30)
# Nivel H3: 7 (~5 km^2 por celda, ~190 celdas en zona estudio). Buen
# balance entre detalle y volumen para la muestra EOD.
H3_LEVEL = 7
# Filtros del analisis original.
FILTRO_TOTAL_MIN = 300
FILTRO_VIAJES_CAT = 50
SUAVIZADO_PMI = 0.5
N_PERMUTACIONES = 200
# Para los mapas de redes mostramos solo las aristas mas importantes
# dentro del subgrafo PMI > 0. Criterio: media geometrica entre PMI y
# peso (sqrt(pmi * peso)). Esto premia aristas que son al mismo tiempo
# sobre-representadas y voluminosas. Tomar el top 30% reduce el
# clutter visual sin perder el patron principal.
TOP_ARISTAS_PCT = 0.30

# Paleta del proyecto Viajes_Encadenados.
COLOR_HOMBRE = "#641a80"
COLOR_MUJER = "#de4968"
COLORES_CAT = {
    "Hombre_Cuidado":        "#7b5ea7",
    "Hombre_Empleo/Estudio": "#641a80",
    "Hombre_Personal":       "#9e7fbf",
    "Mujer_Cuidado":         "#de4968",
    "Mujer_Empleo/Estudio":  "#e87a90",
    "Mujer_Personal":        "#f2a5b5",
}

# %% [markdown]
# ## PARTE 1. Por que viajes de cuidado?
#
# Sanchez de Madariaga (2013) introduce el concepto de movilidad del
# cuidado para nombrar el conjunto de viajes asociados al trabajo
# reproductivo: llevar y buscar ninos, acompanar adultos mayores,
# hacer tramites y compras. Tiene tres caracteristicas:
#
#   1. Mayoritariamente femenino en sociedades con division
#      tradicional del trabajo.
#   2. Compuesto por trayectos cortos y encadenados.
#   3. Invisibilizado en estadisticas tradicionales de transporte que
#      registran un solo viaje principal por persona y por dia.
#
# Las encuestas O-D agregadas sin desglose tratan estos viajes como
# variacion de fondo. La pregunta es si su estructura espacial es
# distinta de la de viajes laborales o personales.

# %% [markdown]
# ## PARTE 2. Anatomia de la EOD 2012
#
# Cargamos los viajes y los unimos con personas para tener sexo y
# edad. El peso es el producto de los dos factores de expansion (uno
# del viaje, uno de la persona). Filtramos:
#
#   - SectorOrigen y SectorDestino dentro de la RM (excluye exterior).
#   - Adultos en edad laboral plena (30-65). Excluye estudiantes y
#     adultos mayores que tienen patrones distintos.
#   - Ingreso en tramos bajos a medios (TramoIngresoFinal <= 3) para
#     concentrar el analisis en hogares dependientes de transporte
#     publico.

# %%
viajes_raw = eod.read_trips(EOD_PATH)
personas_raw = eod.read_people(EOD_PATH)

print(f"Viajes crudos: {len(viajes_raw)}")
print(f"Personas: {len(personas_raw)}")
print()
print("Sexo en personas:")
print(personas_raw["Sexo"].value_counts())
print()
print("Propositos en viajes (top 10):")
print(viajes_raw["Proposito"].value_counts().head(10))

# %%
# Merge viajes-personas + peso combinado.
viajes = viajes_raw.merge(personas_raw, on=["Hogar", "Persona"])
viajes["Peso"] = viajes["FactorExpansion"] * viajes["FactorPersona"]
viajes["Edad"] = 2012 - viajes["AnoNac"]

print(f"Viajes con persona: {len(viajes)}")
print(f"Total expandido: {viajes['Peso'].sum():,.0f}")

# %%
# Filtros: sector valido + adultos + ingreso medio-bajo.
sectores_invalidos = {"Exterior a RM", "Extensión Sur-Poniente"}
viajes = viajes[
    ~viajes["SectorOrigen"].isin(sectores_invalidos)
    & ~viajes["SectorDestino"].isin(sectores_invalidos)
    & viajes["SectorOrigen"].notna()
    & viajes["SectorDestino"].notna()
]
viajes = viajes[(viajes["Edad"] >= 30) & (viajes["Edad"] <= 65)]
viajes = viajes[viajes["TramoIngresoFinal"] <= 3]

print(f"Despues de filtros: {len(viajes)} viajes")
print(f"Total expandido: {viajes['Peso'].sum():,.0f}")
print()
print("Distribucion por sexo:")
print(viajes.groupby("Sexo")["Peso"].sum().apply(lambda x: f"{x:,.0f}"))

# %%
# Agrupacion de propositos: Cuidado, Empleo/Estudio, Personal, Hogar.
# Hogar se excluye (no es actividad de destino, cierra el dia).
viajes["PropositoAgregado"] = viajes["Proposito"].map(eod.GRUPOS_PROPOSITOS)
viajes = viajes.dropna(subset=["PropositoAgregado"])
viajes = viajes[viajes["PropositoAgregado"] != "Hogar"]

print("Distribucion por proposito agregado (expandida):")
tabla_prop = viajes.groupby("PropositoAgregado")["Peso"].sum().sort_values(ascending=False)
print(tabla_prop.apply(lambda x: f"{x:,.0f}"))

print("\nProporcion mujer/hombre por proposito:")
tabla_prop_sexo = (
    viajes.groupby(["PropositoAgregado", "Sexo"])["Peso"].sum().unstack()
)
tabla_prop_sexo["%Mujer"] = (
    tabla_prop_sexo["Mujer"] / tabla_prop_sexo.sum(axis=1) * 100
)
print(tabla_prop_sexo.round(0))

# %% [markdown]
# Lectura del descriptivo: en Empleo/Estudio los hombres aportan mas
# del 55% del volumen; en Cuidado las mujeres son mayoria clara. Esta
# es la primera evidencia de la asimetria de genero en la movilidad.

# %% [markdown]
# ## PARTE 3. Matriz O-D por comuna y proposito (vista agregada)
#
# Antes de pasar a H3-7 conviene una vista en una unidad familiar:
# las comunas del Gran Santiago. La matriz comuna x comuna es mas
# grosera (43 comunas x 43 comunas = 1849 celdas) pero las
# diferencias estructurales entre propositos ya deberian ser visibles
# como bloques en el heatmap. Si esta vista no muestra nada, no tiene
# sentido subir la resolucion a H3.
#
# Ordenamos las comunas por volumen total de viajes (origen +
# destino), de mayor a menor. Las comunas mas activas (Santiago, Maipu,
# Puente Alto, Las Condes, Providencia) quedan en la esquina superior
# izquierda; las periferias en la esquina inferior derecha.

# %%
from matplotlib.colors import LogNorm

# Comunas RM presentes en la cartografia (43 sin Pirque ni San Jose de Maipo
# tras el clip al bbox del Gran Santiago).
carto_para_comunas = gpd.read_parquet(CARTO_PATH)
if "SHAPE" in carto_para_comunas.columns and "geometry" not in carto_para_comunas.columns:
    carto_para_comunas = (
        carto_para_comunas.set_geometry("SHAPE").rename_geometry("geometry")
    )
carto_para_comunas = carto_para_comunas.to_crs("EPSG:4326")
comunas_rm_set = set(
    carto_para_comunas[carto_para_comunas["COD_REGION"] == 13]
    .clip(box(*BBOX_GRAN_STGO))["COMUNA"]
    .str.title()
)

# Filtramos viajes a pares cuyas dos comunas esten en la RM (excluye
# viajes con comuna NA o fuera del area de estudio).
viajes_com = viajes[
    viajes["ComunaOrigen"].isin(comunas_rm_set)
    & viajes["ComunaDestino"].isin(comunas_rm_set)
].copy()
print(f"Viajes inter/intra comunales en zona estudio: {len(viajes_com)}")
print(f"Expandidos: {viajes_com['Peso'].sum():,.0f}")

# Orden de comunas por volumen total (origen + destino).
volumen_origen = viajes_com.groupby("ComunaOrigen")["Peso"].sum()
volumen_destino = viajes_com.groupby("ComunaDestino")["Peso"].sum()
volumen_total = (volumen_origen.add(volumen_destino, fill_value=0)).sort_values(
    ascending=False
)
orden_comunas = volumen_total.index.tolist()
print(f"Comunas en heatmap: {len(orden_comunas)}")
print("Top 5 por volumen total:", orden_comunas[:5])

# %%
# Construimos la matriz comuna x comuna por proposito.
def matriz_comunas_proposito(df, proposito, orden):
    sub = df[df["PropositoAgregado"] == proposito]
    pivot = (
        sub.groupby(["ComunaOrigen", "ComunaDestino"])["Peso"].sum()
        .unstack(fill_value=0)
    )
    return pivot.reindex(index=orden, columns=orden, fill_value=0)

propositos = ["Cuidado", "Empleo/Estudio", "Personal"]
matrices_com = {p: matriz_comunas_proposito(viajes_com, p, orden_comunas)
                for p in propositos}

# Volumen total por proposito (para titulos).
totales_prop = {p: matrices_com[p].values.sum() for p in propositos}
print("Volumen total por proposito:")
for p in propositos:
    print(f"  {p:18s}: {totales_prop[p]:>12,.0f}")

# %%
# Heatmap 1x3 con escala log compartida. Usamos imshow + LogNorm para
# que las pocas celdas con volumen alto no aplasten el resto.
valores_no_cero = np.concatenate([
    matrices_com[p].values[matrices_com[p].values > 0].ravel()
    for p in propositos
])
vmin = max(valores_no_cero.min(), 1)
vmax = valores_no_cero.max()
norm = LogNorm(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
for ax, prop in zip(axes, propositos):
    M = matrices_com[prop].values
    # Reemplazamos ceros por NaN para que se pinten en blanco.
    M_plot = np.where(M > 0, M, np.nan)
    im = ax.imshow(M_plot, cmap="magma", norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_title(f"{prop}\n({totales_prop[prop]:,.0f} viajes expandidos)",
                 fontsize=11)
    ax.set_xlabel("Destino (comuna)")
    n = len(orden_comunas)
    paso = max(1, n // 12)
    ticks = list(range(0, n, paso))
    ax.set_xticks(ticks)
    ax.set_xticklabels([orden_comunas[i] for i in ticks],
                       rotation=60, ha="right", fontsize=8)
axes[0].set_ylabel("Origen (comuna)")
n = len(orden_comunas)
paso = max(1, n // 12)
ticks = list(range(0, n, paso))
axes[0].set_yticks(ticks)
axes[0].set_yticklabels([orden_comunas[i] for i in ticks], fontsize=8)

# Colorbar comun debajo.
fig.subplots_adjust(bottom=0.18)
cax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
fig.colorbar(im, cax=cax, orientation="horizontal",
             label="Viajes expandidos (escala log)")
fig.suptitle(
    "Matriz O-D por comuna y proposito (orden: mayor volumen total arriba/izquierda)",
    fontsize=13,
)
fig.savefig("images/10-matriz-comunas.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# Lectura del heatmap:
#
#   - **Diagonal brillante** = muchos viajes intra-comunales. Cuidado
#     y Personal tienen diagonal mas marcada (las personas se mueven
#     poco). Empleo/Estudio tiene diagonal mas tenue (commuting).
#   - **Columnas brillantes** = comunas que atraen viajes desde
#     muchos origenes. En Empleo/Estudio las primeras columnas
#     (Santiago, Providencia, Las Condes) concentran el flujo.
#   - **Bloques en la esquina inferior-derecha** = pares periferia
#     con periferia (Maipu, Puente Alto, La Florida), mas frecuentes
#     en Cuidado.
#
# Cuantificamos la diagonal con la proporcion de viajes intra-comuna:

# %%
print("Proporcion de viajes intra-comuna por proposito:")
for p in propositos:
    M = matrices_com[p]
    diag = np.diag(M.values).sum()
    total = M.values.sum()
    print(f"  {p:18s}: {diag / total:.1%}  (diag={diag:,.0f}, total={total:,.0f})")

# %% [markdown]
# El ranking confirma la intuicion: el porcentaje intra-comunal es
# mayor en Cuidado/Personal que en Empleo/Estudio. Los viajes
# laborales cruzan la ciudad; los de cuidado se quedan cerca. Este
# patron tambien sostendra las diferencias de clustering y triangulos
# que veremos en la version H3 con PMI.

# %% [markdown]
# ## PARTE 4. Asignacion de viajes a celdas H3
#
# Convertimos las coordenadas EOD (UTM 19S con coma decimal) a Points,
# reproyectamos a WGS84 y asignamos a una grilla H3-7 sobre el bbox
# del Gran Santiago. Cada viaje queda etiquetado con grid_origen y
# grid_destino. Excluimos viajes intra-celda (origen == destino) para
# que el analisis sea de movilidad real entre zonas.

# %%
viajes_geo = eod.georreferenciar_viajes(
    viajes.copy(), "OrigenCoordX", "OrigenCoordY",
    crs_origen="EPSG:32719", crs_destino="EPSG:4326",
).rename(columns={"geometry": "geom_origen"})

destinos_geo = eod.georreferenciar_viajes(
    viajes.copy(), "DestinoCoordX", "DestinoCoordY",
    crs_origen="EPSG:32719", crs_destino="EPSG:4326",
).rename(columns={"geometry": "geom_destino"})

viajes_geo["geom_destino"] = destinos_geo["geom_destino"].values

# Asignamos H3 por lat/lon a cada Point de origen y destino.
def asignar_h3(geom):
    return h3.latlng_to_cell(geom.y, geom.x, H3_LEVEL)

viajes_geo["grid_origen"] = viajes_geo["geom_origen"].apply(asignar_h3)
viajes_geo["grid_destino"] = viajes_geo["geom_destino"].apply(asignar_h3)

# Filtramos viajes que tengan al menos origen o destino dentro del bbox.
hex_zona = h3_grid_from_bounds(BBOX_GRAN_STGO, grid_level=H3_LEVEL).reset_index()
celdas_zona = set(hex_zona["h3_cell_id"])
viajes_geo = viajes_geo[
    viajes_geo["grid_origen"].isin(celdas_zona)
    & viajes_geo["grid_destino"].isin(celdas_zona)
]

# Excluimos intra-celda (origen == destino) para enfocarnos en flujos.
viajes_od = viajes_geo[viajes_geo["grid_origen"] != viajes_geo["grid_destino"]].copy()

print(f"Viajes inter-celda: {len(viajes_od)} ({viajes_od['Peso'].sum():,.0f} expandidos)")
print(f"Celdas H3-{H3_LEVEL} en zona estudio: {len(hex_zona)}")
print(f"Celdas con viajes: {len(set(viajes_od['grid_origen']).union(viajes_od['grid_destino']))}")

# %% [markdown]
# ### Anatomia de la matriz O-D
#
# La matriz O-D es el insumo fundamental del analisis. Cada fila es un
# par (origen, destino), cada columna es una categoria, el valor es la
# suma de pesos. La funcion `matriz_od_por_categoria` de gdsutils.redes
# hace pivot directo y agrega la columna Total + filtro de volumen
# minimo.
#
# El filtro `total_minimo = 300` descarta pares con muy pocos viajes
# expandidos (ruido por muestreo). Sin filtro, la mayoria de las
# aristas son espurias y los estadisticos se distorsionan.

# %%
# Categoria 1: solo Sexo (para empezar con la version mas simple).
matriz_sexo = matriz_od_por_categoria(
    viajes_od,
    col_origen="grid_origen",
    col_destino="grid_destino",
    col_categoria="Sexo",
    col_peso="Peso",
    total_minimo=FILTRO_TOTAL_MIN,
)
print("Matriz O-D por sexo (head):")
print(matriz_sexo.head().round(1))
print(f"\nPares O-D con Total > {FILTRO_TOTAL_MIN}: {len(matriz_sexo)}")
print(f"Volumen total expandido: {matriz_sexo['Total'].sum():,.0f}")

# %% [markdown]
# ## PARTE 5. Red O-D basica por sexo
#
# Antes de calcular PMI veamos la red cruda. Construimos dos
# GeoDataFrames de aristas (uno por sexo) con las lineas entre
# centroides de celdas, y los pintamos en small multiples con grosor
# proporcional al peso.

# %%
# Centroides de las celdas (necesarios para dibujar las lineas).
hex_zona["centroide"] = hex_zona.geometry.centroid
centroide_por_celda = dict(zip(hex_zona["h3_cell_id"], hex_zona["centroide"]))

# Comunas de la RM clipeadas al bbox para usar como fondo.
carto = gpd.read_parquet(CARTO_PATH)
if "SHAPE" in carto.columns and "geometry" not in carto.columns:
    carto = carto.set_geometry("SHAPE").rename_geometry("geometry")
carto = carto.to_crs("EPSG:4326")
comunas_rm = carto[carto["COD_REGION"] == 13].clip(box(*BBOX_GRAN_STGO))
comunas_rm = comunas_rm[
    comunas_rm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
].copy()
print(f"Comunas RM en zona estudio: {len(comunas_rm)}")

# %%
aristas_h = matriz_od_a_aristas_geodataframe(
    matriz_sexo, "Hombre", centroide_por_celda
)
aristas_m = matriz_od_a_aristas_geodataframe(
    matriz_sexo, "Mujer", centroide_por_celda
)
print(f"Aristas hombres: {len(aristas_h)}, mujeres: {len(aristas_m)}")
print(f"Peso medio hombres: {aristas_h['peso'].mean():.0f}, "
      f"mujeres: {aristas_m['peso'].mean():.0f}")

# %%
# Pintamos las dos redes lado a lado. Grosor proporcional al peso
# escalado al maximo comun para que las dos sean comparables.
peso_max = max(aristas_h["peso"].max(), aristas_m["peso"].max())
fig, axes = small_multiples_from_geodataframe(comunas_rm, n_variables=2,
                                              col_wrap=2, height=6)
for ax, (titulo, gdf, color) in zip(
    axes,
    [
        (f"Hombres ({aristas_h['peso'].sum():,.0f} viajes)", aristas_h, COLOR_HOMBRE),
        (f"Mujeres ({aristas_m['peso'].sum():,.0f} viajes)", aristas_m, COLOR_MUJER),
    ],
):
    comunas_rm.plot(ax=ax, facecolor="#fafaf5", edgecolor="#aaaaaa", linewidth=0.4)
    anchos = 0.2 + 3.0 * gdf["peso"] / peso_max
    gdf.plot(ax=ax, color=color, linewidth=anchos.values, alpha=0.55)
    ax.set_title(titulo)
fig.savefig("images/10-red-od-sexo.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# A simple vista las dos redes parecen iguales: ambas tienen flujos
# pesados sobre los corredores principales (Alameda, Vespucio). Los
# conteos crudos no revelan la diferencia. Para encontrarla necesitamos
# normalizar por las marginales: ahi entra PMI.

# %% [markdown]
# ## PARTE 6. PMI por sexo
#
# El Pointwise Mutual Information mide cuanto se aleja la probabilidad
# condicional de un par O-D dado un sexo respecto a la probabilidad
# marginal del par O-D:
#
#   PMI(arista, sexo) = log2( P(arista | sexo) / P(arista) )
#
#   - PMI > 0: el sexo usa esa arista mas de lo esperado bajo independencia.
#   - PMI < 0: la usa menos.
#   - PMI = 0: ocurre como se esperaria si sexo y arista fueran independientes.
#
# Usamos suavizado aditivo (Laplace) con alpha = 0.5 para evitar log(0)
# cuando un sexo tiene 0 viajes en una arista.

# %%
pmi_sexo = pmi_od(matriz_sexo, ["Hombre", "Mujer"], suavizado=SUAVIZADO_PMI)
print("PMI por sexo (head):")
print(pmi_sexo.head().round(3))
print()
print("Distribucion de PMI:")
print(pmi_sexo.describe().round(3))

# %%
# Histograma comparativo. Las dos distribuciones deberian ser casi
# simetricas alrededor de 0 si la marginal de Hombre y Mujer es similar.
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-1.5, 1.5, 50)
ax.hist(pmi_sexo["Hombre"], bins=bins, color=COLOR_HOMBRE, alpha=0.55,
        label=f"Hombre (mediana={pmi_sexo['Hombre'].median():+.2f})", edgecolor="white")
ax.hist(pmi_sexo["Mujer"], bins=bins, color=COLOR_MUJER, alpha=0.55,
        label=f"Mujer (mediana={pmi_sexo['Mujer'].median():+.2f})", edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("PMI (bits)")
ax.set_ylabel("Pares O-D")
ax.set_title("Distribucion del PMI por sexo (sin desglose por proposito)")
ax.legend()
fig.savefig("images/10-pmi-sexo-hist.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# Sin desglose por proposito las dos distribuciones son casi
# simetricas: hay aristas masculinas y femeninas, pero el patron no se
# concentra. Agreguemos el proposito.

# %% [markdown]
# ## PARTE 7. Matriz O-D por sexo x proposito
#
# Cada viaje queda etiquetado con su par (sexo, proposito), formando
# seis categorias: Hombre_Cuidado, Hombre_Empleo/Estudio,
# Hombre_Personal, Mujer_Cuidado, Mujer_Empleo/Estudio,
# Mujer_Personal. La matriz tiene seis columnas + Total. Filtramos
# pares O-D con Total > 300 (igual que antes).

# %%
viajes_od["GenProp"] = viajes_od["Sexo"] + "_" + viajes_od["PropositoAgregado"]
matriz_gp = matriz_od_por_categoria(
    viajes_od,
    col_origen="grid_origen",
    col_destino="grid_destino",
    col_categoria="GenProp",
    col_peso="Peso",
    total_minimo=FILTRO_TOTAL_MIN,
)
categorias = [c for c in matriz_gp.columns if c != "Total"]
print(f"Pares O-D filtrados: {len(matriz_gp)}")
print(f"Categorias: {categorias}")
print()
print("Matriz O-D por sexo x proposito (head, redondeada):")
print(matriz_gp.head().round(1))

# %%
# PMI por categoria sexo x proposito.
pmi_gp = pmi_od(matriz_gp, categorias, suavizado=SUAVIZADO_PMI)
print("PMI por sexo x proposito (head):")
print(pmi_gp.head().round(3))

# %%
# Distribuciones de PMI por categoria. Si Mujer_Cuidado tiene mas masa
# en la cola positiva que Hombre_Cuidado, la categoria femenina del
# cuidado concentra el patron.
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for ax, cat in zip(axes.flat, categorias):
    vals = pmi_gp[cat].dropna()
    color = COLORES_CAT.get(cat, "gray")
    ax.hist(vals, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(vals.median(), color="darkred", linewidth=1.5,
               label=f"mediana={vals.median():+.2f}")
    ax.set_title(cat, fontsize=10)
    ax.set_xlabel("PMI (bits)")
    ax.legend(fontsize=8)
fig.savefig("images/10-pmi-genprop-hist.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 8. Subgrafos PMI > 0 por categoria
#
# Para cada categoria construimos el subgrafo dirigido que contiene
# solo las aristas con PMI > 0 (sobre-representadas) y al menos 50
# viajes expandidos en esa categoria. Esto es el conjunto de pares
# O-D distintivos de la categoria.

# %%
subgrafos = {}
for cat in categorias:
    g = subgrafo_pmi(
        matriz_gp, cat, categorias,
        suavizado=SUAVIZADO_PMI,
        umbral_pmi=0.0,
        min_viajes=FILTRO_VIAJES_CAT,
    )
    subgrafos[cat] = g
    m = metricas_subgrafo(g)
    r = m["resumen"]
    print(
        f"{cat:30s} | aristas={r['n_aristas']:4d} | nodos={r['n_nodos']:3d} "
        f"| triangulos={r['triangulos_total']:4d} "
        f"| clustering={r['clustering_medio']:.3f}"
    )


# %% [markdown]
# Mapa comparativo: las seis subredes en un grid 2x3 con aristas en
# triangulo destacadas (mas oscuras y gruesas) y aristas lineales en
# color claro. Pintar los seis paneles permite ver de un vistazo donde
# se concentra el cierre local.

# %%
# Para mantener escala comparable: maximo peso comun entre todas las
# categorias, calculado sobre las aristas que se pintaran.
gdfs_top = {}
for cat in categorias:
    gdf = aristas_pmi_a_geodataframe(subgrafos[cat], centroide_por_celda)
    gdfs_top[cat] = top_aristas(gdf, TOP_ARISTAS_PCT)
peso_max_gp = max(
    (gdfs_top[c]["peso"].max() for c in categorias if len(gdfs_top[c]) > 0),
    default=1.0,
)

# In-degree por nodo dentro del subgrafo filtrado de cada categoria.
# Cada arista (u, v) contribuye 1 al in-degree de v.
in_degrees_por_cat = {
    cat: gdfs_top[cat].groupby("v").size().to_dict() for cat in categorias
}
in_deg_max = max(
    (max(d.values(), default=0) for d in in_degrees_por_cat.values()),
    default=1,
)

fig, axes = small_multiples_from_geodataframe(
    comunas_rm, n_variables=6, col_wrap=3, height=5
)
for ax, cat in zip(axes, categorias):
    gdf_t = gdfs_top[cat]
    g = subgrafos[cat]
    in_deg = in_degrees_por_cat[cat]

    comunas_rm.plot(ax=ax, facecolor="#fafaf5", edgecolor="#bbbbbb", linewidth=0.3)
    if len(gdf_t) > 0:
        lw = 0.4 + 3.0 * gdf_t["peso"] / peso_max_gp
        gdf_t.plot(ax=ax, color=COLORES_CAT[cat], linewidth=lw.values, alpha=0.75)

        # Nodos con tamano proporcional al in-degree.
        nodos_x, nodos_y, nodos_s = [], [], []
        for nodo, deg in in_deg.items():
            if nodo in centroide_por_celda:
                p = centroide_por_celda[nodo]
                nodos_x.append(p.x)
                nodos_y.append(p.y)
                nodos_s.append(8 + 220 * deg / in_deg_max)
        ax.scatter(nodos_x, nodos_y, s=nodos_s, c="white",
                   edgecolors="black", linewidths=0.5, zorder=5)

    ax.set_title(
        f"{cat}\ntop {len(gdf_t)} de {g.number_of_edges()} aristas",
        fontsize=10,
    )
fig.suptitle(
    f"Aristas con mayor sqrt(PMI x peso): top {int(TOP_ARISTAS_PCT * 100)}% "
    "del subgrafo PMI > 0. Grosor = peso. Tamano de nodo = in-degree.",
    fontsize=12,
)
fig.savefig("images/10-subgrafos-pmi.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 9. Metricas topologicas observadas
#
# Resumimos en una tabla las metricas de cada subgrafo PMI > 0. Las
# diferencias relevantes para el caso son:
#
#   - clustering_medio: cierre local. Mas alto = mas patron triangular.
#   - triangulos_total: cuantos triangulos hay en la red.
#   - n_aristas: tamano del subgrafo.

# %%
filas = []
for cat in categorias:
    g = subgrafos[cat]
    m = metricas_subgrafo(g)
    r = m["resumen"].copy()
    r["categoria"] = cat
    r["sexo"], r["proposito"] = cat.split("_", 1)
    r["viajes_total"] = float(matriz_gp[cat].sum())
    filas.append(r)

resumen = pd.DataFrame(filas).set_index("categoria")
print("Metricas observadas:")
print(resumen[
    ["sexo", "proposito", "n_aristas", "n_nodos", "clustering_medio",
     "triangulos_total", "viajes_total"]
].round(3))

print("\nDelta Mujer - Hombre por proposito:")
for prop in ["Cuidado", "Empleo/Estudio", "Personal"]:
    cat_m = f"Mujer_{prop}"
    cat_h = f"Hombre_{prop}"
    if cat_m in resumen.index and cat_h in resumen.index:
        d_clust = resumen.loc[cat_m, "clustering_medio"] - resumen.loc[cat_h, "clustering_medio"]
        d_tri = resumen.loc[cat_m, "triangulos_total"] - resumen.loc[cat_h, "triangulos_total"]
        d_edges = resumen.loc[cat_m, "n_aristas"] - resumen.loc[cat_h, "n_aristas"]
        print(
            f"  {prop:18s} | dclustering={d_clust:+.3f} "
            f"| dtriangulos={d_tri:+4d} | dedges={d_edges:+4d}"
        )

# %% [markdown]
# ## PARTE 10. Modelo nulo por permutacion
#
# Las diferencias observadas son distinguibles del azar? Construimos
# un modelo nulo permutando la columna Sexo entre viajes (conservando
# origen, destino, proposito, peso). Bajo la hipotesis nula sexo y
# par (O-D, proposito) son independientes; cualquier diferencia
# significativa entre Mujer_Cuidado y Hombre_Cuidado deberia
# desaparecer.
#
# Hacemos 200 permutaciones. La implementacion vectorizada con
# np.bincount termina en menos de un minuto.

# %%
print(f"Corriendo {N_PERMUTACIONES} permutaciones (puede tomar 30-60s)...")
nulo_df = modelo_nulo_permutacion(
    viajes_od,
    categorias=categorias,
    col_origen="grid_origen",
    col_destino="grid_destino",
    col_sexo="Sexo",
    col_peso="Peso",
    col_proposito="PropositoAgregado",
    n_permutaciones=N_PERMUTACIONES,
    total_minimo=FILTRO_TOTAL_MIN,
    min_viajes=FILTRO_VIAJES_CAT,
    umbral_pmi=0.0,
    suavizado=SUAVIZADO_PMI,
    seed=42,
)
print(f"Shape del DataFrame nulo: {nulo_df.shape}")
print(nulo_df.head())

# %%
# Calculamos p-values para las diferencias observadas. Test bilateral:
# proporcion de permutaciones donde |delta_nulo| >= |delta_observado|.
print("p-values (test bilateral por permutacion):\n")
metricas_test = [
    ("clustering_medio", "Clustering medio"),
    ("triangulos_total", "Triangulos"),
    ("n_aristas", "Aristas"),
    ("viajes_total", "Viajes total"),
]
filas_pval = []
for col, titulo in metricas_test:
    for prop in ["Cuidado", "Empleo/Estudio", "Personal"]:
        cat_m, cat_h = f"Mujer_{prop}", f"Hombre_{prop}"
        if cat_m not in resumen.index or cat_h not in resumen.index:
            continue
        delta_obs = resumen.loc[cat_m, col] - resumen.loc[cat_h, col]

        nulo_m = nulo_df[nulo_df["categoria"] == cat_m].set_index("permutacion")[col]
        nulo_h = nulo_df[nulo_df["categoria"] == cat_h].set_index("permutacion")[col]
        delta_nulo = nulo_m - nulo_h

        p_value = (delta_nulo.abs() >= abs(delta_obs)).mean()
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        filas_pval.append({"metrica": col, "proposito": prop,
                           "delta_obs": delta_obs, "p_value": p_value,
                           "sig": sig})
        print(f"  {titulo:16s} | {prop:16s} | delta_obs={delta_obs:+10.3f} "
              f"| p={p_value:.4f} {sig}")

pval_df = pd.DataFrame(filas_pval)

# %% [markdown]
# ## PARTE 11. Barras observadas vs modelo nulo (95% CI)
#
# Pintamos las cuatro metricas con barras para Hombre/Mujer + error
# bars con el rango 95% del modelo nulo. Si la barra observada sale
# del rango del nulo, la diferencia es significativa.

# %%
fig, axes = plt.subplots(1, len(metricas_test), figsize=(5 * len(metricas_test), 5))
propositos = ["Cuidado", "Empleo/Estudio", "Personal"]

for ax, (col, titulo) in zip(axes, metricas_test):
    x = np.arange(len(propositos))
    ancho = 0.35

    for offset, sexo, color in [
        (-ancho / 2, "Hombre", COLOR_HOMBRE),
        (ancho / 2, "Mujer", COLOR_MUJER),
    ]:
        vals_obs, lo, hi = [], [], []
        for prop in propositos:
            cat = f"{sexo}_{prop}"
            vals_obs.append(resumen.loc[cat, col] if cat in resumen.index else 0)
            nulo_cat = nulo_df[nulo_df["categoria"] == cat][col]
            lo.append(nulo_cat.quantile(0.025))
            hi.append(nulo_cat.quantile(0.975))
        vals_obs = np.array(vals_obs)
        lo = np.array(lo)
        hi = np.array(hi)

        ax.bar(x + offset, vals_obs, ancho, color=color, alpha=0.85, label=sexo)
        ax.errorbar(x + offset, (lo + hi) / 2, yerr=(hi - lo) / 2,
                    fmt="none", ecolor="black", capsize=4, linewidth=1.3,
                    label="95% null" if (offset < 0 and col == "clustering_medio") else None)

    ax.set_xticks(x)
    ax.set_xticklabels(propositos, fontsize=9)
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Observado vs modelo nulo por permutacion (intervalo 95%)",
             fontsize=13)
fig.savefig("images/10-observado-vs-nulo.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 12. Figura final: viajes de cuidado, mujeres vs hombres
#
# Comparamos las dos redes con mas relevancia para el caso de estudio:
# Mujer_Cuidado y Hombre_Cuidado. Las aristas en triangulo en color
# del genero correspondiente, las aristas lineales (sin cierre local)
# en gris. Los nodos pintados como circulos con tamano proporcional al
# volumen de viajes incidentes.

# %%
g_h_cuidado = subgrafos["Hombre_Cuidado"]
g_m_cuidado = subgrafos["Mujer_Cuidado"]

# Sin filtro top: pintamos todo el subgrafo PMI > 0.
tri_h = aristas_en_triangulos(g_h_cuidado)
tri_m = aristas_en_triangulos(g_m_cuidado)
gdf_h = aristas_pmi_a_geodataframe(g_h_cuidado, centroide_por_celda,
                                   aristas_en_triangulo=tri_h)
gdf_m = aristas_pmi_a_geodataframe(g_m_cuidado, centroide_por_celda,
                                   aristas_en_triangulo=tri_m)

vol_h = strength_por_nodo(g_h_cuidado)
vol_m = strength_por_nodo(g_m_cuidado)
vol_max_final = max(max(vol_h.values()) if vol_h else 1,
                    max(vol_m.values()) if vol_m else 1)
peso_max_final = max(
    gdf_h["peso"].max() if len(gdf_h) > 0 else 1,
    gdf_m["peso"].max() if len(gdf_m) > 0 else 1,
)

fig, axes = small_multiples_from_geodataframe(
    comunas_rm, n_variables=2, col_wrap=2, height=7
)
for ax, (titulo, g, gdf_t, color_tri, vol) in zip(
    axes,
    [
        ("Hombres - Cuidado", g_h_cuidado, gdf_h, COLOR_HOMBRE, vol_h),
        ("Mujeres - Cuidado", g_m_cuidado, gdf_m, COLOR_MUJER, vol_m),
    ],
):
    comunas_rm.plot(ax=ax, facecolor="#fafaf5", edgecolor="#bbbbbb", linewidth=0.4)

    if len(gdf_t) > 0:
        lw = 0.5 + 3.0 * gdf_t["peso"] / peso_max_final
        mask_lin = ~gdf_t["en_triangulo"]
        mask_tri = gdf_t["en_triangulo"]
        if mask_lin.any():
            gdf_t[mask_lin].plot(ax=ax, color="#cccccc",
                                 linewidth=lw[mask_lin].values, alpha=0.65)
        if mask_tri.any():
            gdf_t[mask_tri].plot(ax=ax, color=color_tri,
                                 linewidth=lw[mask_tri].values, alpha=0.9)

    # Nodos con tamano proporcional al volumen incidente.
    nodos_x, nodos_y, nodos_s = [], [], []
    for n in vol:
        if n in centroide_por_celda:
            p = centroide_por_celda[n]
            nodos_x.append(p.x)
            nodos_y.append(p.y)
            nodos_s.append(20 + 480 * vol[n] / vol_max_final)
    ax.scatter(nodos_x, nodos_y, s=nodos_s, c="white", edgecolors="black",
               linewidths=0.7, zorder=5)

    m = metricas_subgrafo(g)
    r = m["resumen"]
    n_tri = int(gdf_t["en_triangulo"].sum()) if len(gdf_t) > 0 else 0
    txt = (f"aristas={r['n_aristas']} (en triangulo={n_tri})\n"
           f"clustering={r['clustering_medio']:.3f}, "
           f"triangulos={r['triangulos_total']}")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_title(titulo, fontsize=12)

fig.suptitle(
    "Viajes de cuidado: subgrafo PMI > 0 completo. "
    "Color = arista en triangulo. Tamano de nodo = volumen incidente.",
    fontsize=12,
)
fig.savefig("images/10-cuidado-comparacion.png", dpi=128, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## PARTE 13. Flowmap interactivo con flowmap.gl
#
# Generamos un HTML self-contained que usa flowmap.gl/layers (deck.gl)
# para visualizar los flujos por categoria con toggle de visibilidad.
# Cada arista del subgrafo PMI > 0 (sin filtro top) se incluye como un
# flow.
#
# Para visualizar (los modulos ES no cargan desde file://, hay que
# servir el archivo por HTTP):
#
#     uv run python -m http.server -d images 8000
#
# y abrir http://localhost:8000/10-flowmap.html en el navegador.

# %%
# Datos para flowmap.gl: locations + flows con category.
flows_export = []
celdas_usadas = set()
for cat in categorias:
    gdf = aristas_pmi_a_geodataframe(subgrafos[cat], centroide_por_celda)
    for _, fila in gdf.iterrows():
        u, v = fila["u"], fila["v"]
        if u not in centroide_por_celda or v not in centroide_por_celda:
            continue
        flows_export.append({
            "origin": u,
            "dest": v,
            "count": round(float(fila["peso"]), 1),
            "category": cat,
            "pmi": round(float(fila["pmi"]), 3),
        })
        celdas_usadas.add(u)
        celdas_usadas.add(v)

locations_export = [
    {
        "id": c,
        "lon": round(float(centroide_por_celda[c].x), 6),
        "lat": round(float(centroide_por_celda[c].y), 6),
    }
    for c in celdas_usadas
]
print(f"Flowmap: {len(locations_export)} locations, "
      f"{len(flows_export)} flows en {len(categorias)} categorias")

# %%
# Generamos el HTML con el helper de gdsutils.redes. Template y
# logica de substitucion viven en gdsutils/templates/flowmap.html y
# gdsutils.redes.generar_html_flowmap.
descripcion_flowmap = (
    "Aristas: subgrafo PMI &gt; 0 de cada categoria. Grosor = volumen "
    "de viajes. Color = categoria (gradiente de claro a oscuro por "
    "magnitud). Hover sobre un arco para detalles."
    "<br/><br/>Fuente: EOD Santiago 2012."
)
ruta_html = generar_html_flowmap(
    locations=locations_export,
    flows=flows_export,
    colores=COLORES_CAT,
    ruta_salida="images/10-flowmap.html",
    title="Flujos OD por sexo x proposito (EOD Santiago 2012)",
    panel_titulo="Sexo x proposito",
    descripcion_html=descripcion_flowmap,
    centro=(-70.65, -33.45),
    zoom=10.3,
    pitch=0,
)
print(f"HTML generado: {ruta_html}")
print()
print("Para visualizar:")
print("  uv run python -m http.server -d images 8000")
print("  Abrir http://localhost:8000/10-flowmap.html")

# %% [markdown]
# ## Cierre
#
# Las redes de cuidado de mujeres y hombres tienen un volumen
# distinto, pero la diferencia que importa es topologica: las
# aristas femeninas estan mas trianguladas (mas vecinos compartidos),
# lo que es consistente con la hipotesis de viajes encadenados
# cortos. Las redes masculinas de empleo/estudio son mas radiales:
# aristas largas y poco cerradas.
#
# El modelo nulo confirma que las diferencias no son atribuibles al
# azar: permutar la columna Sexo conservando todo lo demas reproduce
# la marginal pero borra la asimetria. Las metricas observadas para
# Mujer_Cuidado caen fuera del intervalo del 95% del nulo en
# clustering y triangulos.
#
# Esta clase mostro tres ideas transversales:
#
#   - Una matriz O-D es un grafo. Lo que veiamos en clase 09 como
#     calles tambien funciona para flujos.
#   - Las metricas crudas (volumen, grado) ocultan patrones cuando hay
#     subgrupos. PMI normaliza por marginales y los recupera.
#   - Las diferencias entre subgrupos deben contrastarse contra un
#     modelo nulo. Sin ese paso, cualquier ranking de pares O-D
#     amplifica ruido de muestreo.
