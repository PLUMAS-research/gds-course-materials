# %%
# Descarga de datos
# - Encuesta Origen-Destino 2012 (viajes + personas + zonificación)
# - Cartografía censal 2024 (para bordes comunales de contexto)

from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/eod2012.tgz")
# %%

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chiricoca.colors import categorical_color_legend
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.geo.grid import h3_grid_from_bounds
from chiricoca.maps import choropleth_map
from chiricoca.tables import marimekko, slopegraph, streamgraph
from matplotlib.ticker import FuncFormatter

import gdsutils.eodscl as eod
from gdsutils.censo2024 import regiones
from gdsutils.eodscl import (
    COLORES_PROPOSITO,
    GRUPOS_PROPOSITOS,
    PROPOSITOS,
    calcular_viajes_proposito,
    georreferenciar_viajes,
)
from gdsutils.geo import clip_geodataframe
from gdsutils.spatial import COLORES_LISA, calcular_lisa, graficar_lisa

setup_style(dpi=96)

# %%
# Parámetros geográficos
BBOX = (-70.85, -33.65, -70.45, -33.3)
EOD_PATH = Path("data") / "eod2012" / "EOD_STGO"
ZONAS_PATH = EOD_PATH / "Zonificacion_EOD2012"
RUTA_CARTO = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

# Colores por sexo
COLOR_HOMBRE = "#641a80"
COLOR_MUJER = "#de4968"


# %%
# ========================================
# PARTE 1: Carga y preparación de datos
# ========================================
# Leemos los viajes de la EOD 2012 y los combinamos con datos de personas
# para obtener factores de expansión.

viajes = eod.read_trips(EOD_PATH)

# Descartamos viajes con origen o destino fuera de la RM
viajes = viajes[
    (viajes["SectorOrigen"] != "Exterior a RM")
    & (viajes["SectorDestino"] != "Exterior a RM")
    & pd.notnull(viajes["SectorOrigen"])
    & pd.notnull(viajes["SectorDestino"])
]

personas = eod.read_people(EOD_PATH)

viajes_personas = viajes.merge(personas)
viajes_personas["Peso"] = (
    viajes_personas["FactorExpansion"] * viajes_personas["FactorPersona"]
)

print(f"Viajes: {len(viajes_personas):,}")
print(f"Viajes expandidos: {viajes_personas['Peso'].sum():,.0f}")


# %%
# ========================================
# PARTE 2: Qué (propósitos de viaje)
# ========================================
# La EOD registra 14 propósitos de viaje. Los agrupamos en cuatro categorías:
# Cuidado (salud, compras, trámites, acompañar), Empleo/Estudio,
# Personal (recreación, comer) y Hogar (regreso a casa).

print("Propósitos en los datos (valores únicos):")
print(sorted(viajes_personas["Proposito"].dropna().unique()))

viajes_personas["PropositoAgregado"] = viajes_personas["Proposito"].map(
    GRUPOS_PROPOSITOS
)

print("\nPropositoAgregado (incluyendo NaN por propósitos no mapeados):")
print(viajes_personas["PropositoAgregado"].value_counts(dropna=False))


# %%
# ========================================
# PARTE 3: Cómo (modo de transporte)
# ========================================
# ¿En qué se mueve la gente? El marimekko muestra volumen (ancho) y
# composición por sexo (alto) de cada modo. El slopegraph compara las
# razones declaradas para no usar transporte público entre hombres y mujeres.

MODOS_AGREGADOS = {
    "Auto": "Auto",
    "Bicicleta": "Bicicleta",
    "Bip!": "Transporte público",
    "Bip! - Otros Privado": "Transporte público",
    "Bip! - Otros Público": "Transporte público",
    "Caminata": "Caminata",
    "Otros": "Otros",
    "Taxi": "Taxi/colectivo",
    "Taxi Colectivo": "Taxi/colectivo",
}

viajes_personas["ModoAgregado2"] = (
    viajes_personas["ModoDifusion"].map(MODOS_AGREGADOS).fillna("Otros")
)

tabla_modo = (
    viajes_personas.groupby(["ModoAgregado2", "Sexo"])["Peso"]
    .sum()
    .unstack(fill_value=0)
)

print("Distribución por modo de transporte (viajes expandidos):")
print(tabla_modo)

fig, ax = plt.subplots(figsize=(11, 4))
marimekko(
    tabla_modo,
    sort_items=True,
    sort_categories=False,
    annotate=True,
    xlabel_levels=True,
    legend_args="right",
    palette=[COLOR_HOMBRE, COLOR_MUJER],
    ax=ax,
)
ax.set_ylabel("Proporción")
fig.suptitle("Modo de transporte por sexo")
sns.despine(ax=ax)
fig.savefig(IMAGES / "04-modo-por-sexo.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# Razones de no-uso de transporte público por sexo
# La EOD pregunta a quienes no usaron Transantiago por qué no lo hicieron
# (selección múltiple). El slopegraph muestra si las razones difieren
# entre hombres y mujeres: cada línea es una razón, la pendiente indica
# quién la menciona más.

no_tp_raw = eod.read_transantiago_usage(EOD_PATH)

personas_info = viajes_personas[["Persona", "Sexo", "FactorPersona"]].drop_duplicates(
    "Persona"
)
no_tp = no_tp_raw.merge(personas_info, on="Persona", how="inner")

tabla_notp = (
    no_tp.groupby(["NoUsaTransantiago", "Sexo"])["FactorPersona"]
    .sum()
    .unstack(fill_value=0)
)
tabla_notp_norm = tabla_notp.div(tabla_notp.sum(axis=1), axis=0)
tabla_notp_norm = tabla_notp_norm.sort_values("Hombre")

print("Razones de no-uso de TP (proporción por sexo):")
print(tabla_notp_norm.round(3))

# %%
ax = slopegraph(
    tabla_notp_norm[["Hombre", "Mujer"]],
    colors=(COLOR_MUJER, COLOR_HOMBRE),
    reference_line=0.5,
)
ax.set_xticklabels(["Hombres", "Mujeres"])
fig = ax.get_figure()
fig.suptitle("Razones de no uso de transporte público por sexo", fontsize=10)
ax.set_title(
    "Proporción de personas que menciona cada razón (selección múltiple)",
    fontsize=8,
)
sns.despine(ax=ax, left=True)
fig.savefig(IMAGES / "04-no-uso-tp-sexo.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# ========================================
# PARTE 4: Cuándo (perfil temporal)
# ========================================
# Cada panel muestra la distribución horaria de viajes por propósito,
# separada por sexo. La comparación revela si hombres y mujeres viajan
# a horas distintas o con propósitos distintos en cada franja del día.

viajes_personas["hora_inicio"] = (
    viajes_personas["HoraIni"].dt.total_seconds().div(3600).astype(int) % 24
)

viajes_con_hogar = viajes_personas[viajes_personas["PropositoAgregado"].notna()].copy()

print("Viajes por propósito (con hogar):")
print(viajes_con_hogar["PropositoAgregado"].value_counts())

hora_min = viajes_con_hogar["hora_inicio"].min()
hora_max = viajes_con_hogar["hora_inicio"].max()
rango_horas = range(hora_min, hora_max + 1)

_fmt_miles = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}")

# Máximo global para compartir escala entre paneles
y_max_global = max(
    viajes_con_hogar[viajes_con_hogar["Sexo"] == sexo]
    .groupby("hora_inicio")["Peso"]
    .sum()
    .max()
    for sexo in ["Hombre", "Mujer"]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, sexo in zip(axes, ["Hombre", "Mujer"]):
    sub = viajes_con_hogar[viajes_con_hogar["Sexo"] == sexo]
    tabla_temporal = (
        sub.groupby(["hora_inicio", "PropositoAgregado"])["Peso"]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            index=rango_horas, columns=list(COLORES_PROPOSITO.keys()), fill_value=0
        )
    )
    streamgraph(
        tabla_temporal,
        baseline="zero",
        colors=COLORES_PROPOSITO,
        labels=True,
        label_args={"fontsize": 9},
        edgecolor="white",
        linewidth=0.3,
        skip_set_ylim=True,
        ax=ax,
    )
    ax.set_ylim(0, y_max_global * 1.05)
    n_total = int(sub["Peso"].sum())
    ax.set_xlabel("Hora del día")
    ax.yaxis.set_major_formatter(_fmt_miles)
    ax.text(
        0.02,
        0.95,
        f"{sexo} (n={n_total:,})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    sns.despine(ax=ax)

axes[0].set_ylabel("Viajes expandidos")
fig.suptitle("Perfil temporal de viajes por propósito y sexo")
fig.savefig(IMAGES / "04-perfil-temporal.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# ========================================
# PARTE 5: Dónde (distribución geográfica)
# ========================================
# Pasamos de las dimensiones tabulares (qué, cómo, cuándo) a la espacial.
# Primero mostramos la distribución de destinos como puntos; después
# agregamos por zona EOD y visualizamos como mapas coropléticos.

# Cartografía de referencia para toda la sesión
carto_rm = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(
    "EPSG:4326"
)
carto_stgo = clip_geodataframe(carto_rm, BBOX)

# %%
# GeoDataFrame de viajes con destino georreferenciado.
# Se reutiliza en el análisis MAUP (parte 8) y LISA H3 (parte 9).
_destino_df = viajes_personas.dropna(
    subset=["DestinoCoordX", "DestinoCoordY", "PropositoAgregado"]
)

viajes_destino = georreferenciar_viajes(_destino_df, "DestinoCoordX", "DestinoCoordY")

print(f"Viajes con destino georreferenciado: {len(viajes_destino):,}")

# %%
# Destinos de viaje como puntos (muestra de hasta 5 000 por propósito)
_N_MUESTRA = 5_000

_viajes_bbox = viajes_destino.cx[BBOX[0] : BBOX[2], BBOX[1] : BBOX[3]]
fig, axes = small_multiples_from_geodataframe(
    _viajes_bbox, len(PROPOSITOS), col_wrap=2, height=6
)

for ax, proposito in zip(axes, PROPOSITOS):
    sub = _viajes_bbox[_viajes_bbox["PropositoAgregado"] == proposito]
    muestra = sub.sample(min(_N_MUESTRA, len(sub)), random_state=42)
    ax.scatter(
        muestra.geometry.x,
        muestra.geometry.y,
        s=1,
        alpha=0.9,
        color=COLORES_PROPOSITO[proposito],
        edgecolor="none",
    )
    carto_stgo.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.3)
    ax.set_title(f"{proposito}  (n={len(sub):,})")

for ax in axes[len(PROPOSITOS) :]:
    ax.set_visible(False)

fig.suptitle("Destinos de viajes por propósito (EOD Santiago 2012)")
fig.savefig(IMAGES / "04-destinos-proposito.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# Agregación por zonas EOD
# Usamos la zona de destino como unidad de análisis: cada zona refleja
# qué actividades concentra, no dónde residen quienes las realizan.
# La variable es la transformación de Anscombe (sqrt(x + 3/8)) de los
# conteos ponderados, que estabiliza la varianza de datos Poisson.

zonas_eod = gpd.read_file(ZONAS_PATH / "Zonificacion_EOD2012.shp").to_crs("EPSG:4326")
zonas_eod = clip_geodataframe(zonas_eod, BBOX)
print(f"Zonas EOD en el área de estudio: {len(zonas_eod)}")

viajes_dest_zona = viajes_personas[
    viajes_personas["ZonaDestino"].notna()
    & viajes_personas["PropositoAgregado"].notna()
].copy()

proporciones_zonas = calcular_viajes_proposito(viajes_dest_zona, "ZonaDestino")

viajes_por_zona = viajes_dest_zona.groupby("ZonaDestino")["Peso"].sum()
zonas_suficientes = viajes_por_zona[viajes_por_zona >= 500].index

geodf_zonas = (
    zonas_eod.merge(
        proporciones_zonas.reset_index(),
        left_on="Zona",
        right_on="ZonaDestino",
        how="inner",
    )
    .loc[lambda df: df["ZonaDestino"].isin(zonas_suficientes)]
    .copy()
)

print(f"Zonas con >= 500 viajes recibidos: {len(geodf_zonas)}")
for p in PROPOSITOS:
    print(f"  {p}: media={geodf_zonas[p].mean():.3f}, std={geodf_zonas[p].std():.3f}")

# %%
# Mapas coropléticos: Anscombe(viajes recibidos) por propósito

fig, axes = small_multiples_from_geodataframe(
    zonas_eod, len(PROPOSITOS), col_wrap=2, height=6
)

for ax, proposito in zip(axes, PROPOSITOS):
    choropleth_map(
        geodf_zonas,
        proposito,
        k=5,
        binning="fisher_jenks",
        palette="magma",
        edgecolor="white",
        linewidth=0.3,
        ax=ax,
    )
    carto_stgo.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.3)
    ax.set_title(proposito)

for ax in axes[len(PROPOSITOS) :]:
    ax.set_visible(False)

fig.suptitle("Anscombe(viajes recibidos) por propósito en zonas EOD (destinos)")
fig.savefig(IMAGES / "04-prop-cuidado-zonas-eod.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# ========================================
# PARTE 6: Autocorrelación espacial global
# ========================================
# Los mapas coropléticos sugieren agrupación espacial, pero no la cuantifican.
# El I de Moran mide si los valores están agrupados (I > 0), dispersos
# (I < 0), o distribuidos al azar (I ~ 0).

from esda.moran import Moran
from libpysal.weights import Queen
from libpysal.weights.spatial_lag import lag_spatial

w_zonas = Queen.from_dataframe(geodf_zonas)
w_zonas.transform = "r"

moran_por_proposito = {}
for proposito in PROPOSITOS:
    moran_por_proposito[proposito] = Moran(geodf_zonas[proposito], w_zonas)
    m = moran_por_proposito[proposito]
    print(f"{proposito}: I={m.I:.4f}, p={m.p_norm:.4f}, z={m.z_norm:.4f}")

# Se conserva moran_zonas apuntando a Empleo/Estudio para la tabla MAUP (parte 8)
moran_zonas = moran_por_proposito["Empleo/Estudio"]

# %%
# Diagramas de dispersión de Moran, uno por propósito
# Eje X: valor estandarizado; eje Y: promedio ponderado de vecinos (spatial lag).

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, proposito in zip(axes.flatten(), PROPOSITOS):
    x = geodf_zonas[proposito]
    z = (x - x.mean()) / x.std()
    lag_z = lag_spatial(w_zonas, z)
    m = moran_por_proposito[proposito]

    ax.scatter(z, lag_z, s=8, alpha=0.4, color="#636363", edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    x_line = np.array([z.min(), z.max()])
    ax.plot(x_line, m.I * x_line, color="#e07b54", linewidth=1.5)

    for txt, ha, va, xp, yp in [
        ("HH", "right", "top", 0.97, 0.97),
        ("LL", "left", "bottom", 0.03, 0.03),
        ("LH", "left", "top", 0.03, 0.97),
        ("HL", "right", "bottom", 0.97, 0.03),
    ]:
        ax.text(
            xp,
            yp,
            txt,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=11,
            color="#aaaaaa",
        )

    ax.set_title(f"{proposito}  (I = {m.I:.3f}, p = {m.p_norm:.3f})")
    ax.set_xlabel("Anscombe(viajes) (estandarizado)")
    ax.set_ylabel("Lag espacial")
    sns.despine(ax=ax)

fig.suptitle(
    "Diagramas de Moran sobre Anscombe(viajes) por propósito (destinos, zonas EOD)"
)
fig.savefig(IMAGES / "04-moran-scatter-zonas.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# ========================================
# PARTE 7: LISA (indicadores locales)
# ========================================
# LISA (Local Indicators of Spatial Association) descompone el I de Moran
# global en contribuciones locales. Identifica clusters espaciales:
#   HH: zona con valor alto rodeada de valores altos
#   LL: zona con valor bajo rodeada de valores bajos
#   HL: zona con valor alto rodeada de valores bajos (outlier)
#   LH: zona con valor bajo rodeada de valores altos (outlier)

lisas_por_proposito = {}
for proposito in PROPOSITOS:
    lisa, moran = calcular_lisa(geodf_zonas, proposito)
    lisas_por_proposito[proposito] = (lisa, moran)
    print(f"\n{proposito}:")
    print(lisa["lisa_cluster"].value_counts())

# Se conserva lisa_zonas apuntando a Empleo/Estudio para la comparación MAUP (parte 8)
lisa_zonas, _ = lisas_por_proposito["Empleo/Estudio"]

# %%
# Mapas LISA, uno por propósito

fig, axes = small_multiples_from_geodataframe(
    zonas_eod, len(PROPOSITOS), col_wrap=2, height=7
)

for ax, proposito in zip(axes, PROPOSITOS):
    lisa, moran = lisas_por_proposito[proposito]
    graficar_lisa(lisa, ax, f"{proposito}  (I = {moran.I:.3f})", carto_stgo)

categorical_color_legend(axes[-1], COLORES_LISA, loc="lower right")

for ax in axes[len(PROPOSITOS) :]:
    ax.set_visible(False)

fig.suptitle("LISA: Anscombe(viajes recibidos) por propósito (zonas EOD, destinos)")
fig.savefig(IMAGES / "04-lisa-zonas-eod.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# ========================================
# PARTE 8: MAUP (efecto de la escala espacial)
# ========================================
# El Problema de la Unidad de Área Modificable (MAUP) muestra que los
# resultados de un análisis espacial dependen de la unidad de agregación.
# Repetimos el análisis LISA con tres unidades adicionales:
# grilla H3 nivel 7, grilla H3 nivel 8, y comunas.
# En todas las escalas usamos Anscombe(viajes de empleo/estudio) en destinos.


def asignar_a_h3(viajes_gdf, grid_level, bbox, proposito="Empleo/Estudio"):
    """Crea grilla H3 y calcula Anscombe(viajes) con el propósito dado por celda."""
    print(f"Construyendo grilla H3 nivel {grid_level}...")
    bounds = [bbox[0], bbox[1], bbox[2], bbox[3]]
    gdf_hex = h3_grid_from_bounds(
        bounds, extra_margin=0.025, grid_level=grid_level, crs="EPSG:4326"
    ).reset_index()
    print(f"  Celdas H3: {len(gdf_hex)}")

    viajes_h3 = gpd.sjoin(
        viajes_gdf[["geometry", "PropositoAgregado", "Peso"]],
        gdf_hex[["h3_cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    sqrt_df = calcular_viajes_proposito(viajes_h3, "h3_cell_id", [proposito])
    sqrt_serie = sqrt_df[proposito].rename("sqrt_viajes")

    total = viajes_h3.groupby("h3_cell_id")["Peso"].sum()
    celdas_suficientes = total[total >= 50].index
    geodf_h3 = (
        gdf_hex.set_index("h3_cell_id").join(sqrt_serie).loc[celdas_suficientes].copy()
    )
    geodf_h3 = geodf_h3.dropna(subset=["sqrt_viajes"]).reset_index()

    print(f"  Celdas con >= 50 viajes: {len(geodf_h3)}")
    return geodf_h3


# %%
# LISA con grilla H3 nivel 7

geodf_h3_7 = asignar_a_h3(viajes_destino, 7, BBOX)
lisa_h3_7, moran_h3_7 = calcular_lisa(geodf_h3_7, "sqrt_viajes")

print(f"\nMoran I (H3-7): {moran_h3_7.I:.4f} (p = {moran_h3_7.p_norm:.4f})")
print(lisa_h3_7["lisa_cluster"].value_counts())

# %%
# LISA con grilla H3 nivel 8

geodf_h3_8 = asignar_a_h3(viajes_destino, 8, BBOX)
lisa_h3_8, moran_h3_8 = calcular_lisa(geodf_h3_8, "sqrt_viajes")

print(f"\nMoran I (H3-8): {moran_h3_8.I:.4f} (p = {moran_h3_8.p_norm:.4f})")
print(lisa_h3_8["lisa_cluster"].value_counts())

# %%
# LISA a nivel comunal

sqrt_empleo_comunas = calcular_viajes_proposito(
    viajes_destino, "ComunaDestino", ["Empleo/Estudio"]
)["Empleo/Estudio"].rename("sqrt_viajes")

comunas_stgo = carto_stgo.dissolve(by="COMUNA").reset_index()

sqrt_comunas_df = sqrt_empleo_comunas.reset_index()
sqrt_comunas_df["COMUNA"] = sqrt_comunas_df["ComunaDestino"].str.upper()

geodf_comunas = comunas_stgo.merge(sqrt_comunas_df, on="COMUNA", how="inner")

viajes_por_comuna = viajes_destino.groupby("ComunaDestino")["Peso"].sum()
comunas_suficientes = viajes_por_comuna[viajes_por_comuna >= 200].index
geodf_comunas = geodf_comunas[
    geodf_comunas["ComunaDestino"].isin(comunas_suficientes)
].copy()

print(f"Comunas con >= 200 viajes: {len(geodf_comunas)}")

lisa_comunas, moran_comunas = calcular_lisa(geodf_comunas, "sqrt_viajes")

print(f"\nMoran I (Comunas): {moran_comunas.I:.4f} (p = {moran_comunas.p_norm:.4f})")
print(lisa_comunas["lisa_cluster"].value_counts())

# %%
# Comparación visual: cuatro mapas LISA muestran cómo la misma variable
# produce patrones diferentes según la unidad de agregación.

escalas = [
    (lisa_zonas, "Zonas EOD (866 zonas)"),
    (lisa_h3_7, "Grilla H3 nivel 7"),
    (lisa_h3_8, "Grilla H3 nivel 8"),
    (lisa_comunas, "Comunas"),
]

fig, axes = small_multiples_from_geodataframe(
    carto_stgo, len(escalas), col_wrap=2, height=7
)

for ax, (geodf_lisa, titulo) in zip(axes, escalas):
    graficar_lisa(geodf_lisa, ax, titulo, carto_stgo)

categorical_color_legend(axes[-1], COLORES_LISA, loc="lower right")

fig.suptitle(
    "LISA: Anscombe(viajes de empleo/estudio) a distintas escalas espaciales (MAUP)",
    fontsize=13,
)
fig.savefig(IMAGES / "04-lisa-comparacion-maup.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# Tabla resumen: Moran I global por escala

resumen = pd.DataFrame(
    [
        {
            "Escala": "Zonas EOD",
            "N unidades": len(geodf_zonas),
            "Moran I": moran_zonas.I,
            "p-value": moran_zonas.p_norm,
            "z-score": moran_zonas.z_norm,
        },
        {
            "Escala": "H3 nivel 7",
            "N unidades": len(geodf_h3_7),
            "Moran I": moran_h3_7.I,
            "p-value": moran_h3_7.p_norm,
            "z-score": moran_h3_7.z_norm,
        },
        {
            "Escala": "H3 nivel 8",
            "N unidades": len(geodf_h3_8),
            "Moran I": moran_h3_8.I,
            "p-value": moran_h3_8.p_norm,
            "z-score": moran_h3_8.z_norm,
        },
        {
            "Escala": "Comunas",
            "N unidades": len(geodf_comunas),
            "Moran I": moran_comunas.I,
            "p-value": moran_comunas.p_norm,
            "z-score": moran_comunas.z_norm,
        },
    ]
)

print(resumen.to_string(index=False, float_format="{:.4f}".format))

# %%
# Notas:
# - El I de Moran cambia con la escala: las unidades más pequeñas capturan variación local
#   que desaparece al agregar.
# - Con pocas unidades (comunas), los clusters LISA son menos estables y tienen menor poder estadístico.
# - Las grillas H3 tienen la ventaja de ser uniformes, pero los polígonos irregulares
#   (zonas EOD, comunas) reflejan mejor la estructura urbana y administrativa.
# - Este efecto se conoce como MAUP (Modifiable Areal Unit Problem): los resultados
#   del análisis espacial dependen de la definición arbitraria de las unidades de agregación.


# %%
# ========================================
# PARTE 9: LISA por propósito en grilla H3 nivel 8
# ========================================
# Las zonas EOD tienen tamaños heterogéneos: las zonas centrales son pequeñas
# y las periféricas grandes. Esto puede distorsionar los clusters LISA.
# Una grilla H3 uniforme permite comparar propósitos en igualdad de condiciones.

print("Construyendo grilla H3 nivel 8 para todos los propósitos...")
_bounds_h3 = [BBOX[0], BBOX[1], BBOX[2], BBOX[3]]
_gdf_hex_8 = h3_grid_from_bounds(
    _bounds_h3, extra_margin=0.025, grid_level=8, crs="EPSG:4326"
).reset_index()
print(f"  Celdas H3: {len(_gdf_hex_8)}")

_viajes_h3_8 = gpd.sjoin(
    viajes_destino[["geometry", "PropositoAgregado", "Peso"]],
    _gdf_hex_8[["h3_cell_id", "geometry"]],
    how="inner",
    predicate="within",
)

_anscombe_h3 = calcular_viajes_proposito(_viajes_h3_8, "h3_cell_id")

_total_h3 = _viajes_h3_8.groupby("h3_cell_id")["Peso"].sum()
_celdas_suf = _total_h3[_total_h3 >= 50].index
_geodf_h3_props = (
    _gdf_hex_8.set_index("h3_cell_id")
    .join(_anscombe_h3)
    .loc[_celdas_suf]
    .dropna(subset=PROPOSITOS)
    .reset_index()
)
print(f"  Celdas con >= 50 viajes: {len(_geodf_h3_props)}")

# %%
# LISA por propósito
lisas_h3_proposito = {}
for proposito in PROPOSITOS:
    lisa, moran = calcular_lisa(_geodf_h3_props, proposito)
    lisas_h3_proposito[proposito] = (lisa, moran)
    print(f"{proposito}: I={moran.I:.4f} (p={moran.p_norm:.4f})")

# %%
# Mapas LISA, grilla H3 nivel 8 con un panel por propósito

fig, axes = small_multiples_from_geodataframe(
    carto_stgo, len(PROPOSITOS), col_wrap=2, height=7
)

for ax, proposito in zip(axes, PROPOSITOS):
    lisa, moran = lisas_h3_proposito[proposito]
    graficar_lisa(lisa, ax, f"{proposito}  (I = {moran.I:.3f})", carto_stgo)

categorical_color_legend(axes[-1], COLORES_LISA, loc="lower right")

for ax in axes[len(PROPOSITOS) :]:
    ax.set_visible(False)

fig.suptitle(
    "LISA: Anscombe(viajes recibidos) por propósito en grilla H3 nivel 8 (destinos)"
)
fig.savefig(IMAGES / "04-lisa-h3-proposito.png", dpi=192, bbox_inches="tight")
plt.show()
