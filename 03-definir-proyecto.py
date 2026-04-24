# %%
#
# Datos de Directorio de Transporte Público Metropolitano
# Viajes: https://www.dtpm.cl/index.php/documentos/matrices-de-viaje
# Paraderos: https://www.dtpm.cl/index.php/programa-de-operacion
#
# Situación: Las personas usan transporte público en la ciudad.
# Complicación: Las intervenciones que cambian las rutas de transporte público tienen efectos que no se cuantifican.
# Propuesta: Medir el efecto de una intervención que alteró la red de transporte público (Paseo Bandera).
#
# %%

import os
from pathlib import Path

import dask
import dask.dataframe as dd

# Configuramos dask para que pueda trabajar con datos DTPM
# Evita que NumPy/Pandas usen todos los cores por su cuenta
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

dask.config.set(
    {
        # Gestión de Memoria (Relativa al sistema)
        "distributed.worker.memory.target": 0.60,  # Comienza a volcar a disco (spill)
        "distributed.worker.memory.spill": 0.70,  # Presiona el volcado a disco
        "distributed.worker.memory.pause": 0.80,  # Pausa la ejecución de nuevas tareas
        "distributed.worker.memory.terminate": 0.95,  # Reinicia el worker si llega aquí
        "distributed.worker.nthreads": 4,  # Hilos por cada proceso worker
        "distributed.worker.nprocs": None,  # 'None' deja que Dask decida procesos según el sistema
    }
)

# %%

from gdsutils.dtpm import (
    DIR_CONSOLIDADO,
    MAPEO_DTPM,
    RUTA_PARADEROS_PARQUET,
    consolidar_anio,
    descargar_viajes,
    procesar_paraderos,
    procesar_viajes,
)

ANIOS = list(range(2014, 2026))

# Directorio para archivos temporales de extracción.
# Cambiar si /tmp no tiene suficiente espacio (ej. tmpfs pequeño).
DIR_TEMP = "./tmp"  # None = default del sistema (/tmp)

for anio, urls in MAPEO_DTPM.items():
    if (DIR_CONSOLIDADO / f"dtpm-{anio}.parquet").exists():
        continue
    try:
        descargar_viajes(anio, urls)
    except Exception as e:
        print(f"ERROR DE DESCARGA en año {anio}: {e}")

for anio, urls in MAPEO_DTPM.items():
    if (DIR_CONSOLIDADO / f"dtpm-{anio}.parquet").exists():
        continue
    try:
        procesar_viajes(anio, urls, dir_temp=DIR_TEMP)
    except Exception as e:
        print(f"ERROR DE PROCESAMIENTO en año {anio}: {e}")

procesar_paraderos()

for anio in ANIOS:
    try:
        consolidar_anio(anio)
    except Exception as e:
        print(f"ERROR DE CONSOLIDACIÓN en {anio}: {e}")

# %%
# Análisis básico de los datos de viajes DTPM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chiricoca.config import setup_style
from chiricoca.maps import bubble_map

setup_style(dpi=96)

PROPOSITOS_VALIDOS = ["TRABAJO", "ESTUDIO", "HOGAR", "OTROS"]
ANIO_MAPA = ANIOS[-1]

# %%
# Cómputo de agregados por año

total_anual = {}  # anio -> total de viajes en el período (depende de cuántos días cubre)
promedio_diario = {}  # anio -> promedio diario (comparable entre años)
diasemana_anual = {}  # anio -> Series con viajes promedio por día de semana (0=lun..6=dom)
prop_anual = {}  # anio -> Series con % por propósito (excluye sin bajada)
horas_anual = {}  # anio -> Series con % de viajes por hora del día
dist_prop_anual = {}  # anio -> DataFrame(proposito x dia_semana) con distancia media (km)
origen_ref = None  # paradero -> viajes matinales expandidos (ANIO_MAPA)
destino_ref = None  # paradero_destino -> viajes matinales expandidos (ANIO_MAPA)

for anio in ANIOS:
    ruta = DIR_CONSOLIDADO / f"dtpm-{anio}.parquet"
    if not ruta.exists():
        print(f"OMITIDO: {anio}")
        continue
    print(f"PROCESANDO: {anio} ...", end=" ", flush=True)

    df = dd.read_parquet(ruta)

    # 1. Total y promedio diario
    por_fecha = (
        df.groupby(["fecha", "dia_semana"])["factor"].sum().compute().reset_index()
    )
    total_anual[anio] = por_fecha["factor"].sum()
    promedio_diario[anio] = por_fecha.groupby("fecha")["factor"].sum().mean()
    diasemana_anual[anio] = por_fecha.groupby("dia_semana")["factor"].mean()

    # 2. Propósitos (excluye sin bajada)
    prop = (
        df[df["proposito"].isin(PROPOSITOS_VALIDOS)]
        .groupby("proposito")["factor"]
        .sum()
        .compute()
    )
    prop_anual[anio] = prop / prop.sum() * 100

    # 3. Distribución horaria
    horas = (
        df[df["proposito"] != "SINBAJADA"]
        .groupby("hora_inicio")["factor"]
        .sum()
        .compute()
    )
    horas_anual[anio] = horas / horas.sum() * 100

    # 4. Distancia media ponderada por propósito y día de semana
    if "dist_ruta_mts" in df.columns:
        df_d = df[
            df["dist_ruta_mts"].notnull()
            & (df["dist_ruta_mts"] > 0)
            & df["proposito"].isin(PROPOSITOS_VALIDOS)
        ]
        df_d = df_d.assign(dist_w=df_d["dist_ruta_mts"] * df_d["factor"])
        peso = df_d.groupby(["proposito", "dia_semana"])["dist_w"].sum().compute()
        norm = df_d.groupby(["proposito", "dia_semana"])["factor"].sum().compute()
        ratio = (peso / norm / 1000).unstack("dia_semana")
        if not ratio.empty:
            dist_prop_anual[anio] = ratio

    # 5. Origen y destino matinal para el mapa
    if anio == ANIO_MAPA:
        df_manana = df[
            (df["hora_inicio"] >= 6)
            & (df["hora_inicio"] <= 10)
            & (df["dia_semana"] < 5)
            & (df["proposito"] != "SINBAJADA")
        ]
        origen_ref = df_manana.groupby("paradero")["factor"].sum().compute()
        destino_ref = (
            df_manana[df_manana["paradero_destino"].notnull()]
            .groupby("paradero_destino")["factor"]
            .sum()
            .compute()
        )

    print(f"{total_anual[anio] / 1e6:.1f}M viajes")

print("LISTO.")


# %%
# Gráfico 1: viajes diarios promedio por año + desglose por día de semana

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
colores_dia = {
    0: "#abacab",
    1: "#abacab",
    2: "#abacab",
    3: "#abacab",
    4: "#abacab",
    5: "#e07b54",
    6: "#6aab6e",
}

anios_disp = sorted(total_anual.keys())
ncols_sm = 3
nrows_sm = (len(anios_disp) + ncols_sm - 1) // ncols_sm

fig = plt.figure(figsize=(15, max(6, nrows_sm * 2)))
gs = fig.add_gridspec(nrows_sm, ncols_sm + 2, hspace=0.05, wspace=0.05)
ax_main = fig.add_subplot(gs[:, :2])

df_prom = pd.Series(promedio_diario).sort_index()
ax_main.bar(
    df_prom.index.astype(str),
    df_prom.values / 1e6,
    color="#abacab",
    width=0.9,
    edgecolor="none",
)
ax_main.set_xlabel("")
ax_main.set_ylabel("Viajes promedio por día (millones)")
ax_main.set_title("Viajes diarios promedio por año")
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))

for idx, anio in enumerate(anios_disp):
    ax = fig.add_subplot(gs[idx // ncols_sm, idx % ncols_sm + 2])
    data = diasemana_anual.get(anio, pd.Series(dtype=float))
    vals = [(data[d] / 1e6) if d in data.index else 0 for d in range(7)]
    ax.bar(
        DIAS,
        vals,
        color=[colores_dia[d] for d in range(7)],
        width=0.9,
        edgecolor="none",
    )
    ax.set_title(str(anio), fontsize=8, pad=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.tick_params(labelsize=7)

fig.suptitle("Viajes promedio por día y distribución por día de semana", fontsize=11)
fig.savefig(Path("images") / "03-viajes-totales.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# Gráfico 2: distribución de propósitos por año (stacked bar)

df_prop = pd.DataFrame(prop_anual).T.fillna(0)
colores = {
    "TRABAJO": "#e07b54",
    "ESTUDIO": "#5b8db8",
    "HOGAR": "#6aab6e",
    "OTROS": "#b0b0b0",
}
cols_orden = [
    c for c in ["TRABAJO", "ESTUDIO", "HOGAR", "OTROS"] if c in df_prop.columns
]
df_prop = df_prop[cols_orden]

fig, ax = plt.subplots(figsize=(10, 4))
base = np.zeros(len(df_prop))
for col in cols_orden:
    ax.bar(
        df_prop.index.astype(str),
        df_prop[col],
        bottom=base,
        label=col.capitalize(),
        color=colores[col],
        width=1.0,
        edgecolor="none",
    )
    base += df_prop[col].values

ax.set_xlabel("")
ax.set_ylabel("Porcentaje de viajes (%)")
ax.set_title("Propósito de viajes por año (excluye sin bajada)")
ax.legend(loc="upper right")
ax.set_ylim(0, 100)
ax.set_xlim(0 - 0.5, len(ANIOS) - 0.5)

fig.savefig(Path("images") / "03-propositos.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# Gráfico 3: distribución horaria como small multiples con todos los años como contexto

df_horas = pd.DataFrame(horas_anual).fillna(0).sort_index()
anios_disp = sorted(horas_anual.keys())
cmap = plt.get_cmap("plasma")
n = len(anios_disp)
colores_anio = {a: cmap(i / max(n - 1, 1)) for i, a in enumerate(anios_disp)}

ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(
    nrows, ncols, figsize=(14, nrows * 2.5), sharey=True, sharex=True
)
axes_flat = axes.flatten()

for i, anio in enumerate(anios_disp):
    ax = axes_flat[i]
    for otro in anios_disp:
        if otro != anio and otro in df_horas.columns:
            ax.plot(
                df_horas.index, df_horas[otro], color="#cccccc", linewidth=0.8, zorder=1
            )
    ax.plot(
        df_horas.index, df_horas[anio], color=colores_anio[anio], linewidth=2, zorder=2
    )
    ax.set_title(str(anio), fontsize=9)
    ax.set_xticks(range(0, 25, 6))
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.supxlabel("Hora del día", fontsize=9)
fig.supylabel("% de viajes", fontsize=9)
fig.suptitle("Distribución horaria de viajes con cada año en contexto", fontsize=11)
fig.savefig(Path("images") / "03-hora-de-viaje.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# Gráfico 4: heatmap de distancia media de viaje por propósito, año y día de semana

import seaborn as sns

anios_dist = sorted(dist_prop_anual.keys())
n_anios = len(anios_dist)

hm_data = {}
for prop in PROPOSITOS_VALIDOS:
    filas = {}
    for anio in anios_dist:
        df_p = dist_prop_anual[anio]
        if prop in df_p.index:
            filas[anio] = df_p.loc[prop].rename(lambda d: DIAS[d])
    if filas:
        hm_data[prop] = pd.DataFrame(filas).T.reindex(columns=DIAS)

paneles = [p for p in PROPOSITOS_VALIDOS if p in hm_data]
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, max(6, n_anios * 0.55 + 2)),
    gridspec_kw={"hspace": 0.15, "wspace": 0.15},
)
axes_flat = axes.flatten()

for ax, prop in zip(axes_flat, paneles):
    sns.heatmap(
        hm_data[prop],
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        cbar_kws={"label": "km", "shrink": 0.7},
    )
    ax.set_title(prop.capitalize(), fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)
    ax.tick_params(axis="y", rotation=0)

for ax in axes_flat[len(paneles) :]:
    ax.set_visible(False)

fig.suptitle(
    "Distancia media de viaje (km) por propósito, año y día de semana", fontsize=11
)
fig.savefig(Path("images") / "03-distancias.png", dpi=192, bbox_inches="tight")
plt.show()


# %%
# Gráfico 5: origen y destino de viajes matinales (06:00–10:30), sin basemap

import geopandas as gpd
import numpy as np
from chiricoca.geo.figures import small_multiples_from_geodataframe

from gdsutils.censo2024 import regiones
from gdsutils.geo import clip_geodataframe

RUTA_CARTO = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)

paraderos = gpd.read_parquet(RUTA_PARADEROS_PARQUET).to_crs("EPSG:4326")
bbox = regiones["RM"]["capital_bbox"]  # [xmin, ymin, xmax, ymax]

carto_rm = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(
    "EPSG:4326"
)
carto_stgo = clip_geodataframe(carto_rm, bbox)


def preparar_mapa(trips_series, paraderos, bbox):
    """Une conteos por paradero con su geometría y recorta al bbox."""
    gdf = paraderos.set_index("Código paradero TS").join(
        trips_series.rename("viajes"), how="inner"
    )
    x, y = gdf.geometry.x, gdf.geometry.y
    mask = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
    return gdf[mask].copy()


origen_rm = preparar_mapa(origen_ref, paraderos, bbox)
destino_rm = preparar_mapa(destino_ref, paraderos, bbox)
print(f"Paraderos origen: {len(origen_rm):,} | destino: {len(destino_rm):,}")

panel_h = 6
fig, (ax1, ax2) = small_multiples_from_geodataframe(carto_stgo, 2, panel_h)

max_viajes = max(origen_rm["viajes"].max(), destino_rm["viajes"].max())
scale = 300 / max_viajes

for ax in (ax1, ax2):
    carto_stgo.plot(ax=ax, facecolor="#e8e8e8", edgecolor="white", linewidth=0.4)

bubble_map(
    origen_rm,
    size="viajes",
    scale=scale,
    alpha=0.6,
    color="#5b8db8",
    edgecolor="none",
    add_legend=True,
    ax=ax1,
)
bubble_map(
    destino_rm,
    size="viajes",
    scale=scale,
    alpha=0.6,
    color="#e07b54",
    edgecolor="none",
    add_legend=True,
    ax=ax2,
)

for ax, titulo in [(ax1, "Paradero de subida"), (ax2, "Paradero de bajada")]:
    ax.set_title(titulo)

fig.suptitle(f"Origen y destino de viajes matinales (06:00 a 10:30) en DTPM {ANIO_MAPA}")
fig.savefig(Path("images") / "03-burbujas-od.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# Notas:
# - Los primeros años (2014--2016) tienen patrones diferentes. ¡Probablemente se aplicó un procesamiento u algoritmo diferente!
# - Hay datos consistentes en el resto de los años, aunque hay dimensiones donde hay que tener cuidado (propósitos de viaje, patrones por día específico).
# - Hay viajes en toda la ciudad, por lo que disponemos de información suficiente para medir el efecto de la intervención que nos interesa.
# - Hacerlo entregará evidencia a quienes planifican el transporte y diseñan la ciudad, impactando la calidad de vida de las personas en Santiago.
