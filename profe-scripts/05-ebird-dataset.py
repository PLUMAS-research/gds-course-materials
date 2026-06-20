# %%
"""Construye dataset eBird Santiago agregado a hexágonos H3-8 para regresión.

Lee el EBD crudo de Chile (reciente release relMar-2026), filtra a bbox
Santiago + ventana de verano austral + protocolos comparables, agrega a
H3-8 con riqueza (total, nativa, exótica) y esfuerzo (checklists desde el
Sampling Event Dataset), pega NDVI + luminosidad nocturna + densidad
comunal, guarda el parquet dentro de `data/ebird-santiago-2024/` y lo
sube por scp al servidor del curso.

Inputs crudos (paso manual una vez)
-----------------------------------
Solicitar acceso al EBD en https://ebird.org/data/download. Descargar el
paquete "smp" para Chile (incluye EBD species + SED) y descomprimirlo. La
carpeta se configura como `EBD_DIR` en config_local.py (o env var GDS_EBD_DIR);
debe contener:
    - ebd_CL_smp_relMar-2026.txt (species, ~3.7 GB, 9.5M filas)
    - ebd_CL_smp_relMar-2026_sampling.txt (SED, ~244 MB, 950K checklists)

Por qué dos archivos
--------------------
El EBD tiene una fila por especie observada en cada checklist. Si una
checklist no reportó ninguna especie, no aparece en el EBD pero sí en el
SED. Contar checklists usando `SAMPLING EVENT IDENTIFIER` del EBD
subestima el esfuerzo y sesga la riqueza. El SED da el conteo correcto
de esfuerzo.

Salidas
-------
- `data/ebird-santiago-2024/h3-8-2024.parquet`: hexágonos con riqueza,
  esfuerzo, covariables.
- `data/ebird-santiago-2024.tgz`: empaquetado para subir al servidor.
- Subida por scp al servidor del curso (requiere ssh-key configurada).
"""

# %%
import sys
from pathlib import Path

# Permitir `import config` y `from gdsutils...` al ejecutar desde la raíz con
# `uv run python profe-scripts/<script>.py`.
_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from chiricoca.geo.grid import h3_grid_from_bounds

from gdsutils.geo import clip_geodataframe
from gdsutils.ndvi import (
    BBOX_SANTIAGO,
    descargar_ndvi_santiago,
    ndvi_por_zona,
)

# %% Configuración
DIR_EBD_CRUDO = config.requerir(config.EBD_DIR, "carpeta del EBD", "GDS_EBD_DIR")
RUTA_EBD = DIR_EBD_CRUDO / "ebd_CL_smp_relMar-2026.txt"
RUTA_SED = DIR_EBD_CRUDO / "ebd_CL_smp_relMar-2026_sampling.txt"

NOMBRE_DATASET = "ebird-santiago-2024"
DIR_SALIDA = Path("data") / NOMBRE_DATASET
RUTA_PARQUET = DIR_SALIDA / "h3-8-2024.parquet"

BBOX = BBOX_SANTIAGO
FECHA_INICIO = "2023-10-01"
FECHA_FIN = "2024-03-31"
MIN_CHECKLISTS = 2

RUTA_CARTO_COMUNAL = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
RUTA_MICRODATA_PERSONAS = Path("data") / "censo2024-microdatos" / "personas"

DIR_SALIDA.mkdir(parents=True, exist_ok=True)

# %%
# ========================================
# Paso 1.SED: esfuerzo (checklists) por hexágono
# ========================================
# Leer columnas mínimas con pyarrow, filtrar a Santiago + ventana + calidad.
print(f"[1/6] Leyendo SED {RUTA_SED}")
if not RUTA_SED.exists():
    raise FileNotFoundError(f"Falta el SED en {RUTA_SED}")

COLS_SED = [
    "LATITUDE",
    "LONGITUDE",
    "OBSERVATION DATE",
    "SAMPLING EVENT IDENTIFIER",
    "PROTOCOL NAME",
    "ALL SPECIES REPORTED",
]
sed = pd.read_csv(
    RUTA_SED,
    sep="\t",
    usecols=COLS_SED,
    dtype=str,
    engine="pyarrow",
)
print(f"  filas crudas: {len(sed):,}")

sed["LATITUDE"] = pd.to_numeric(sed["LATITUDE"], errors="coerce")
sed["LONGITUDE"] = pd.to_numeric(sed["LONGITUDE"], errors="coerce")
sed["OBSERVATION DATE"] = pd.to_datetime(sed["OBSERVATION DATE"], errors="coerce")

m_sed = (
    (sed["LONGITUDE"] >= BBOX[0])
    & (sed["LONGITUDE"] <= BBOX[2])
    & (sed["LATITUDE"] >= BBOX[1])
    & (sed["LATITUDE"] <= BBOX[3])
    & (sed["OBSERVATION DATE"] >= FECHA_INICIO)
    & (sed["OBSERVATION DATE"] <= FECHA_FIN)
    & (sed["PROTOCOL NAME"].isin(["Stationary", "Traveling"]))
    & (sed["ALL SPECIES REPORTED"] == "1")
)
sed = sed.loc[m_sed, ["LATITUDE", "LONGITUDE", "SAMPLING EVENT IDENTIFIER"]].copy()
print(f"  checklists en bbox+ventana+calidad: {len(sed):,}")

# %%
# ========================================
# Paso 2.EBD: riqueza (total, nativa, exótica) por checklist
# ========================================
# 9.5M filas. Con engine="pyarrow" y usecols cabe en memoria; si un día
# crece, migrar a pyarrow.csv.open_csv batched.
print(f"[2/6] Leyendo EBD {RUTA_EBD}")
if not RUTA_EBD.exists():
    raise FileNotFoundError(f"Falta el EBD en {RUTA_EBD}")

COLS_EBD = [
    "SCIENTIFIC NAME",
    "CATEGORY",
    "EXOTIC CODE",
    "LATITUDE",
    "LONGITUDE",
    "OBSERVATION DATE",
    "SAMPLING EVENT IDENTIFIER",
    "PROTOCOL NAME",
    "ALL SPECIES REPORTED",
    "APPROVED",
]
ebd = pd.read_csv(
    RUTA_EBD,
    sep="\t",
    usecols=COLS_EBD,
    dtype=str,
    engine="pyarrow",
)
print(f"  filas crudas: {len(ebd):,}")

ebd["LATITUDE"] = pd.to_numeric(ebd["LATITUDE"], errors="coerce")
ebd["LONGITUDE"] = pd.to_numeric(ebd["LONGITUDE"], errors="coerce")
ebd["OBSERVATION DATE"] = pd.to_datetime(ebd["OBSERVATION DATE"], errors="coerce")

m_ebd = (
    (ebd["LONGITUDE"] >= BBOX[0])
    & (ebd["LONGITUDE"] <= BBOX[2])
    & (ebd["LATITUDE"] >= BBOX[1])
    & (ebd["LATITUDE"] <= BBOX[3])
    & (ebd["OBSERVATION DATE"] >= FECHA_INICIO)
    & (ebd["OBSERVATION DATE"] <= FECHA_FIN)
    & (ebd["CATEGORY"] == "species")
    & (ebd["PROTOCOL NAME"].isin(["Stationary", "Traveling"]))
    & (ebd["ALL SPECIES REPORTED"] == "1")
    & (ebd["APPROVED"] == "1")
)
ebd = ebd.loc[m_ebd].copy()
print(f"  observaciones species en bbox+ventana+calidad: {len(ebd):,}")
print(f"  especies únicas totales: {ebd['SCIENTIFIC NAME'].nunique():,}")

# EXOTIC CODE: N (naturalized), P (provisional), X (introduced).
# Nativa = sin código exótico (NaN o vacío).
codigos_exoticos = {"N", "P", "X"}
ebd["es_exotica"] = ebd["EXOTIC CODE"].isin(codigos_exoticos)
print(f"  exóticas: {ebd.loc[ebd['es_exotica'], 'SCIENTIFIC NAME'].nunique()}")
print(f"  nativas:  {ebd.loc[~ebd['es_exotica'], 'SCIENTIFIC NAME'].nunique()}")

# Exportar observaciones filtradas (antes del sjoin) para visualización
# exploratoria. No se aplica MIN_CHECKLISTS acá: son todas las observaciones
# válidas, no solo las de hexes bien muestreados.
ruta_obs = DIR_SALIDA / "observaciones.parquet"
observaciones_export = ebd[[
    "SCIENTIFIC NAME", "LATITUDE", "LONGITUDE", "OBSERVATION DATE",
    "SAMPLING EVENT IDENTIFIER", "es_exotica",
]].rename(columns={
    "SCIENTIFIC NAME": "nombre_cientifico",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "OBSERVATION DATE": "fecha",
    "SAMPLING EVENT IDENTIFIER": "checklist_id",
})
observaciones_export.to_parquet(ruta_obs)
print(f"  Observaciones exportadas: {ruta_obs} "
      f"({len(observaciones_export):,} filas, "
      f"{ruta_obs.stat().st_size / 1024 / 1024:.1f} MB)")

# %%
# ========================================
# Paso 3.Spatial join SED y EBD a la grilla H3-8
# ========================================
print("[3/6] Grilla H3-8 y spatial join")
hexagonos = h3_grid_from_bounds(
    list(BBOX),
    extra_margin=0.025,
    grid_level=8,
    crs="EPSG:4326",
).reset_index()
print(f"  hexágonos en bbox: {len(hexagonos):,}")

puntos_sed = gpd.GeoDataFrame(
    sed,
    geometry=gpd.points_from_xy(sed["LONGITUDE"], sed["LATITUDE"]),
    crs="EPSG:4326",
)
sed_hex = gpd.sjoin(
    puntos_sed[["SAMPLING EVENT IDENTIFIER", "geometry"]],
    hexagonos[["h3_cell_id", "geometry"]],
    how="inner",
    predicate="within",
)
checklists_por_hex = (
    sed_hex.groupby("h3_cell_id")["SAMPLING EVENT IDENTIFIER"]
    .nunique()
    .rename("checklists")
)
print(f"  hexágonos con al menos 1 checklist: {len(checklists_por_hex):,}")

puntos_ebd = gpd.GeoDataFrame(
    ebd[["SCIENTIFIC NAME", "es_exotica", "LATITUDE", "LONGITUDE"]],
    geometry=gpd.points_from_xy(ebd["LONGITUDE"], ebd["LATITUDE"]),
    crs="EPSG:4326",
)
ebd_hex = gpd.sjoin(
    puntos_ebd[["SCIENTIFIC NAME", "es_exotica", "geometry"]],
    hexagonos[["h3_cell_id", "geometry"]],
    how="inner",
    predicate="within",
)
agr_ebd = ebd_hex.groupby("h3_cell_id").agg(
    riqueza=("SCIENTIFIC NAME", "nunique"),
    observaciones=("SCIENTIFIC NAME", "count"),
)
riqueza_exotica = (
    ebd_hex[ebd_hex["es_exotica"]]
    .groupby("h3_cell_id")["SCIENTIFIC NAME"]
    .nunique()
    .rename("riqueza_exotica")
)
agr_ebd = agr_ebd.join(riqueza_exotica, how="left")
agr_ebd["riqueza_exotica"] = agr_ebd["riqueza_exotica"].fillna(0).astype(int)
agr_ebd["riqueza_nativa"] = agr_ebd["riqueza"] - agr_ebd["riqueza_exotica"]

# Diversidad de Shannon por hex desde frecuencias de observaciones.
# Cada fila de ebd_hex es una observación (especie x checklist). n_i es el
# conteo por especie i en el hex, N la suma total de observaciones del hex,
# p_i = n_i / N la frecuencia relativa. H = -sum(p_i log p_i). Es Shannon
# basado en ocurrencias, no en abundancia absoluta (que requeriría OBS
# COUNT, que puede venir como "X" no cuantificado).
conteo_especie_hex = ebd_hex.groupby(["h3_cell_id", "SCIENTIFIC NAME"]).size()
total_por_hex = conteo_especie_hex.groupby(level=0).sum()
p = conteo_especie_hex.div(total_por_hex, level=0)
shannon = (-p * np.log(p)).groupby(level=0).sum().rename("shannon")
agr_ebd = agr_ebd.join(shannon, how="left")

hex_geo = hexagonos.merge(
    checklists_por_hex.reset_index(), on="h3_cell_id", how="inner"
).merge(agr_ebd.reset_index(), on="h3_cell_id", how="left")
# Hex con checklists pero sin especies reportadas (checklists vacías):
# riqueza = 0, no NaN.
for col in ["riqueza", "observaciones", "riqueza_exotica", "riqueza_nativa"]:
    hex_geo[col] = hex_geo[col].fillna(0).astype(int)
hex_geo["shannon"] = hex_geo["shannon"].fillna(0.0)

hex_geo = hex_geo[hex_geo["checklists"] >= MIN_CHECKLISTS].reset_index(drop=True)
print(f"  hexágonos con >= {MIN_CHECKLISTS} checklists: {len(hex_geo)}")

# Pielou = H / log(S): evenness en [0, 1]. Indefinido para S <= 1.
hex_geo["pielou"] = np.where(
    hex_geo["riqueza"] > 1,
    hex_geo["shannon"] / np.log(hex_geo["riqueza"].clip(lower=2)),
    np.nan,
)
print(
    hex_geo[
        ["checklists", "riqueza", "riqueza_nativa", "riqueza_exotica",
         "observaciones", "shannon", "pielou"]
    ]
    .describe()
    .round(2)
)

# Transformación Anscombe para estabilizar varianza Poisson (consistente
# con clases 04 y 05-sosafe).
for col in ["riqueza", "riqueza_nativa", "riqueza_exotica"]:
    hex_geo[f"sqrt_{col}"] = np.sqrt(hex_geo[col] + 3 / 8)

# %%
# ========================================
# Paso 4.NDVI y luminosidad nocturna por hexágono
# ========================================
print("[4/6] NDVI y luminosidad por hexágono")
ruta_ndvi = descargar_ndvi_santiago(
    bbox=BBOX,
    fecha_inicio=FECHA_INICIO,
    fecha_fin=FECHA_FIN,
)
hex_geo = ndvi_por_zona(hex_geo, ruta_ndvi, columna="ndvi", estadistico="median")
print(
    f"  NDVI: min={hex_geo['ndvi'].min():.2f} "
    f"max={hex_geo['ndvi'].max():.2f} "
    f"median={hex_geo['ndvi'].median():.2f}"
)

# Luminosidad la genera `profe-scripts/05-preparar-luminosidad.py` desde
# NASA Earthdata. No usamos `descargar_luminosidad_santiago()` acá porque
# esa función baja del servidor; en el profe-script asumimos generación
# local + upload posterior.
ruta_lum = Path("data") / "luminosidad-santiago-2023.tif"
if not ruta_lum.exists():
    raise FileNotFoundError(
        f"Falta {ruta_lum}. Generalo una vez con:\n"
        "  uv run python profe-scripts/05-preparar-luminosidad.py"
    )
hex_geo = ndvi_por_zona(hex_geo, ruta_lum, columna="luminosidad", estadistico="mean")
print(
    f"  Luminosidad: min={hex_geo['luminosidad'].min():.2f} "
    f"max={hex_geo['luminosidad'].max():.2f} "
    f"median={hex_geo['luminosidad'].median():.2f}"
)

# %%
# ========================================
# Paso 5.Densidad poblacional comunal (Censo 2024)
# ========================================
print("[5/6] Densidad poblacional por comuna")
if not RUTA_CARTO_COMUNAL.exists() or not RUTA_MICRODATA_PERSONAS.exists():
    raise FileNotFoundError(
        "Faltan insumos censales. Corre antes en la raíz:\n"
        '  uv run python -c "from gdsutils.general import descargar_datos;'
        " descargar_datos('https://dcc.uchile.cl/~egraells/gds-data/"
        "censo2024-cartografia.tgz'); descargar_datos('https://dcc.uchile.cl/"
        "~egraells/gds-data/censo2024-microdatos.tgz')\""
    )

personas_rm = dd.read_parquet(
    RUTA_MICRODATA_PERSONAS, filters=[("region", "=", 13)], columns=["comuna"]
)
poblacion = (
    personas_rm.groupby("comuna").size().compute().rename("poblacion").reset_index()
)
poblacion["comuna"] = poblacion["comuna"].astype(int)

comunas_rm = gpd.read_parquet(
    RUTA_CARTO_COMUNAL, filters=[("COD_REGION", "=", 13)]
).to_crs("EPSG:4326")
comunas_stgo = clip_geodataframe(comunas_rm, BBOX)
comunas_stgo["CUT"] = comunas_stgo["CUT"].astype(int)
comunas_stgo["area_km2"] = comunas_stgo.to_crs("EPSG:32719").area / 1e6
comunas_stgo = comunas_stgo.merge(
    poblacion, left_on="CUT", right_on="comuna", how="left"
)
comunas_stgo["densidad"] = comunas_stgo["poblacion"] / comunas_stgo["area_km2"]
print(f"  comunas: {len(comunas_stgo)}")

centroides = hex_geo.copy()
centroides["geometry"] = centroides.geometry.centroid
hex_comuna = gpd.sjoin(
    centroides[["h3_cell_id", "geometry"]],
    comunas_stgo[["COMUNA", "densidad", "geometry"]],
    how="left",
    predicate="within",
).drop(columns="index_right")

hex_geo = hex_geo.merge(
    hex_comuna[["h3_cell_id", "COMUNA", "densidad"]], on="h3_cell_id", how="left"
)

# Dejar solo hex con todas las covariables disponibles
n_antes = len(hex_geo)
hex_geo = hex_geo.dropna(subset=["ndvi", "luminosidad", "densidad"]).reset_index(
    drop=True
)
print(f"  hex con todas las covariables: {n_antes} → {len(hex_geo)}")
print(
    hex_geo[["sqrt_riqueza", "ndvi", "luminosidad", "densidad", "checklists", "COMUNA"]]
    .describe(include="all")
    .iloc[:, :-1]
    .round(2)
)

# %%
# ========================================
# Paso 6.Guardar, empaquetar y subir
# ========================================
print(f"[6/6] Guardando dataset")
hex_geo.to_parquet(RUTA_PARQUET)
print(
    f"  Parquet: {RUTA_PARQUET} ({len(hex_geo)} filas, {len(hex_geo.columns)} columnas)"
)

# Junto al .tgz se suben los rasters NDVI y luminosidad, que luego se bajan
# sin Planetary Computer ni Earthdata vía `descargar_ndvi_santiago_precomputado`
# y `descargar_luminosidad_santiago`.
config.publicar(NOMBRE_DATASET, DIR_SALIDA, extras=(ruta_ndvi, ruta_lum))
