# %%
"""Construye dataset SOSAFE 2024 agregado a hexágonos H3-8 para regresión.

Salidas en `data/sosafe/`:
- `reportes-2024.parquet`: puntos limpios con `categoria` y `grupo`.
- `h3-8-censo.parquet`: copia del grid H3-8 con variables censo 2024.
- `h3-8-2024.parquet`: hexágonos con conteos por grupo (Ambiental, Disturbios,
  Delitos), Anscombe, densidad poblacional, migración, educación, NDVI,
  luminosidad nocturna (VIIRS Black Marble) y densidad de paraderos DTPM.

Categorías propias para esta clase (ver `gdsutils/sosafe.py`): se conservan
tipos ambientales (basura, luminaria, árboles) que el pipeline original del
paper descarta.
"""

# %%
import shutil
import sys
from pathlib import Path

# Permitir `import config` y `from gdsutils...` al ejecutar desde la raíz con
# `uv run python profe-scripts/<script>.py`.
_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import geopandas as gpd
import numpy as np
import pandas as pd

from gdsutils.censo2024 import regiones
from gdsutils.dtpm import RUTA_PARADEROS_PARQUET
from gdsutils.ndvi import (
    BBOX_SANTIAGO,
    descargar_luminosidad_santiago,
    descargar_ndvi_santiago,
    ndvi_por_zona,
)
from gdsutils.sosafe import cargar_reportes, filtrar_relevantes

# %% Configuración
SOSAFE_RAW = config.requerir(config.SOSAFE_RAW, "reportes SOSAFE crudos", "GDS_SOSAFE_RAW")
H3_GRID_ORIGEN = config.requerir(
    config.SOSAFE_H3_GRID, "grid H3-8 censal", "GDS_SOSAFE_H3_GRID"
)
DIR_SALIDA = Path("data") / "sosafe"
DIR_SALIDA.mkdir(exist_ok=True, parents=True)
ANIO = 2024
BBOX_RM = tuple(regiones["RM"]["capital_bbox"])
LOTE_DIAS = 30

# %% Paso 1 — reportes SOSAFE crudos, limpios y categorizados
ruta_reportes = DIR_SALIDA / f"reportes-{ANIO}.parquet"
if ruta_reportes.exists():
    print(f"[1/6] Cache: {ruta_reportes}")
    reportes = gpd.read_parquet(ruta_reportes)
else:
    archivos = sorted(SOSAFE_RAW.glob(f"{ANIO}-*.json"))
    print(f"[1/6] Leyendo {len(archivos)} archivos JSON de {ANIO}")
    partes = []
    n_lotes = (len(archivos) + LOTE_DIAS - 1) // LOTE_DIAS
    for k, i in enumerate(range(0, len(archivos), LOTE_DIAS), start=1):
        print(f"  Lote {k}/{n_lotes}: días {i+1}-{min(i+LOTE_DIAS, len(archivos))}")
        parte = cargar_reportes(archivos[i : i + LOTE_DIAS], bbox=BBOX_RM)
        partes.append(filtrar_relevantes(parte))
    reportes = gpd.GeoDataFrame(
        pd.concat(partes, ignore_index=True), crs=partes[0].crs
    )
    reportes.to_parquet(ruta_reportes)
    print(f"  Guardado: {ruta_reportes}")

print(f"  Reportes relevantes: {len(reportes):,}")
print(reportes["grupo"].value_counts())
print(reportes["categoria"].value_counts().head(10))

# %% Paso 2 — grid H3-8 con covariables censales
ruta_h3 = DIR_SALIDA / "h3-8-censo.parquet"
if not ruta_h3.exists():
    shutil.copy(H3_GRID_ORIGEN, ruta_h3)
    print(f"[2/6] Grid H3-8 copiada: {ruta_h3}")
else:
    print(f"[2/6] Cache: {ruta_h3}")
h3 = gpd.read_parquet(ruta_h3)
print(f"  Hexágonos: {len(h3)}")

# %% Paso 3 — indicadores censales derivados por hex
pred = h3[["geometry"]].copy()
pred["poblacion"] = h3["poblacion_total"].fillna(0)

h3_utm = h3.to_crs(32719)
pred["area_km2"] = h3_utm.geometry.area / 1e6
pred["densidad_hab_km2"] = np.where(
    pred["area_km2"] > 0, pred["poblacion"] / pred["area_km2"], 0.0
)

cols_mig = [c for c in h3.columns if c.startswith("migracion_")]
migrantes = h3[cols_mig].fillna(0).sum(axis=1)
pred["n_migrantes"] = migrantes
pred["frac_migrantes"] = np.where(
    pred["poblacion"] > 0, migrantes / pred["poblacion"], 0.0
)

cols_reciente = [
    "llegada_entre_2017_y_2019",
    "llegada_entre_2020_y_2022",
    "llegada_entre_2023_y_2024",
]
llegada_reciente = h3[cols_reciente].fillna(0).sum(axis=1)
# Denominador: total con año de llegada conocido (no usar `migrantes` por
# país de origen, ya que la persona con doble nacionalidad reporta llegada
# pero no aparece en `migracion_*`).
cols_llegada_conocida = [
    c for c in h3.columns
    if c.startswith("llegada_") and c != "llegada_no_respuesta"
]
llegada_total = h3[cols_llegada_conocida].fillna(0).sum(axis=1)
pred["frac_migracion_reciente"] = np.where(
    llegada_total > 0, llegada_reciente / llegada_total, 0.0
)

edu_superior = (
    h3["educacion_tecnica"].fillna(0)
    + h3["educacion_universitaria/posgrado"].fillna(0)
)
pred["frac_edu_superior"] = np.where(
    pred["poblacion"] > 0, edu_superior / pred["poblacion"], 0.0
)

print(f"[3/6] Indicadores censales listos")
print(
    pred[
        [
            "poblacion",
            "densidad_hab_km2",
            "frac_migrantes",
            "frac_migracion_reciente",
            "frac_edu_superior",
        ]
    ]
    .describe()
    .round(2)
)

# %% Paso 4 — spatial join reportes → hex, conteos por grupo
print(f"[4/6] Spatial join reportes → hexágonos")
reportes_m = reportes.to_crs(pred.crs)
pred_idx = pred.reset_index()
r_hex = gpd.sjoin(
    reportes_m[["geometry", "grupo"]],
    pred_idx[["geometry", "h3_cell_id"]],
    how="left",
    predicate="within",
)
conteos = (
    r_hex.dropna(subset=["h3_cell_id"])
    .groupby(["h3_cell_id", "grupo"])
    .size()
    .unstack(fill_value=0)
)
conteos.columns = [f"n_{c.lower()}" for c in conteos.columns]
conteos["n_total"] = conteos.sum(axis=1)
pred = pred.join(conteos, how="left")
for col in conteos.columns:
    pred[col] = pred[col].fillna(0).astype(int)
# Anscombe sqrt(x + 3/8), consistente con la clase 04
for col in conteos.columns:
    pred[f"sqrt_{col}"] = np.sqrt(pred[col] + 3 / 8)

print(f"  Totales por grupo:")
print(pred[list(conteos.columns)].sum())

# %% Paso 5 — NDVI mediano por hex
print(f"[5/7] NDVI mediano por hexágono")
ruta_ndvi = Path("data") / "ndvi-santiago-2023.tif"
if not ruta_ndvi.exists():
    descargar_ndvi_santiago(
        bbox=BBOX_SANTIAGO,
        fecha_inicio="2023-10-01",
        fecha_fin="2024-03-31",
        ruta_salida=ruta_ndvi,
    )
pred = ndvi_por_zona(pred, ruta_ndvi, columna="ndvi", estadistico="median")
print(f"  NDVI: min={pred['ndvi'].min():.2f} max={pred['ndvi'].max():.2f} "
      f"median={pred['ndvi'].median():.2f}")

# %% Paso 6 — Luminosidad nocturna (VIIRS Black Marble anual 2023)
# Usamos la media zonal porque la radiancia nocturna se interpreta
# tradicionalmente como "sum of lights": mean × área ≈ radiancia total.
print(f"[6/7] Luminosidad nocturna por hexágono")
ruta_lum = descargar_luminosidad_santiago()
pred = ndvi_por_zona(pred, ruta_lum, columna="luminosidad", estadistico="mean")
print(
    f"  Luminosidad: min={pred['luminosidad'].min():.2f} "
    f"max={pred['luminosidad'].max():.2f} "
    f"median={pred['luminosidad'].median():.2f}"
)

# %% Paso 7 — densidad de paraderos DTPM
print(f"[7/7] Paraderos DTPM por hexágono")
paraderos = gpd.read_parquet(RUTA_PARADEROS_PARQUET).to_crs(pred.crs)
pred_idx = pred.reset_index()
p_hex = gpd.sjoin(
    paraderos[["geometry"]],
    pred_idx[["geometry", "h3_cell_id"]],
    how="left",
    predicate="within",
)
conteo_par = p_hex.dropna(subset=["h3_cell_id"]).groupby("h3_cell_id").size()
pred["n_paraderos"] = pred.index.map(conteo_par).fillna(0).astype(int)
pred["densidad_paraderos"] = np.where(
    pred["area_km2"] > 0, pred["n_paraderos"] / pred["area_km2"], 0.0
)
print(f"  Total paraderos geolocalizados en hex: {pred['n_paraderos'].sum():,}")

# %% Guardar dataset final
ruta_final = DIR_SALIDA / f"h3-8-{ANIO}.parquet"
pred.to_parquet(ruta_final)
print(f"\nDataset guardado: {ruta_final}")
print(f"Filas: {len(pred)}, columnas: {len(pred.columns)}")
print(f"\nResumen numérico:")
print(pred.drop(columns="geometry").describe().round(2).T[["mean", "std", "min", "max"]])
