# %%
"""Subset Santiago del dataset global de check-ins de Foursquare.

Fuente: Dingqi Yang, Bingqing Qu, Jie Yang, Philippe Cudre-Mauroux,
"Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph
Embedding Approach", WWW'19. Check-ins globales de abril 2012 a enero 2014.
Citar a los autores en cualquier material derivado.

Lee los parquet crudos (CHECKINS y POIS), filtra a venues de Chile dentro del
bbox del Gran Santiago, calcula la hora local (UTC + offset) y deja un dataset
manejable: check-ins (usuario, venue, fecha local) y venues (coordenadas,
categoría). El análisis de unicidad (clase 14) corre sobre esto.

Salidas en `data/foursquare-santiago/`:
- `checkins.parquet`: (user_id, venue_id, datetime_local).
- `venues.parquet`: (venue_id, lat, lon, category).
"""

# %%
import glob
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import pandas as pd
import pyarrow.parquet as pq

# %% Configuración
NOMBRE_DATASET = "foursquare-santiago"
DIR_SALIDA = Path("data") / NOMBRE_DATASET

DIR_4SQ = config.requerir(
    config.FOURSQUARE_DIR, "dataset Foursquare (CHECKINS/ y POIS/)", "GDS_FOURSQUARE_DIR"
)
DIR_CHECKINS = DIR_4SQ / "CHECKINS"
DIR_POIS = DIR_4SQ / "POIS"

BBOX = (-70.85, -33.65, -70.45, -33.30)  # Gran Santiago (lon_min, lat_min, lon_max, lat_max)
FORMATO_FECHA = "%a %b %d %H:%M:%S %z %Y"  # "Tue Apr 03 18:00:22 +0000 2012"

DIR_SALIDA.mkdir(parents=True, exist_ok=True)

# %%
# ========================================
# Paso 1. Venues de Santiago (país CL + bbox)
# ========================================
print("[1/3] Filtrando venues a Santiago")
venues = []
for f in sorted(glob.glob(str(DIR_POIS / "part*.parquet"))):
    t = pq.read_table(f, columns=["venue_id", "lat", "lon", "category", "country"])
    venues.append(t.to_pandas())
venues = pd.concat(venues, ignore_index=True)
print(f"  venues globales: {len(venues):,}")

venues = venues[venues["country"] == "CL"]
venues = venues[
    venues["lon"].between(BBOX[0], BBOX[2]) & venues["lat"].between(BBOX[1], BBOX[3])
].copy()
venues = venues.drop(columns="country").drop_duplicates("venue_id").reset_index(drop=True)
venue_ids = set(venues["venue_id"])
print(f"  venues en Santiago: {len(venues):,}")

# %%
# ========================================
# Paso 2. Check-ins de esos venues, con hora local
# ========================================
print("[2/3] Filtrando check-ins de Santiago")
partes = []
archivos = sorted(glob.glob(str(DIR_CHECKINS / "part*.parquet")))
for i, f in enumerate(archivos, 1):
    t = pq.read_table(f, columns=["user_id", "venue_id", "datetime", "utc_offset"])
    df = t.to_pandas()
    df = df[df["venue_id"].isin(venue_ids)]
    if len(df):
        partes.append(df)
    if i % 20 == 0 or i == len(archivos):
        print(f"  {i}/{len(archivos)} partes")
checkins = pd.concat(partes, ignore_index=True)
print(f"  check-ins en Santiago: {len(checkins):,}")

# Hora local = UTC + offset (en minutos). datetime cruda trae tz +0000 (UTC).
utc = pd.to_datetime(checkins["datetime"], format=FORMATO_FECHA, utc=True)
checkins["datetime_local"] = (
    utc + pd.to_timedelta(checkins["utc_offset"], unit="m")
).dt.tz_localize(None)
checkins = checkins[["user_id", "venue_id", "datetime_local"]].dropna()
print(f"  rango local: {checkins['datetime_local'].min()} a "
      f"{checkins['datetime_local'].max()}")
print(f"  usuarios: {checkins['user_id'].nunique():,}")

# %%
# ========================================
# Paso 3. Guardar, empaquetar y (opcional) publicar
# ========================================
print("[3/3] Guardando")
venues.to_parquet(DIR_SALIDA / "venues.parquet", index=False)
checkins.to_parquet(DIR_SALIDA / "checkins.parquet", index=False)
print(f"  venues.parquet ({len(venues):,}) | checkins.parquet ({len(checkins):,})")

config.publicar(NOMBRE_DATASET, DIR_SALIDA)
