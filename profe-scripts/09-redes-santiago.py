# %%
"""Prepara las redes urbanas del Gran Santiago para la clase 09.

Combina dos fuentes de datos:

1. OpenStreetMap (via quackosm): calles vehiculares y ciclovías
   etiquetadas como `highway=cycleway`. La cobertura OSM es completa
   para calles pero parcial para ciclovías (depende del mapeo
   colaborativo).
2. Catálogo nacional de ciclovías de MINVU (Geoportal de Chile):
   shapefile oficial con atributos de etapa (existente, planificada,
   diseño), tipo (ciclovía, smp, zona30, cicloparque), emplazamiento
   (calzada, acera, berma, parque), año de ejecución, normativa.

Fuente MINVU
------------
https://geoportal.cl/geoportal/catalog/36392/Ciclovias

Bajar manualmente el shapefile (zip) y dejarlo en
`~/Descargas/CICLOVchile_shp.zip`. La clase usa solo los registros
RM con `ETAPA = existentes` para la red real; el resto queda fuera
del dataset publicado pero se puede recuperar reejecutando este
script con un filtro distinto.

Salidas
-------
- `data/redes-santiago/calles.parquet`: aristas de calles vehiculares
  OSM.
- `data/redes-santiago/ciclovias-osm.parquet`: ciclovías OSM
  (`highway=cycleway`).
- `data/redes-santiago/ciclovias-minvu.parquet`: catálogo MINVU RM
  filtrado a `ETAPA = existentes`.
- `data/redes-santiago.tgz`: empaquetado para subir al servidor.

Notas técnicas
--------------
Quackosm corre workers con `multiprocessing` start method "spawn" para
ser compatible con bibliotecas Rust (Polars, DuckDB). Esto requiere que
el código pesado quede dentro del guard `if __name__ == "__main__":`
para evitar recursión en los procesos hijos.
"""

# %%
import multiprocessing as mp
import sys
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import geopandas as gpd
import pandas as pd
import quackosm as qosm
from shapely.geometry import box

from gdsutils.ndvi import BBOX_SANTIAGO

# %%
NOMBRE_DATASET = "redes-santiago"
DIR_SALIDA = Path("data") / NOMBRE_DATASET
DIR_TRABAJO_QUACKOSM = Path("data") / "_quackosm_cache"

RUTA_MINVU_ZIP = config.MINVU_ZIP
SHP_MINVU_EN_ZIP = "CICLOVchile_shp1/SHP/CICLOVnac_dic25.shp"

TAGS_CALLES = {
    "highway": [
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    ]
}

ATRIBUTOS_CALLES = ["highway", "name", "oneway", "maxspeed", "surface", "lanes"]
ATRIBUTOS_CICLOVIAS_OSM = ["highway", "name", "surface", "lit", "oneway"]
ATRIBUTOS_CICLOVIAS_MINVU = [
    "IDENTIFICA",
    "REGION",
    "COMUNA",
    "EJE_VIA",
    "KM",
    "TIPO",
    "CARAC_FUNC",
    "EMPLAZA_TE",
    "URBANA",
    "ETAPA",
    "YEAR_EJECU",
]


def extraer_minvu_a_rm(ruta_zip: Path) -> gpd.GeoDataFrame:
    """Lee el shapefile MINVU del zip, filtra a RM y existentes."""
    if not ruta_zip.exists():
        raise FileNotFoundError(
            f"No existe {ruta_zip}. Bajar el shapefile desde "
            "https://geoportal.cl/geoportal/catalog/36392/Ciclovias"
        )
    uri = f"zip://{ruta_zip}!{SHP_MINVU_EN_ZIP}"
    print(f"Leyendo MINVU shapefile: {uri}")
    gdf = gpd.read_file(uri)
    print(f"  Filas crudas: {len(gdf)}")
    rm = gdf[gdf["REGION"] == "13-Metropolitana"].copy()
    print(f"  RM (todas las etapas): {len(rm)}")
    existentes = rm[rm["ETAPA"] == "existentes"].copy()
    print(f"  RM existentes: {len(existentes)}")
    return existentes


# %%
if __name__ == "__main__":
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    DIR_TRABAJO_QUACKOSM.mkdir(parents=True, exist_ok=True)

    LIMITE_CPU = max(1, mp.cpu_count() // 2)
    print(f"Usando hasta {LIMITE_CPU} CPUs de {mp.cpu_count()} disponibles")

    POLIGONO = box(*BBOX_SANTIAGO)
    print(f"BBOX Gran Santiago: {BBOX_SANTIAGO}")

    # ========================================
    # Paso 1. Calles vehiculares (OSM)
    # ========================================
    print("[1/4] Extrayendo calles vehiculares con quackosm")
    calles = qosm.convert_geometry_to_geodataframe(
        geometry_filter=POLIGONO,
        tags_filter=TAGS_CALLES,
        working_directory=DIR_TRABAJO_QUACKOSM,
        verbosity_mode="transient",
        keep_all_tags=True,
        explode_tags=True,
        cpu_limit=LIMITE_CPU,
    )
    calles = calles[calles.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    print(f"  Filas crudas: {len(calles)}")
    columnas = [c for c in ATRIBUTOS_CALLES if c in calles.columns] + ["geometry"]
    calles = calles[columnas].copy()
    calles.to_parquet(DIR_SALIDA / "calles.parquet")
    print(f"  Guardado: {DIR_SALIDA / 'calles.parquet'} ({len(calles)} filas)")

    # ========================================
    # Paso 2. Ciclovías OSM (highway=cycleway)
    # ========================================
    print("[2/4] Extrayendo ciclovías OSM")
    ciclovias_osm = qosm.convert_geometry_to_geodataframe(
        geometry_filter=POLIGONO,
        tags_filter={"highway": "cycleway"},
        working_directory=DIR_TRABAJO_QUACKOSM,
        verbosity_mode="transient",
        keep_all_tags=True,
        explode_tags=True,
        cpu_limit=LIMITE_CPU,
    )
    ciclovias_osm = ciclovias_osm[
        ciclovias_osm.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ]
    print(f"  Filas crudas: {len(ciclovias_osm)}")
    columnas = [c for c in ATRIBUTOS_CICLOVIAS_OSM if c in ciclovias_osm.columns] + [
        "geometry"
    ]
    ciclovias_osm = ciclovias_osm[columnas].copy()
    ciclovias_osm.to_parquet(DIR_SALIDA / "ciclovias-osm.parquet")
    print(
        f"  Guardado: {DIR_SALIDA / 'ciclovias-osm.parquet'} "
        f"({len(ciclovias_osm)} filas)"
    )

    # ========================================
    # Paso 3. Ciclovías MINVU (catálogo nacional, RM, existentes)
    # ========================================
    print("[3/4] Procesando catálogo MINVU")
    ciclovias_minvu = extraer_minvu_a_rm(RUTA_MINVU_ZIP)
    ciclovias_minvu = ciclovias_minvu.to_crs("EPSG:4326")
    columnas = [
        c for c in ATRIBUTOS_CICLOVIAS_MINVU if c in ciclovias_minvu.columns
    ] + ["geometry"]
    ciclovias_minvu = ciclovias_minvu[columnas].copy()
    if "KM" in ciclovias_minvu.columns:
        ciclovias_minvu["KM"] = pd.to_numeric(ciclovias_minvu["KM"], errors="coerce")
    ciclovias_minvu.to_parquet(DIR_SALIDA / "ciclovias-minvu.parquet")
    print(
        f"  Guardado: {DIR_SALIDA / 'ciclovias-minvu.parquet'} "
        f"({len(ciclovias_minvu)} filas)"
    )

    # ========================================
    # Paso 4. Empaquetar y subir
    # ========================================
    print("[4/4] Empaquetando y publicando")
    config.publicar(NOMBRE_DATASET, DIR_SALIDA)
