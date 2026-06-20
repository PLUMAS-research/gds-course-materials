# %%
"""Construye dataset de asignacion censal a H3 para el Gran Santiago.

Para cada comuna RM con manzanas dentro del bbox del Gran Santiago:
  1. Construye la tabla wide vivienda x atributos desde los microdatos.
  2. Construye un grid H3-9 sobre la comuna.
  3. Asigna manzanas a celdas H3-9 por areal weighting (idxmax) y agrega
     los conteos publicados por manzana al grid.
  4. Llama a `gdsutils.asignacion.asignar_viviendas` con una config que
     prioriza sexo, inmigrantes, ocupacion CIUO, edad, educacion CINE,
     transporte, dormitorios y tipo de vivienda como restriccion dura.

Salidas
-------
- `data/censo2024-asignacion-rm/asignaciones-h3-9/{cod_comuna}.parquet`
  (id_vivienda, h3_cell_id, comuna)
- `data/censo2024-asignacion-rm/errores/{cod_comuna}.parquet`
  (errores por atributo, para diagnostico)
- `data/censo2024-asignacion-rm/asignaciones-h3-9.parquet` (concat
  ordenado por comuna; pega los conteos por vivienda como columnas n_*)
- `data/censo2024-asignacion-rm/asignaciones-h3-8.parquet` (reagregacion
  a H3-8 con sumas por hexagono).
- `data/censo2024-asignacion-rm.tgz`: empaquetado para subir al servidor.

Idempotente. Si ya existen los parquet por comuna (incluyendo copias
importadas desde un directorio de asignaciones precomputadas, configurado en
`LEGADO_ASIGNACION`), los reusa.

Uso:
    uv run python profe-scripts/08-asignacion-rm.py

Tiempo total esperado en mi maquina: ~3 a 5 horas para las 43 comunas
del Gran Santiago si no hay nada precomputado. La comuna mas grande
(Puente Alto, ~183k viviendas) toma ~10 min. Para comunas chicas <2 min.
"""

# %%
import gc
import shutil
import sys
import tarfile
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import geopandas as gpd
import pandas as pd
from chiricoca.base.weights import normalize_rows
from chiricoca.geo.grid import h3_grid_from_bounds
from h3 import cell_to_parent

from gdsutils.asignacion import (
    asignar_viviendas,
    diagnosticar_atributos,
    reportar_errores,
)
from gdsutils.censo2024 import regiones, tabla_viviendas_desde_microdatos

# %% Configuracion
NOMBRE_DATASET = "censo2024-asignacion-rm"
DIR_SALIDA = Path("data") / NOMBRE_DATASET
DIR_VIVIENDAS = DIR_SALIDA / "viviendas"
DIR_ASIGNACIONES_H9 = DIR_SALIDA / "asignaciones-h3-9"
DIR_ERRORES = DIR_SALIDA / "errores"
RUTA_H9 = DIR_SALIDA / "asignaciones-h3-9.parquet"
RUTA_H8 = DIR_SALIDA / "asignaciones-h3-8.parquet"
RUTA_TGZ = Path("data") / f"{NOMBRE_DATASET}.tgz"

REGION = regiones["RM"]["region"]
BBOX = regiones["RM"]["capital_bbox"]

RUTA_CARTO_MANZANAS = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Manzanas.parquet"
)

# Si existe, intenta importar desde un directorio de asignaciones
# precomputadas antes de regenerar (los nombres de archivo son distintos:
# ahi son `asignacion-censal-2024-rm-{cod}.parquet`). Configurar en
# config_local.py o por env var GDS_LEGADO_ASIGNACION.
RUTA_LEGADO = config.LEGADO_ASIGNACION

CONFIG_ASIGNACION = {
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

PARAMS_SA = dict(
    peso_alta=15.0,
    peso_media=1.0,
    peso_baja=0.1,
    peso_tipo=20.0,
    sa_iteraciones=500_000,
    verbose=True,
)

for d in (DIR_SALIDA, DIR_VIVIENDAS, DIR_ASIGNACIONES_H9, DIR_ERRORES):
    d.mkdir(parents=True, exist_ok=True)


# %%
print(f"[1/5] Cargando manzanas RM y recortando al bbox del Gran Santiago")
manzanas_rm = (
    gpd.read_parquet(RUTA_CARTO_MANZANAS, filters=[("COD_REGION", "=", REGION)])
    .assign(
        geometry=lambda x: x.clip_by_rect(*BBOX),
        MANZENT=lambda x: x["MANZENT"].astype(int),
        CUT=lambda x: x["CUT"].astype(int),
    )
    .set_geometry("geometry")
    .pipe(lambda x: x[~x.is_empty])
    .drop("SHAPE", axis=1, errors="ignore")
)
print(f"  manzanas en bbox: {len(manzanas_rm):,}")

comunas = sorted(manzanas_rm["CUT"].unique())
print(f"  comunas a procesar: {len(comunas)}")


# %%
def importar_legado(cod_comuna):
    """Si hay parquets equivalentes en `RUTA_LEGADO`, los copia con el nombre nuevo."""
    if not RUTA_LEGADO or not RUTA_LEGADO.exists():
        return False

    fuente_asignacion = RUTA_LEGADO / f"asignacion-censal-2024-rm-{cod_comuna}.parquet"
    fuente_errores = (
        RUTA_LEGADO / f"asignacion-censal-2024-rm-errores-{cod_comuna}.parquet"
    )
    fuente_viviendas = RUTA_LEGADO / f"censo2024-viviendas-rm-{cod_comuna}.parquet"

    destino_asignacion = DIR_ASIGNACIONES_H9 / f"{cod_comuna}.parquet"
    destino_errores = DIR_ERRORES / f"{cod_comuna}.parquet"
    destino_viviendas = DIR_VIVIENDAS / f"{cod_comuna}.parquet"

    importado = False
    if fuente_asignacion.exists() and not destino_asignacion.exists():
        shutil.copy(fuente_asignacion, destino_asignacion)
        importado = True
    if fuente_errores.exists() and not destino_errores.exists():
        shutil.copy(fuente_errores, destino_errores)
    if fuente_viviendas.exists() and not destino_viviendas.exists():
        shutil.copy(fuente_viviendas, destino_viviendas)

    return importado


def procesar_comuna(cod_comuna):
    destino_asignacion = DIR_ASIGNACIONES_H9 / f"{cod_comuna}.parquet"
    if destino_asignacion.exists():
        print(f"  comuna {cod_comuna}: ya existe, skip")
        return

    if importar_legado(cod_comuna):
        print(f"  comuna {cod_comuna}: importada desde {RUTA_LEGADO}")
        return

    print(f"  comuna {cod_comuna}: procesando")
    manzanas_comuna = manzanas_rm[manzanas_rm["CUT"] == cod_comuna]
    if len(manzanas_comuna) == 0:
        print("    sin manzanas, skip")
        return

    count_cols = manzanas_comuna.filter(like="n_").columns

    grid = h3_grid_from_bounds(
        manzanas_comuna.total_bounds, extra_margin=0.05, grid_level=9
    )

    cell_to_manzana = (
        gpd.overlay(manzanas_comuna, grid.reset_index(), how="intersection")
        .assign(area=lambda x: x.area)
        .set_index(["MANZENT", "h3_cell_id"])["area"]
        .unstack(fill_value=0)
        .pipe(normalize_rows)
        .idxmax(axis=1)
        .rename("h3_cell_id")
    )

    grid_variables = grid.join(
        manzanas_comuna.join(cell_to_manzana, on="MANZENT")
        .groupby("h3_cell_id")[count_cols]
        .sum()
    ).dropna()

    if len(grid_variables) == 0:
        print("    grid vacio tras dropna, skip")
        return

    tabla_viv, _ = tabla_viviendas_desde_microdatos(cod_comuna, region=REGION)
    tabla_viv.to_parquet(DIR_VIVIENDAS / f"{cod_comuna}.parquet")

    diagnostico, config_filtrada = diagnosticar_atributos(tabla_viv, CONFIG_ASIGNACION)

    resultado, errores = asignar_viviendas(
        tabla_viv.reset_index(drop=True),
        grid_variables.reset_index(),
        config_filtrada,
        **PARAMS_SA,
    )

    df_errores = reportar_errores(errores, tabla_viv, config_filtrada)

    resultado["comuna"] = cod_comuna
    resultado.to_parquet(destino_asignacion)
    df_errores.to_parquet(DIR_ERRORES / f"{cod_comuna}.parquet")

    print(
        f"    asignadas {len(resultado):,} viviendas a {len(grid_variables):,} celdas H3-9"
    )


# %%
print("[2/5] Asignando comunas")
for cod_comuna in comunas:
    procesar_comuna(cod_comuna)
    gc.collect()


# %%
print("[3/5] Concatenando asignaciones a H3-9")
asignaciones = []
for cod_comuna in comunas:
    f = DIR_ASIGNACIONES_H9 / f"{cod_comuna}.parquet"
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    if "comuna" not in df.columns:
        df["comuna"] = cod_comuna
    asignaciones.append(df)

asignaciones = pd.concat(asignaciones, ignore_index=True)
print(f"  total viviendas asignadas: {len(asignaciones):,}")

asignaciones.to_parquet(RUTA_H9)


# %%
print("[4/5] Pegando conteos por vivienda y reagregando a H3-8")
viviendas_concat = []
for cod_comuna in comunas:
    f = DIR_VIVIENDAS / f"{cod_comuna}.parquet"
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    df["comuna"] = cod_comuna
    viviendas_concat.append(df)

viviendas_concat = pd.concat(viviendas_concat, ignore_index=True)
print(f"  total viviendas en tabla wide: {len(viviendas_concat):,}")

cols_n = [c for c in viviendas_concat.columns if c.startswith("n_")]
join = asignaciones.merge(
    viviendas_concat[["id_vivienda"] + cols_n], on="id_vivienda", how="left"
)

asignaciones_h8 = (
    join.assign(
        h3_cell_id=lambda x: x["h3_cell_id"].map(lambda c: cell_to_parent(c, 8))
    )
    .groupby("h3_cell_id")[cols_n]
    .sum()
    .reset_index()
)
asignaciones_h8["n_vp"] = (
    join.assign(
        h3_cell_id=lambda x: x["h3_cell_id"].map(lambda c: cell_to_parent(c, 8))
    )
    .groupby("h3_cell_id")
    .size()
    .values
)
asignaciones_h8.to_parquet(RUTA_H8)
print(f"  hexagonos H3-8: {len(asignaciones_h8):,}")


# %%
print(f"[5/5] Empaquetando {RUTA_TGZ}")
with tarfile.open(RUTA_TGZ, "w:gz") as tar:
    tar.add(RUTA_H9, arcname=f"{NOMBRE_DATASET}/asignaciones-h3-9.parquet")
    tar.add(RUTA_H8, arcname=f"{NOMBRE_DATASET}/asignaciones-h3-8.parquet")
print(f"  Tamanho: {RUTA_TGZ.stat().st_size / 1024 / 1024:.1f} MB")


# %%
# Empaquetado custom (solo los dos parquet agregados, no las subcarpetas
# intermedias), por eso no se usa config.publicar.
if config.SUBIR_AL_SERVIDOR:
    config.subir_scp(RUTA_TGZ)
else:
    print("SUBIR_AL_SERVIDOR=False (GDS_SUBIR=1 para subir). Subir manualmente:")
    print(f"  scp {RUTA_TGZ} {config.DESTINO_SCP}")
