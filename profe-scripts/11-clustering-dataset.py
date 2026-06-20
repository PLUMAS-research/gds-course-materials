# %%
"""Dataset de la clase de clustering: dos semanas de reportes SOSAFE anonimizados.

Produce un subconjunto liviano de los reportes ciudadanos para detectar
hotspots con DBSCAN/HDBSCAN. Dos decisiones respecto al dataset completo de
reportes:

- Solo una quincena (configurable), no el año entero: tamaño manejable y sin el
  pico de petardos de fin de año que distorsiona la densidad.
- El texto libre (`description`) se anonimiza: se reemplazan correos y teléfonos
  por marcas fijas antes de publicar el dataset.

Salidas en `data/sosafe-clustering/`:
- `reportes-quincena.parquet`: puntos limpios, categorizados y anonimizados.

El grid H3-8 con perfil censal para SKATER se reutiliza de la clase 08
(`censo2024-asignacion-rm.tgz`), no se reconstruye aquí.
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
import geopandas as gpd
import pandas as pd

from gdsutils.censo2024 import regiones
from gdsutils.sosafe import (
    PATRON_CORREO,
    PATRON_TELEFONO,
    anonimizar_texto,
    cargar_reportes,
    filtrar_relevantes,
)

# %% Configuración
NOMBRE_DATASET = "sosafe-clustering"
DIR_SALIDA = Path("data") / NOMBRE_DATASET

SOSAFE_RAW = config.requerir(config.SOSAFE_RAW, "reportes SOSAFE crudos", "GDS_SOSAFE_RAW")
BBOX_RM = tuple(regiones["RM"]["capital_bbox"])

# Quincena: lunes a domingo x2. Abril es periodo regular (post-verano, sin
# feriados largos) con buena mezcla de los tres grupos.
FECHA_INICIO = pd.Timestamp("2024-04-08")
N_DIAS = 14

# Columnas que sobreviven al recorte. La descripción se conserva anonimizada;
# `comments`/`likes` son conteos, no texto.
COLUMNAS_SALIDA = [
    "geometry",
    "created_at",
    "hora",
    "dia_semana",
    "type",
    "categoria",
    "grupo",
    "likes",
    "comments",
    "description",
]


# %% Paso 1 — leer los JSON crudos de la quincena
DIR_SALIDA.mkdir(exist_ok=True, parents=True)
fechas = [FECHA_INICIO + pd.Timedelta(days=i) for i in range(N_DIAS)]
archivos = [SOSAFE_RAW / f"{f:%Y-%m-%d}.json" for f in fechas]
faltantes = [a.name for a in archivos if not a.exists()]
if faltantes:
    raise FileNotFoundError(
        f"Faltan {len(faltantes)} JSON crudos en {SOSAFE_RAW}: {faltantes[:3]}..."
    )

print(f"[1/4] Leyendo {len(archivos)} JSON: {fechas[0]:%Y-%m-%d} a {fechas[-1]:%Y-%m-%d}")
reportes = filtrar_relevantes(cargar_reportes(archivos, bbox=BBOX_RM))
print(f"  Reportes relevantes: {len(reportes):,}")
print(reportes["grupo"].value_counts())

# %% Paso 2 — anonimizar el texto libre
print("[2/4] Anonimizando correos y teléfonos en `description`")
desc = reportes["description"].fillna("")
n_correo = desc.str.contains(PATRON_CORREO).sum()
n_tel = desc.str.contains(PATRON_TELEFONO).sum()
reportes["description"] = anonimizar_texto(reportes["description"])
print(f"  Reportes con correo: {n_correo:,} | con teléfono: {n_tel:,}")
# Verificación: no debe quedar ningún patrón sin marcar.
restante_correo = reportes["description"].fillna("").str.contains(PATRON_CORREO).sum()
restante_tel = reportes["description"].fillna("").str.contains(PATRON_TELEFONO).sum()
assert restante_correo == 0 and restante_tel == 0, (
    f"Quedó PII sin anonimizar: {restante_correo} correos, {restante_tel} teléfonos"
)
print("  Verificación OK: sin correos ni teléfonos en el texto de salida")

# %% Paso 3 — recortar columnas y guardar
print("[3/4] Recortando columnas y guardando parquet")
reportes = gpd.GeoDataFrame(
    reportes[COLUMNAS_SALIDA], geometry="geometry", crs=reportes.crs
)
ruta_parquet = DIR_SALIDA / "reportes-quincena.parquet"
reportes.to_parquet(ruta_parquet)
print(f"  Guardado: {ruta_parquet} ({len(reportes):,} filas, {len(reportes.columns)} columnas)")

# %% Paso 4 — empaquetar y (opcional) subir
print("[4/4] Empaquetando y publicando")
config.publicar(NOMBRE_DATASET, DIR_SALIDA)
