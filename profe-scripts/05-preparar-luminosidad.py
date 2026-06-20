# %%
"""Prepara el GeoTIFF de luminosidad nocturna que consumen los scripts 05 y 06.

Se ejecuta UNA sola vez y genera `data/luminosidad-santiago-2023.tif`, que
luego hay que subir al servidor con el mismo nombre para que
`gdsutils.ndvi.descargar_luminosidad_santiago` lo encuentre.

Requisitos previos (paso manual)
---------------------------------
1. Crear una cuenta gratuita en https://urs.earthdata.nasa.gov.
2. Autorizar la aplicación "LAADS DAAC Cumulus (PROD)" desde
   https://urs.earthdata.nasa.gov/profile → Applications → Authorized Apps.
3. Agregar earthaccess al entorno:
       uv add earthaccess
4. Guardar credenciales para que earthaccess las lea automáticamente:
       uv run python -c "import earthaccess; earthaccess.login(persist=True)"
   Esto escribe `~/.netrc` con las credenciales.

Producto usado
--------------
VNP46A4 (VIIRS Black Marble annual composite). Capa
`AllAngle_Composite_Snow_Free`: radiancia promedio anual en nW/cm²/sr,
corregida por efectos atmosféricos y lunares, excluyendo noches con nieve.
Resolución 15 arc-seg (~500 m), grilla geográfica 10×10°.
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
import earthaccess
import h5py
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.warp import Resampling

from gdsutils.ndvi import BBOX_SANTIAGO, CRS_UTM_19S

ANIO = 2023
RUTA_SALIDA = Path("data") / f"luminosidad-santiago-{ANIO}.tif"
DIR_HDF5 = Path("data") / "black-marble-raw"
DIR_HDF5.mkdir(parents=True, exist_ok=True)
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

# %% 1 — autenticación Earthdata
print("Autenticando contra NASA Earthdata")
earthaccess.login(strategy="netrc")

# %% 2 — búsqueda del granulo VNP46A4 para Santiago
print(f"Buscando granulos VNP46A4 del año {ANIO} sobre el bbox de Santiago")
resultados = earthaccess.search_data(
    short_name="VNP46A4",
    temporal=(f"{ANIO}-01-01", f"{ANIO}-12-31"),
    bounding_box=BBOX_SANTIAGO,
)
print(f"  granulos encontrados: {len(resultados)}")
if not resultados:
    raise SystemExit(
        "Sin resultados: verificar que VNP46A4 ya publicó el año pedido "
        "o bajar temporal=('2022-01-01','2022-12-31')."
    )

# %% 3 — descarga local del HDF5
print("Descargando HDF5 a", DIR_HDF5)
archivos = earthaccess.download(resultados, str(DIR_HDF5))
print("  archivos:", archivos)

# earthaccess devuelve los granulos del rango publicación, puede incluir
# años vecinos. Filtrar por el año pedido usando el patrón `A{YYYY}001`.
archivos_anio = [p for p in archivos if f"A{ANIO}001" in Path(p).name]
if not archivos_anio:
    raise SystemExit(f"No se descargó granulo para el año {ANIO}")
ruta_hdf5 = Path(archivos_anio[0])
print(f"  usando: {ruta_hdf5.name}")

# %% 4 — extracción de la capa AllAngle_Composite_Snow_Free
print(f"Leyendo capa AllAngle_Composite_Snow_Free desde {ruta_hdf5.name}")
with h5py.File(ruta_hdf5, "r") as f:
    campos = f["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"]
    capa = campos["AllAngle_Composite_Snow_Free"]
    datos = capa[:].astype("float32")
    escala = float(capa.attrs.get("scale_factor", [1.0])[0])
    offset = float(capa.attrs.get("offset", [0.0])[0])
    fill = float(capa.attrs.get("_FillValue", [-999.9])[0])
    datos = np.where(datos == fill, np.nan, datos * escala + offset)
    # Radiancia negativa residual (ruido bajo cero): tratar como 0.
    datos = np.where(datos < 0, 0.0, datos)

    # Las coordenadas vienen como arrays 1D en el mismo HDF5.
    y_coords = campos["lat"][:].astype("float64")
    x_coords = campos["lon"][:].astype("float64")

print(
    f"  tile extent lon=[{x_coords.min():.3f},{x_coords.max():.3f}] "
    f"lat=[{y_coords.min():.3f},{y_coords.max():.3f}]"
)
print(f"  shape={datos.shape}")

# %% 5 — armar DataArray en WGS84 con coordenadas geográficas
da = xr.DataArray(
    datos,
    coords={"y": y_coords, "x": x_coords},
    dims=("y", "x"),
    name="radiance_nW_cm2_sr",
)
da = da.rio.write_crs("EPSG:4326")

# %% 6 — recorte al bbox de Santiago (WGS84) y reproyección a UTM 19S
print("Recortando a bbox Santiago y reproyectando a UTM 19S")
lon_min, lat_min, lon_max, lat_max = BBOX_SANTIAGO
da_clip = da.rio.clip_box(
    minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max
)
da_utm = da_clip.rio.reproject(
    CRS_UTM_19S, resampling=Resampling.bilinear, resolution=500,
)
print(f"  shape final: {da_utm.shape}, res: 500 m")

# %% 7 — exportar GeoTIFF
da_utm.rio.to_raster(RUTA_SALIDA, compress="deflate", tiled=True)
print(f"Guardado: {RUTA_SALIDA}")

# %% 8 — verificación rápida
with rasterio.open(RUTA_SALIDA) as src:
    arr = src.read(1)
    validos = arr[~np.isnan(arr) & (arr > 0)]
    print(
        f"  píxeles válidos: {len(validos):,}, "
        f"radiancia min={validos.min():.3f} "
        f"median={np.median(validos):.3f} max={validos.max():.3f}"
    )

# %%
# Paso final: subir el GeoTIFF al servidor del curso (GDS_SUBIR=1)
if config.SUBIR_AL_SERVIDOR:
    config.subir_scp(RUTA_SALIDA)
else:
    print(
        f"\nSubir {RUTA_SALIDA} al servidor del curso:\n"
        f"  scp {RUTA_SALIDA} {config.DESTINO_SCP}"
    )
