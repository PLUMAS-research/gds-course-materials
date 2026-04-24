"""NDVI desde Sentinel-2 vía Microsoft Planetary Computer.

Flujo típico en un script de clase:

    from gdsutils.ndvi import descargar_ndvi_santiago, ndvi_por_zona
    ruta = descargar_ndvi_santiago()
    hexagonos_con_ndvi = ndvi_por_zona(hexagonos, ruta)

Para luminosidad nocturna (VIIRS Black Marble VNP46A4, radiancia anual en
nW/cm²/sr), usar `descargar_luminosidad_santiago` y reutilizar
`ndvi_por_zona` indicando `columna="luminosidad"`.
"""

from pathlib import Path
import urllib.request

import numpy as np
import planetary_computer
import rioxarray  # noqa: F401  registra accessor .rio en xarray
from pystac_client import Client

BBOX_SANTIAGO = (-70.85, -33.65, -70.45, -33.30)
CRS_UTM_19S = "EPSG:32719"
URL_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

URL_LUMINOSIDAD_SANTIAGO = (
    "https://dcc.uchile.cl/~egraells/gds-data/luminosidad-santiago-2023.tif"
)
URL_NDVI_SANTIAGO = (
    "https://dcc.uchile.cl/~egraells/gds-data/ndvi-santiago-2023.tif"
)

# SCL (Scene Classification Layer) de Sentinel-2:
# 4 = vegetación, 5 = suelo desnudo, 6 = agua, 7 = sin clasificar.
# 3 = sombra de nube, 8-10 = nubes, 11 = nieve.
SCL_LIMPIO = [4, 5, 6, 7]


def descargar_ndvi_santiago(
    bbox=BBOX_SANTIAGO,
    fecha_inicio="2023-10-01",
    fecha_fin="2024-03-31",
    max_nubosidad=20,
    resolucion=30,
    limite_escenas=15,
    ruta_salida=None,
):
    """Calcula NDVI mediano sobre un bbox WGS84 y guarda un GeoTIFF.

    Descarga escenas Sentinel-2 L2A desde Planetary Computer, agrupa
    las escenas del mismo día solar (fusiona tiles MGRS vecinos),
    se queda con las más limpias, aplica máscara de nubes con SCL,
    calcula NDVI por escena y toma la mediana temporal pixel a pixel.
    El resultado queda reproyectado a UTM 19S (EPSG:32719).

    Si `ruta_salida` ya existe, la función no recalcula nada y devuelve
    la ruta existente.

    Parámetros
    ----------
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max) en WGS84.
    fecha_inicio, fecha_fin : str
        Rango temporal en formato ISO. Por defecto cubre primavera-verano
        austral 2023-2024, cuando la actividad de avifauna es mayor.
    max_nubosidad : int
        Filtro STAC por `eo:cloud_cover`. Escenas con más nubes se descartan.
    resolucion : int
        Resolución de salida en metros. 30 m basta para agregar a
        hexágonos H3-8 (~460 m de lado); subir a 10 m solo tiene sentido
        si se quiere detalle intraparque.
    limite_escenas : int o None
        Si se indica, se retienen las `limite_escenas` fechas con menor
        nubosidad promedio. None usa todas.
    ruta_salida : str o Path, opcional
        Destino del GeoTIFF. Por defecto: `data/ndvi-santiago-{año_inicio}.tif`.

    Retorna
    -------
    Path
        Ruta al GeoTIFF con NDVI mediano.
    """
    import odc.stac

    if ruta_salida is None:
        ruta_salida = Path("data") / f"ndvi-santiago-{fecha_inicio[:4]}.tif"
    else:
        ruta_salida = Path(ruta_salida)

    if ruta_salida.exists():
        print(f"NDVI ya calculado en {ruta_salida}")
        return ruta_salida

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Buscando escenas Sentinel-2 L2A entre {fecha_inicio} y {fecha_fin}")
    print(f"  bbox = {bbox}")
    print(f"  nubosidad máxima = {max_nubosidad}%")

    cliente = Client.open(URL_STAC, modifier=planetary_computer.sign_inplace)
    busqueda = cliente.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{fecha_inicio}/{fecha_fin}",
        query={"eo:cloud_cover": {"lt": max_nubosidad}},
    )
    items = list(busqueda.items())
    print(f"  escenas candidatas: {len(items)}")
    if not items:
        raise ValueError("No se encontraron escenas en el rango indicado")

    if limite_escenas is not None and len(items) > limite_escenas:
        fechas = {}
        for item in items:
            clave = item.datetime.date()
            nubosidad = item.properties.get("eo:cloud_cover", 100)
            fechas.setdefault(clave, []).append((nubosidad, item))
        nubosidad_por_fecha = [
            (sum(n for n, _ in lst) / len(lst), fecha)
            for fecha, lst in fechas.items()
        ]
        nubosidad_por_fecha.sort()
        fechas_elegidas = {
            fecha for _, fecha in nubosidad_por_fecha[:limite_escenas]
        }
        items = [
            item for item in items if item.datetime.date() in fechas_elegidas
        ]
        print(
            f"  retenidas {len(items)} escenas de {len(fechas_elegidas)} "
            f"fechas más limpias"
        )

    print("Cargando bandas B04 (rojo), B08 (NIR) y SCL (clasificación)")
    data = odc.stac.load(
        items,
        bands=["B04", "B08", "SCL"],
        bbox=bbox,
        resolution=resolucion,
        crs=CRS_UTM_19S,
        groupby="solar_day",
        chunks={"time": 1, "x": 2048, "y": 2048},
    )
    print(f"  dimensiones: {dict(data.sizes)}")

    print("Aplicando máscara de nubes vía SCL")
    mascara = data.SCL.isin(SCL_LIMPIO)
    b04 = data.B04.where(mascara)
    b08 = data.B08.where(mascara)

    print("Calculando NDVI y mediana temporal")
    suma = b08 + b04
    ndvi = (b08 - b04) / suma.where(suma != 0)
    ndvi = ndvi.where((ndvi >= -1) & (ndvi <= 1))
    ndvi_mediana = ndvi.median(dim="time", skipna=True)

    ndvi_mediana = ndvi_mediana.rio.write_crs(CRS_UTM_19S)
    ndvi_mediana = ndvi_mediana.astype("float32")

    print(f"Escribiendo {ruta_salida}")
    ndvi_mediana.rio.to_raster(ruta_salida, compress="deflate", tiled=True)
    print("Listo")
    return ruta_salida


def descargar_ndvi_santiago_precomputado(ruta_salida=None):
    """Descarga el GeoTIFF de NDVI mediano para Santiago desde el servidor.

    Recorte UTM 19S del NDVI mediano sobre escenas Sentinel-2 L2A entre
    octubre 2023 y marzo 2024 (verano austral). Equivalente al output de
    `descargar_ndvi_santiago()`, pero sin requerir Planetary Computer ni
    sus dependencias pesadas (odc.stac, planetary_computer, pystac_client).
    Pensado para que los estudiantes accedan al mismo raster de entrada
    sin tener que correr el cómputo desde STAC.

    Parámetros
    ----------
    ruta_salida : str o Path, opcional
        Ruta local del TIFF. Por defecto: `data/ndvi-santiago-2023.tif`.

    Retorna
    -------
    Path
        Ruta al GeoTIFF local (existente o recién descargado).
    """
    if ruta_salida is None:
        ruta_salida = Path("data") / "ndvi-santiago-2023.tif"
    else:
        ruta_salida = Path(ruta_salida)

    if ruta_salida.exists():
        print(f"NDVI ya descargado: {ruta_salida}")
        return ruta_salida

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL_NDVI_SANTIAGO}")
    urllib.request.urlretrieve(URL_NDVI_SANTIAGO, ruta_salida)
    print(f"  Guardado: {ruta_salida}")
    return ruta_salida


def descargar_luminosidad_santiago(ruta_salida=None):
    """Descarga el GeoTIFF de luminosidad nocturna anual 2023 para Santiago.

    El archivo es un recorte pre-procesado del producto VIIRS Black Marble
    VNP46A4 (annual `AllAngle_Composite_Snow_Free`), reproyectado a UTM 19S
    y alojado en el servidor del curso. La radiancia queda en nW/cm²/sr.
    La preparación original exige una cuenta NASA Earthdata; el recorte se
    publica para que los estudiantes no necesiten autenticarse.

    Parámetros
    ----------
    ruta_salida : str o Path, opcional
        Ruta local del TIFF. Por defecto: `data/luminosidad-santiago-2023.tif`.

    Retorna
    -------
    Path
        Ruta al GeoTIFF local (existente o recién descargado).
    """
    if ruta_salida is None:
        ruta_salida = Path("data") / "luminosidad-santiago-2023.tif"
    else:
        ruta_salida = Path(ruta_salida)

    if ruta_salida.exists():
        print(f"Luminosidad ya descargada: {ruta_salida}")
        return ruta_salida

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL_LUMINOSIDAD_SANTIAGO}")
    urllib.request.urlretrieve(URL_LUMINOSIDAD_SANTIAGO, ruta_salida)
    print(f"  Guardado: {ruta_salida}")
    return ruta_salida


def ndvi_por_zona(geodf, ruta_raster, columna="ndvi", estadistico="mean"):
    """Calcula estadística zonal de NDVI por geometría.

    Parámetros
    ----------
    geodf : GeoDataFrame
        Zonas sobre las cuales agregar (por ejemplo, hexágonos H3-8).
    ruta_raster : str o Path
        GeoTIFF con NDVI generado por `descargar_ndvi_santiago`.
    columna : str
        Nombre de la columna de salida.
    estadistico : str o lista
        Estadístico pasado a `rasterstats.zonal_stats` (mean, median,
        std, min, max, count, ...).

    Retorna
    -------
    GeoDataFrame
        Copia de `geodf` con la columna de NDVI agregada.
    """
    import rasterio
    from rasterstats import zonal_stats

    with rasterio.open(ruta_raster) as src:
        crs_raster = src.crs

    geodf_proj = geodf.to_crs(crs_raster)
    stats = zonal_stats(
        geodf_proj.geometry,
        str(ruta_raster),
        stats=[estadistico] if isinstance(estadistico, str) else estadistico,
        nodata=np.nan,
    )

    resultado = geodf.copy()
    if isinstance(estadistico, str):
        resultado[columna] = [s[estadistico] for s in stats]
    else:
        for est in estadistico:
            resultado[f"{columna}_{est}"] = [s[est] for s in stats]
    return resultado
