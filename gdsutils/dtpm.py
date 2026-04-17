"""Descarga, extracción y consolidación de datos DTPM.

Esquemas de viajes:
  Antiguo (2014–2022): tiemposubida/tiempobajada, factorexpansion, netapa,
                       tipotransporte_1era…4ta  (BUS/METRO/ZP/METROTREN)
  Nuevo   (2023–2025): tiempo_inicio_viaje/tiempo_fin_viaje, factor_expansion, n_etapas,
                       tipo_transporte_1…4  (1=BUS, 2=METRO, 3=ZP, 4=METROTREN)

Dataset consolidado (columnas):
  anio, fecha, dia_semana (0=lun…6=dom), tipodia,
  ts_inicio, ts_fin, hora_inicio, minuto_inicio, hora_fin,
  tviaje_min, dist_ruta_mts, factor, proposito, contiene_metro,
  paradero, paradero_destino, comuna, n_etapas

Dataset paraderos: combina el Excel de paradas (BUS) con estaciones de
  metro y metrotren extraídas del GTFS. Columna "modo": BUS/METRO/METROTREN.
"""

import csv
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd

DIR_VIAJES = Path("data") / "dtpm-viajes"
DIR_CONSOLIDADO = Path("data") / "dtpm-consolidado"
RUTA_PARADEROS_RAW = Path("data") / "dtpm-raw" / "paraderos.xlsx"
RUTA_PARADEROS_PARQUET = Path("data") / "dtpm-paraderos.parquet"
RUTA_GTFS = Path("data") / "dtpm-raw" / "gtfs.zip"

URL_PARADEROS = "https://www.dtpm.cl/descargas/pops26/2026-03-21_consolidado_Registro-Paradas_anual.xlsx"
URL_GTFS = "https://www.dtpm.cl/descargas/gtfs/GTFS_20260321_v3.zip"

# Año → lista de URLs. Algunos años tienen archivos separados por tipo de día.
MAPEO_DTPM = {
    "2014": ["https://www.dtpm.cl/descargas/tablas/viajes201405_transparencia.rar"],
    "2015": ["https://www.dtpm.cl/descargas/tablas/viajes201504_transparencia.rar"],
    "2016": ["https://www.dtpm.cl/descargas/tablas/viajes201605_transparencia.rar"],
    "2017": [
        "https://www.dtpm.cl/descargas/tablas/viajes201704_laboral_transparencia.rar"
    ],
    "2018": [
        "https://www.dtpm.cl/descargas/tablas/viajes201804_laboral_transparencia.rar",
        "https://www.dtpm.cl/descargas/tablas/viajes_octubre_18.zip",
    ],
    "2019": [
        "https://www.dtpm.cl/descargas/tablas/tabla-viajes.rar",
        "https://www.dtpm.cl/descargas/tablas/viajes_19.zip",
    ],
    "2020": [
        # Semana del 9–12 mar (lun–jue) + dom 8 mar. Viernes ausente del dataset fuente.
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202003_laboral_transparencia.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202003_sab_dom_transparencia.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/nov21/viajes202011_transparencia_9al15.zip",
    ],
    "2021": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202104_transparencia.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202111_noviembre_8al14_transparencia.zip",
    ],
    "2022": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202204_abril_4al10_transparencia-.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes082022_8al14_transparencia.zip",
    ],
    "2023": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes_042023_17al23_transparencia.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes_082023_7al13-.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes_112023_6al12.zip",
    ],
    "2024": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202404_transparencia_15al21.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/Tabla-de-Viajes-Nov-24.zip",
    ],
    "2025": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/Tabla-de-viajes-011025.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/Tablas_viajes_NOV_2025.zip",
    ],
}

TIPODIA_NORM = {"0": "LABORAL", "1": "SABADO", "2": "DOMINGO"}
PROPOSITO_NORM = {"SIN_BAJADA": "SINBAJADA", "ACTIVIDAD1MINUTO": "MENOS1MINUTO"}

_TRANS_OLD = [
    "tipotransporte_1era",
    "tipotransporte_2da",
    "tipotransporte_3era",
    "tipotransporte_4ta",
]
_TRANS_NEW = [
    "tipo_transporte_1",
    "tipo_transporte_2",
    "tipo_transporte_3",
    "tipo_transporte_4",
]
_METRO_OLD = {"METRO", "METROTREN"}
_METRO_NEW = {
    "2",
    "4",
}  # 2=METRO, 4=METROTREN (verificado por comparación de magnitudes)

# Normalización de nombres: esquemas antiguo e intermedio → esquema nuevo.
# Se aplica en procesar_viajes antes de concatenar archivos con schemas distintos.
_NORM_COLUMNAS = {
    # Antiguo (2014–2022) → nuevo
    "tiemposubida": "tiempo_inicio_viaje",
    "tiempobajada": "tiempo_fin_viaje",
    "factorexpansion": "factor_expansion",
    "netapa": "n_etapas",
    "paraderosubida": "paradero_inicio_viaje",
    "paraderobajada": "paradero_fin_viaje",
    "comunasubida": "comuna_inicio_viaje",
    "comunabajada": "comuna_fin_viaje",
    "dviajeenruta_mts": "distancia_ruta",
    "dviajeeuclidiana_mts": "distancia_eucl",
    "tipotransporte_1era": "tipo_transporte_1",
    "tipotransporte_2da": "tipo_transporte_2",
    "tipotransporte_3era": "tipo_transporte_3",
    "tipotransporte_4ta": "tipo_transporte_4",
    # Intermedio (~2023) → nuevo
    "tiempo_inicio": "tiempo_inicio_viaje",
    "tiempo_fin": "tiempo_fin_viaje",
    "paradero_inicio": "paradero_inicio_viaje",
    "paradero_fin": "paradero_fin_viaje",
    "comuna_inicio": "comuna_inicio_viaje",
    "comuna_fin": "comuna_fin_viaje",
    "tipotransporte_1": "tipo_transporte_1",
    "tipotransporte_2": "tipo_transporte_2",
    "tipotransporte_3": "tipo_transporte_3",
    "tipotransporte_4": "tipo_transporte_4",
}


# ---------------------------------------------------------------------------
# Descarga y extracción
# ---------------------------------------------------------------------------


def _detectar_encoding(ruta_archivo):
    """Detecta si un archivo tiene BOM UTF-8. Devuelve el encoding adecuado."""
    with open(ruta_archivo, "rb") as f:
        return "utf-8-sig" if f.read(3) == b"\xef\xbb\xbf" else "latin-1"


def detectar_separador(ruta_archivo, encoding=None):
    """Detecta el delimitador de un CSV analizando sus primeras líneas."""
    if encoding is None:
        encoding = _detectar_encoding(ruta_archivo)
    with open(ruta_archivo, "r", encoding=encoding) as f:
        muestra = f.readline() + f.readline()
        try:
            dialect = csv.Sniffer().sniff(muestra, delimiters=",;||\t")
            return dialect.delimiter
        except Exception:
            for sep in ["|", ";", ",", "\t"]:
                if sep in muestra:
                    return sep
            return ","


def _ruta_raw(anio, idx, ext):
    """Ruta del archivo comprimido en disco.

    El índice 0 usa el nombre sin sufijo para no invalidar descargas previas;
    los siguientes añaden _{idx}.
    """
    dir_raw = Path("data") / "dtpm-raw"
    if idx == 0:
        return dir_raw / f"viajes_{anio}.{ext}"
    return dir_raw / f"viajes_{anio}_{idx}.{ext}"


def descargar_viajes(anio, urls):
    """Descarga los archivos comprimidos de viajes para un año.

    Si una URL falla (404, timeout, etc.) se imprime el error y se continúa
    con las siguientes.
    """
    (Path("data") / "dtpm-raw").mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls):
        url = url.strip()
        ext = url.split(".")[-1].lower()
        ruta = _ruta_raw(anio, idx, ext)
        if not ruta.exists():
            print(f"DESCARGANDO: {url}")
            try:
                urllib.request.urlretrieve(url, ruta)
            except Exception as e:
                print(f"ERROR DESCARGA ({url}): {e}")
                if ruta.exists():
                    ruta.unlink()
                continue


def procesar_viajes(anio, urls, dir_temp=None):
    """Extrae y convierte a Parquet los viajes de un año.

    Si el Parquet ya existe, no hace nada. Descarga los archivos fuente
    si no están en disco (omitiendo URLs que fallen).

    Maneja CSVs con BOM UTF-8, separadores distintos entre archivos y
    schemas mixtos (columnas diferentes) dentro del mismo año.

    Parameters
    ----------
    dir_temp : str o Path, opcional
        Directorio base para archivos temporales de extracción. Si es None
        usa el default del sistema (generalmente /tmp).
    """
    DIR_VIAJES.mkdir(parents=True, exist_ok=True)
    ruta_parquet = DIR_VIAJES / f"dtpm-{anio}.parquet"

    if ruta_parquet.exists():
        print(f"OMITIDO: {anio} ya procesado.")
        return

    rutas_raw = []
    for idx, url in enumerate(urls):
        url = url.strip()
        ext = url.split(".")[-1].lower()
        ruta = _ruta_raw(anio, idx, ext)
        if ruta.exists() and ruta.stat().st_size < 1024:
            print(f"CORRUPTO: {ruta.name}, se eliminará y reintentará.")
            ruta.unlink()
        if not ruta.exists():
            print(f"DESCARGANDO: {url}")
            try:
                urllib.request.urlretrieve(url, ruta)
                print(
                    f"DESCARGADO: {ruta.name} ({ruta.stat().st_size / 1024:.1f} KB)"
                )
            except Exception as e:
                print(f"ERROR DESCARGA ({url}): {e}")
                if ruta.exists():
                    ruta.unlink()
                continue
        else:
            print(f"LOCAL: {ruta.name}")
        rutas_raw.append(ruta)

    if not rutas_raw:
        print(f"ERROR: No hay archivos disponibles para el año {anio}")
        return

    if dir_temp is not None:
        Path(dir_temp).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dir_temp) as dir_temporal:
        path_temp = Path(dir_temporal)

        for ruta_comprimido in rutas_raw:
            ext = ruta_comprimido.suffix[1:].lower()
            if ext == "rar":
                cmd = ["unrar", "x", "-y", str(ruta_comprimido), str(path_temp) + "/"]
            else:
                cmd = ["7z", "x", str(ruta_comprimido), f"-o{path_temp}", "-y"]

            resultado = subprocess.run(cmd, capture_output=True, text=True)
            if resultado.returncode != 0 and not list(path_temp.rglob("*")):
                print(f"ERROR extracción ({ruta_comprimido.name}): {resultado.stderr}")
                return
            elif resultado.returncode != 0:
                print(
                    f"ADVERTENCIA ({ruta_comprimido.name}): {resultado.stderr.strip()}"
                )

        for zip_anidado in list(path_temp.rglob("*.zip")):
            print(f"EXTRAYENDO ZIP ANIDADO: {zip_anidado.name}")
            subprocess.run(
                ["unzip", "-o", str(zip_anidado), "-d", str(zip_anidado.parent)],
                capture_output=True,
            )
            zip_anidado.unlink()

        for gz_anidado in list(path_temp.rglob("*.gz")):
            print(f"EXTRAYENDO GZ ANIDADO: {gz_anidado.name}")
            subprocess.run(["gunzip", "-f", str(gz_anidado)], capture_output=True)

        archivos_datos = list(path_temp.rglob("*.csv")) + list(
            path_temp.rglob("*.txt")
        )
        if not archivos_datos:
            print(f"ERROR: No se hallaron datos para el año {anio}")
            return

        # Agrupar archivos por (encoding, separador) para manejar schemas mixtos
        grupos = {}
        for archivo in archivos_datos:
            enc = _detectar_encoding(archivo)
            sep = detectar_separador(archivo, encoding=enc)
            grupos.setdefault((enc, sep), []).append(archivo)

        partes = []
        for (enc, sep), archivos in grupos.items():
            print(
                f"LEYENDO: {len(archivos)} archivo(s) (enc={enc}, sep='{sep}')"
            )
            parte = dd.read_csv(
                archivos,
                sep=sep,
                encoding=enc,
                dtype=str,
                on_bad_lines="skip",
            )
            # Normalizar nombres: quitar comillas/espacios y unificar esquemas
            cols_limpias = {c: c.strip().strip('"') for c in parte.columns}
            cols_norm = {
                orig: _NORM_COLUMNAS.get(limpio, limpio)
                for orig, limpio in cols_limpias.items()
            }
            renombradas = [
                orig for orig, dest in cols_norm.items() if orig != dest
            ]
            if renombradas:
                print(f"  NORMALIZANDO: {len(renombradas)} columnas renombradas")
            parte = parte.rename(columns=cols_norm)
            partes.append(parte)

        if len(partes) == 1:
            df = partes[0]
        else:
            # Unificar schemas distintos
            df = dd.concat(partes, join="outer")

        print(
            f"CONVIRTIENDO: {len(archivos_datos)} archivo(s) → dtpm-{anio}.parquet"
        )
        df.to_parquet(ruta_parquet, engine="pyarrow", compression="brotli")
        print(f"FINALIZADO: {ruta_parquet}")


def _estaciones_desde_gtfs():
    """Extrae estaciones de metro y metrotren desde el GTFS.

    Descarga el ZIP si no existe, identifica rutas de tipo metro (route_type=1)
    y metrotren (route_type=0), y devuelve un GeoDataFrame en EPSG:32719.

    Metro: usa parent stations (location_type=1), una por estación física.
    Metrotren: los stops no tienen parent station ni location_type; se
    deduplicam por nombre (eliminando el sufijo de andén).
    """
    Path("data/dtpm-raw").mkdir(parents=True, exist_ok=True)

    if not RUTA_GTFS.exists():
        print(f"DESCARGANDO: {URL_GTFS}")
        urllib.request.urlretrieve(URL_GTFS, RUTA_GTFS)
        print(
            f"DESCARGADO: {RUTA_GTFS.name} ({RUTA_GTFS.stat().st_size / 1024:.1f} KB)"
        )
    else:
        print(f"LOCAL: {RUTA_GTFS.name}")

    with zipfile.ZipFile(RUTA_GTFS) as zf:
        routes = pd.read_csv(zf.open("routes.txt"), dtype=str)
        trips = pd.read_csv(zf.open("trips.txt"), dtype=str)
        stop_times = pd.read_csv(zf.open("stop_times.txt"), dtype=str)
        stops = pd.read_csv(zf.open("stops.txt"), dtype=str)

    # Separar rutas por modo: metro (subway=1) y metrotren (tram=0)
    rutas_metro = set(routes.loc[routes["route_type"] == "1", "route_id"])
    rutas_metrotren = set(routes.loc[routes["route_type"] == "0", "route_id"])
    print(f"GTFS: metro {sorted(rutas_metro)} | metrotren {sorted(rutas_metrotren)}")

    # Mapeo trip → route_id (solo rutas de interés)
    rutas_todas = rutas_metro | rutas_metrotren
    trip_route = dict(
        zip(
            trips.loc[trips["route_id"].isin(rutas_todas), "trip_id"],
            trips.loc[trips["route_id"].isin(rutas_todas), "route_id"],
        )
    )

    # stop_id → conjunto de route_ids que lo sirven
    from collections import defaultdict

    stop_rutas = defaultdict(set)
    mask = stop_times["trip_id"].isin(trip_route)
    for sid, tid in zip(
        stop_times.loc[mask, "stop_id"], stop_times.loc[mask, "trip_id"]
    ):
        stop_rutas[sid].add(trip_route[tid])

    stops_idx = stops.set_index("stop_id")
    partes = []

    # --- Metro: parent stations ---
    parents_metro = set()
    for sid in stop_rutas:
        if stop_rutas[sid] & rutas_metro:
            parent = stops_idx.at[sid, "parent_station"]
            if parent:
                parents_metro.add(parent)

    est_metro = stops_idx.loc[stops_idx.index.isin(parents_metro)].copy()
    est_metro["modo"] = "METRO"
    partes.append(est_metro)
    print(f"GTFS: {len(est_metro)} estaciones de metro")

    # --- Metrotren: stops sin parent, deduplicar por nombre ---
    sids_mt = [sid for sid in stop_rutas if stop_rutas[sid] & rutas_metrotren]
    est_mt = stops_idx.loc[stops_idx.index.isin(sids_mt)].copy()
    # Limpiar nombre: "Estación X (Anden1)" → "Estación X"
    est_mt["stop_name"] = est_mt["stop_name"].str.replace(
        r"\s*\(Anden\d+\)\s*$", "", regex=True
    )
    est_mt = est_mt.drop_duplicates(subset="stop_name")
    est_mt["modo"] = "METROTREN"
    partes.append(est_mt)
    print(f"GTFS: {len(est_mt)} estaciones de metrotren")

    estaciones = pd.concat(partes)
    estaciones["stop_lat"] = pd.to_numeric(estaciones["stop_lat"])
    estaciones["stop_lon"] = pd.to_numeric(estaciones["stop_lon"])

    gdf = gpd.GeoDataFrame(
        estaciones,
        geometry=gpd.points_from_xy(estaciones["stop_lon"], estaciones["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32719")

    gdf = gdf.reset_index().rename(
        columns={
            "stop_id": "Código paradero TS",
            "stop_code": "Código  paradero Usuario",
            "stop_name": "Nombre Paradero",
        }
    )
    gdf["Operación con Zona Paga"] = None
    gdf["Comuna"] = None
    gdf["Eje"] = None
    gdf["Desde ( Cruce 1)"] = None
    gdf["Hacia ( Cruce 2)"] = None

    return gdf[
        [
            "Código paradero TS",
            "Código  paradero Usuario",
            "Nombre Paradero",
            "Operación con Zona Paga",
            "Comuna",
            "Eje",
            "Desde ( Cruce 1)",
            "Hacia ( Cruce 2)",
            "modo",
            "geometry",
        ]
    ]


def procesar_paraderos():
    """Descarga el Excel de paraderos y lo convierte a Parquet con geometría.

    Coordenadas x/y en UTM Zone 19S (EPSG:32719). Deduplica por código TS.
    Incorpora estaciones de metro/metrotren desde el GTFS (no incluidas en
    el archivo de paraderos).
    """
    Path("data/dtpm-raw").mkdir(parents=True, exist_ok=True)

    if not RUTA_PARADEROS_RAW.exists():
        print(f"DESCARGANDO: {URL_PARADEROS}")
        urllib.request.urlretrieve(URL_PARADEROS, RUTA_PARADEROS_RAW)
        print(
            f"DESCARGADO: {RUTA_PARADEROS_RAW.name} ({RUTA_PARADEROS_RAW.stat().st_size / 1024:.1f} KB)"
        )
    else:
        print(f"LOCAL: {RUTA_PARADEROS_RAW.name}")

    if RUTA_PARADEROS_PARQUET.exists():
        print(f"OMITIDO: {RUTA_PARADEROS_PARQUET.name} ya existe.")
        return

    xls = pd.ExcelFile(RUTA_PARADEROS_RAW)
    hojas = []
    for hoja in xls.sheet_names:
        print(f"LEYENDO hoja: {hoja}")
        df_hoja = pd.read_excel(RUTA_PARADEROS_RAW, sheet_name=hoja, dtype=str)
        df_hoja["periodo"] = hoja
        hojas.append(df_hoja)

    df = pd.concat(hojas, ignore_index=True)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")  # 'POR DEFINIR' y similares → NaN
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    paraderos = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:32719"
    )
    print(f"Registros paraderos: {len(paraderos):,} | CRS: {paraderos.crs}")

    paraderos["modo"] = "BUS"
    cols = [
        "Código paradero TS",
        "Código  paradero Usuario",
        "Nombre Paradero",
        "Operación con Zona Paga",
        "Comuna",
        "Eje",
        "Desde ( Cruce 1)",
        "Hacia ( Cruce 2)",
        "modo",
        "geometry",
    ]
    paraderos = paraderos.drop_duplicates(subset="Código paradero TS")[cols]

    # Agregar estaciones de metro/metrotren desde GTFS
    estaciones = _estaciones_desde_gtfs()
    resultado = pd.concat([paraderos, estaciones], ignore_index=True)
    resultado = resultado.drop_duplicates(subset="Código paradero TS")

    print(f"Total (paraderos + estaciones): {len(resultado):,}")
    resultado.to_parquet(RUTA_PARADEROS_PARQUET, engine="pyarrow", compression="brotli")
    print(f"FINALIZADO: {RUTA_PARADEROS_PARQUET}")


# ---------------------------------------------------------------------------
# Consolidación
# ---------------------------------------------------------------------------


def _mapa_columnas(df):
    nuevo = "tiempo_inicio_viaje" in df.columns
    return {
        ("tiempo_inicio_viaje" if nuevo else "tiemposubida"): "tiempo",
        ("tiempo_fin_viaje" if nuevo else "tiempobajada"): "t_fin",
        ("factor_expansion" if nuevo else "factorexpansion"): "factor",
        ("paradero_inicio_viaje" if nuevo else "paraderosubida"): "paradero",
        ("paradero_fin_viaje" if nuevo else "paraderobajada"): "paradero_destino",
        ("comuna_inicio_viaje" if nuevo else "comunasubida"): "comuna",
        ("n_etapas" if nuevo else "netapa"): "n_etapas",
        ("distancia_ruta" if nuevo else "dviajeenruta_mts"): "dist_ruta",
    }


def _to_num(s, errors="coerce"):
    return pd.to_numeric(s, errors=errors)


def consolidar_anio(anio):
    """Normaliza el schema de un año y guarda en DIR_CONSOLIDADO.

    Lee desde dtpm-viajes/, unifica nombres de columnas, convierte tipos
    y escribe dtpm-{anio}.parquet. Idempotente: no hace nada si ya existe.
    """
    ruta_in = DIR_VIAJES / f"dtpm-{anio}.parquet"
    ruta_out = DIR_CONSOLIDADO / f"dtpm-{anio}.parquet"

    if ruta_out.exists():
        print(f"OMITIDO: {anio} ya consolidado.")
        return
    if not ruta_in.exists():
        print(f"OMITIDO: {anio} sin datos.")
        return

    print(f"CONSOLIDANDO: {anio} ...", end=" ", flush=True)
    DIR_CONSOLIDADO.mkdir(parents=True, exist_ok=True)
    df = dd.read_parquet(ruta_in)

    nuevo = "tiempo_inicio_viaje" in df.columns
    trans_cols = [c for c in (_TRANS_NEW if nuevo else _TRANS_OLD) if c in df.columns]
    metro_vals = _METRO_NEW if nuevo else _METRO_OLD

    df = df.rename(columns=_mapa_columnas(df))

    cols = [
        c
        for c in [
            "tiempo",
            "t_fin",
            "tipodia",
            "factor",
            "proposito",
            "paradero",
            "paradero_destino",
            "comuna",
            "n_etapas",
            "dist_ruta",
        ]
        if c in df.columns
    ] + trans_cols
    df = df[cols]

    if trans_cols:
        metro_mask = df[trans_cols[0]].isin(metro_vals)
        for col in trans_cols[1:]:
            metro_mask = metro_mask | df[col].isin(metro_vals)
        df["contiene_metro"] = metro_mask
    else:
        df["contiene_metro"] = False
    df = df.drop(columns=trans_cols)

    df["factor"] = (
        df["factor"]
        .map_partitions(_to_num, meta=("factor", "float64"))
        .fillna(0)
        .astype("float32")
    )
    df["tipodia"] = df["tipodia"].replace(TIPODIA_NORM)
    df["proposito"] = df["proposito"].replace(PROPOSITO_NORM)
    df["paradero_destino"] = df["paradero_destino"].replace({"-": None})

    df["ts_inicio"] = df["tiempo"].map_partitions(
        lambda s: pd.to_datetime(s, format="ISO8601", errors="coerce"),
        meta=("ts_inicio", "datetime64[ns]"),
    )
    df["ts_fin"] = df["t_fin"].map_partitions(
        lambda s: pd.to_datetime(s, format="ISO8601", errors="coerce"),
        meta=("ts_fin", "datetime64[ns]"),
    )
    df = df.drop(columns=["tiempo", "t_fin"])

    # Descartar filas sin timestamp de inicio (datos corruptos o schema incompatible)
    total = len(df)
    n_nulos = df["ts_inicio"].isnull().sum().compute()
    df = df[df["ts_inicio"].notnull()]
    print(f"({n_nulos:,} filas sin timestamp / {total:,} total) ", end="", flush=True)

    df["fecha"] = df["ts_inicio"].dt.strftime("%Y-%m-%d")
    df["anio"] = df["ts_inicio"].dt.year.astype("int16")
    df["dia_semana"] = df["ts_inicio"].dt.dayofweek.astype("int8")
    df["hora_inicio"] = df["ts_inicio"].dt.hour.astype("int8")
    df["minuto_inicio"] = df["ts_inicio"].dt.minute.astype("int8")
    df["hora_fin"] = df["ts_fin"].dt.hour.astype("float32")
    df["tviaje_min"] = (
        (df["ts_fin"] - df["ts_inicio"]).dt.total_seconds() / 60
    ).astype("float32")

    if "dist_ruta" in df.columns:
        df["dist_ruta_mts"] = (
            df["dist_ruta"]
            .map_partitions(_to_num, meta=("dist_ruta_mts", "float64"))
            .astype("float32")
        )
        df = df.drop(columns=["dist_ruta"])

    if "n_etapas" in df.columns:
        df["n_etapas"] = (
            df["n_etapas"]
            .map_partitions(_to_num, meta=("n_etapas", "float64"))
            .fillna(0)
            .astype("int8")
        )

    orden = [
        "anio",
        "fecha",
        "dia_semana",
        "tipodia",
        "ts_inicio",
        "ts_fin",
        "hora_inicio",
        "minuto_inicio",
        "hora_fin",
        "tviaje_min",
        "dist_ruta_mts",
        "factor",
        "proposito",
        "contiene_metro",
        "paradero",
        "paradero_destino",
        "comuna",
        "n_etapas",
    ]
    df = df[[c for c in orden if c in df.columns]]

    df.to_parquet(ruta_out, engine="pyarrow", compression="brotli")
    print(f"OK ({ruta_out.stat().st_size / 1e6:.0f} MB)")
