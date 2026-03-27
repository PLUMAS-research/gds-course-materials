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
"""

import csv
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import pandas as pd

DIR_VIAJES = Path("data") / "dtpm-viajes"
DIR_CONSOLIDADO = Path("data") / "dtpm-consolidado"
RUTA_PARADEROS_RAW = Path("data") / "dtpm-raw" / "paraderos.xlsx"
RUTA_PARADEROS_PARQUET = Path("data") / "dtpm-paraderos.parquet"

URL_PARADEROS = "https://www.dtpm.cl/descargas/pops26/2026-03-21_consolidado_Registro-Paradas_anual.xlsx"

# Año → lista de URLs. Algunos años tienen archivos separados por tipo de día.
MAPEO_DTPM = {
    "2014": ["https://www.dtpm.cl/descargas/tablas/viajes201405_transparencia.rar"],
    "2015": ["https://www.dtpm.cl/descargas/tablas/viajes201504_transparencia.rar"],
    "2016": ["https://www.dtpm.cl/descargas/tablas/viajes201605_transparencia.rar"],
    "2017": [
        "https://www.dtpm.cl/descargas/tablas/viajes201704_laboral_transparencia.rar"
    ],
    "2018": [
        "https://www.dtpm.cl/descargas/tablas/viajes201804_laboral_transparencia.rar"
    ],
    "2019": ["https://www.dtpm.cl/descargas/tablas/tabla-viajes.rar"],
    "2020": [
        # Semana del 9–12 mar (lun–jue) + dom 8 mar. Viernes ausente del dataset fuente.
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202003_laboral_transparencia.zip",
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202003_sab_dom_transparencia.zip",
    ],
    "2021": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202104_transparencia.zip"
    ],
    "2022": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202204_abril_4al10_transparencia-.zip"
    ],
    "2023": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes_042023_17al23_transparencia.zip"
    ],
    "2024": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/viajes202404_transparencia_15al21.zip"
    ],
    "2025": [
        "https://www.dtpm.cl/descargas/modelos_y_matrices/Tabla-de-viajes-011025.zip"
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


# ---------------------------------------------------------------------------
# Descarga y extracción
# ---------------------------------------------------------------------------


def detectar_separador(ruta_archivo, encoding="latin-1"):
    """Detecta el delimitador de un CSV analizando sus primeras líneas."""
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
    """Descarga los archivos comprimidos de viajes para un año."""
    (Path("data") / "dtpm-raw").mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls):
        ext = url.split(".")[-1].lower()
        ruta = _ruta_raw(anio, idx, ext)
        if not ruta.exists():
            print(f"DESCARGANDO: {url}")
            urllib.request.urlretrieve(url, ruta)


def procesar_viajes(anio, urls):
    """Extrae y convierte a Parquet los viajes de un año.

    Si el Parquet ya existe, no hace nada. Descarga los archivos fuente
    si no están en disco.
    """
    DIR_VIAJES.mkdir(parents=True, exist_ok=True)
    ruta_parquet = DIR_VIAJES / f"dtpm-{anio}.parquet"

    if ruta_parquet.exists():
        print(f"OMITIDO: {anio} ya procesado.")
        return

    rutas_raw = []
    for idx, url in enumerate(urls):
        ext = url.split(".")[-1].lower()
        ruta = _ruta_raw(anio, idx, ext)
        if ruta.exists() and ruta.stat().st_size < 1024:
            print(f"CORRUPTO: {ruta.name}, se eliminará y reintentará.")
            ruta.unlink()
        if not ruta.exists():
            print(f"DESCARGANDO: {url}")
            urllib.request.urlretrieve(url, ruta)
            print(f"DESCARGADO: {ruta.name} ({ruta.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"LOCAL: {ruta.name}")
        rutas_raw.append(ruta)

    with tempfile.TemporaryDirectory() as dir_temporal:
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

        archivos_datos = list(path_temp.rglob("*.csv")) + list(path_temp.rglob("*.txt"))
        if not archivos_datos:
            print(f"ERROR: No se hallaron datos para el año {anio}")
            return

        separador = detectar_separador(archivos_datos[0])
        print(
            f"CONVIRTIENDO: {len(archivos_datos)} archivo(s) → dtpm-{anio}.parquet (sep='{separador}')"
        )

        df = dd.read_csv(
            archivos_datos,
            sep=separador,
            encoding="latin-1",
            dtype=str,
            on_bad_lines="skip",
        )
        df.to_parquet(ruta_parquet, engine="pyarrow", compression="brotli")
        print(f"FINALIZADO: {ruta_parquet}")


def procesar_paraderos():
    """Descarga el Excel de paraderos y lo convierte a Parquet con geometría.

    Coordenadas x/y en UTM Zone 19S (EPSG:32719). Deduplica por código TS.
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
    print(f"Registros: {len(paraderos):,} | CRS: {paraderos.crs}")

    paraderos.drop_duplicates(subset="Código paradero TS")[
        [
            "Código paradero TS",
            "Código  paradero Usuario",
            "Nombre Paradero",
            "Operación con Zona Paga",
            "Comuna",
            "Eje",
            "Desde ( Cruce 1)",
            "Hacia ( Cruce 2)",
            "geometry",
        ]
    ].to_parquet(RUTA_PARADEROS_PARQUET, engine="pyarrow", compression="brotli")
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
        lambda s: pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce"),
        meta=("ts_inicio", "datetime64[ns]"),
    )
    df["ts_fin"] = df["t_fin"].map_partitions(
        lambda s: pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce"),
        meta=("ts_fin", "datetime64[ns]"),
    )
    df = df.drop(columns=["tiempo", "t_fin"])

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
