# %%
"""Preprocesamiento de la clase espacio-temporal: el Metro como experimento natural.

Este script SOLO preprocesa los datos. Los cálculos del análisis (clasificar
tratamiento/control, sustitución de modo, gravedad, etc.) viven en
`13-analisis-espacio-temporal.py`.

Qué hace el preprocesamiento:

1. Lee la base DTPM consolidada (días laborales, 2014-2019). La base trae, por
   viaje, si usó Metro y si usó bus (`contiene_metro`, `contiene_bus`), derivados
   por etapa en `gdsutils/dtpm.py`. Ojo: el Metro está SIEMPRE en la base (un
   45-58% de los viajes lo usan); el flag estaba mal calculado y se corrigió.
2. Normaliza por fecha laboral: la encuesta DTPM abarca de una a tres semanas
   tipo según el año. Se divide `factor` por el número de fechas laborales para
   dejar el volumen de un día representativo, comparable entre años.
3. Deja una submuestra manejable de viajes (`viajes-dtpm.parquet`) con su modo,
   tiempo, distancia, hora, propósito y número de etapas, más los paraderos de
   origen y destino. El análisis clasifica y agrega sobre esta submuestra.

Salidas: `viajes-dtpm.parquet`, `paraderos.parquet`, `metro-estaciones.parquet`,
`comunas.parquet`.
"""

# %%
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box


def norm(s):
    """Texto en MAYÚSCULAS sin acentos (para cruzar nombres de estación)."""
    return "".join(c for c in unicodedata.normalize("NFD", str(s).upper())
                   if unicodedata.category(c) != "Mn").strip()

from gdsutils.dtpm import RUTA_GTFS, URL_GTFS

# %% Configuración
NOMBRE_DATASET = "lineas-metro"
DIR_SALIDA = Path("data") / NOMBRE_DATASET

DIR_CONSOLIDADO = Path("data") / "dtpm-consolidado"
RUTA_PARADEROS = Path("data") / "dtpm-paraderos.parquet"
RUTA_COMUNAL = Path("data") / "censo2024-cartografia" / "Cartografia_censo2024_Pais_Comunal.parquet"

BBOX_URBANO = (-70.85, -33.65, -70.45, -33.30)
ANIOS = list(range(2014, 2020))  # 2014-2019: antes y después de L6 (2017) y L3 (2019)
N_MUESTRA = 300_000              # submuestra de viajes por año
SEMILLA = 42


def extraer_metro_estaciones():
    """Estaciones de Metro del GTFS, marcando las nuevas (solo L3/L6).

    Una estación es `nueva` si sirve únicamente a las Líneas 3 (2019) o 6 (2017);
    los intercambios con líneas viejas ya existían antes. Se usa para clasificar
    qué paraderos ganaron acceso a Metro (tratamiento del experimento natural).
    """
    if not RUTA_GTFS.exists():
        RUTA_GTFS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_GTFS, RUTA_GTFS)
    with zipfile.ZipFile(RUTA_GTFS) as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        stops = pd.read_csv(z.open("stops.txt"), dtype=str)
        stop_times = pd.read_csv(z.open("stop_times.txt"), dtype=str)
    rutas_metro = set(routes.loc[routes["route_type"] == "1", "route_id"])
    nuevas = {"L3", "L6"}
    trip2route = dict(zip(trips["trip_id"], trips["route_id"]))
    st = stop_times.assign(linea=stop_times["trip_id"].map(trip2route))
    st = st[st["linea"].isin(rutas_metro)]
    lineas_por_estacion = st.groupby("stop_id")["linea"].agg(lambda s: set(s))
    est = stops[stops["stop_id"].isin(lineas_por_estacion.index)].drop_duplicates("stop_id").copy()
    est["lineas"] = est["stop_id"].map(lineas_por_estacion)
    est = gpd.GeoDataFrame(
        est, geometry=[Point(float(x), float(y)) for x, y in zip(est.stop_lon, est.stop_lat)],
        crs="EPSG:4326")
    est["nueva"] = est["lineas"].apply(lambda s: bool(s) and s <= nuevas)
    est["estacion"] = est["stop_name"].str.replace(r" Dirección.*", "", regex=True)
    est = est.drop_duplicates("estacion")[["estacion", "nueva", "geometry"]]
    print(f"  Metro: {len(est)} estaciones, {est['nueva'].sum()} nuevas (solo L3/L6)")
    return est.reset_index(drop=True)


def extraer_metro_lineas():
    """Trazado de cada línea de Metro (GTFS), con la marca de nueva (L3/L6)."""
    if not RUTA_GTFS.exists():
        RUTA_GTFS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_GTFS, RUTA_GTFS)
    with zipfile.ZipFile(RUTA_GTFS) as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips = pd.read_csv(z.open("trips.txt"), dtype=str)
        shapes = pd.read_csv(z.open("shapes.txt"), dtype=str)
    for c in ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]:
        shapes[c] = pd.to_numeric(shapes[c])
    rutas_metro = set(routes.loc[routes["route_type"] == "1", "route_id"])
    shape2route = dict(zip(trips["shape_id"], trips["route_id"]))
    filas = []
    for sid, g in shapes.sort_values("shape_pt_sequence").groupby("shape_id"):
        rid = shape2route.get(sid)
        if rid in rutas_metro and len(g) > 1:
            filas.append({"linea": rid, "geometry": LineString(list(zip(g.shape_pt_lon, g.shape_pt_lat)))})
    lineas = gpd.GeoDataFrame(filas, crs="EPSG:4326")
    lineas["_l"] = lineas.to_crs(32719).length
    lineas = lineas.sort_values("_l").groupby("linea").tail(1)[["linea", "geometry"]].reset_index(drop=True)
    lineas["nueva"] = lineas["linea"].isin({"L3", "L6"})
    print(f"  Líneas de Metro: {sorted(lineas['linea'])}")
    return lineas


# %%
if __name__ == "__main__":
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    # ===== 1. Metadatos: estaciones de Metro, paraderos y comunas =====
    print("[1/3] Estaciones de Metro, paraderos y comunas")
    estaciones = extraer_metro_estaciones()
    estaciones.to_parquet(DIR_SALIDA / "metro-estaciones.parquet")
    extraer_metro_lineas().to_parquet(DIR_SALIDA / "metro-lineas.parquet")

    comunal = gpd.read_parquet(RUTA_COMUNAL).to_crs("EPSG:4326").rename_geometry("geometry")
    comunas = comunal.clip(box(*BBOX_URBANO))[["COMUNA", "geometry"]]
    comunas.to_parquet(DIR_SALIDA / "comunas.parquet")
    print(f"  Comunas del Gran Santiago: {len(comunas)}")

    # Paraderos = paradas de bus (código) + estaciones de Metro (nombre). Los
    # viajes en Metro usan el NOMBRE de la estación como código de paradero
    # ("TOBALABA"), no un código de parada; sin esto, los viajes de Metro no se
    # ubicarían en el mapa. Se normaliza el nombre para cruzarlo.
    par = gpd.read_parquet(RUTA_PARADEROS).to_crs("EPSG:4326")
    bus_par = pd.DataFrame({
        "paradero": par["Código paradero TS"].astype(str),
        "lon": par.geometry.x, "lat": par.geometry.y,
        "comuna": par["Comuna"].astype(str), "es_metro": False, "nueva": False,
    })
    est_c = estaciones.sjoin(comunal[["COMUNA", "geometry"]], how="left", predicate="within")
    metro_par = pd.DataFrame({
        "paradero": est_c["estacion"].map(norm),
        "lon": est_c.geometry.x, "lat": est_c.geometry.y,
        "comuna": est_c["COMUNA"], "es_metro": True, "nueva": est_c["nueva"],
    })
    paraderos = (pd.concat([bus_par, metro_par], ignore_index=True)
                 .dropna(subset=["lon", "lat"]).drop_duplicates("paradero"))
    paraderos.to_parquet(DIR_SALIDA / "paraderos.parquet", index=False)
    print(f"  Paraderos: {len(bus_par):,} de bus + {len(metro_par)} de Metro")

    # ===== 2. Submuestra de viajes con modo (días laborales, por fecha) =====
    print("[2/3] Submuestra de viajes DTPM con modo (días laborales)")
    rng = np.random.default_rng(SEMILLA)
    cols = ["tipodia", "fecha", "factor", "paradero", "paradero_destino", "contiene_metro",
            "contiene_bus", "n_etapas", "tviaje_min", "dist_ruta_mts", "hora_inicio", "proposito"]
    filas = []
    for anio in ANIOS:
        ruta = DIR_CONSOLIDADO / f"dtpm-{anio}.parquet"
        if not ruta.exists():
            print(f"  {anio}: sin datos")
            continue
        df = pd.read_parquet(ruta, columns=cols)
        df = df[df["tipodia"] == "LABORAL"]
        n_fechas = df["fecha"].nunique()
        df["factor"] = df["factor"] / n_fechas  # día representativo
        if len(df) > N_MUESTRA:
            df = df.iloc[rng.choice(len(df), N_MUESTRA, replace=False)]
        df = df.assign(anio=anio).drop(columns=["tipodia", "fecha"])
        filas.append(df)
        print(f"  {anio}: {n_fechas} fechas | submuestra {len(df):,} | "
              f"metro {df['contiene_metro'].mean() * 100:.0f}% | bus {df['contiene_bus'].mean() * 100:.0f}%")
    viajes = pd.concat(filas, ignore_index=True)

    # ===== 3. Guardar, empaquetar, (opcional) subir =====
    print("[3/3] Guardando y empaquetando")
    viajes.to_parquet(DIR_SALIDA / "viajes-dtpm.parquet", index=False)
    print(f"  viajes-dtpm {len(viajes):,} filas, {viajes.memory_usage(deep=True).sum() / 1e6:.0f} MB en memoria")
    config.publicar(NOMBRE_DATASET, DIR_SALIDA)
