"""Procesamiento de redes urbanas a partir de geometrías y tablas GTFS.

Funciones puras: reciben DataFrames y GeoDataFrames ya cargados, devuelven
grafos NetworkX o estructuras tabulares. La descarga de datos vive en los
scripts (clase 09) o en módulos específicos (`gdsutils.dtpm` para GTFS,
`gdsutils.general.descargar_datos` para artefactos del curso).

Las funciones devuelven `(grafo, nodos_gdf, aristas_gdf)` para que el
usuario pueda trabajar sobre el grafo (networkx) y sobre las geometrías
(geopandas) simultáneamente.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring

_TEMPLATES_DIR = Path(__file__).parent / "templates"

CRS_METRICO = "EPSG:32719"


def _clusterizar_coords(coords: np.ndarray, tolerancia: float) -> np.ndarray:
    """Asigna a cada coordenada un id de cluster: coordenadas a distancia
    menor o igual a `tolerancia` quedan en el mismo cluster.

    Usa cKDTree + union-find (no transitivo: una cadena de coords cada
    una a `tolerancia` de la anterior queda en el mismo cluster). Es más
    robusto que redondear a una grilla porque no sufre artefactos de
    borde.

    Parámetros
    ----------
    coords : array (n, 2) de coordenadas en alguna unidad métrica.
    tolerancia : distancia máxima en la misma unidad.

    Retorna
    -------
    array (n,) de ids consecutivos 0..k-1.
    """
    n = len(coords)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if tolerancia > 0:
        tree = cKDTree(coords)
        pares = tree.query_pairs(r=tolerancia, output_type="ndarray")
        for a, b in pares:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb

    raices = np.array([find(i) for i in range(n)])
    _, ids_consecutivos = np.unique(raices, return_inverse=True)
    return ids_consecutivos


def construir_grafo_desde_lineas(
    gdf_lineas: gpd.GeoDataFrame,
    tolerancia_metros: float = 1.0,
    tolerancia_endpoints_metros: float | None = None,
    crs_metrico: str = CRS_METRICO,
    dirigido: bool = False,
) -> tuple[nx.Graph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convierte un GeoDataFrame de LineStrings en un grafo NetworkX.

    Las ways OSM no siempre se publican partidas en intersecciones: una
    misma LineString puede atravesar varias intersecciones internas. En
    catálogos cartográficos como MINVU, además, dos segmentos
    contiguos pueden tener endpoints separados por unos metros por
    errores de digitalización. Esta función:

        1. Clusteriza todas las coordenadas con tolerancia
           `tolerancia_metros` (KDTree + union-find). Dos coords más
           cercanas que esa tolerancia quedan en el mismo cluster.
        2. Detecta intersecciones: clusters compartidos por dos o más
           líneas.
        3. Parte cada LineString en sus puntos-intersección. Los
           endpoints siempre son nodos.
        4. Opcionalmente, fusiona aún más nodos terminales cercanos con
           `tolerancia_endpoints_metros` para cerrar conexiones que la
           cartografía dejó separadas.

    Parámetros
    ----------
    gdf_lineas : GeoDataFrame con LineString o MultiLineString. CRS
        arbitrario.
    tolerancia_metros : distancia máxima para considerar que dos
        coordenadas pertenecen al mismo nodo (snap general). Más alto
        agrupa más. Para OSM calles 1-2 m; para catálogos oficiales con
        ruido cartográfico 5-15 m.
    tolerancia_endpoints_metros : si no es None, aplica un segundo
        clustering más permisivo SOLO sobre nodos que actúan como
        endpoint (sin nodos internos). Útil para reducir componentes
        falsamente desconectados en catálogos de ciclovías. Default
        None (no aplicar).
    crs_metrico : CRS proyectado en metros para clustering y largos.
    dirigido : si True, MultiDiGraph respetando la dirección de la
        LineString.

    Retorna
    -------
    grafo : nx.MultiGraph o nx.MultiDiGraph con `largo_m` por arista.
    nodos : GeoDataFrame (`node_id`, geometry=Point) en CRS original.
    aristas : GeoDataFrame (`u`, `v`, `largo_m`, geometry, atributos).
    """
    if gdf_lineas.empty:
        raise ValueError("gdf_lineas está vacío.")
    crs_origen = gdf_lineas.crs
    gdf = gdf_lineas.copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf_m = gdf.to_crs(crs_metrico)

    # Paso 1: recolectar todas las coordenadas con índice global.
    coords_globales = []
    rangos_por_linea = []
    for geom in gdf_m.geometry:
        inicio = len(coords_globales)
        for c in geom.coords:
            coords_globales.append((c[0], c[1]))
        rangos_por_linea.append((inicio, len(coords_globales)))
    coords_globales = np.array(coords_globales, dtype=float)

    # Paso 2: clusterizar.
    ids_cluster = _clusterizar_coords(coords_globales, tolerancia_metros)

    coords_por_linea_id = [
        list(ids_cluster[ini:fin]) for ini, fin in rangos_por_linea
    ]
    coords_por_linea_orig = [
        [tuple(c) for c in coords_globales[ini:fin]]
        for ini, fin in rangos_por_linea
    ]

    # Paso 3: detectar intersecciones. Un cluster es nodo si aparece en
    # 2+ líneas distintas, o si es endpoint de alguna línea.
    lineas_por_cluster: dict[int, set[int]] = defaultdict(set)
    for idx_linea, ids in enumerate(coords_por_linea_id):
        for cid in ids:
            lineas_por_cluster[int(cid)].add(idx_linea)

    es_nodo: set[int] = set()
    for ids in coords_por_linea_id:
        es_nodo.add(int(ids[0]))
        es_nodo.add(int(ids[-1]))
    for cid, lineas in lineas_por_cluster.items():
        if len(lineas) >= 2:
            es_nodo.add(cid)

    # Paso 4 (opcional): fusión adicional de endpoints terminales por
    # ruido cartográfico. Sólo aplica a clusters que son endpoint de
    # exactamente una línea y no intersección.
    cid_a_nodo = {cid: cid for cid in es_nodo}
    if tolerancia_endpoints_metros is not None and tolerancia_endpoints_metros > 0:
        endpoints_solo = []
        cids_endpoints = []
        for cid in es_nodo:
            lineas = lineas_por_cluster[cid]
            es_endpoint_terminal = len(lineas) == 1
            if es_endpoint_terminal:
                # cualquier coord representativa del cluster
                idx_rep = int(np.where(ids_cluster == cid)[0][0])
                endpoints_solo.append(coords_globales[idx_rep])
                cids_endpoints.append(cid)
        if len(endpoints_solo) > 1:
            endpoints_arr = np.array(endpoints_solo)
            sub_ids = _clusterizar_coords(
                endpoints_arr, tolerancia_endpoints_metros
            )
            # Mapear cada cid_endpoint a un cid representativo del
            # subcluster (el primero que aparezca).
            rep_por_sub: dict[int, int] = {}
            for cid, sub in zip(cids_endpoints, sub_ids):
                sub_int = int(sub)
                if sub_int not in rep_por_sub:
                    rep_por_sub[sub_int] = cid
                cid_a_nodo[cid] = rep_por_sub[sub_int]

    # Renumerar a ids consecutivos.
    nodos_finales = sorted(set(cid_a_nodo.values()))
    nodo_consecutivo = {n: i for i, n in enumerate(nodos_finales)}

    def cluster_a_nodo(cid: int) -> int | None:
        if cid not in cid_a_nodo:
            return None
        return nodo_consecutivo[cid_a_nodo[cid]]

    # Coordenada representativa por nodo final.
    coord_orig_por_nodo: dict[int, tuple[float, float]] = {}
    for ids, coords in zip(coords_por_linea_id, coords_por_linea_orig):
        for cid, c in zip(ids, coords):
            nodo = cluster_a_nodo(int(cid))
            if nodo is not None:
                coord_orig_por_nodo.setdefault(nodo, c)

    atrs_originales = [c for c in gdf.columns if c != "geometry"]
    filas_aristas = []
    for idx_linea, (ids, coords) in enumerate(
        zip(coords_por_linea_id, coords_por_linea_orig)
    ):
        atrs = {k: gdf.iloc[idx_linea][k] for k in atrs_originales}
        u_actual = cluster_a_nodo(int(ids[0]))
        segmento_orig = [coords[0]]
        for cid, c in zip(ids[1:], coords[1:]):
            segmento_orig.append(c)
            v = cluster_a_nodo(int(cid))
            if v is None:
                continue
            if u_actual is None:
                u_actual = v
                segmento_orig = [c]
                continue
            if v != u_actual and len(segmento_orig) >= 2:
                geom_seg_m = LineString(segmento_orig)
                fila = {
                    "u": u_actual,
                    "v": v,
                    "largo_m": geom_seg_m.length,
                    "geometry_m": geom_seg_m,
                    **atrs,
                }
                filas_aristas.append(fila)
                u_actual = v
                segmento_orig = [c]

    if not filas_aristas:
        raise ValueError("No se generaron aristas tras la partición.")

    aristas = gpd.GeoDataFrame(filas_aristas, geometry="geometry_m", crs=crs_metrico)
    aristas = aristas.to_crs(crs_origen)
    aristas = aristas.rename(columns={"geometry_m": "geometry"}).set_geometry("geometry")

    ids_ordenados = sorted(coord_orig_por_nodo)
    nodos_geom_m = [Point(coord_orig_por_nodo[i]) for i in ids_ordenados]
    nodos = gpd.GeoDataFrame(
        {"node_id": ids_ordenados},
        geometry=nodos_geom_m,
        crs=crs_metrico,
    ).to_crs(crs_origen)

    # x, y de los nodos del grafo en el CRS de salida (consistente con
    # las geometrías de aristas y de nodos en el GeoDataFrame).
    xy_por_nodo = {
        int(nid): (float(pt.x), float(pt.y))
        for nid, pt in zip(nodos["node_id"], nodos.geometry)
    }

    ClaseGrafo = nx.MultiDiGraph if dirigido else nx.MultiGraph
    grafo = ClaseGrafo()
    for nid, (x, y) in xy_por_nodo.items():
        grafo.add_node(nid, x=x, y=y)

    atrs_arista = [c for c in aristas.columns if c not in ("u", "v")]
    for fila in aristas.itertuples(index=False):
        d = {k: getattr(fila, k) for k in atrs_arista}
        grafo.add_edge(int(fila.u), int(fila.v), **d)

    return grafo, nodos, aristas


def red_paraderos_desde_gtfs(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes: pd.DataFrame,
    modos: tuple[int, ...] = (3,),
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Construye una red de paraderos conectando pares consecutivos en GTFS.

    Conecta paraderos consecutivos en cada `trip`. El peso por arista
    (`servicios`) es el número de rutas distintas que recorren ese tramo
    en al menos un sentido.

    Parámetros
    ----------
    stops, stop_times, trips, routes : DataFrames con la estructura
        estándar de GTFS (columnas tipadas como string).
    modos : tipos de servicio GTFS a incluir. Por defecto bus (3). En
        GTFS estándar: 0=tranvía, 1=metro, 2=tren, 3=bus.

    Retorna
    -------
    grafo : nx.MultiDiGraph con paraderos como nodos.
    nodos, aristas : GeoDataFrames con la geometría de la red.
    """
    routes = routes[routes["route_type"].astype(int).isin(modos)]
    trips = trips.merge(routes[["route_id"]], on="route_id")
    stop_times = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id")

    stop_times = stop_times.copy()
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    stop_times["next_stop"] = stop_times.groupby("trip_id")["stop_id"].shift(-1)
    pares = stop_times.dropna(subset=["next_stop"])

    aristas_df = (
        pares.groupby(["stop_id", "next_stop"])["route_id"]
        .nunique()
        .reset_index(name="servicios")
        .rename(columns={"stop_id": "u_id", "next_stop": "v_id"})
    )

    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    nodos = gpd.GeoDataFrame(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )

    ids_validos = set(nodos["stop_id"])
    aristas_df = aristas_df[
        aristas_df["u_id"].isin(ids_validos) & aristas_df["v_id"].isin(ids_validos)
    ]

    coords_por_id = dict(zip(nodos["stop_id"], nodos.geometry))
    geom_lineas = [
        LineString([coords_por_id[u], coords_por_id[v]])
        for u, v in zip(aristas_df["u_id"], aristas_df["v_id"])
    ]
    aristas = gpd.GeoDataFrame(aristas_df, geometry=geom_lineas, crs="EPSG:4326")

    grafo = nx.MultiDiGraph()
    for fila in nodos.itertuples(index=False):
        grafo.add_node(
            fila.stop_id,
            x=float(fila.stop_lon),
            y=float(fila.stop_lat),
            nombre=fila.stop_name,
        )
    for fila in aristas.itertuples(index=False):
        grafo.add_edge(fila.u_id, fila.v_id, servicios=int(fila.servicios))

    return grafo, nodos, aristas


def geometrias_aristas_desde_shapes_gtfs(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    shapes: pd.DataFrame,
    crs_metrico: str = CRS_METRICO,
) -> dict[tuple[str, str], LineString]:
    """Calcula la geometría real de cada arista (par de paraderos consecutivos)
    siguiendo las shapes del GTFS.

    Sin esta corrección, las aristas se dibujan como líneas rectas entre
    paraderos y la red parece errática (los buses cortan calles en
    diagonal en el mapa). Con shapes, cada arista sigue el recorrido
    real del bus.

    Algoritmo:
        1. Construye un LineString por cada shape_id (en `crs_metrico`).
        2. Para cada trip, proyecta cada paradero sobre la shape de su
           trip para obtener la distancia recorrida.
        3. Entre paraderos consecutivos, extrae el substring de la
           shape entre las dos distancias proyectadas. Esa es la
           geometría real del tramo.
        4. Para cada par (u, v), conserva la primera geometría
           encontrada (puede haber varias trips usando shapes distintas
           para el mismo par).

    Las proyecciones se cachean por (shape_id, stop_id) para no repetir
    cálculos cuando varios trips comparten shape.

    Retorna
    -------
    dict {(stop_a, stop_b): LineString WGS84}.
    """
    stops = stops.copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops_gdf = gpd.GeoDataFrame(
        stops[["stop_id"]],
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(crs_metrico)
    stop_pt = {
        sid: Point(x, y)
        for sid, x, y in zip(
            stops_gdf["stop_id"], stops_gdf.geometry.x, stops_gdf.geometry.y
        )
    }

    shapes = shapes.copy()
    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(int)
    shapes = shapes.dropna(subset=["shape_pt_lat", "shape_pt_lon"])
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    shapes_gdf = gpd.GeoDataFrame(
        shapes,
        geometry=gpd.points_from_xy(shapes["shape_pt_lon"], shapes["shape_pt_lat"]),
        crs="EPSG:4326",
    ).to_crs(crs_metrico)

    shape_lines: dict[str, LineString] = {}
    for sid, grp in shapes_gdf.groupby("shape_id"):
        coords = list(zip(grp.geometry.x, grp.geometry.y))
        if len(coords) >= 2:
            shape_lines[sid] = LineString(coords)

    trip_to_shape = dict(zip(trips["trip_id"], trips["shape_id"]))

    stop_times = stop_times.copy()
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    proyecciones_cache: dict[tuple[str, str], float] = {}
    edge_geoms_m: dict[tuple[str, str], LineString] = {}

    for trip_id, grp in stop_times.groupby("trip_id"):
        shape_id = trip_to_shape.get(trip_id)
        if shape_id not in shape_lines:
            continue
        shape = shape_lines[shape_id]
        stops_seq = grp["stop_id"].tolist()
        dists = []
        for sid in stops_seq:
            if sid not in stop_pt:
                dists.append(None)
                continue
            key = (shape_id, sid)
            if key not in proyecciones_cache:
                proyecciones_cache[key] = shape.project(stop_pt[sid])
            dists.append(proyecciones_cache[key])

        for i in range(len(stops_seq) - 1):
            a, b = stops_seq[i], stops_seq[i + 1]
            d_a, d_b = dists[i], dists[i + 1]
            if d_a is None or d_b is None or (a, b) in edge_geoms_m:
                continue
            if d_b <= d_a:
                # Stop B aparece antes que A en la shape (ruido en la
                # cartografía o stops fuera de orden); saltar.
                continue
            try:
                seg = substring(shape, d_a, d_b)
            except Exception:
                continue
            if seg.length < 1e-3:
                continue
            edge_geoms_m[(a, b)] = seg

    # Reproyectar a WGS84.
    if not edge_geoms_m:
        return {}
    aristas_gdf_m = gpd.GeoDataFrame(
        {"u": [k[0] for k in edge_geoms_m], "v": [k[1] for k in edge_geoms_m]},
        geometry=list(edge_geoms_m.values()),
        crs=crs_metrico,
    ).to_crs("EPSG:4326")
    return {
        (u, v): geom
        for u, v, geom in zip(aristas_gdf_m["u"], aristas_gdf_m["v"], aristas_gdf_m.geometry)
    }


def red_od_desde_viajes(
    viajes: pd.DataFrame,
    geo_zonas: gpd.GeoDataFrame,
    col_origen: str,
    col_destino: str,
    col_zona: str,
    col_peso: str | None = None,
    umbral_minimo: float = 0.0,
) -> tuple[nx.DiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Construye una red OD: zonas como nodos, viajes como aristas.

    Parámetros
    ----------
    viajes : DataFrame con columnas `col_origen`, `col_destino` que
        identifican zonas, y opcionalmente `col_peso` con un factor de
        expansión o frecuencia. Si `col_peso` es None, se cuenta cada
        viaje como 1.
    geo_zonas : GeoDataFrame con la geometría de cada zona y la columna
        `col_zona` identificadora.
    umbral_minimo : descarta aristas con peso menor a este valor para
        evitar ruido. Default 0.

    Retorna
    -------
    grafo : nx.DiGraph dirigido, nodos = zonas, aristas = viajes
        agregados, atributo `peso`.
    nodos, aristas : GeoDataFrames con la geometría (centroides de zona).
    """
    if col_peso is None:
        agregados = (
            viajes.groupby([col_origen, col_destino]).size().reset_index(name="peso")
        )
    else:
        agregados = (
            viajes.groupby([col_origen, col_destino])[col_peso]
            .sum()
            .reset_index(name="peso")
        )
    agregados = agregados[agregados["peso"] > umbral_minimo]

    geo_zonas = geo_zonas.copy()
    geo_zonas["_centroide"] = geo_zonas.geometry.representative_point()
    centroides = dict(zip(geo_zonas[col_zona], geo_zonas["_centroide"]))
    geo_zonas = geo_zonas.drop(columns="_centroide")

    agregados = agregados[
        agregados[col_origen].isin(centroides) & agregados[col_destino].isin(centroides)
    ]
    geom_lineas = [
        LineString([centroides[o], centroides[d]])
        for o, d in zip(agregados[col_origen], agregados[col_destino])
    ]
    aristas = gpd.GeoDataFrame(
        agregados.rename(columns={col_origen: "u", col_destino: "v"}),
        geometry=geom_lineas,
        crs=geo_zonas.crs,
    )

    ids_usados = set(aristas["u"]).union(aristas["v"])
    nodos = geo_zonas[geo_zonas[col_zona].isin(ids_usados)].copy()
    nodos = nodos.rename(columns={col_zona: "node_id"})
    nodos.geometry = nodos.geometry.representative_point()

    grafo = nx.DiGraph()
    for fila in nodos.itertuples(index=False):
        grafo.add_node(fila.node_id, x=fila.geometry.x, y=fila.geometry.y)
    for fila in aristas.itertuples(index=False):
        grafo.add_edge(fila.u, fila.v, peso=float(fila.peso))

    return grafo, nodos, aristas


def grafo_a_geodataframe(
    grafo: nx.Graph, crs: str = "EPSG:4326"
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Vuelca un grafo NetworkX a (nodos_gdf, aristas_gdf).

    Los nodos deben tener atributos `x`, `y` en el CRS indicado.
    """
    nodos = gpd.GeoDataFrame(
        {"node_id": list(grafo.nodes)},
        geometry=[Point(grafo.nodes[n]["x"], grafo.nodes[n]["y"]) for n in grafo.nodes],
        crs=crs,
    )
    coords = {n: (grafo.nodes[n]["x"], grafo.nodes[n]["y"]) for n in grafo.nodes}

    es_multi = grafo.is_multigraph()
    aristas_filas = []
    geoms = []
    if es_multi:
        iterador = grafo.edges(keys=True, data=True)
        for u, v, k, d in iterador:
            geoms.append(LineString([coords[u], coords[v]]))
            fila = {"u": u, "v": v, "key": k}
            fila.update({kk: vv for kk, vv in d.items() if kk != "geometry"})
            aristas_filas.append(fila)
    else:
        for u, v, d in grafo.edges(data=True):
            geoms.append(LineString([coords[u], coords[v]]))
            fila = {"u": u, "v": v}
            fila.update({kk: vv for kk, vv in d.items() if kk != "geometry"})
            aristas_filas.append(fila)

    aristas = gpd.GeoDataFrame(aristas_filas, geometry=geoms, crs=crs)
    return nodos, aristas


# ---------------------------------------------------------------------------
# Análisis OD con PMI y modelo nulo
# ---------------------------------------------------------------------------


def matriz_od_por_categoria(
    viajes: pd.DataFrame,
    col_origen: str,
    col_destino: str,
    col_categoria: str,
    col_peso: str = "Peso",
    total_minimo: float = 0.0,
) -> pd.DataFrame:
    """Construye la matriz O-D pivotada por categoría.

    Cada fila es un par O-D (MultiIndex), cada columna es una categoría
    (e.g. Sexo, Sexo_Proposito), con valor igual a la suma de pesos.
    Agrega una columna Total con la suma por fila y filtra a pares con
    Total > `total_minimo`.

    Parámetros
    ----------
    viajes : DataFrame
        Tabla larga de viajes con una fila por viaje.
    col_origen, col_destino : str
        Columnas que identifican origen y destino (e.g. ids H3).
    col_categoria : str
        Columna con la categoría del viaje (e.g. "Sexo" o "GenProp").
    col_peso : str
        Columna con el peso del viaje (default "Peso", factor de expansión).
    total_minimo : float
        Filtra pares O-D con Total <= total_minimo.

    Retorna
    -------
    DataFrame con MultiIndex (col_origen, col_destino), una columna por
    categoría más la columna Total. Las columnas tienen los nombres de
    las categorías observadas en los datos.
    """
    pivot = (
        viajes.groupby([col_origen, col_destino, col_categoria])[col_peso]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    if total_minimo > 0:
        pivot = pivot[pivot["Total"] > total_minimo]
    return pivot


def pmi_od(
    od_df: pd.DataFrame,
    columnas: list[str],
    suavizado: float = 0.5,
) -> pd.DataFrame:
    """Calcula el Pointwise Mutual Information por categoría para cada par O-D.

    PMI(arista, categoria) = log2( P(arista | categoria) / P(arista) )

    Positivo = la categoría sobre-representa esa arista; negativo = la sub-representa.
    Se aplica suavizado aditivo (Laplace) para evitar log(0).

    Parámetros
    ----------
    od_df : DataFrame
        Matriz O-D pivotada (e.g. salida de `matriz_od_por_categoria`).
        El índice puede ser cualquiera; las columnas relevantes son `columnas`.
    columnas : list[str]
        Nombres de las columnas de categoría sobre las que calcular PMI.
    suavizado : float
        Constante de suavizado aditivo. Default 0.5.

    Retorna
    -------
    DataFrame con las mismas filas que `od_df` y una columna por categoría
    en `columnas`, conteniendo el valor PMI en bits.
    """
    df = od_df[columnas].copy().astype(float) + suavizado

    total_arista = df.sum(axis=1)
    total_categoria = df.sum(axis=0)
    total_global = df.values.sum()

    resultado = pd.DataFrame(index=df.index, dtype=float)
    for col in columnas:
        p_arista_dado_cat = df[col] / total_categoria[col]
        p_arista = total_arista / total_global
        resultado[col] = np.log2(p_arista_dado_cat / p_arista)
    return resultado


def subgrafo_pmi(
    od_df: pd.DataFrame,
    categoria: str,
    columnas: list[str],
    suavizado: float = 0.5,
    umbral_pmi: float = 0.0,
    min_viajes: float = 0.0,
) -> nx.DiGraph:
    """Construye el subgrafo dirigido inducido por PMI para una categoría.

    Incluye solo las aristas con PMI(arista, categoria) > umbral_pmi
    y al menos min_viajes en esa categoría.

    El DataFrame od_df debe tener MultiIndex (origen, destino) o columnas
    grid_origen, grid_destino.

    Parámetros
    ----------
    od_df : DataFrame con la matriz O-D (resultado de matriz_od_por_categoria).
    categoria : str
        Categoría para la que se construye el subgrafo (e.g. "Mujer_Cuidado").
    columnas : list[str]
        Todas las categorías presentes (necesario para calcular PMI).
    suavizado : float
        Suavizado para PMI.
    umbral_pmi : float
        PMI mínimo para incluir la arista.
    min_viajes : float
        Mínimo de peso en la categoría para incluir la arista.

    Retorna
    -------
    nx.DiGraph con atributos `pmi` y `peso` en cada arista.
    """
    pmi_df = pmi_od(od_df, columnas, suavizado=suavizado)

    if isinstance(od_df.index, pd.MultiIndex):
        od_plano = od_df.reset_index()
        pmi_plano = pmi_df.reset_index(drop=True)
        col_o, col_d = od_df.index.names
    else:
        od_plano = od_df
        pmi_plano = pmi_df
        col_o, col_d = "grid_origen", "grid_destino"

    grafo = nx.DiGraph()
    valores_pmi = pmi_plano[categoria].values
    valores_peso = od_plano[categoria].astype(float).values
    origenes = od_plano[col_o].values
    destinos = od_plano[col_d].values

    for i in range(len(od_plano)):
        v_pmi = valores_pmi[i]
        v_peso = valores_peso[i]
        if pd.notna(v_pmi) and v_pmi > umbral_pmi and v_peso >= min_viajes:
            grafo.add_edge(
                origenes[i], destinos[i],
                pmi=float(v_pmi), peso=float(v_peso),
            )
    return grafo


def aristas_en_triangulos(grafo: nx.DiGraph) -> set[tuple]:
    """Identifica aristas dirigidas que participan en al menos un triángulo.

    Usa la proyección no dirigida: una arista (u, v) participa en triángulo
    si u y v comparten al menos un vecino (sin importar dirección).

    Retorna
    -------
    Set de tuplas (u, v) para las aristas dirigidas en triángulo.
    """
    grafo_und = grafo.to_undirected()
    en_triangulo = set()
    for u, v in grafo.edges():
        if u == v:
            continue
        comunes = set(grafo_und.neighbors(u)) & set(grafo_und.neighbors(v))
        if comunes:
            en_triangulo.add((u, v))
    return en_triangulo


def metricas_subgrafo(grafo: nx.DiGraph) -> dict:
    """Métricas topológicas del subgrafo: clustering, triángulos, densidad.

    Las métricas de clustering y triángulos se calculan sobre la versión
    no dirigida del grafo (los algoritmos clásicos están definidos así).

    Retorna
    -------
    dict con:
        - clustering : Series por nodo
        - triangulos : Series por nodo
        - resumen : dict con n_nodos, n_aristas, densidad, clustering_medio,
                    triangulos_total, triangulos_por_nodo_medio
    """
    if grafo.number_of_nodes() == 0:
        vacio = pd.Series(dtype=float)
        return {
            "clustering": vacio,
            "triangulos": vacio,
            "resumen": {
                "n_nodos": 0, "n_aristas": 0, "densidad": 0.0,
                "clustering_medio": 0.0, "triangulos_total": 0,
                "triangulos_por_nodo_medio": 0.0,
            },
        }

    grafo_und = grafo.to_undirected()
    clustering = pd.Series(nx.clustering(grafo_und), name="clustering")
    triangulos = pd.Series(nx.triangles(grafo_und), name="triangulos")

    resumen = {
        "n_nodos": grafo.number_of_nodes(),
        "n_aristas": grafo.number_of_edges(),
        "densidad": nx.density(grafo),
        "clustering_medio": float(clustering.mean()),
        "triangulos_total": int(triangulos.sum() // 3),
        "triangulos_por_nodo_medio": float(triangulos.mean()),
    }
    return {
        "clustering": clustering,
        "triangulos": triangulos,
        "resumen": resumen,
    }


def modelo_nulo_permutacion(
    viajes: pd.DataFrame,
    categorias: list[str],
    col_origen: str = "grid_origen",
    col_destino: str = "grid_destino",
    col_sexo: str = "Sexo",
    col_peso: str = "Peso",
    col_proposito: str | None = "PropositoAgregado",
    n_permutaciones: int = 200,
    total_minimo: float = 300.0,
    min_viajes: float = 50.0,
    umbral_pmi: float = 0.0,
    suavizado: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Modelo nulo por permutación: rompe la asociación sexo - movilidad.

    En cada permutación, baraja la columna de sexo entre todos los viajes
    (conserva origen, destino, propósito, peso), reconstruye la matriz O-D
    por categoría sexo o sexo x propósito, calcula PMI y mide la topología
    del subgrafo PMI > umbral para cada categoría.

    Implementación vectorizada con `np.bincount`: codifica cada viaje como
    un entero (od_code * n_categorias + cat_code) y reduce todo el groupby
    a una sola operación por permutación.

    Parámetros
    ----------
    viajes : DataFrame con una fila por viaje y columnas col_origen,
        col_destino, col_sexo, col_peso, opcionalmente col_proposito.
    categorias : list[str]
        Lista de categorías a evaluar (e.g. ["Mujer_Cuidado", ...] si
        col_proposito está presente, ["Mujer", "Hombre"] si no).
    n_permutaciones : int
        Número de permutaciones (default 200).
    total_minimo : float
        Filtro de total por par O-D (invariante bajo permutación de sexo).
    min_viajes : float
        Filtro de peso por categoría.
    umbral_pmi : float
        Umbral PMI para incluir aristas en el subgrafo.

    Retorna
    -------
    DataFrame con n_permutaciones * len(categorias) filas. Columnas:
    permutacion, categoria, sexo, proposito, viajes_total, n_nodos,
    n_aristas, densidad, clustering_medio, triangulos_total,
    triangulos_por_nodo_medio.
    """
    rng = np.random.default_rng(seed)
    con_proposito = (col_proposito is not None
                     and col_proposito in viajes.columns)

    od_pares = list(zip(viajes[col_origen].values, viajes[col_destino].values))
    od_codes, od_unique = pd.factorize(pd.Index(od_pares))
    n_od = len(od_unique)

    sexo_codes, sexo_unique = pd.factorize(viajes[col_sexo].values)
    n_sexo = len(sexo_unique)

    pesos = viajes[col_peso].to_numpy(dtype=float)

    if con_proposito:
        prop_codes, prop_unique = pd.factorize(viajes[col_proposito].values)
        n_prop = len(prop_unique)
        n_cat = n_sexo * n_prop
        cat_a_idx = {}
        for i_s, s in enumerate(sexo_unique):
            for i_p, p in enumerate(prop_unique):
                cat_a_idx[f"{s}_{p}"] = i_s * n_prop + i_p
    else:
        n_cat = n_sexo
        cat_a_idx = {s: i for i, s in enumerate(sexo_unique)}

    cat_idxs = np.array([cat_a_idx.get(c, -1) for c in categorias])

    total_por_od = np.bincount(od_codes, weights=pesos, minlength=n_od)
    mask_od = total_por_od > total_minimo
    od_validas = np.where(mask_od)[0]
    n_od_validas = len(od_validas)

    remap = -np.ones(n_od, dtype=np.int64)
    remap[od_validas] = np.arange(n_od_validas)
    trip_mask = mask_od[od_codes]
    od_codes_f = remap[od_codes[trip_mask]]
    pesos_f = pesos[trip_mask]
    sexo_codes_f = sexo_codes[trip_mask]
    if con_proposito:
        prop_codes_f = prop_codes[trip_mask]

    pares_od_validos = [od_unique[i] for i in od_validas]
    minlen = n_od_validas * n_cat

    registros = []
    for it in range(n_permutaciones):
        sexo_perm = rng.permutation(sexo_codes_f)
        if con_proposito:
            cat_codes = sexo_perm * n_prop + prop_codes_f
        else:
            cat_codes = sexo_perm

        bin_idx = od_codes_f * n_cat + cat_codes
        M = np.bincount(bin_idx, weights=pesos_f, minlength=minlen)
        M = M.reshape(n_od_validas, n_cat)

        M_suav = M + suavizado
        total_arista = M_suav.sum(axis=1)
        total_categoria = M_suav.sum(axis=0)
        total_global = M_suav.sum()
        pmi_matrix = np.log2(
            (M_suav / total_categoria[np.newaxis, :])
            / (total_arista[:, np.newaxis] / total_global)
        )

        for cat, ci in zip(categorias, cat_idxs):
            if ci < 0:
                g = nx.DiGraph()
            else:
                mask_edge = (
                    (pmi_matrix[:, ci] > umbral_pmi)
                    & (M[:, ci] >= min_viajes)
                )
                edge_idx = np.where(mask_edge)[0]
                g = nx.DiGraph()
                g.add_weighted_edges_from(
                    (pares_od_validos[j][0], pares_od_validos[j][1],
                     float(M[j, ci]))
                    for j in edge_idx
                )

            metricas = metricas_subgrafo(g)
            r = metricas["resumen"].copy()
            r["permutacion"] = it
            r["categoria"] = cat
            if "_" in cat:
                sexo, proposito = cat.split("_", 1)
            else:
                sexo, proposito = cat, None
            r["sexo"] = sexo
            r["proposito"] = proposito
            r["viajes_total"] = float(M[:, ci].sum()) if ci >= 0 else 0.0
            registros.append(r)

        if verbose and (it + 1) % 50 == 0:
            print(f"  Permutacion {it + 1}/{n_permutaciones}")

    return pd.DataFrame(registros)


def matriz_od_a_aristas_geodataframe(
    matriz_od: pd.DataFrame,
    columna: str,
    centroides: dict,
    col_origen: str | None = None,
    col_destino: str | None = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convierte una columna de matriz O-D pivotada a aristas GeoDataFrame.

    Para cada par (origen, destino) con el peso indicado por `columna`,
    emite una LineString entre los centroides correspondientes.

    Parámetros
    ----------
    matriz_od : DataFrame con MultiIndex (origen, destino) o columnas
        explicitas `col_origen` y `col_destino`, y al menos la columna
        `columna` con el peso.
    columna : nombre de la columna a leer como peso de la arista.
    centroides : dict de id_zona -> shapely.Point (en `crs`).
    col_origen, col_destino : nombres de las columnas si no hay
        MultiIndex (defecto: usa los nombres del index).
    crs : CRS de los centroides.
    """
    if isinstance(matriz_od.index, pd.MultiIndex):
        plano = matriz_od.reset_index()
        col_o = col_origen or matriz_od.index.names[0]
        col_d = col_destino or matriz_od.index.names[1]
    else:
        plano = matriz_od
        col_o = col_origen
        col_d = col_destino

    geoms, pesos, origenes, destinos = [], [], [], []
    for fila in plano.itertuples(index=False):
        o = getattr(fila, col_o)
        d = getattr(fila, col_d)
        if o in centroides and d in centroides:
            geoms.append(LineString([centroides[o], centroides[d]]))
            pesos.append(float(getattr(fila, columna)))
            origenes.append(o)
            destinos.append(d)
    return gpd.GeoDataFrame(
        {"u": origenes, "v": destinos, "peso": pesos},
        geometry=geoms, crs=crs,
    )


def aristas_pmi_a_geodataframe(
    grafo: nx.DiGraph,
    centroides: dict,
    aristas_en_triangulo: set | None = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convierte aristas de un subgrafo PMI a GeoDataFrame con metadata.

    Cada arista del grafo se vuelca a una LineString entre los
    centroides de origen y destino, conservando los atributos `pmi` y
    `peso` y agregando `importancia = sqrt(max(pmi, 0) * peso)`. Si se
    pasa `aristas_en_triangulo`, se agrega la columna booleana
    `en_triangulo` para distinguir aristas en triangulo en visualizaciones.

    Parámetros
    ----------
    grafo : nx.DiGraph con atributos `pmi` y `peso` en cada arista
        (resultado de `subgrafo_pmi`).
    centroides : dict de id_celda -> shapely.Point en `crs`.
    aristas_en_triangulo : set opcional de tuplas (u, v) en triangulo
        (resultado de `aristas_en_triangulos`).
    crs : CRS de los centroides.
    """
    tri = aristas_en_triangulo if aristas_en_triangulo is not None else set()
    geoms, pmis, pesos, en_tri = [], [], [], []
    origenes, destinos = [], []
    for u, v, datos in grafo.edges(data=True):
        if u not in centroides or v not in centroides:
            continue
        geoms.append(LineString([centroides[u], centroides[v]]))
        pmis.append(datos.get("pmi", 0))
        pesos.append(datos.get("peso", 0))
        en_tri.append((u, v) in tri)
        origenes.append(u)
        destinos.append(v)
    gdf = gpd.GeoDataFrame(
        {"u": origenes, "v": destinos, "pmi": pmis, "peso": pesos,
         "en_triangulo": en_tri},
        geometry=geoms, crs=crs,
    )
    gdf["importancia"] = np.sqrt(np.clip(gdf["pmi"], 0, None) * gdf["peso"])
    return gdf


def top_aristas(
    gdf: gpd.GeoDataFrame,
    top_pct: float,
    columna: str = "importancia",
) -> gpd.GeoDataFrame:
    """Filtra el top `top_pct` (fraccion en [0, 1]) de aristas por una columna.

    Default ordena por `importancia` (la columna que agrega
    `aristas_pmi_a_geodataframe`). Sirve para reducir el clutter en
    mapas de redes sin tocar el calculo de metricas topologicas.
    """
    if len(gdf) == 0 or top_pct >= 1.0:
        return gdf
    n_top = max(1, int(np.ceil(len(gdf) * top_pct)))
    return gdf.nlargest(n_top, columna)


def strength_por_nodo(grafo: nx.Graph, atributo_peso: str = "peso") -> dict:
    """Suma de pesos de aristas incidentes por nodo (strength).

    En grafos dirigidos cuenta aristas tanto de entrada como de salida.
    Util como medida de actividad de un nodo en una red ponderada.
    """
    s = {}
    for u, v, datos in grafo.edges(data=True):
        w = datos.get(atributo_peso, 0)
        s[u] = s.get(u, 0) + w
        s[v] = s.get(v, 0) + w
    return s


# URL del basemap por defecto: Carto positron (sin token).
BASEMAP_CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


def generar_html_flowmap(
    locations: list[dict],
    flows: list[dict],
    colores: dict[str, str],
    ruta_salida: str | Path,
    title: str = "Flujos origen-destino",
    panel_titulo: str = "Categorias",
    descripcion_html: str = "",
    centro: tuple[float, float] = (-70.65, -33.45),
    zoom: float = 10.3,
    pitch: float = 0.0,
    basemap_url: str = BASEMAP_CARTO_POSITRON,
) -> Path:
    """Genera un HTML self-contained con flowmap.gl/layers.

    Una `FlowmapLayer` por categoria, con panel de checkboxes para
    toggle de visibilidad. Las dependencias se cargan desde esm.sh
    (modulos ES via import map) y maplibre-gl desde unpkg (UMD). Sin
    React, sin pydeck. El template HTML vive en
    `gdsutils/templates/flowmap.html`.

    Para visualizar el HTML hay que servirlo por HTTP (los modulos ES
    no cargan desde `file://` por CORS):

        uv run python -m http.server -d <directorio> 8000

    Parámetros
    ----------
    locations : lista de dicts con keys `id`, `lon`, `lat`.
    flows : lista de dicts con keys `origin`, `dest`, `count`,
        `category` (y opcionales `pmi`, etc. accesibles via hover).
    colores : dict de categoria -> color hex (por ejemplo "#641a80").
        El orden define el orden del panel de toggles.
    ruta_salida : path al archivo HTML a escribir.
    title : titulo del documento (etiqueta `<title>`).
    panel_titulo : titulo del panel de toggles.
    descripcion_html : HTML opcional a mostrar debajo de los toggles
        (envuelto en `<small>`). Vacio omite la seccion.
    centro : (longitud, latitud) del centro inicial del mapa.
    zoom, pitch : configuracion inicial de la camara.
    basemap_url : URL al style.json del basemap (default Carto positron).
    """
    template = (_TEMPLATES_DIR / "flowmap.html").read_text(encoding="utf-8")

    if descripcion_html:
        descripcion_bloque = (
            '<hr style="border: 0; border-top: 1px solid #eee; margin: 10px 0;"/>\n'
            f"<small>{descripcion_html}</small>"
        )
    else:
        descripcion_bloque = ""

    html = (
        template
        .replace("__TITLE__", title)
        .replace("__PANEL_TITULO__", panel_titulo)
        .replace("__DESCRIPCION_HTML__", descripcion_bloque)
        .replace("__BASEMAP_URL__", basemap_url)
        .replace("__CENTER_LON__", str(centro[0]))
        .replace("__CENTER_LAT__", str(centro[1]))
        .replace("__ZOOM__", str(zoom))
        .replace("__PITCH__", str(pitch))
        .replace("__LOCATIONS__", json.dumps(locations))
        .replace("__FLOWS__", json.dumps(flows))
        .replace("__COLORES__", json.dumps(colores))
    )

    ruta = Path(ruta_salida)
    ruta.write_text(html, encoding="utf-8")
    return ruta
