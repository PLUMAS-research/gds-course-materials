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

from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring

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
