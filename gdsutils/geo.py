import geopandas as gpd


def clip_geodataframe(geodf, bbox):
    """Recorta un GeoDataFrame a un bounding box, eliminando geometrías vacías.

    Maneja la columna SHAPE de la cartografía censal (la descarta si existe,
    ya que clip_by_rect crea una nueva columna geometry).

    Parameters
    ----------
    geodf : GeoDataFrame
        GeoDataFrame con geometrías a recortar.
    bbox : list
        [xmin, ymin, xmax, ymax].

    Returns
    -------
    GeoDataFrame
        GeoDataFrame recortado.
    """
    result = (
        geodf.assign(geometry=lambda x: x.clip_by_rect(*bbox))
        .set_geometry("geometry")
        .pipe(lambda x: x[~x.is_empty])
    )
    if "SHAPE" in result.columns:
        result = result.drop("SHAPE", axis=1)
    return result
