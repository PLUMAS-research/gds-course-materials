from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de análisis
# ---------------------------------------------------------------------------

# Mapeo de propósitos EOD a categorías agregadas
GRUPOS_PROPOSITOS = {
    "De salud": "Cuidado",
    "Visitar a alguien": "Cuidado",
    "Buscar o Dejar a alguien": "Cuidado",
    "Buscar o dejar algo": "Cuidado",
    "De compras": "Cuidado",
    "Trámites": "Cuidado",
    "Al trabajo": "Empleo/Estudio",
    "Por trabajo": "Empleo/Estudio",
    "Al estudio": "Empleo/Estudio",
    "Por estudio": "Empleo/Estudio",
    "Recreación": "Personal",
    "Comer o Tomar algo": "Personal",
    "Otra actividad (especifique)": "Personal",
    "volver a casa": "Hogar",
}

PROPOSITOS = ["Cuidado", "Empleo/Estudio", "Personal", "Hogar"]

COLORES_PROPOSITO = {
    "Cuidado": "#e08214",
    "Empleo/Estudio": "#2d708e",
    "Personal": "#35b779",
    "Hogar": "#aaaaaa",
}


# ---------------------------------------------------------------------------
# Funciones de preparación de datos
# ---------------------------------------------------------------------------


def georreferenciar_viajes(df, col_x, col_y, crs_origen="EPSG:32719", crs_destino="EPSG:4326"):
    """Convierte columnas de coordenadas EOD (coma decimal, UTM 19S) a GeoDataFrame.

    La EOD 2012 almacena coordenadas como texto con coma decimal. Esta función
    las convierte a float, construye geometrías Point y reproyecta.

    Parameters
    ----------
    df : DataFrame
    col_x, col_y : str
        Nombres de las columnas con coordenadas X e Y.
    crs_origen : str
        CRS de las coordenadas originales (UTM 19S por defecto).
    crs_destino : str
        CRS de salida.

    Returns
    -------
    GeoDataFrame
    """
    result = df.copy()
    for col in [col_x, col_y]:
        result[col] = (
            result[col].astype(str).str.replace(",", ".", regex=False).astype(float)
        )
    return gpd.GeoDataFrame(
        result,
        geometry=gpd.points_from_xy(result[col_x], result[col_y]),
        crs=crs_origen,
    ).to_crs(crs_destino)


def calcular_viajes_proposito(df, col_zona, propositos=None, col_peso="Peso"):
    """Calcula la transformación de Anscombe de los viajes ponderados por propósito y zona.

    Aplica sqrt(x + 3/8) a los conteos ponderados. Esta transformación
    estabiliza la varianza de datos de conteo (Poisson): la varianza del
    resultado es aproximadamente 1/4 independiente de la media, lo que
    permite usar estadísticos espaciales (Moran, LISA) que asumen
    homocedasticidad. Corrige el sesgo de la raíz cuadrada simple en
    conteos bajos.

    Parameters
    ----------
    df : DataFrame
        Debe contener columnas ``PropositoAgregado`` y ``col_peso``.
    col_zona : str
        Columna con la unidad espacial de agregación.
    propositos : list of str, optional
        Propósitos a calcular. Por defecto usa ``PROPOSITOS``.
    col_peso : str
        Columna de pesos / factor de expansión.

    Returns
    -------
    DataFrame con una columna por propósito e índice = zona.
    """
    if propositos is None:
        propositos = PROPOSITOS
    resultado = {}
    for proposito in propositos:
        sub = df[df["PropositoAgregado"] == proposito]
        conteo = sub.groupby(col_zona)[col_peso].sum()
        resultado[proposito] = np.sqrt(conteo + 3 / 8)
    return pd.DataFrame(resultado).fillna(0)


def calcular_poblacion_flotante(
    viajes,
    geodf_celdas,
    columna_id_celda,
    propositos_excluidos=("volver a casa", "Visitar a alguien"),
    franja_diurna=(7, 19),
    col_peso="Peso",
    col_persona="Persona",
    col_x="DestinoCoordX",
    col_y="DestinoCoordY",
    col_hora_ini="HoraIni",
    col_hora_fin="HoraFin",
    crs_eod="EPSG:32719",
):
    """Población flotante promedio diurna por celda, a partir de viajes EOD.

    Para cada viaje, la permanencia en el destino se estima como
    HoraIni del siguiente viaje de la misma persona menos HoraFin actual.
    Si no hay viaje siguiente, se capea al cierre de la franja diurna.
    Se descartan viajes con propósito en ``propositos_excluidos`` (destinos
    residenciales). Se suma col_peso * permanencia_diurna por celda y se
    divide por la duración de la franja para obtener personas presentes en
    promedio durante el día.

    Parameters
    ----------
    viajes : DataFrame
        Viajes EOD. Debe contener col_persona, col_hora_ini, col_hora_fin,
        col_x, col_y, col_peso y "Proposito".
    geodf_celdas : GeoDataFrame
        Celdas espaciales (H3, comunas, zonas, etc.). Cualquier CRS.
    columna_id_celda : str
        Identificador de celda en geodf_celdas.
    propositos_excluidos : tuple
        Propósitos cuyo destino no aporta a población flotante pública.
    franja_diurna : tuple of (lo, hi)
        Franja horaria considerada, en horas (0-24).
    col_peso : str
        Factor de expansión por viaje (típicamente FactorExpansion * FactorPersona).

    Returns
    -------
    pd.Series indexada por columna_id_celda con la población flotante.
    """
    lo, hi = franja_diurna
    duracion_franja = hi - lo

    df = viajes.copy()

    # Normalizar HoraIni y HoraFin a horas (float)
    def _a_horas(serie):
        if pd.api.types.is_timedelta64_dtype(serie):
            return serie.dt.total_seconds() / 3600.0
        return pd.to_timedelta(serie.astype(str) + ":00").dt.total_seconds() / 3600.0

    df["_h_ini"] = _a_horas(df[col_hora_ini])
    df["_h_fin"] = _a_horas(df[col_hora_fin])

    # Viajes que cruzan medianoche: HoraFin aparente menor que HoraIni
    cruza = df["_h_fin"] < df["_h_ini"]
    df.loc[cruza, "_h_fin"] += 24.0

    df = df.sort_values([col_persona, "_h_ini"]).reset_index(drop=True)

    # Shift global y máscara de cambio de persona (evita groupby)
    h_siguiente = df["_h_ini"].shift(-1)
    persona_siguiente = df[col_persona].shift(-1)
    ultimo = df[col_persona] != persona_siguiente
    h_siguiente = h_siguiente.where(~ultimo, hi)

    # Solapamiento del intervalo [HoraFin, HoraIni_siguiente] con [lo, hi]
    inicio_d = df["_h_fin"].clip(lower=lo, upper=hi)
    fin_d = h_siguiente.clip(lower=lo, upper=hi)
    permanencia = (fin_d - inicio_d).clip(lower=0)

    valido = (
        (~df["Proposito"].isin(propositos_excluidos))
        & (permanencia > 0)
        & df[col_x].notna()
        & df[col_y].notna()
        & df[col_peso].notna()
    )

    contribucion = (
        df.loc[valido, col_peso].astype(float)
        * permanencia[valido].astype(float)
    )

    puntos = gpd.GeoDataFrame(
        {"_contribucion": contribucion.values},
        geometry=gpd.points_from_xy(
            df.loc[valido, col_x].values, df.loc[valido, col_y].values
        ),
        crs=crs_eod,
    )

    celdas = geodf_celdas[[columna_id_celda, "geometry"]]
    if celdas.crs != puntos.crs:
        celdas = celdas.to_crs(puntos.crs)

    asignado = gpd.sjoin(puntos, celdas, how="inner", predicate="within")

    pob = asignado.groupby(columna_id_celda)["_contribucion"].sum() / duracion_franja
    pob.name = "poblacion_flotante"
    return pob


def cargar_zonas_estudio(census_parquet, bbox, comunas_excluidas=None):
    """
    Carga el parquet comunal censal 2024 (región 13), recorta al bounding box,
    excluye comunas, reproyecta a EPSG:4326 y calcula centroides.

    Parameters
    ----------
    census_parquet : str o Path
        Ruta al parquet de cartografía comunal censal 2024.
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max).
    comunas_excluidas : list, optional
        Nombres de comunas a excluir.

    Returns
    -------
    GeoDataFrame con columna 'centroide'.
    """
    comunas_excluidas = comunas_excluidas or []

    zones = gpd.read_parquet(
        census_parquet, filters=[("COD_REGION", "=", 13)]
    ).to_crs("EPSG:4326")

    zones_clip = zones.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    zones_filtrado = zones_clip[
        ~zones_clip["NOM_COMUNA"].isin(comunas_excluidas)
    ].copy()

    zones_filtrado["centroide"] = zones_filtrado.geometry.centroid

    return zones_filtrado


def decode_column(
    df,
    fname,
    col_name,
    index_col="Id",
    value_col=None,
    sep=";",
    encoding="utf-8",
    index_dtype=np.float64,
):
    """
    Decodifica los valores de una columna, reemplazando identificadores por su correspondiente valor según la tabla de códigos.

    Parameters
    ------------
    df : pandas.dataframe
        Dataframe del que se leerá una columna.
    fname: string
        Nombre del archivo que contiene los valores a decodificar.
    col_name: string
        Nombre de la columna que queremos decodificar.
    index_col: string, default="Id"
        Nombre de la columna en el archivo ` fname `  que tiene los índices que codifican ` col_name ` .
    value_col: string, default=None
        Nombre de la columna en el archivo ` fname `  que tiene los valores decodificados.
    sep: string, default=";"
        Caracter que separa los valores en ` fname ` .
    encoding: string, default="utf-8"
        Identificación del character set que utiliza el archivo. Usualmente es utf-8, si no funciona,
          se puede probar con iso-8859-1.
    index_dtype: dtype, default=np.float64

    Returns
    -------
    pd.DataFrame
        Dataframe decodificado en la columna señalada.

    """
    if value_col is None:
        value_col = "value"

    values_df = pd.read_csv(
        fname,
        sep=sep,
        index_col=index_col,
        names=[index_col, value_col],
        header=0,
        dtype={index_col: index_dtype},
        encoding=encoding,
    )

    src_df = df.loc[:, (col_name,)]

    return src_df.join(values_df, on=col_name)[value_col]


# Funcion para asignar tipo de día
def etiquetar_tipo_dia(row):
    if pd.notna(row["FactorLaboralNormal"]):
        return "Laboral"
    if pd.notna(row["FactorDomingoNormal"]):
        return "Domingo"
    if pd.notna(row["FactorSabadoNormal"]):
        return "Sábado"
    if pd.notna(row["FactorLaboralEstival"]):
        return "LaboralEstival"
    if pd.notna(row["FactorFindesemanaEstival"]):
        return "FindesemanaEstival"
    else:
        return "No Definido"


# Funcion para asignar Factor Externo
def etiquetar_FactorExp(row):
    if pd.notna(row["FactorLaboralNormal"]):
        return row["FactorLaboralNormal"]
    if pd.notna(row["FactorDomingoNormal"]):
        return row["FactorDomingoNormal"]
    if pd.notna(row["FactorSabadoNormal"]):
        return row["FactorSabadoNormal"]
    if pd.notna(row["FactorLaboralEstival"]):
        return row["FactorLaboralEstival"]
    if pd.notna(row["FactorFindesemanaEstival"]):
        return row["FactorFindesemanaEstival"]
    else:
        return None


def read_trips(path, decode_columns=True, remove_invalid=True, fix_clock_times=True):
    """
    Busca los archivos "viajes.csv", "ViajesDifusion.csv" y "DistanciaViaje.csv" dentro
    del directorio especificado o en su defecto en el definido en la variable global "_EOD_PATH".
    Unifica la información de estos archivos en un dataframe de pandas.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path : string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_column: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado
        según las tablas de decodificación ubicadas en el directorio "Tablas_parametros".
    remove_invalid: bool, default=True
        Indica si se quiere eliminar filas que no tienen hora o que han sido inputadas.
    fix_clock_times: bool, default=True
        Indica si se desea estandarizar la hora de inicio al formato timedelta.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información de viajes de la encuesta origen-destino.
    """
    DATA_PATH = Path(path)

    df = (
        pd.read_csv(DATA_PATH / "viajes.csv", sep=";", decimal=",")
        .join(
            pd.read_csv(DATA_PATH / "ViajesDifusion.csv", sep=";", index_col="Viaje"),
            on="Viaje",
        )
        .join(
            pd.read_csv(DATA_PATH / "DistanciaViaje.csv", sep=";", index_col="Viaje"),
            on="Viaje",
        )
    )

    if decode_columns:
        df["ModoAgregado"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ModoAgregado.csv",
            "ModoAgregado",
            index_col="ID",
            value_col="Modo",
        )
        df["ModoDifusion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ModoDifusion.csv",
            "ModoDifusion",
            encoding="latin-1",
            index_col="ID",
        )
        df["SectorOrigen"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Sector.csv",
            col_name="SectorOrigen",
            index_col="Sector",
            value_col="Nombre",
            sep=";",
        )
        df["SectorDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Sector.csv",
            col_name="SectorDestino",
            index_col="Sector",
            value_col="Nombre",
            sep=";",
        )
        df["Proposito"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "Proposito.csv", col_name="Proposito"
        )
        df["ComunaOrigen"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Comunas.csv",
            "ComunaOrigen",
            value_col="Comuna",
            sep=",",
        )
        df["ComunaDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Comunas.csv",
            "ComunaDestino",
            value_col="Comuna",
            sep=",",
        )
        df["ActividadDestino"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "ActividadDestino.csv",
            "ActividadDestino",
        )
        df["Periodo"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Periodo.csv",
            "Periodo",
            sep=";",
            value_col="Periodos",
        )

    if remove_invalid:
        df = df[pd.notnull(df["HoraIni"])]
        df = df[df["Imputada"] == 0].copy()
        df = df[df["DistManhattan"] != -1].copy()

    if fix_clock_times:
        df["HoraIni"] = pd.to_timedelta(df["HoraIni"] + ":00")

    # Aplicamos la función a cada fila y creamos una nueva columna llamada 'TipoDia'
    df["TipoDia"] = df.apply(etiquetar_tipo_dia, axis=1)

    df["FactorExpansion"] = df.apply(etiquetar_FactorExp, axis=1)

    return df


def read_homes(path):
    """
    Carga el contenido del archivo "Hogares.csv", que contiene las respuestas sobre hogares participantes de la
    encuesta origen destino, a un dataframe.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre hogares, con las columnas decodificadas.

    """
    DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "Hogares.csv", sep=";", decimal=",", encoding="utf-8"
    ).rename(columns={"Factor": "FactorHogar"})

    df["Sector"] = decode_column(
        df, DATA_PATH / "Tablas_parametros" / "Sector.csv", "Sector"
    )

    return df


def read_people(path, decode_columns=True):
    """
    Carga el contenido del archivo "personas.csv", que contiene información sobre las personas encuestadas, a un dataframe.
    En el notebook ubicado en notebooks/gds-course/01-scl-travel-survey-maps.ipynb se pueden encontrar ejemplos de uso
    de esta función.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre personas.

    """
    DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "personas.csv", sep=";", decimal=",", encoding="utf-8"
    ).rename(columns={"Factor": "FactorPersona"})

    if decode_columns:
        df["Sexo"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "Sexo.csv", "Sexo"
        )
        df["TramoIngreso"] = decode_column(
            df, DATA_PATH / "Tablas_parametros" / "TramoIngreso.csv", "TramoIngreso"
        )
        df["Relacion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Relacion.csv",
            "Relacion",
            value_col="relacion",
        )
        df["Ocupacion"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "Ocupacion.csv",
            "Ocupacion",
            value_col="ocupacion",
        )
    return df


def read_transantiago_usage(path, decode_columns=True):
    """
    Crea un dataframe que contiene a las personas que no usaron el sistema Transantiago en su viaje y la razón.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con una fila por persona que no usó el Transantiago y su razón para no hacerlo.

    """
    DATA_PATH = Path(path)

    df = (
        pd.read_csv(DATA_PATH / "personas.csv", sep=";", decimal=",", encoding="utf-8")
        .pipe(lambda x: x[pd.notnull(x.NoUsaTransantiago)])
        .set_index("Persona")["NoUsaTransantiago"]
        .str.split(";")
        .explode()
        .reset_index()
    )

    if decode_columns:
        df["NoUsaTransantiago"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "NoUsaTransantiago.csv",
            "NoUsaTransantiago",
            index_dtype=str,
        )

    return df


def read_zone_design(path):
    """
    Carga la geometría de la zonificación de las comunas participantes en la encuesta.
    Podemos encontrar un tutorial de uso de esta función en el notebook ` notebooks/vis-course/03-python-mapas-preliminario.ipynb`

    Parameters
    ----------
    path: string, default=None
        Ubicación del archivo shapefile que contiene la geometría de las comunas. Si no se especifica, se usará el valor
        almacenado en la variable global _EOD_MAPS.
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataframe con la geometría de la zonificación de las comunas, el sistema de coordenadas usado es [`EPSG:32719`](https://epsg.io/32719)

    """

    DATA_PATH = Path(path) / "Zonificacion_EOD2012"

    return gpd.read_file(DATA_PATH)


def read_vehicles(path=None, decode_columns=True):
    """
    Carga el contenido del archivo "Vehiculo.csv", que contiene información sobre los vehículos de los
    hogares encuestados.

    Parameters
    ----------
    path: string, default=None
        Ubicación de los archivos csv con la data de la encuesta origen destino.
    decode_columns: bool, default=True
        Indica si se quiere decodificar el contenido de las columnas, reemplazando IDs por su significado

    Returns
    -------
    pd.DataFrame
        Dataframe con la información sobre personas.

    """
    DATA_PATH = Path(path)

    df = pd.read_csv(
        DATA_PATH / "Vehiculo.csv", sep=";", decimal=",", encoding="iso-8859-1"
    )

    if decode_columns:
        df["TipoVeh"] = decode_column(
            df,
            DATA_PATH / "Tablas_parametros" / "TipoVeh.csv",
            value_col="vehiculo",
            col_name="TipoVeh",
            index_dtype=int,
            encoding="iso-8859-1",
        )

    return df
