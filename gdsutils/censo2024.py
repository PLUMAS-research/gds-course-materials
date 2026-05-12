from pathlib import Path

import numpy as np
import pandas as pd

territorios = pd.read_csv(
    Path("data") / "censo2024-microdatos" / "cod_territoriales_especificos.csv",
    index_col=0,
)["Territorio específico"].to_dict()

dic_personas = pd.read_csv(
    Path("data") / "censo2024-microdatos" / "tabla_personas.csv",
    index_col="Nombre variable",
    thousands=",",
    decimal=".",
).drop("Entidad", axis=1)

dic_viviendas = pd.read_csv(
    Path("data") / "censo2024-microdatos" / "tabla_viviendas.csv",
    index_col="Nombre variable",
    thousands=",",
    decimal=".",
).drop("Entidad", axis=1)


def diccionario(col, cast_int=True, drop_na=True, tipo="personas"):
    if tipo == "viviendas":
        result = dic_viviendas.loc[col]
    else:
        result = dic_personas.loc[col]
    if drop_na:
        result = result.dropna(subset="Valor")
    if cast_int:
        result = result.assign(Valor=lambda x: x["Valor"].astype(int))
    else:
        result = result.assign(Valor=lambda x: x["Valor"].str.strip())

    return result.set_index("Valor")["Etiqueta de categoría"].to_dict()


def reassign(df, col, territorio=False):
    if not territorio:
        return df.assign(**{col: lambda x: x[col].map(diccionario(col))})
    else:
        return df.assign(**{col: lambda x: x[col].map(territorios)})


def rango_edad(edad):
    if edad <= 5:
        return "n_edad_0_5"
    elif edad <= 13:
        return "n_edad_6_13"
    elif edad <= 17:
        return "n_edad_14_17"
    elif edad <= 24:
        return "n_edad_18_24"
    elif edad <= 44:
        return "n_edad_25_44"
    elif edad <= 59:
        return "n_edad_45_59"
    else:
        return "n_edad_60_mas"


regiones = {
    "RM": {
        "region": 13,
        "capital_bbox": [-70.88006218, -33.67612715, -70.43015094, -33.31069169],
        "nombre": "Región Metropolitana",
        "nombre_capital": "Gran Santiago",
    },
    "VALPO": {
        "region": 5,
        "capital_bbox": [-71.669428, -33.074331, -71.324728, -32.906420],
        "nombre": "Región de Valparaíso",
        "nombre_capital": "Gran Valparaíso",
    },
    "BIOBIO": {
        "region": 8,
        "capital_bbox": [-73.243089, -36.953576, -72.941832, -36.680511],
        "nombre": "Región del Biobío",
        "nombre_capital": "Gran Concepción",
    },
}


COLUMNAS_TRANSPORTE = [
    "n_transporte_auto",
    "n_transporte_publico",
    "n_transporte_camina",
    "n_transporte_bicicleta",
    "n_transporte_motocicleta",
    "n_transporte_cab_lan_bote",
    "n_transporte_otros",
]

_RENOMBRA_TRANSPORTE = {
    "Auto particular": "n_transporte_auto",
    "Transporte público (bus, micro, metro, tren, taxi, colectivo)": "n_transporte_publico",
    "Caminando": "n_transporte_camina",
    "Bicicleta (incluye scooter)": "n_transporte_bicicleta",
    "Motocicleta": "n_transporte_motocicleta",
    "Caballo, lancha o bote": "n_transporte_cab_lan_bote",
    "Otro": "n_transporte_otros",
}

COLUMNAS_EDAD = [
    "n_edad_0_5",
    "n_edad_6_13",
    "n_edad_14_17",
    "n_edad_18_24",
    "n_edad_25_44",
    "n_edad_45_59",
    "n_edad_60_mas",
]

COLUMNAS_CINE = [
    "n_cine_nunca_curso_primera_infancia",
    "n_cine_primaria",
    "n_cine_secundaria",
    "n_cine_terciaria_maestria_doctorado",
    "n_cine_especial_diferencial",
]

_MAPA_CINE = {
    1: "n_cine_nunca_curso_primera_infancia",
    2: "n_cine_primaria",
    3: "n_cine_primaria",
    4: "n_cine_primaria",
    5: "n_cine_primaria",
    6: "n_cine_secundaria",
    7: "n_cine_secundaria",
    8: "n_cine_secundaria",
    9: "n_cine_terciaria_maestria_doctorado",
    10: "n_cine_terciaria_maestria_doctorado",
    11: "n_cine_terciaria_maestria_doctorado",
    12: "n_cine_especial_diferencial",
    -99: "nr",
}

COLUMNAS_CIUO = [f"n_ciuo_{i}" for i in range(10)]

COLUMNAS_TIPO_VIVIENDA = [
    "n_tipo_viv_casa",
    "n_tipo_viv_depto",
    "n_tipo_viv_pieza",
    "n_tipo_viv_mediagua",
    "n_tipo_viv_movil",
    "n_tipo_viv_indigena",
    "n_tipo_viv_otro",
]

_RENOMBRA_TIPO_VIVIENDA = {
    "Casa": "n_tipo_viv_casa",
    "Departamento": "n_tipo_viv_depto",
    "Pieza en casa antigua o en conventillo": "n_tipo_viv_pieza",
    "Mediagua, mejora, vivienda de emergencia, rancho o choza": "n_tipo_viv_mediagua",
    "Móvil (carpa, casa rodante o similar)": "n_tipo_viv_movil",
    "Vivienda tradicional indígena (ruka u otras)": "n_tipo_viv_indigena",
    "Otro tipo de vivienda particular": "n_tipo_viv_otro",
}

COLUMNAS_DORMITORIOS = [
    "n_dormitorios_1",
    "n_dormitorios_2",
    "n_dormitorios_3",
    "n_dormitorios_4",
    "n_dormitorios_5",
    "n_dormitorios_6_o_mas",
]


def _conteos_por_vivienda(personas, columna_categoria, mapa_renombre, columnas_finales):
    tabla = (
        personas.groupby(["id_vivienda", columna_categoria])
        .size()
        .unstack(fill_value=0)
        .rename(columns=mapa_renombre)
    )
    for col in columnas_finales:
        if col not in tabla.columns:
            tabla[col] = 0
    return tabla[columnas_finales]


def tabla_viviendas_desde_microdatos(
    cod_comuna,
    region=13,
    ruta_microdatos=Path("data") / "censo2024-microdatos",
):
    """
    Construye la tabla wide de viviendas para una comuna desde los microdatos.

    Cada fila es una vivienda; las columnas son conteos por sexo, edad,
    inmigrantes, educacion CINE, ocupacion CIUO, transporte (p45), tipo de
    vivienda y dormitorios. La salida sirve como entrada de
    `gdsutils.asignacion.asignar_viviendas`.

    Retorna (tabla_viviendas, personas) donde `personas` es el DataFrame de
    microdatos filtrado por la comuna (util para analisis posterior).
    """
    ruta_microdatos = Path(ruta_microdatos)

    personas = pd.read_parquet(
        ruta_microdatos / "personas",
        filters=[("region", "==", region), ("comuna", "==", cod_comuna)],
    )

    transporte = (
        personas.groupby(["id_vivienda", "p45_medio_transporte"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=diccionario("p45_medio_transporte"))
        .rename(columns=_RENOMBRA_TRANSPORTE)
    )
    for col in COLUMNAS_TRANSPORTE:
        if col not in transporte.columns:
            transporte[col] = 0
    transporte = transporte[COLUMNAS_TRANSPORTE]

    sexo = (
        personas.groupby(["id_vivienda", "sexo"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=diccionario("sexo"))
        .rename(columns={"Hombre": "n_hombres", "Mujer": "n_mujeres"})
    )

    edades = (
        personas.assign(rango_edad=personas["edad"].apply(rango_edad))
        .groupby(["id_vivienda", "rango_edad"])
        .size()
        .unstack(fill_value=0)
    )
    for col in COLUMNAS_EDAD:
        if col not in edades.columns:
            edades[col] = 0
    edades = edades[COLUMNAS_EDAD]

    origen = (
        personas.pipe(lambda x: reassign(x, "p25_lug_nacimiento"))
        .pipe(lambda x: reassign(x, "p27_nacionalidad"))
        .assign(
            es_inmigrante=lambda x: (
                (x["p25_lug_nacimiento"] == "En otro país")
                & (x["p27_nacionalidad"] == "Otra nacionalidad")
            ).astype(int)
        )
        .groupby("id_vivienda")["es_inmigrante"]
        .sum()
        .rename("n_inmigrantes")
    )

    educacion = (
        personas.assign(cine11=lambda x: x["cine11"].map(_MAPA_CINE))
        .groupby(["id_vivienda", "cine11"])
        .size()
        .unstack(fill_value=0)
    )
    for col in COLUMNAS_CINE:
        if col not in educacion.columns:
            educacion[col] = 0
    educacion = educacion[COLUMNAS_CINE]

    ciuo = (
        personas.groupby(["id_vivienda", "cod_ciuo"])
        .size()
        .unstack(fill_value=0)
    )
    ciuo.columns = [f"n_ciuo_{int(c)}" if c != -99 else "n_ciuo_nr" for c in ciuo.columns]
    for col in COLUMNAS_CIUO:
        if col not in ciuo.columns:
            ciuo[col] = 0
    ciuo = ciuo[COLUMNAS_CIUO]

    viviendas_raw = pd.read_parquet(
        ruta_microdatos / "viviendas",
        filters=[("region", "==", region), ("comuna", "==", cod_comuna)],
    )

    tipo_vivienda = (
        viviendas_raw.set_index("id_vivienda")["p2_tipo_vivienda"]
        .map(diccionario("p2_tipo_vivienda", tipo="viviendas"))
        .to_frame()
        .assign(valor=1)
        .reset_index()
        .set_index(["id_vivienda", "p2_tipo_vivienda"])["valor"]
        .unstack(fill_value=0)
        .rename(columns=_RENOMBRA_TIPO_VIVIENDA)
    )
    if np.nan in tipo_vivienda.columns:
        tipo_vivienda = tipo_vivienda.drop(np.nan, axis=1)
    for col in COLUMNAS_TIPO_VIVIENDA:
        if col not in tipo_vivienda.columns:
            tipo_vivienda[col] = 0
    tipo_vivienda = tipo_vivienda[COLUMNAS_TIPO_VIVIENDA]

    dormitorios = pd.get_dummies(viviendas_raw.set_index("id_vivienda")["p5_num_dormitorios"])
    for valor_invalido in (-99, 0):
        if valor_invalido in dormitorios.columns:
            dormitorios = dormitorios.drop(valor_invalido, axis=1)
    dormitorios = dormitorios.astype(int)
    dormitorios.columns = [
        f"n_dormitorios_{int(i)}" if i < 6 else "n_dormitorios_6_o_mas" for i in dormitorios.columns
    ]
    for col in COLUMNAS_DORMITORIOS:
        if col not in dormitorios.columns:
            dormitorios[col] = 0
    dormitorios = dormitorios[COLUMNAS_DORMITORIOS]

    tabla_viviendas = (
        sexo.join(edades, how="left")
        .join(origen, how="left")
        .join(ciuo, how="left")
        .join(tipo_vivienda, how="left")
        .join(dormitorios, how="left")
        .join(transporte, how="left")
        .join(educacion, how="left")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    return tabla_viviendas, personas
