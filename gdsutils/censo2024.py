import pandas as pd

territorios = pd.read_csv(
    "data/censo2024-microdatos/cod_territoriales_especificos.csv", index_col=0
)["Territorio específico"].to_dict()

dic_personas = pd.read_csv(
    "data/censo2024-microdatos/tabla_personas.csv",
    index_col="Nombre variable",
    thousands=",",
    decimal=".",
).drop("Entidad", axis=1)

dic_viviendas = pd.read_csv(
    "data/censo2024-microdatos/tabla_viviendas.csv",
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
    'RM': {
        "region": 13,
        "capital_bbox": [-70.88006218, -33.67612715, -70.43015094, -33.31069169],
        "basemap": "data/censo2024-microdatos/basemap_rm.tif",
        "basemap_capital": "data/censo2024-microdatos/basemap_rm_gran_stgo.tif",
        "nombre": "Región Metropolitana",
        "nombre_capital": "Gran Santiago",
    },
    'VALPO': {
        "region": 5,
        "capital_bbox": [-71.669428,-33.074331,-71.324728,-32.906420],
        "basemap": "data/censo2024-microdatos/basemap_valpo.tif",
        "basemap_capital": "data/censo2024-microdatos/basemap_valpo_gran_valpo.tif",
        "nombre": "Región de Valparaíso",
        "nombre_capital": "Gran Valparaíso",
    },
    'BIOBIO': {
        "region": 8,
        "capital_bbox": [-73.243089,-36.953576,-72.941832,-36.680511],
        "basemap": "data/censo2024-microdatos/basemap_biobio.tif",
        "basemap_capital": "data/censo2024-microdatos/basemap_biobio_gran_conce.tif",
        "nombre": "Región del Biobío",
        "nombre_capital": "Gran Concepción",
    },
}