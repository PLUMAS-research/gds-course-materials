"""Carga y limpieza de reportes SOSAFE para análisis espacial."""

import os
import re
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

CATEGORIAS = {
    # Ambiental y mantención urbana
    20: "Árboles y ramas",
    21: "Basura",
    22: "Luminaria",
    24: "Señalización",
    26: "Semáforos",
    29: "Alcantarillado",
    41: "Veredas y pavimento",
    42: "Fuga de agua",
    49: "Quema de basura",
    50: "Mantención general",
    # Disturbios y uso del espacio público
    13: "Ruido",
    14: "Disturbios públicos",
    17: "Desorden público",
    23: "Comercio ambulante",
    28: "Graffitis",
    43: "Situación de calle",
    103: "Mal estacionamiento",
    124: "Disturbios vecinales",
    # Delitos
    0: "Robo en vía pública",
    1: "Robo de vehículo",
    2: "Robo a vivienda o local",
    6: "Actividad sospechosa",
    101: "Robo de cables",
    112: "Delito en moto",
    146: "Intento de robo",
    152: "Robo de cables",
}

GRUPOS = {
    "Árboles y ramas": "Ambiental",
    "Basura": "Ambiental",
    "Luminaria": "Ambiental",
    "Señalización": "Ambiental",
    "Semáforos": "Ambiental",
    "Alcantarillado": "Ambiental",
    "Veredas y pavimento": "Ambiental",
    "Fuga de agua": "Ambiental",
    "Quema de basura": "Ambiental",
    "Mantención general": "Ambiental",
    "Ruido": "Disturbios",
    "Disturbios públicos": "Disturbios",
    "Desorden público": "Disturbios",
    "Comercio ambulante": "Disturbios",
    "Graffitis": "Disturbios",
    "Situación de calle": "Disturbios",
    "Mal estacionamiento": "Disturbios",
    "Disturbios vecinales": "Disturbios",
    "Robo en vía pública": "Delitos",
    "Robo de vehículo": "Delitos",
    "Robo a vivienda o local": "Delitos",
    "Actividad sospechosa": "Delitos",
    "Robo de cables": "Delitos",
    "Delito en moto": "Delitos",
    "Intento de robo": "Delitos",
}

COLORES_GRUPO = {
    "Ambiental": "#2a9d8f",
    "Disturbios": "#e9c46a",
    "Delitos": "#e76f51",
}

DESCRIPCIONES_EXCLUIR = {
    "",
    "¡Ayuda! Tengo una emergencia de seguridad",
    "Prueba",
    ".",
    "Hola",
    "Help! I have a security emergency",
    "Solo comparto",
}

PREFIJOS_EXCLUIR = (
    "¡Mira este reporte",
    "¡Hola! Únete a mi grupo SOSAFE llamado",
)

PATRONES_EXCLUIR = [
    r"probando la",
    r"probando ap",
    r"probando esta",
]


def _corregir_coordenadas_string(ruta):
    """Corrige coordenadas almacenadas como strings. Retorna (ruta, es_temp)."""
    with open(ruta, encoding="utf-8") as f:
        contenido = f.read()
    patron = r'"coordinates"\s*:\s*\[\s*"[-+]?[0-9]*\.?[0-9]+".*?"[-+]?[0-9]*\.?[0-9]+"'
    if not re.search(patron, contenido):
        return ruta, False
    contenido_corregido = re.sub(
        r'"coordinates"\s*:\s*\[\s*"([-+]?[0-9]*\.?[0-9]+)"\s*,\s*"([-+]?[0-9]*\.?[0-9]+)"\s*\]',
        r'"coordinates": [\1, \2]',
        contenido,
    )
    fd, ruta_temp = tempfile.mkstemp(suffix=".geojson")
    with os.fdopen(fd, "w") as f:
        f.write(contenido_corregido)
    return ruta_temp, True


def _filtrar_spam(geodf):
    desc = geodf["description"].fillna("").str.strip()
    mascara = ~desc.isin(DESCRIPCIONES_EXCLUIR) & ~desc.str.startswith(PREFIJOS_EXCLUIR)
    for patron in PATRONES_EXCLUIR:
        mascara &= ~desc.str.contains(patron, case=False, regex=True, na=False)
    return geodf[mascara].copy()


def cargar_reportes(archivos, bbox=None):
    """Lee uno o varios GeoJSON diarios y devuelve un GeoDataFrame categorizado.

    Aplica la limpieza mínima del pipeline original (bbox, spam, bots) y
    asigna `categoria` y `grupo` según CATEGORIAS y GRUPOS.
    """
    partes = []
    for ruta in archivos:
        ruta_uso, es_temp = _corregir_coordenadas_string(str(ruta))
        try:
            gdf = gpd.read_file(ruta_uso)
        finally:
            if es_temp:
                try:
                    os.unlink(ruta_uso)
                except OSError:
                    pass
        if bbox is not None:
            gdf = gdf.assign(geometry=lambda x: x.clip_by_rect(*bbox))
            gdf = gdf[~gdf.is_empty]
        partes.append(gdf)
    if not partes:
        return gpd.GeoDataFrame()
    reportes = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=partes[0].crs)
    reportes["created_at_cl"] = pd.DatetimeIndex(
        pd.to_datetime(reportes["created_at"], dayfirst=True, utc=True)
    ).tz_convert("America/Santiago")
    reportes["hora"] = reportes["created_at_cl"].dt.hour
    reportes["dia_semana"] = reportes["created_at_cl"].dt.dayofweek
    reportes["mes"] = reportes["created_at_cl"].dt.month
    reportes = (
        reportes.pipe(_filtrar_spam)
        .drop(columns=["created_at", "uuid"], errors="ignore")
        .rename(columns={"created_at_cl": "created_at"})
        .reset_index(drop=True)
    )
    reportes["categoria"] = reportes["type"].map(CATEGORIAS)
    reportes["grupo"] = reportes["categoria"].map(GRUPOS)
    return reportes


def filtrar_relevantes(reportes):
    """Conserva solo reportes cuyo `type` está en CATEGORIAS."""
    return reportes.dropna(subset=["grupo"]).copy()


# Información personal en el texto libre de los reportes.
PATRON_CORREO = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)

# Teléfonos chilenos: móvil 9 + 8 dígitos, fijo 2 + 8 dígitos. Acepta prefijo
# de país (+56 o 56) y separadores comunes (espacio, punto, guion). Exige un
# bloque de 8 dígitos para no confundir patentes, fechas u horas con teléfonos.
PATRON_TELEFONO = re.compile(
    r"(?<!\w)\+?(?:56)?[\s.\-]?(?:9|2)?[\s.\-]?\d{4}[\s.\-]?\d{4}(?!\w)"
)


def anonimizar_texto(serie, marca_correo="[correo]", marca_telefono="[telefono]"):
    """Reemplaza correos y teléfonos en una serie de texto por marcas fijas.

    Filtro conservador: prefiere borrar de más (un RUT de 8 dígitos también cae)
    a dejar pasar información de contacto. Retorna una serie nueva.
    """
    texto = serie.fillna("")
    texto = texto.str.replace(PATRON_CORREO, marca_correo, regex=True)
    texto = texto.str.replace(PATRON_TELEFONO, marca_telefono, regex=True)
    return texto.where(serie.notna(), serie)
