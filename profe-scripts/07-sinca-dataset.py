# %%
"""Construye dataset SINCA PM2.5 para estaciones de la Región Metropolitana.

Scrapea la página `https://sinca.mma.gob.cl/index.php/region/index/id/M`
para listar estaciones de la RM, sigue cada `estacion/index/id/<n>` para
extraer nombre, coordenadas WGS84 y código interno (tipo D11, D28) usado
en el endpoint de descarga `cgi-bin/APUB-MMA/apub.tsindico2.cgi`. Luego
baja PM2.5 promedio diario para todas las estaciones a lo largo del
invierno austral 2024 (mayo a agosto), arma un parquet en formato largo
y un parquet con metadatos de estación.

Salidas
-------
- `data/sinca-santiago-2024/pm25-diario.parquet`: filas
  `(codigo, fecha, pm25)`.
- `data/sinca-santiago-2024/estaciones.parquet`: metadatos
  `(codigo, id, nombre, comuna, lat, lon)`.
- `data/sinca-santiago-2024.tgz`: empaquetado para subir al servidor.
- Subida por scp al servidor del curso si `GDS_SUBIR=1`.

Uso
---
    uv run python profe-scripts/07-sinca-dataset.py

Si SINCA cambia el HTML o el endpoint cambia, los regex `RE_*` y la URL
`URL_DESCARGA` son los lugares para ajustar.
"""

# %% Imports
import html
import re
import sys
import time
import urllib.request
from io import StringIO
from pathlib import Path

_AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(_AQUI.parent))
sys.path.insert(0, str(_AQUI))

import config
import pandas as pd

# %% Configuración
NOMBRE_DATASET = "sinca-santiago-2024"
DIR_SALIDA = Path("data") / NOMBRE_DATASET
RUTA_PM25 = DIR_SALIDA / "pm25-diario.parquet"
RUTA_ESTACIONES = DIR_SALIDA / "estaciones.parquet"

URL_REGION_RM = "https://sinca.mma.gob.cl/index.php/region/index/id/M"
URL_ESTACION = "https://sinca.mma.gob.cl/index.php/estacion/index/id/{id}"
URL_DESCARGA = (
    "https://sinca.mma.gob.cl/cgi-bin/APUB-MMA/apub.tsindico2.cgi"
    "?outtype=txt"
    "&macro=./RM/{codigo}/Cal/PM25/PM25.diario.diario.ic"
    "&from={desde}&to={hasta}"
    "&path=/usr/airviro/data/CONAMA/&lang=esp"
)
USER_AGENT = "Mozilla/5.0 (gds-course-materials profe-script)"

# Ventana de invierno austral 2024 (formato YYMMDD para SINCA).
DESDE = "240501"
HASTA = "240831"

DIR_SALIDA.mkdir(parents=True, exist_ok=True)


# %%
# ========================================
# Paso 1. Listar estaciones de la RM
# ========================================
# La página de la región lista anchors `/index.php/estacion/index/id/<n>`,
# uno por estación. No hay un JSON publico, pero el HTML es estable y
# regex basta.
RE_ID_ESTACION = re.compile(r"estacion/index/id/(\d+)")
RE_LATLNG = re.compile(r"LatLng\(([-\d.]+),\s*([-\d.]+)\)")
RE_CODIGO = re.compile(r"\.\/RM\/([A-Z0-9]+)/")
RE_TITULO = re.compile(r"<title>Estaci[oó]n\s+([^<]+?)\s*-\s*Sistema", re.I)
RE_COMUNA = re.compile(
    r"<th[^>]*>\s*Comuna\s*</th>\s*<td[^>]*>\s*([^<]+?)\s*</td>", re.I | re.S
)


def obtener_html(url, intentos=3, espera=2.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"  reintento {i + 1}/{intentos} tras error: {e}")
            time.sleep(espera * (i + 1))


print(f"[1/4] Listando estaciones desde {URL_REGION_RM}")
html_region = obtener_html(URL_REGION_RM)
ids_estaciones = sorted({int(m) for m in RE_ID_ESTACION.findall(html_region)})
print(f"  estaciones encontradas: {len(ids_estaciones)}")


# %%
# ========================================
# Paso 2. Metadatos por estación (nombre, comuna, lat/lon, código)
# ========================================
# Cada página /estacion/index/id/<n> contiene:
# - <title>Estación NOMBRE - Sistema...</title>
# - LatLng(lat, lon) en JS embebido
# - macropath=./RM/<CODIGO>/Cal/<param> en los enlaces de descarga.
# El código interno (D11, D28...) es el que el endpoint cgi-bin acepta;
# el id numérico solo sirve para enlazar dentro del sitio.
print("[2/4] Extrayendo metadatos por estación")
filas = []
for i_est, id_est in enumerate(ids_estaciones):
    url = URL_ESTACION.format(id=id_est)
    try:
        html_estacion = obtener_html(url)
    except Exception as e:
        print(f"  id={id_est}: error al bajar página, salto: {e}")
        continue

    m_lat = RE_LATLNG.search(html_estacion)
    m_cod = RE_CODIGO.search(html_estacion)
    m_tit = RE_TITULO.search(html_estacion)
    m_com = RE_COMUNA.search(html_estacion)
    if not (m_lat and m_cod):
        print(f"  id={id_est}: sin lat/lon o código, salto")
        continue

    fila = {
        "id": id_est,
        "codigo": m_cod.group(1),
        "nombre": html.unescape(m_tit.group(1).strip())
        if m_tit
        else f"Estacion {id_est}",
        "comuna": html.unescape(m_com.group(1).strip()) if m_com else None,
        "lat": float(m_lat.group(1)),
        "lon": float(m_lat.group(2)),
    }
    filas.append(fila)
    print(
        f"  [{i_est + 1}/{len(ids_estaciones)}] id={id_est} "
        f"codigo={fila['codigo']} {fila['nombre']} "
        f"({fila['lat']:.4f}, {fila['lon']:.4f})"
    )
    time.sleep(0.3)

estaciones = pd.DataFrame(filas)
if estaciones.empty:
    raise RuntimeError("No se obtuvieron estaciones de la RM")
estaciones = estaciones.drop_duplicates(subset="codigo").reset_index(drop=True)
print(f"  estaciones únicas por código: {len(estaciones)}")


# %%
# ========================================
# Paso 3. Descargar PM2.5 diario por estación
# ========================================
# El endpoint cgi-bin retorna un archivo de texto con encabezado de
# metadatos terminado en "EOH" y el bloque de datos delimitado por
# "#DATA" y "EOF". Cada fila trae:
#     YYMMDD, HHMM, validados, preliminares, no_validados,
# Donde validados/preliminares/no_validados pueden estar vacíos. Para
# el promedio diario, HHMM siempre es 0000 y nos quedamos con el primer
# valor disponible (validado > preliminar > no validado), reflejando la
# política de visualización del propio SINCA.
RE_DATA = re.compile(r"#DATA\s*\n(.*?)(?:^EOF|^#)", re.S | re.M)


def parsear_diario(texto, codigo):
    bloque = RE_DATA.search(texto)
    if not bloque:
        return pd.DataFrame(columns=["codigo", "fecha", "pm25"])
    csv = bloque.group(1)
    df = pd.read_csv(
        StringIO(csv),
        header=None,
        names=["fecha_raw", "hora", "validado", "preliminar", "no_validado", "_"],
        engine="python",
        skipinitialspace=True,
    )
    df["fecha"] = pd.to_datetime(
        df["fecha_raw"].astype(str), format="%y%m%d", errors="coerce"
    )
    for col in ["validado", "preliminar", "no_validado"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pm25"] = (
        df["validado"].combine_first(df["preliminar"]).combine_first(df["no_validado"])
    )
    df = df.dropna(subset=["fecha"])
    df = df[["fecha", "pm25"]].copy()
    df["codigo"] = codigo
    return df[["codigo", "fecha", "pm25"]]


print(f"[3/4] Descargando PM2.5 diario {DESDE} a {HASTA}")
piezas = []
for i_est, fila in enumerate(estaciones.itertuples(index=False)):
    url = URL_DESCARGA.format(codigo=fila.codigo, desde=DESDE, hasta=HASTA)
    try:
        texto = obtener_html(url)
    except Exception as e:
        print(f"  {fila.codigo} {fila.nombre}: error de descarga, salto: {e}")
        continue
    df = parsear_diario(texto, fila.codigo)
    n_validos = int(df["pm25"].notna().sum())
    print(
        f"  [{i_est + 1}/{len(estaciones)}] {fila.codigo} {fila.nombre}: "
        f"{n_validos} días con PM2.5"
    )
    if n_validos > 0:
        piezas.append(df)
    time.sleep(0.3)

if not piezas:
    raise RuntimeError("Ninguna estación devolvió PM2.5 en el rango")
pm25 = pd.concat(piezas, ignore_index=True)
pm25 = pm25.dropna(subset=["pm25"]).reset_index(drop=True)
print(f"  filas totales con PM2.5 válido: {len(pm25):,}")


# %%
# ========================================
# Paso 4. Resumen, guardado, empaque y subida
# ========================================
# Para que la clase pueda elegir el día más útil, mostrar las fechas con
# mayor promedio y mayor cobertura de estaciones simultáneas.
resumen = (
    pm25.groupby("fecha")
    .agg(estaciones_activas=("codigo", "nunique"), pm25_medio=("pm25", "mean"))
    .reset_index()
    .sort_values("pm25_medio", ascending=False)
)
print("Top 10 días por PM2.5 medio en la red:")
top = resumen.head(10).copy()
top["pm25_medio"] = top["pm25_medio"].round(1)
print(top.to_string(index=False))

# Filtrar metadatos de estaciones a las que efectivamente reportaron.
codigos_validos = pm25["codigo"].unique()
estaciones = estaciones[estaciones["codigo"].isin(codigos_validos)].reset_index(
    drop=True
)
print(f"\nEstaciones con datos PM2.5 en el rango: {len(estaciones)}")

print(f"\n[4/4] Guardando")
estaciones.to_parquet(RUTA_ESTACIONES)
print(f"  {RUTA_ESTACIONES} ({len(estaciones)} filas)")
pm25.to_parquet(RUTA_PM25)
print(f"  {RUTA_PM25} ({len(pm25):,} filas)")

# SINCA puede traer huecos por estación: verificar el parquet antes de
# publicar (GDS_SUBIR=1).
config.publicar(NOMBRE_DATASET, DIR_SALIDA)
