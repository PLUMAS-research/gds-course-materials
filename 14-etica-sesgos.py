# %%
"""Clase 14: unicidad de trayectorias como caso de reidentificación.

Replica el argumento de de Montjoye et al. (2013): cuántas personas quedan
identificadas de forma única según cuántos puntos espacio-temporales de su
trayectoria se conozcan. Primero con trayectorias sintéticas (curva limpia) y
luego sobre la EOD Santiago 2012.

Advertencia central: la EOD captura un solo día, así que esto demuestra el
mecanismo de unicidad, no la reidentificación multi-día del CDR original. Aun
así, con pocos puntos una fracción alta de personas queda identificada de forma
única.
"""

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/eod2012.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/foursquare-santiago.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.geo.grid import h3_grid_from_bounds
from chiricoca.maps import choropleth_map
from matplotlib.colors import BoundaryNorm
from shapely.geometry import box

import gdsutils.eodscl as eod
from gdsutils.ndvi import BBOX_SANTIAGO

setup_style(dpi=128)

# Tipografía Urbanist y figuras chicas a alto DPI (texto relativamente más grande).
from matplotlib import font_manager  # noqa: E402

for _f in Path.home().glob(".fonts/Urbanist/*.otf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": "Urbanist", "font.size": 13, "axes.titlesize": 14,
    "legend.fontsize": 11, "savefig.dpi": 300, "figure.dpi": 300,
})

EOD_PATH = Path("data") / "eod2012" / "EOD_STGO"
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)
SEMILLA = 42

# bbox del Gran Santiago: mapas y filtros se recortan a esta caja.
CRS_UTM = "EPSG:32719"
CAJA_UTM = gpd.GeoSeries([box(*BBOX_SANTIAGO)], crs="EPSG:4326").to_crs(CRS_UTM).iloc[0]
_b = CAJA_UTM.bounds  # (xmin, ymin, xmax, ymax) en UTM 19S
XLIM, YLIM = (_b[0], _b[2]), (_b[1], _b[3])

# Comunas (cartografía del censo) como contexto geográfico de los mapas.
RUTA_CARTO = (
    Path("data") / "censo2024-cartografia" / "Cartografia_censo2024_Pais_Comunal.parquet"
)
comunas = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(CRS_UTM)
comunas = comunas.rename_geometry("geometry")[["COMUNA", "geometry"]]
comunas_bb = comunas[comunas.intersects(CAJA_UTM)]

# %%
# ========================================
# PARTE 1 — Mecánica de la unicidad
# ========================================
# Un "punto espacio-temporal" es un par (lugar, franja horaria): dónde estuvo
# una persona y cuándo. Conociendo k de esos puntos de una persona, ella es
# ÚNICA si ninguna otra trayectoria de la base contiene esos k puntos. Si la
# fracción de personas únicas es alta con k chico, borrar el nombre no
# anonimiza: la propia trayectoria identifica.
#
# Para evaluarlo sin comparar todas las personas contra todas, usamos un índice
# invertido punto -> personas que lo tienen. Los candidatos compatibles con k
# puntos son la intersección de esos conjuntos; la persona es única si la
# intersección la contiene solo a ella.


def indice_invertido(serie_puntos):
    """Mapea cada punto espacio-temporal al conjunto de personas que lo tienen."""
    idx = {}
    for persona, puntos in serie_puntos.items():
        for punto in puntos:
            idx.setdefault(punto, set()).add(persona)
    return idx


def fraccion_unicos(serie_puntos, k, indice, semilla=SEMILLA, n_objetivo=None):
    """Fracción de personas únicas conocidos k puntos al azar de su trayectoria.

    El índice se construye sobre TODAS las personas; `n_objetivo` solo limita
    cuántas se evalúan como objetivo (estimación insesgada para bases grandes).
    """
    rng = np.random.default_rng(semilla)
    objetivos = serie_puntos
    if n_objetivo is not None and n_objetivo < len(serie_puntos):
        objetivos = serie_puntos.sample(n=n_objetivo, random_state=semilla)
    unicos = 0
    for puntos in objetivos:
        items = list(puntos)
        sel = rng.choice(len(items), min(k, len(items)), replace=False)
        # Intersección de candidatos, empezando por el punto más raro.
        conjuntos = sorted((indice[items[i]] for i in sel), key=len)
        candidatos = set(conjuntos[0])
        for c in conjuntos[1:]:
            candidatos &= c
            if len(candidatos) == 1:
                break
        unicos += len(candidatos) == 1
    return unicos / len(objetivos)


def curva(serie_puntos, ks, semilla=SEMILLA, n_objetivo=None):
    """Curva de unicidad: fracción única en función de k (índice construido una vez)."""
    indice = indice_invertido(serie_puntos)
    return pd.Series(
        {k: fraccion_unicos(serie_puntos, k, indice, semilla, n_objetivo) for k in ks},
        name="unicos",
    )


# %%
# ========================================
# PARTE 2 — Curva sintética (referencia limpia)
# ========================================
# Población artificial: cada persona visita `n_puntos` pares (zona, hora) al
# azar. Mostramos dos resoluciones espaciales (zonas finas vs gruesas) para
# anticipar la lección: con pocos puntos la unicidad es alta incluso en grillas
# gruesas.


def trayectorias_sinteticas(n_personas, n_puntos, n_zonas, n_horas, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    filas = []
    for pid in range(n_personas):
        zonas = rng.integers(0, n_zonas, n_puntos)
        horas = rng.integers(0, n_horas, n_puntos)
        filas.append(frozenset(zip(zonas.tolist(), horas.tolist())))
    return pd.Series(filas, name="puntos")

ks = list(range(1, 7))
sint_fina = trayectorias_sinteticas(50_000, n_puntos=6, n_zonas=866, n_horas=24)
sint_gruesa = trayectorias_sinteticas(50_000, n_puntos=6, n_zonas=52, n_horas=8)
curva_fina = curva(sint_fina, ks)
curva_gruesa = curva(sint_gruesa, ks)

print("Curva sintética (fracción única):")
print(pd.DataFrame({"zonas=866, horas=24": curva_fina,
                    "zonas=52, horas=8": curva_gruesa}).round(3))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, curva_fina.values, "o-", color="#0A0E50", label="866 zonas, 24 horas")
ax.plot(ks, curva_gruesa.values, "s--", color="#CF3889", label="52 zonas, 8 franjas")
ax.axvline(4, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Puntos conocidos (k)")
ax.set_ylabel("Fracción de personas únicas")
ax.set_ylim(0, 1.02)
ax.set_title("Unicidad sobre trayectorias sintéticas")
ax.legend(frameon=False)
fig.savefig(IMAGES / "14-unicidad-sintetica.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 3 — Construir puntos espacio-temporales desde la EOD
# ========================================
# Cada viaje aporta dos visitas: origen al inicio y destino al final. El
# conjunto de pares (lugar, franja) por persona es su trayectoria del día.
print("\nLeyendo viajes EOD 2012")
viajes = eod.read_trips(EOD_PATH)
print(f"  viajes válidos: {len(viajes):,}")


def _hora_desde_timedelta(serie):
    return (serie.dt.total_seconds() // 3600 % 24).astype("Int64")


def _hora_desde_texto(serie):
    td = pd.to_timedelta(serie.astype(str) + ":00", errors="coerce")
    return (td.dt.total_seconds() // 3600 % 24).astype("Int64")


viajes = viajes.assign(
    hora_ini=_hora_desde_timedelta(viajes["HoraIni"]),
    hora_fin=_hora_desde_texto(viajes["HoraFin"]),
)


def construir_puntos(viajes, col_loc_o, col_loc_d, ancho_franja):
    """Serie por persona con su conjunto de pares (lugar, franja horaria).

    `ancho_franja` en horas: 1 = resolución horaria, 3 = bloques de 3 horas.
    """
    origen = viajes[["Persona", col_loc_o, "hora_ini"]].rename(
        columns={col_loc_o: "lugar", "hora_ini": "hora"}
    )
    destino = viajes[["Persona", col_loc_d, "hora_fin"]].rename(
        columns={col_loc_d: "lugar", "hora_fin": "hora"}
    )
    largo = pd.concat([origen, destino], ignore_index=True).dropna(
        subset=["lugar", "hora"]
    )
    largo["franja"] = (largo["hora"].astype(int) // ancho_franja)
    largo["punto"] = list(zip(largo["lugar"], largo["franja"]))
    return largo.groupby("Persona")["punto"].apply(frozenset)


puntos_base = construir_puntos(viajes, "ZonaOrigen", "ZonaDestino", ancho_franja=1)
print(f"  personas con trayectoria: {len(puntos_base):,}")
print(f"  puntos por persona: mediana={puntos_base.apply(len).median():.0f}, "
      f"máx={puntos_base.apply(len).max()}")

# %%
# ========================================
# PARTE 4 — Unicidad sobre la EOD y efecto de la resolución
# ========================================
# Variamos la celda espacial (zona de 866 vs comuna) y la franja horaria (1h vs
# 3h) para ver cuánto protege bajar la resolución.
configs = {
    "Zona, franja 1 h": ("ZonaOrigen", "ZonaDestino", 1),
    "Zona, franja 3 h": ("ZonaOrigen", "ZonaDestino", 3),
    "Comuna, franja 1 h": ("ComunaOrigen", "ComunaDestino", 1),
    "Comuna, franja 3 h": ("ComunaOrigen", "ComunaDestino", 3),
}

curvas_eod = {}
for nombre, (co, cd, ancho) in configs.items():
    pts = construir_puntos(viajes, co, cd, ancho)
    curvas_eod[nombre] = curva(pts, ks)

tabla_eod = pd.DataFrame(curvas_eod)
print("\nCurva de unicidad EOD (fracción única por k):")
print(tabla_eod.round(3))
print(f"\nCon k=2 puntos, fracción única (zona, 1 h): "
      f"{tabla_eod.loc[2, 'Zona, franja 1 h']:.1%}")
print(f"Con k=4 puntos, fracción única (zona, 1 h): "
      f"{tabla_eod.loc[4, 'Zona, franja 1 h']:.1%}")

estilos = {
    "Zona, franja 1 h": ("o-", "#0A0E50"),
    "Zona, franja 3 h": ("o--", "#0A0E50"),
    "Comuna, franja 1 h": ("s-", "#CF3889"),
    "Comuna, franja 3 h": ("s--", "#CF3889"),
}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for nombre, serie in curvas_eod.items():
    fmt, color = estilos[nombre]
    ax.plot(ks, serie.values, fmt, color=color, label=nombre,
            alpha=0.6 if "3 h" in nombre else 1.0)
ax.set_xlabel("Puntos conocidos (k)")
ax.set_ylabel("Fracción de personas únicas")
ax.set_ylim(0, 1.02)
ax.set_title("Unicidad de trayectorias en la EOD 2012 (un día)")
ax.legend(frameon=False, loc="lower right")
fig.savefig(IMAGES / "14-unicidad-eod.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 5 — Casos concretos: trayectorias que quedan únicas
# ========================================
# Las curvas muestran el fenómeno agregado. Para discutirlo conviene un caso
# concreto: una persona real de la EOD cuya trayectoria del día no la comparte
# nadie más. Seleccionamos personas con pocos puntos cuya trayectoria completa
# es única (a resolución de zona y franja de 1 hora) y las dibujamos sobre la
# zonificación EOD.
zonas = eod.read_zone_design(EOD_PATH)
zonas = zonas[zonas["Zona"].notna()].copy()
zona_a_centroide = {
    int(z): (g.centroid.x, g.centroid.y)
    for z, g in zip(zonas["Zona"], zonas.geometry)
}
zona_a_comuna = {int(z): c for z, c in zip(zonas["Zona"], zonas["Comuna"])}

indice_base = indice_invertido(puntos_base)
COLORES_EJEMPLO = ["#0A0E50", "#CF3889", "#1b9e77"]


def seleccionar_ejemplos(serie_puntos, indice, n=6, min_lugares=3, max_pts=8, semilla=SEMILLA):
    """Personas con una trayectoria de varios lugares distintos, única, en comunas distintas."""
    rng = np.random.default_rng(semilla)
    personas = list(serie_puntos.index)
    rng.shuffle(personas)
    ejemplos = []
    comunas_usadas = set()
    for persona in personas:
        pts = sorted(serie_puntos[persona], key=lambda t: t[1])
        lugares = {int(round(z)) for z, _ in pts}
        if len(pts) > max_pts or len(lugares) < min_lugares:
            continue
        if any(z not in zona_a_centroide for z in lugares):
            continue
        candidatos = set(indice[pts[0]])
        for p in pts[1:]:
            candidatos &= indice[p]
        if len(candidatos) != 1:
            continue
        comuna = zona_a_comuna.get(int(round(pts[0][0])))
        if comuna in comunas_usadas:
            continue
        comunas_usadas.add(comuna)
        ejemplos.append((persona, pts))
        if len(ejemplos) >= n:
            break
    return ejemplos


def grilla_ejemplos(ejemplos, dibujar, titulo_de, suptitle, ruta, ncols=3):
    """Grilla de mini-mapas, uno por persona, con su trayectoria sobre las comunas.

    Usa ejes independientes (no `small_multiples`, que comparte ejes) para poder
    hacer zoom distinto en cada panel.
    """
    nrows = (len(ejemplos) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(ejemplos):]:
        ax.set_visible(False)
    for ax, ej in zip(axes, ejemplos):
        comunas_bb.boundary.plot(ax=ax, color="0.8", linewidth=0.4)
        dibujar(ax, ej)
        ax.set_title(titulo_de(ej), fontsize=10)
        ax.set_axis_off()
    fig.suptitle(suptitle)
    fig.savefig(ruta, bbox_inches="tight")


def _zoom(ax, xs, ys, margen_min=900):
    """Acerca el panel a la trayectoria, con un margen para ver la comuna."""
    m = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.3
    m = max(m, margen_min)
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")


def _dibujar_eod(ax, ej):
    _, pts = ej
    xy = [zona_a_centroide[int(round(z))] for z, _ in pts]
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#CF3889", lw=2.2, alpha=0.9))
    ax.scatter([p[0] for p in xy], [p[1] for p in xy], color="#0A0E50", s=45, zorder=4)
    for (x, y), (_, f) in zip(xy, pts):
        ax.annotate(f"{int(f)}h", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color="#0A0E50")
    _zoom(ax, [p[0] for p in xy], [p[1] for p in xy])


ejemplos = seleccionar_ejemplos(puntos_base, indice_base, n=6, min_lugares=3, max_pts=8)
print(f"\nEjemplos EOD únicos con su trayectoria completa: {len(ejemplos)}")
grilla_ejemplos(
    ejemplos, _dibujar_eod,
    lambda ej: f"{zona_a_comuna.get(int(round(ej[1][0][0])), '?').title()}: "
               f"{len({int(round(z)) for z, _ in ej[1]})} lugares",
    "Cada día es una huella: trayectorias EOD únicas entre 38.719 personas",
    IMAGES / "14-unicidad-ejemplos.png",
)

# %%
# ========================================
# PARTE 6 — El embudo de reidentificación
# ========================================
# Para esas mismas personas, cuántas siguen siendo compatibles a medida que se
# revela un punto más de su trayectoria, en orden cronológico. La población
# arranca en decenas de miles y colapsa a una sola persona en pocos pasos.
total = len(puntos_base)
max_pasos = 0
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for (persona, pts), color in zip(ejemplos, COLORES_EJEMPLO):
    # Antes de revelar nada, cualquiera es candidato (población completa).
    tamanos = [total]
    candidatos = None
    for p in pts:
        conj = indice_base[p]
        candidatos = set(conj) if candidatos is None else (candidatos & conj)
        tamanos.append(len(candidatos))
    max_pasos = max(max_pasos, len(tamanos) - 1)
    comuna = zona_a_comuna.get(int(round(pts[0][0])), "?")
    ax.plot(range(len(tamanos)), tamanos, "o-", color=color, label=comuna)
    ax.annotate("única", (len(tamanos) - 1, tamanos[-1]), textcoords="offset points",
                xytext=(6, 0), fontsize=9, color=color, va="center")
ax.axhline(1, color="gray", linestyle=":", linewidth=1)
ax.set_yscale("log")
ax.set_xticks(range(max_pasos + 1))
ax.set_xlabel("Puntos revelados (orden cronológico)")
ax.set_ylabel("Personas compatibles")
ax.set_title("De toda la ciudad a una persona en pocos puntos")
ax.legend(frameon=False, title="Persona (comuna de origen)")
fig.savefig(IMAGES / "14-unicidad-embudo.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 7 — Foursquare multi-día: carga de datos
# ========================================
# La EOD es de un solo día y por eso su curva satura: no hay puntos suficientes.
# Los check-ins de Foursquare (Yang et al., WWW'19) cubren 22 meses en Santiago,
# que es el escenario multi-día de de Montjoye.
print("\nLeyendo check-ins Foursquare Santiago")
DIR_4SQ = Path("data") / "foursquare-santiago"
checkins = pd.read_parquet(DIR_4SQ / "checkins.parquet")
venues = pd.read_parquet(DIR_4SQ / "venues.parquet")
checkins = checkins.merge(venues[["venue_id", "lat", "lon"]], on="venue_id")
print(f"  check-ins: {len(checkins):,} | usuarios: {checkins['user_id'].nunique():,}")
print(f"  rango: {checkins['datetime_local'].min()} a {checkins['datetime_local'].max()}")

# Celda H3 fina (res 9) y sus padres (8, 7): así una sola llamada por check-in.
cel9 = [h3.latlng_to_cell(la, lo, 9) for la, lo in zip(checkins["lat"], checkins["lon"])]
checkins["h3_9"] = cel9
checkins["h3_8"] = [h3.cell_to_parent(c, 8) for c in cel9]
checkins["h3_7"] = [h3.cell_to_parent(c, 7) for c in cel9]

# %%
# ========================================
# PARTE 8 — Qué captura cada fuente: EOD vs Foursquare
# ========================================
# Antes de modelar conviene ver en qué se parecen y en qué difieren las dos
# fuentes. No se comparan en niveles (una es una encuesta de un día, la otra son
# 22 meses de check-ins voluntarios), sino en forma: dónde, cuándo y cuánto
# registra cada una. Las diferencias son sesgo de cobertura, no de la ciudad.

# --- Cobertura espacial por zona EOD (proporción del total de cada fuente) ---
venues_geo = gpd.GeoDataFrame(
    venues, geometry=gpd.points_from_xy(venues["lon"], venues["lat"]),
    crs="EPSG:4326",
).to_crs(CRS_UTM)
zonas_id = zonas.assign(zona_id=zonas["Zona"].astype(int))
venue_zona = gpd.sjoin(
    venues_geo[["venue_id", "geometry"]],
    zonas_id[["zona_id", "geometry"]],
    predicate="within",
)[["venue_id", "zona_id"]]
fsq_por_zona = (
    checkins.merge(venue_zona, on="venue_id").groupby("zona_id").size()
)
eod_endpoints = pd.concat([viajes["ZonaOrigen"], viajes["ZonaDestino"]]).dropna().astype(int)
eod_por_zona = eod_endpoints.value_counts()

zs = zonas_id.copy()
zs["eod_share"] = zs["zona_id"].map(eod_por_zona).fillna(0)
zs["fsq_share"] = zs["zona_id"].map(fsq_por_zona).fillna(0)
zs["eod_share"] /= zs["eod_share"].sum()
zs["fsq_share"] /= zs["fsq_share"].sum()
zs_bb = zs[zs.intersects(CAJA_UTM)].copy()
zs_bb["eod_log"] = np.log10(zs_bb["eod_share"].replace(0, np.nan))
zs_bb["fsq_log"] = np.log10(zs_bb["fsq_share"].replace(0, np.nan))

vmin = float(np.nanmin([zs_bb["eod_log"].min(), zs_bb["fsq_log"].min()]))
vmax = float(np.nanmax([zs_bb["eod_log"].max(), zs_bb["fsq_log"].max()]))
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, col, titulo in [(axes[0], "eod_log", "EOD 2012 (viajes)"),
                        (axes[1], "fsq_log", "Foursquare (check-ins)")]:
    zs_bb.plot(column=col, cmap="magma", ax=ax, vmin=vmin, vmax=vmax,
               legend=True, missing_kwds={"color": "0.92"},
               legend_kwds={"label": "log10 proporción", "shrink": 0.6})
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(titulo)
fig.suptitle("Cobertura espacial por zona (proporción de cada fuente)")
fig.savefig(IMAGES / "14-comparacion-cobertura.png", bbox_inches="tight")

# --- Misma cobertura, comparada zona a zona ---
amb = zs[(zs["eod_share"] > 0) & (zs["fsq_share"] > 0)]
rho = amb["eod_share"].corr(amb["fsq_share"], method="spearman")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(amb["eod_share"], amb["fsq_share"], s=10, alpha=0.4, color="#0A0E50")
lim = [min(amb["eod_share"].min(), amb["fsq_share"].min()),
       max(amb["eod_share"].max(), amb["fsq_share"].max())]
ax.plot(lim, lim, "--", color="0.5", linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Proporción de viajes EOD")
ax.set_ylabel("Proporción de check-ins Foursquare")
ax.set_title(f"Actividad por zona (Spearman = {rho:.2f})")
fig.savefig(IMAGES / "14-comparacion-zonas.png", bbox_inches="tight")
print(f"\nCorrelación espacial EOD vs Foursquare (zonas en común): rho={rho:.2f}")

# --- Ritmo horario y volumen por persona ---
# EOD: hora de salida (HoraIni) y de llegada (HoraFin). Foursquare marca la
# llegada al lugar del check-in, así que debería parecerse más a la llegada EOD.
eod_salida = viajes["hora_ini"].dropna().astype(int).value_counts(normalize=True).sort_index()
eod_llegada = viajes["hora_fin"].dropna().astype(int).value_counts(normalize=True).sort_index()
fsq_hora = checkins["datetime_local"].dt.hour.value_counts(normalize=True).sort_index()
eod_pp = viajes.groupby("Persona").size().sort_values()
fsq_pp = checkins.groupby("user_id").size().sort_values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(eod_salida.index, eod_salida.values, "o-", color="#1b9e77", label="EOD: salida")
ax1.plot(eod_llegada.index, eod_llegada.values, "^--", color="#0A0E50", label="EOD: llegada")
ax1.plot(fsq_hora.index, fsq_hora.values, "s-", color="#CF3889", label="Foursquare: check-in")
ax1.set_xlabel("Hora local")
ax1.set_ylabel("Proporción")
ax1.set_title("Ritmo horario")
ax1.legend(frameon=False)
for serie, color, etiqueta in [(eod_pp, "#1b9e77", "EOD: viajes/persona"),
                               (fsq_pp, "#CF3889", "Foursquare: check-ins/persona")]:
    ecdf = np.arange(1, len(serie) + 1) / len(serie)
    ax2.plot(serie.values, ecdf, color=color, label=etiqueta)
ax2.set_xscale("log")
ax2.set_xlabel("Registros por persona")
ax2.set_ylabel("Fracción acumulada de personas")
ax2.set_title("Volumen por persona")
ax2.legend(frameon=False, loc="lower right")
fig.suptitle("Qué captura cada fuente: tiempo y volumen")
fig.savefig(IMAGES / "14-comparacion-perfiles.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 9 — Qué son los venues y cómo se relacionan con el propósito EOD
# ========================================
# Los venues traen una categoría (Foursquare). Las categorías más visitadas
# mapean de forma natural a los propósitos de viaje de la EOD: oficinas a
# empleo, universidades a estudio, malls y supermercados a compras, hospitales a
# salud, casas a hogar. Primero vemos qué categorías concentran los check-ins.
ci_cat = checkins.merge(venue_zona, on="venue_id").merge(
    venues[["venue_id", "category"]], on="venue_id"
)
cat_checkins = ci_cat["category"].value_counts()
print(f"\nVenues: {venues['category'].nunique()} categorías. Top check-ins:")
print(cat_checkins.head(12).to_string())

# Mapeo de las categorías más frecuentes a los grupos de propósito de la EOD.
CAT_A_GRUPO = {
    "Home (private)": "Hogar",
    "Residential Building (Apartment / Condo)": "Hogar",
    "Office": "Empleo/Estudio", "Coworking Space": "Empleo/Estudio",
    "University": "Empleo/Estudio",
    "Mall": "Cuidado", "Grocery Store": "Cuidado",
    "Hospital": "Cuidado", "Medical Center": "Cuidado",
    "Gym": "Personal", "Coffee Shop": "Personal", "Bar": "Personal",
    "Restaurant": "Personal", "Park": "Personal", "Plaza": "Personal",
    "Other Great Outdoors": "Personal",
    "Subway": "Transporte", "Bus Station": "Transporte", "Airport": "Transporte",
    "Building": "Otro",
}
GRUPO_COLOR = {**eod.COLORES_PROPOSITO, "Transporte": "#7570b3", "Otro": "#444444"}

top_bar = cat_checkins.head(20)[::-1]
colores = [GRUPO_COLOR[CAT_A_GRUPO.get(c, "Otro")] for c in top_bar.index]
fig, ax = plt.subplots(figsize=(7, 7))
ax.barh(range(len(top_bar)), top_bar.values, color=colores)
ax.set_yticks(range(len(top_bar)))
ax.set_yticklabels(top_bar.index, fontsize=10)
ax.set_xlabel("Check-ins")
ax.set_title("Categorías de venues más visitadas en Santiago")
grupos_presentes = dict.fromkeys(CAT_A_GRUPO[c] for c in top_bar.index if c in CAT_A_GRUPO)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRUPO_COLOR[g]) for g in grupos_presentes],
          labels=list(grupos_presentes), frameon=False, title="Propósito EOD asociado")
fig.savefig(IMAGES / "14-foursquare-categorias.png", bbox_inches="tight")

# --- Correspondencia espacial: propósito de destino EOD vs categoría 4sq ---
# Por zona: composición de propósitos de los viajes que llegan (EOD) y
# composición de categorías de los check-ins (Foursquare). Si una categoría
# acompaña a un propósito, sus proporciones por zona deben correlacionar.
dest = viajes.assign(grupo=viajes["Proposito"].map(eod.GRUPOS_PROPOSITOS))
dest = dest.dropna(subset=["ZonaDestino", "grupo"])
dest = dest.assign(ZonaDestino=dest["ZonaDestino"].astype(int))
purp = dest.groupby(["ZonaDestino", "grupo"]).size().unstack(fill_value=0)
purp_share = purp.div(purp.sum(axis=1), axis=0)

top_cats = cat_checkins.head(12).index
total_zona = ci_cat.groupby("zona_id").size()
catz = (
    ci_cat[ci_cat["category"].isin(top_cats)]
    .groupby(["zona_id", "category"]).size().unstack(fill_value=0)
)
cat_share = catz.div(total_zona, axis=0).reindex(columns=top_cats).fillna(0)

zonas_ok = purp.index[purp.sum(axis=1) >= 50].intersection(
    total_zona.index[total_zona >= 50]
)
P, C = purp_share.loc[zonas_ok], cat_share.loc[zonas_ok]
grupos = [g for g in eod.PROPOSITOS if g in P.columns]
corr = pd.DataFrame(
    {c: {g: P[g].corr(C[c], method="spearman") for g in grupos} for c in top_cats}
)
print(f"\nCorrespondencia espacial propósito-categoría ({len(zonas_ok)} zonas):")
print(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(11, 4.2))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6, annot=True,
            fmt=".2f", annot_kws={"size": 8}, cbar_kws={"label": "Spearman"}, ax=ax)
ax.set_xlabel("Categoría de venue (Foursquare)")
ax.set_ylabel("Propósito de destino (EOD)")
ax.set_title("Correspondencia espacial: categoría de venue y propósito de viaje")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
fig.savefig(IMAGES / "14-categoria-proposito.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 10 — Dónde viven: hogar en EOD vs Foursquare
# ========================================
# La EOD trae el hogar declarado de cada persona. En Foursquare lo inferimos
# como la zona donde más hace check-in de noche (20 a 7 h): es un proxy ruidoso
# (la vida nocturna lo sesga hacia el centro-oriente), parte de la lección sobre
# lo difícil que es detectar el hogar con datos pasivos. Comparar ambas
# distribuciones muestra a quién sobre o sub-representa la huella de Foursquare.
personas = eod.read_people(EOD_PATH)
hogares = eod.read_homes(EOD_PATH)
home_eod = personas.merge(hogares[["Hogar", "Zona"]], on="Hogar").dropna(subset=["Zona"])
home_eod["Zona"] = home_eod["Zona"].astype(int)
eod_hogar = home_eod.groupby("Zona")["FactorPersona"].sum()

ci_zona = checkins.merge(venue_zona, on="venue_id")
ci_zona["hora"] = ci_zona["datetime_local"].dt.hour
noche = ci_zona[(ci_zona["hora"] >= 20) | (ci_zona["hora"] < 7)]
hogar_usuario = (
    noche.groupby(["user_id", "zona_id"]).size().reset_index(name="n")
    .sort_values("n").drop_duplicates("user_id", keep="last")
)
fsq_hogar = hogar_usuario["zona_id"].value_counts()
print(f"\nHogar Foursquare inferido para {len(hogar_usuario):,} de "
      f"{checkins['user_id'].nunique():,} usuarios (con check-in nocturno)")

zh = zonas_id.copy()
zh["eod_h"] = zh["zona_id"].map(eod_hogar).fillna(0)
zh["fsq_h"] = zh["zona_id"].map(fsq_hogar).fillna(0)
zh["eod_h"] /= zh["eod_h"].sum()
zh["fsq_h"] /= zh["fsq_h"].sum()
amb_h = zh[(zh["eod_h"] > 0) & (zh["fsq_h"] > 0)]
rho_h = amb_h["eod_h"].corr(amb_h["fsq_h"], method="spearman")
print(f"Correlación de la distribución de hogar (zonas en común): rho={rho_h:.2f}")

zh_bb = zh[zh.intersects(CAJA_UTM)].copy()
zh_bb["eod_log"] = np.log10(zh_bb["eod_h"].replace(0, np.nan))
zh_bb["fsq_log"] = np.log10(zh_bb["fsq_h"].replace(0, np.nan))
vmin = float(np.nanmin([zh_bb["eod_log"].min(), zh_bb["fsq_log"].min()]))
vmax = float(np.nanmax([zh_bb["eod_log"].max(), zh_bb["fsq_log"].max()]))
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, col, titulo in [(axes[0], "eod_log", "EOD: hogar declarado"),
                        (axes[1], "fsq_log", "Foursquare: hogar nocturno")]:
    zh_bb.plot(column=col, cmap="viridis", ax=ax, vmin=vmin, vmax=vmax,
               legend=True, missing_kwds={"color": "0.92"},
               legend_kwds={"label": "log10 proporción", "shrink": 0.6})
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(titulo)
fig.suptitle(f"Distribución del hogar por zona (Spearman = {rho_h:.2f})")
fig.savefig(IMAGES / "14-hogar-mapas.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 11 — Red O-D agregada: EOD vs Foursquare
# ========================================
# Construimos una matriz origen-destino en celdas H3-7 (como la clase 10). En la
# EOD un viaje es origen→destino, ponderado por el factor de expansión. En
# Foursquare un "viaje" son dos check-ins consecutivos del mismo usuario (hasta
# 24 h de diferencia). Comparamos la estructura de flujos de ambas fuentes.
def coords_a_h3(x, y, res=7):
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=CRS_UTM).to_crs("EPSG:4326")
    return [h3.latlng_to_cell(p.y, p.x, res) for p in pts]


vv = viajes.dropna(
    subset=["OrigenCoordX", "OrigenCoordY", "DestinoCoordX", "DestinoCoordY"]
).copy()
vv["o7"] = coords_a_h3(vv["OrigenCoordX"], vv["OrigenCoordY"])
vv["d7"] = coords_a_h3(vv["DestinoCoordX"], vv["DestinoCoordY"])
eod_od = (
    vv[vv["o7"] != vv["d7"]].groupby(["o7", "d7"])["FactorExpansion"].sum()
)

ci = checkins.sort_values(["user_id", "datetime_local"]).copy()
ci["d7"] = ci.groupby("user_id")["h3_7"].shift(-1)
ci["t_sig"] = ci.groupby("user_id")["datetime_local"].shift(-1)
ci["dt_h"] = (ci["t_sig"] - ci["datetime_local"]).dt.total_seconds() / 3600
tr = ci.dropna(subset=["d7"])
tr = tr[(tr["dt_h"] > 0) & (tr["dt_h"] <= 24) & (tr["h3_7"] != tr["d7"])]
fsq_od = tr.groupby(["h3_7", "d7"]).size()
fsq_od.index = fsq_od.index.set_names(["o7", "d7"])

# Grilla H3-7 del bbox (polígonos); las comunas (censo) se cargaron al inicio.
grid7 = (
    h3_grid_from_bounds(list(BBOX_SANTIAGO), grid_level=7, crs="EPSG:4326")
    .reset_index().to_crs(CRS_UTM)
)
celdas_bbox = set(grid7["h3_cell_id"])
g7c = grid7.assign(geometry=grid7.geometry.centroid)
cell_com = dict(
    gpd.sjoin(g7c[["h3_cell_id", "geometry"]], comunas_bb, predicate="within")
    [["h3_cell_id", "COMUNA"]].to_numpy()
)


def norm_cuantiles(valores, k=7):
    """Clasificación por cuantiles con cortes compartidos (preserva contraste)."""
    bordes = np.unique(np.quantile(valores, np.linspace(0, 1, k + 1)))
    return BoundaryNorm(bordes, ncolors=256)


def filtrar(od):
    m = od[[o in celdas_bbox and d in celdas_bbox for o, d in od.index]]
    return m / m.sum()


eod_share, fsq_share = filtrar(eod_od), filtrar(fsq_od)
print(f"\nPares O-D (H3-7 en bbox): EOD {len(eod_share):,} | Foursquare {len(fsq_share):,}")

# --- Fuerza por celda: sectores calientes y fríos ---
# Fuerza = suma de flujos (entrantes + salientes) que tocan la celda. Las celdas
# sin flujo quedan en gris (frías): zonas a las que la fuente no ve viajar.
def fuerza(share):
    s = {}
    for (o, d), w in share.items():
        s[o] = s.get(o, 0) + w
        s[d] = s.get(d, 0) + w
    return pd.Series(s)


grid7["eod"] = grid7["h3_cell_id"].map(fuerza(eod_share))
grid7["fsq"] = grid7["h3_cell_id"].map(fuerza(fsq_share))
norm_f = norm_cuantiles(pd.concat([grid7["eod"], grid7["fsq"]]).dropna())
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, col, titulo in [(axes[0], "eod", "EOD (viajes)"),
                        (axes[1], "fsq", "Foursquare (check-ins)")]:
    grid7.plot(column=col, cmap="magma", ax=ax, norm=norm_f, legend=True,
               missing_kwds={"color": "0.9", "label": "sin flujo"},
               legend_kwds={"label": "fuerza O-D (cuantiles)", "shrink": 0.6})
    comunas_bb.boundary.plot(ax=ax, color="#0A0E50", linewidth=0.5, alpha=0.6)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(titulo)
fig.suptitle("Fuerza O-D por celda H3-7: sectores calientes y fríos")
fig.savefig(IMAGES / "14-od-fuerza.png", bbox_inches="tight")

# --- Matriz O-D comuna x comuna: a qué comunas se viaja (y a cuáles no) ---
def od_comuna(share):
    df = share.rename("w").reset_index()
    df.columns = ["o", "d", "w"]
    df["oc"] = df["o"].map(cell_com)
    df["dc"] = df["d"].map(cell_com)
    return df.dropna(subset=["oc", "dc"]).groupby(["oc", "dc"])["w"].sum()


eod_c, fsq_c = od_comuna(eod_share), od_comuna(fsq_share)
orden = (
    (eod_c.groupby(level=0).sum() + eod_c.groupby(level=1).sum())
    .sort_values(ascending=False).index
)
M_eod = eod_c.unstack().reindex(index=orden, columns=orden)
M_fsq = fsq_c.unstack().reindex(index=orden, columns=orden)
norm_c = norm_cuantiles(pd.concat([eod_c, fsq_c]).dropna())
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, M, titulo in [(axes[0], M_eod, "EOD (viajes)"),
                      (axes[1], M_fsq, "Foursquare (check-ins consecutivos)")]:
    ax.set_facecolor("0.85")  # celdas sin flujo (NaN) quedan grises
    sns.heatmap(M, cmap="magma", norm=norm_c, ax=ax, square=True,
                cbar_kws={"label": "proporción (cuantiles)", "shrink": 0.5},
                xticklabels=True, yticklabels=True)
    ax.set_xlabel("Comuna de destino")
    ax.set_ylabel("Comuna de origen")
    ax.tick_params(labelsize=6)
    ax.set_title(titulo)
fig.suptitle("Matriz O-D comuna a comuna (gris = sin flujo observado)")
fig.savefig(IMAGES / "14-od-comunas.png", bbox_inches="tight")

# --- Acuerdo cuantitativo por par O-D ---
od = pd.DataFrame({"eod": eod_share, "fsq": fsq_share}).fillna(0)
amb_od = od[(od["eod"] > 0) & (od["fsq"] > 0)]
rho_od = amb_od["eod"].corr(amb_od["fsq"], method="spearman")
print(f"Correlación de flujos O-D (pares en común): rho={rho_od:.2f}")
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(amb_od["eod"], amb_od["fsq"], s=8, alpha=0.3, color="#0A0E50")
lim = [min(amb_od.min()), max(amb_od.max())]
ax.plot(lim, lim, "--", color="0.5", linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Proporción del flujo (EOD)")
ax.set_ylabel("Proporción del flujo (Foursquare)")
ax.set_title(f"Flujos O-D por par H3-7 (Spearman = {rho_od:.2f})")
fig.savefig(IMAGES / "14-od-scatter.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 12 — Unicidad sobre Foursquare
# ========================================
# Un "punto" es (celda H3, franja temporal absoluta): la misma celda en días
# distintos cuenta como puntos distintos, así la cobertura multi-día se traduce
# en muchos más puntos por persona que la EOD de un día.


def construir_puntos_4sq(df, col_celda, freq):
    """Serie por usuario con su conjunto de pares (celda H3, franja temporal)."""
    tbin = df["datetime_local"].dt.floor(freq)
    tmp = pd.DataFrame({"user_id": df["user_id"].to_numpy(),
                        "punto": list(zip(df[col_celda], tbin))})
    return tmp.groupby("user_id")["punto"].apply(frozenset)


# Barrido: resolución espacial (H3 7/8/9) x temporal (hora/día).
combos = {
    "H3-9, hora": ("h3_9", "h"), "H3-9, día": ("h3_9", "D"),
    "H3-8, hora": ("h3_8", "h"), "H3-8, día": ("h3_8", "D"),
    "H3-7, hora": ("h3_7", "h"), "H3-7, día": ("h3_7", "D"),
}
curvas_4sq = {}
for nombre, (col, freq) in combos.items():
    curvas_4sq[nombre] = curva(
        construir_puntos_4sq(checkins, col, freq), ks, n_objetivo=10_000
    )
tabla_4sq = pd.DataFrame(curvas_4sq)
print("\nUnicidad Foursquare (fracción única por k):")
print(tabla_4sq.round(3))

# Contraste EOD (1 día) vs Foursquare (multi-día) a resolución comparable.
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(ks, curvas_eod["Zona, franja 1 h"].values, "o-", color="#1b9e77",
        label="EOD: zona, hora (1 día)")
ax.plot(ks, curvas_4sq["H3-8, hora"].values, "s-", color="#CF3889",
        label="Foursquare: H3-8, hora (multi-día)")
ax.plot(ks, curvas_4sq["H3-7, día"].values, "^--", color="#0A0E50",
        label="Foursquare: H3-7, día (multi-día, grueso)")
ax.set_xlabel("Puntos conocidos (k)")
ax.set_ylabel("Fracción de personas únicas")
ax.set_ylim(0, 1.02)
ax.set_title("Un día (EOD) vs multi-día (Foursquare)")
ax.legend(frameon=False, loc="lower right")
fig.savefig(IMAGES / "14-unicidad-contraste.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 13 — Foursquare: huellas individuales
# ========================================
# El análogo de las trayectorias EOD: la constelación de lugares que visita cada
# usuario en 22 meses. No es un día sino meses, así que en vez de un recorrido
# mostramos sus venues (tamaño = cuántas veces fue), un conjunto que no comparte
# nadie más.
uv_counts = checkins.groupby(["user_id", "venue_id"]).size()
n_venues = uv_counts.groupby(level=0).size()
venue_users = checkins.groupby("venue_id")["user_id"].agg(set)
venue_xy = venues.set_index("venue_id")[["lat", "lon"]]

rng = np.random.default_rng(SEMILLA)
candidatos = list(n_venues[(n_venues >= 6) & (n_venues <= 12)].index)
rng.shuffle(candidatos)
ejemplos_4sq = []
for u in candidatos:
    vids = list(uv_counts.loc[u].index)
    conjuntos = sorted((venue_users[v] for v in vids), key=len)
    inter = set(conjuntos[0])
    for s in conjuntos[1:]:
        inter &= s
        if len(inter) == 1:
            break
    if len(inter) == 1:
        ejemplos_4sq.append((u, vids))
    if len(ejemplos_4sq) >= 6:
        break
print(f"Ejemplos Foursquare únicos por su conjunto de venues: {len(ejemplos_4sq)}")


def _dibujar_4sq(ax, ej):
    u, vids = ej
    cuenta = uv_counts.loc[u]
    coords = venue_xy.loc[vids]
    pts = gpd.GeoSeries(
        gpd.points_from_xy(coords["lon"], coords["lat"]), crs="EPSG:4326"
    ).to_crs(CRS_UTM)
    tam = 20 + 80 * (cuenta.to_numpy() / cuenta.max())
    ax.scatter(pts.x, pts.y, s=tam, color="#CF3889", alpha=0.7,
               edgecolor="#0A0E50", linewidth=0.5, zorder=4)
    _zoom(ax, list(pts.x), list(pts.y))


grilla_ejemplos(
    ejemplos_4sq, _dibujar_4sq,
    lambda ej: f"{len(ej[1])} lugares · {int(uv_counts.loc[ej[0]].sum())} check-ins",
    "Cada usuario es una huella: lugares de Foursquare únicos en 22 meses",
    IMAGES / "14-huella-foursquare.png",
)

# %%
# ========================================
# PARTE 14 — Foursquare: el efecto de la resolución
# ========================================
# Con datos multi-día, agrandar la celda espacial o la franja temporal casi no
# reduce la unicidad: es el resultado de de Montjoye et al. (2013).
estilos_4sq = {
    "H3-9, hora": ("o-", "#0A0E50"), "H3-9, día": ("o--", "#0A0E50"),
    "H3-8, hora": ("s-", "#CF3889"), "H3-8, día": ("s--", "#CF3889"),
    "H3-7, hora": ("^-", "#1b9e77"), "H3-7, día": ("^--", "#1b9e77"),
}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for nombre, serie in curvas_4sq.items():
    fmt, color = estilos_4sq[nombre]
    ax.plot(ks, serie.values, fmt, color=color, label=nombre,
            alpha=0.6 if "día" in nombre else 1.0)
ax.set_xlabel("Puntos conocidos (k)")
ax.set_ylabel("Fracción de personas únicas")
ax.set_ylim(0, 1.02)
ax.set_title("Unicidad en Foursquare Santiago (22 meses)")
ax.legend(frameon=False, loc="lower right", ncol=2)
fig.savefig(IMAGES / "14-unicidad-4sq-resolucion.png", bbox_inches="tight")

# %%
# ========================================
# PARTE 15 — Lectura
# ========================================
# En la EOD (un día) ensanchar la franja horaria casi no protege y bajar a
# comuna reduce la unicidad pero inutiliza el dato. En Foursquare (multi-día) ni
# siquiera eso ayuda: con pocos puntos casi todas las personas son únicas aunque
# se agrande la celda. La diferencia entre ambas fuentes es la cobertura
# temporal, no la fuente en sí. La unicidad es una propiedad de la trayectoria:
# por eso seudonimizar (borrar el identificador) no alcanza el estándar de
# anonimización irreversible de la Ley 21.719.
print("\nResolución y unicidad con k=4:")
print("  EOD (zona, hora, 1 día):     "
      f"{curvas_eod['Zona, franja 1 h'].loc[4]:.1%}")
for nombre in combos:
    print(f"  Foursquare ({nombre}):".ljust(32) + f"{curvas_4sq[nombre].loc[4]:.1%}")
