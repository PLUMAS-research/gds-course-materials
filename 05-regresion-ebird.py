# %%
"""Regresión espacial sobre riqueza de aves (eBird 2023-2024) en H3-8.

La variable dependiente es `sqrt_riqueza` (Anscombe sobre el número de
especies por hexágono). Los predictores son:

- `ndvi`: vegetación (proxy de hábitat disponible).
- `log_luminosidad`: radiancia nocturna (proxy de urbanización).
- `log_densidad`: habitantes/km$^2$ por comuna (presión urbana).
- `log_checklists`: control de esfuerzo de observación. Sin este control,
  hexes muy muestreados (parques urbanos, humedales) parecen tener más
  biodiversidad solo porque hay más observadores.
"""

# %% Descargas
from gdsutils.general import descargar_datos
from gdsutils.ndvi import (
    descargar_luminosidad_santiago,
    descargar_ndvi_santiago_precomputado,
)

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/ebird-santiago-2024.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")
RUTA_NDVI = descargar_ndvi_santiago_precomputado()
RUTA_LUMINOSIDAD = descargar_luminosidad_santiago()

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.maps import choropleth_map, heat_map
from esda.moran import Moran
from libpysal.weights import Queen
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from spreg import OLS, ML_Error, ML_Lag

from gdsutils.geo import clip_geodataframe

setup_style(dpi=96)

# %% Configuración
RUTA_DATOS = Path("data") / "ebird-santiago-2024" / "h3-8-2024.parquet"
RUTA_CARTO = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
BBOX = (-70.85, -33.65, -70.45, -33.30)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

# %%
# ========================================
# PARTE 1.Carga del dataset
# ========================================
print(f"Cargando {RUTA_DATOS}")
hex_geo = gpd.read_parquet(RUTA_DATOS)
print(f"  Hexágonos: {len(hex_geo)}")
print(f"  CRS: {hex_geo.crs}")

RUTA_OBSERVACIONES = Path("data") / "ebird-santiago-2024" / "observaciones.parquet"
observaciones = pd.read_parquet(RUTA_OBSERVACIONES)
print(
    f"  Observaciones crudas: {len(observaciones):,} "
    f"({observaciones['nombre_cientifico'].nunique()} especies)"
)

comunas = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(
    hex_geo.crs
)
comunas = clip_geodataframe(comunas, BBOX)
print(f"  Comunas de contexto: {len(comunas)}")

# %%
# ========================================
# PARTE 2.Transformaciones
# ========================================
# Esfuerzo y densidad son muy asimétricos: log estabiliza la varianza.
# Luminosidad tiene ceros en cerros no urbanizados; usamos log(1 + x).
hex_geo["log_checklists"] = np.log(hex_geo["checklists"])
hex_geo["log_densidad"] = np.log(hex_geo["densidad"])
hex_geo["log_luminosidad"] = np.log1p(hex_geo["luminosidad"].clip(lower=0))

VARIABLES = [
    "sqrt_riqueza",
    "riqueza",
    "shannon",
    "pielou",
    "checklists",
    "ndvi",
    "log_luminosidad",
    "log_densidad",
    "log_checklists",
]
print(hex_geo[VARIABLES].describe().round(3))

# %%
# ========================================
# PARTE 3.Mapas exploratorios
# ========================================
# Si la riqueza alta coincide espacialmente con el esfuerzo alto, el
# control por log_checklists será crítico en la regresión.
config_mapas = [
    ("sqrt_riqueza", "Riqueza de aves (Anscombe)", "viridis"),
    ("shannon", "Diversidad de Shannon (ocurrencias)", "viridis"),
    ("log_checklists", "log(checklists), esfuerzo de muestreo", "magma"),
    ("ndvi", "NDVI mediano (Sentinel-2)", "YlGn"),
    ("log_luminosidad", "log(1 + luminosidad)", "cividis"),
    ("log_densidad", "log(densidad hab/km$^2$ comunal)", "plasma"),
]
fig, axes = small_multiples_from_geodataframe(
    comunas,
    len(config_mapas),
    col_wrap=3,
    height=5,
)
for ax, (col, titulo, paleta) in zip(axes, config_mapas):
    choropleth_map(
        hex_geo,
        col,
        k=5,
        binning="fisher_jenks",
        palette=paleta,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(titulo)
fig.suptitle("Variables del análisis de riqueza de aves en Santiago")
fig.savefig(IMAGES / "05-ebird-variables.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# Observaciones por grupos de especies
# ========================================
# Densidad espacial de observaciones por grupo taxonómico. Útil para
# contrastar con áreas verdes (zorzales, picaflores) o con el gradiente
# urbano-periurbano (rapaces).
GRUPOS_ESPECIES = {
    "Zorzales": ["Turdus falcklandii", "Turdus chiguanco"],
    "Rapaces": [
        "Milvago chimango",
        "Caracara plancus",
        "Falco sparverius",
        "Geranoaetus polyosoma",
        "Parabuteo unicinctus",
        "Tyto alba",
        "Athene cunicularia",
    ],
    "Picaflores": [
        "Sephanoides sephaniodes",
        "Patagona gigas",
        "Rhodopis vesper",
    ],
    "Loros": [
        "Enicognathus leptorhynchus",
        "Myiopsitta monachus",
    ],
}
CMAPS_GRUPOS = {
    "Zorzales": "Reds",
    "Rapaces": "Blues",
    "Picaflores": "YlOrBr",
    "Loros": "Greens",
}

# Extent común para que los heatmaps sean comparables entre paneles.
minx, miny, maxx, maxy = comunas.total_bounds

fig, axes = small_multiples_from_geodataframe(
    comunas,
    len(GRUPOS_ESPECIES),
    col_wrap=2,
    height=5.5,
)
for ax, (grupo, especies) in zip(axes, GRUPOS_ESPECIES.items()):
    sub = observaciones[observaciones["nombre_cientifico"].isin(especies)]
    if len(sub) > 0:
        sub_gdf = gpd.GeoDataFrame(
            sub,
            geometry=gpd.points_from_xy(sub["lon"], sub["lat"]),
            crs="EPSG:4326",
        )
        heat_map(
            sub_gdf,
            ax=ax,
            palette=CMAPS_GRUPOS[grupo],
            n_levels=8,
            bandwidth=0.008,
            legend_type="colorbar",
            cbar_label="densidad",
        )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.5)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(
        f"{grupo}: {len(sub):,} obs., {sub['nombre_cientifico'].nunique()} especies"
    )
fig.suptitle("Densidad de observaciones eBird por grupo de especies")
fig.savefig(IMAGES / "05-ebird-grupos.png", dpi=192, bbox_inches="tight")
plt.show()

# %% Riqueza versus cada predictor (exploratorio)
# Lowess como tendencia (no asume linealidad). Spearman en el título
# resume la monotonía de la relación.
predictores_plot = ["ndvi", "log_luminosidad", "log_densidad", "log_checklists"]
fig, axes = plt.subplots(
    1,
    len(predictores_plot),
    figsize=(16, 4.2),
    sharey=True,
)
for ax, predictor in zip(axes, predictores_plot):
    sns.regplot(
        data=hex_geo,
        x=predictor,
        y="sqrt_riqueza",
        scatter_kws={"s": 20, "alpha": 0.5, "edgecolor": "none", "color": "#3b8ea5"},
        line_kws={"color": "#e76f51", "linewidth": 2.0},
        lowess=True,
        ax=ax,
    )
    rho = hex_geo[[predictor, "sqrt_riqueza"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(rf"{predictor}    $\rho$ = {rho:.2f}")
    ax.set_xlabel(predictor)
axes[0].set_ylabel("sqrt(riqueza)")
fig.suptitle("Riqueza de aves vs predictores (tendencia lowess)")
fig.savefig(IMAGES / "05-ebird-correlaciones.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 4.Pesos espaciales Queen
# ========================================
# La cobertura de eBird es heterogénea: hexes en el piedemonte o zonas
# periurbanas quedan aislados del núcleo urbano, formando componentes
# conexos separados e islas. SLM/SEM no están definidos en islas (W @ y
# requiere al menos un vecino), así que restringimos el análisis al
# componente mayor y rearmamos W.
w = Queen.from_dataframe(hex_geo, use_index=True, silence_warnings=True)
print(
    f"Pesos Queen (antes de filtrar): {w.n} celdas, "
    f"{w.n_components} componente(s), islas: {len(w.islands)}"
)

componente_mayor = pd.Series(w.component_labels).value_counts().idxmax()
mascara = w.component_labels == componente_mayor
hex_geo = hex_geo.iloc[mascara].reset_index(drop=True)

w = Queen.from_dataframe(hex_geo, use_index=True, silence_warnings=True)
w.transform = "r"
print(
    f"Pesos Queen (componente mayor): {w.n} celdas, "
    f"{w.n_components} componente(s), islas: {len(w.islands)}"
)

# %%
# ========================================
# PARTE 5.OLS con diagnósticos espaciales
# ========================================
# spreg.OLS calcula LM-lag, LM-error y sus versiones robustas además del
# Moran I sobre residuos. Esos diagnósticos guían la elección entre SLM y
# SEM: si LM-lag (robusto) es más significativo, SLM; si lo es LM-error,
# SEM.
X_COLS = ["ndvi", "log_luminosidad", "log_densidad", "log_checklists"]
y = hex_geo["sqrt_riqueza"].to_numpy(dtype=float).reshape(-1, 1)
X = hex_geo[X_COLS].to_numpy(dtype=float)

ols = OLS(
    y,
    X,
    w=w,
    spat_diag=True,
    moran=True,
    name_y="sqrt_riqueza",
    name_x=X_COLS,
)
print(ols.summary)

hex_geo["residuos_ols"] = ols.u.flatten()
moran_ols = Moran(hex_geo["residuos_ols"], w)
print(f"Moran I residuos OLS: {moran_ols.I:.4f} (p = {moran_ols.p_norm:.4f})")

# Cómo leer el output del OLS:
# - Coeficientes: revisar signo y magnitud. log_checklists debe ser positivo
#   y grande (control de esfuerzo); si no, revisar preparación del dataset.
#   ndvi positivo y log_luminosidad negativo son los efectos ecológicos
#   esperables.
# - Condition number > 30: posible multicolinealidad. En este modelo suele
#   venir de log_luminosidad y log_densidad, ambos proxies de urbanización.
# - Jarque-Bera (normalidad): si p > 0.05, los errores son gaussianos y la
#   inferencia con t es válida.
# - Breusch-Pagan / Koenker-Bassett (heteroscedasticidad): si p < 0.05, los
#   SE de OLS están sesgados. El modelo espacial puede absorber parte.
# - Diagnostico espacial: la regla de decisión es comparar Robust LM-lag vs
#   Robust LM-error. El significativo manda; si los dos lo son, gana el de
#   estadístico mayor. LM-error -> SEM, LM-lag -> SLM.
# - Moran I sobre residuos > 0 con p < 0.05 confirma autocorrelación
#   espacial: OLS no captura toda la estructura.

# %%
# ========================================
# PARTE 6.Spatial Lag Model
# ========================================
# y = rho W y + X beta + epsilon. La riqueza de un hex depende de la riqueza
# vecina: justificable en ecología por dispersión y selección de hábitat a
# corta distancia (un hex rodeado de hábitat bueno recibe especies "sobrantes").
lag = ML_Lag(y, X, w=w, name_y="sqrt_riqueza", name_x=X_COLS)
print(lag.summary)

hex_geo["residuos_lag"] = lag.u.flatten()
moran_lag = Moran(hex_geo["residuos_lag"], w)
print(f"Moran I residuos Lag: {moran_lag.I:.4f} (p = {moran_lag.p_norm:.4f})")

# Cómo leer el SLM:
# - rho (Spatial AR coefficient en el summary): si es significativo y
#   positivo (típicamente 0.2 a 0.6), hay difusión directa entre vecinos.
#   Si no es significativo, el lag no era el modelo correcto.
# - AIC vs OLS: tiene que bajar para que valga la pena.
# - Moran I sobre residuos: debería caer cerca de cero si el lag captura la
#   dependencia. Si todavía es significativo, probablemente Error o Durbin
#   funcionan mejor.
# - Los coeficientes beta NO son efectos marginales: el cambio en x_k afecta
#   a y propio y a y de los vecinos. Para interpretarlos descomponer
#   (I - rho W)^{-1} beta_k en directo, indirecto y total.

# %%
# ========================================
# PARTE 7.Spatial Error Model
# ========================================
# y = X beta + u, u = lambda W u + epsilon. La autocorrelación se atribuye a
# variables omitidas con estructura espacial (microclima, contaminación,
# fragmentación). Usamos err.e_filtered = (I - lambda W) u para el Moran: es
# el residuo que debería ser iid si el modelo captura la dependencia. err.u
# conserva autocorrelación por construcción.
err = ML_Error(y, X, w=w, name_y="sqrt_riqueza", name_x=X_COLS)
print(err.summary)

hex_geo["residuos_err"] = err.e_filtered.flatten()
moran_err = Moran(hex_geo["residuos_err"], w)
print(
    f"Moran I residuos Error (filtrados): "
    f"{moran_err.I:.4f} (p = {moran_err.p_norm:.4f})"
)

# Cómo leer el SEM:
# - lambda significativo (típicamente 0.3 a 0.6): confirma estructura
#   espacial en variables omitidas.
# - AIC: comparar con OLS y SLM. El menor gana.
# - Moran I sobre err.e_filtered debería estar cerca de cero (ya descontada
#   la dependencia). Si usás err.u el Moran sale artificialmente alto porque
#   u = (I - lambda W)^{-1} epsilon conserva la correlación por construcción.
# - Coeficientes beta comparables a los del OLS (no hay multiplicador
#   espacial). Si se atenúan respecto a OLS, parte del efecto que veíamos en
#   OLS venía de la estructura espacial no modelada.

# %%
# ========================================
# PARTE 8.Comparación de los tres modelos globales
# ========================================
comparacion = pd.DataFrame(
    {
        "Modelo": ["OLS", "Spatial Lag", "Spatial Error"],
        "R^2/pseudo-R^2": [ols.r2, lag.pr2, err.pr2],
        "Log-likelihood": [ols.logll, lag.logll, err.logll],
        "AIC": [ols.aic, lag.aic, err.aic],
        "Moran I residuos": [moran_ols.I, moran_lag.I, moran_err.I],
        "p-Moran": [moran_ols.p_norm, moran_lag.p_norm, moran_err.p_norm],
    }
)
print("\nComparación de modelos globales:")
print(comparacion.round(4).to_string(index=False))

# Cómo leer la tabla:
# - AIC es el criterio más limpio para elegir entre modelos no anidados.
#   Diferencias > 2 son sustanciales, > 10 son decisivas.
# - R^2 / pseudo-R^2 NO son comparables directamente entre OLS y modelos ML
#   (definiciones distintas), pero dan idea del orden de magnitud.
# - Moran I cerca de cero en SLM o SEM = el modelo espacial cumplió. Si
#   sigue significativo, queda dependencia sin modelar.

# %% Mapas de residuos lado a lado (paleta divergente resalta clústeres).
fig, axes = small_multiples_from_geodataframe(comunas, 3, col_wrap=3, height=5)
modelos_mapas = [
    ("residuos_ols", "OLS", moran_ols),
    ("residuos_lag", "Spatial Lag", moran_lag),
    ("residuos_err", "Spatial Error (filtrados)", moran_err),
]
for ax, (col, nombre, m) in zip(axes, modelos_mapas):
    choropleth_map(
        hex_geo,
        col,
        k=5,
        binning="fisher_jenks",
        palette="RdBu_r",
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(f"{nombre}\nI = {m.I:.3f}, p = {m.p_norm:.3f}")
fig.suptitle("Residuos por modelo")
fig.savefig(
    IMAGES / "05-ebird-residuos-comparacion.png",
    dpi=192,
    bbox_inches="tight",
)
plt.show()

# Cómo leer los mapas:
# - OLS: si hay clústeres rojos/azules visibles, OLS subestima/sobreestima
#   de forma sistemática en zonas vecinas (autocorrelación residual).
# - Lag/Error: el patrón debería verse fragmentado, sin clústeres claros.
#   Si sigue habiendo clústeres en el modelo espacial, considerar Spatial
#   Durbin (incluir W*X), sugerido por LM-WX en el diagnóstico OLS.

# %%
# ========================================
# PARTE 9.Forest plot de coeficientes (IC 95% Wald)
# ========================================
# beta +- 1.96 * SE por predictor, los tres modelos uno encima del otro. Si el
# intervalo cruza cero el efecto no se distingue de cero al 5%. Ver apunte:
# en OLS el SE viene de sigma^2 (X^T X)^{-1}; en ML, de la inversa de la
# información de Fisher observada. Para n grande, z* ~ 1.96 aplica a ambos.
colores = {"OLS": "#3b8ea5", "Spatial Lag": "#f4a261", "Spatial Error": "#e76f51"}
offsets = {"OLS": 0.22, "Spatial Lag": 0.0, "Spatial Error": -0.22}
modelos_forest = {"OLS": ols, "Spatial Lag": lag, "Spatial Error": err}

fig, ax = plt.subplots(figsize=(8, 4.5))
k = len(X_COLS)
y_pos = np.arange(k)
for nombre, modelo in modelos_forest.items():
    b = np.asarray(modelo.betas).flatten()[1 : 1 + k]
    s = np.asarray(modelo.std_err).flatten()[1 : 1 + k]
    ax.errorbar(
        b,
        y_pos + offsets[nombre],
        xerr=1.96 * s,
        fmt="o",
        color=colores[nombre],
        label=nombre,
        capsize=3,
        markersize=5,
        linewidth=1.2,
    )
ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(X_COLS)
ax.invert_yaxis()
ax.set_xlabel(r"Coeficiente ($\pm$ IC 95%)")
ax.set_title("Coeficientes estimados por modelo para riqueza de aves")
ax.legend(loc="best", frameon=False)
fig.savefig(IMAGES / "05-ebird-coeficientes.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer el forest plot:
# - Si la barra cruza la línea vertical de cero, el efecto NO se distingue
#   de cero al 5%.
# - Comparar magnitudes entre OLS y SLM/SEM: si el modelo espacial atenúa
#   un coeficiente, parte de lo que OLS atribuía al predictor era estructura
#   espacial no modelada.
# - El ancho del IC indica incertidumbre: barras anchas = pocos datos o
#   predictor con baja variación.

# %%
# ========================================
# PARTE 10.GWR (coeficientes que varían en el espacio)
# ========================================
# OLS/SLM/SEM asumen coeficientes constantes. GWR ajusta una regresión
# local por hex ponderando por cercanía con kernel adaptativo (bandwidth
# = número de vecinos). Sel_BW elige el bandwidth óptimo vía AICc.
coords_utm = hex_geo.to_crs("EPSG:32719").geometry.centroid
coords = list(zip(coords_utm.x, coords_utm.y))

print("Buscando bandwidth óptimo (AICc)")
bw = Sel_BW(coords, y, X).search(criterion="AICc")
print(f"  bandwidth óptimo: {bw} vecinos")
# Bandwidth = número de vecinos en cada regresión local. Pequeño (< 30) =
# mucho detalle pero ruido (sobreajuste local); grande (> 100) = casi global
# (poca ganancia sobre OLS). El AICc elige el balance.

print("Ajustando GWR")
gwr = GWR(coords, y, X, bw).fit()
print(gwr.summary())

# Cómo leer el GWR:
# - AICc del GWR vs AIC del OLS/SLM/SEM: si el GWR es sustancialmente menor
#   (diferencia > 10), hay no-estacionariedad y el modelo local mejora. Si
#   la diferencia es chica, los modelos globales bastan.
# - R^2 global del GWR: comparable al OLS, no al pseudo-R^2 de los ML.
# - Los coeficientes locales se interpretan en los mapas (PARTE siguiente).

# %% Coeficientes locales + R^2 local con saneamiento
# En vecindarios con varianza casi nula de y, TSS->0 y R^2_local diverge.
# Enmascarar valores no finitos y fuera de [-1, 1] antes de graficar.
params = np.where(np.isfinite(gwr.params), gwr.params, np.nan)
r2_raw = gwr.localR2.flatten().astype(float)
n_inf = int((~np.isfinite(r2_raw)).sum())
r2_local = np.where(np.isfinite(r2_raw), r2_raw, np.nan)
n_ext = int(np.sum((r2_local < -1) | (r2_local > 1)))
r2_local = np.where((r2_local >= -1) & (r2_local <= 1), r2_local, np.nan)
print(f"R^2 local, enmascarados: {n_inf} no finitos, {n_ext} fuera de [-1, 1]")

nombres_coef = ["intercepto"] + X_COLS
for i, nombre in enumerate(nombres_coef):
    hex_geo[f"beta_{nombre}"] = params[:, i]
hex_geo["r2_local"] = r2_local

# %% Mapas de coeficientes locales
columnas_coef = [f"beta_{c}" for c in X_COLS]
fig, axes = small_multiples_from_geodataframe(
    comunas,
    len(columnas_coef),
    col_wrap=2,
    height=5,
)
for ax, col in zip(axes, columnas_coef):
    hex_validos = hex_geo.dropna(subset=[col])
    choropleth_map(
        hex_validos,
        col,
        k=5,
        binning="fisher_jenks",
        palette="RdBu_r",
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(col.replace("beta_", r"$\beta$ "))
fig.suptitle("Coeficientes locales GWR por hexágono")
fig.savefig(
    IMAGES / "05-ebird-gwr-coeficientes.png",
    dpi=192,
    bbox_inches="tight",
)
plt.show()

# Cómo leer los mapas de coeficientes locales:
# - Signo consistente con el OLS pero con variación espacial: el efecto
#   global se mantiene pero su intensidad cambia por zona.
# - Coeficiente que cambia de signo entre zonas: hay efecto contextual
#   fuerte (la relación se invierte según el contexto urbano).
# - Coeficientes ruidosos (alternados sin patrón claro): el bandwidth puede
#   estar muy chico o el predictor opera de forma global.
# - Para beta_ndvi en Santiago suele verse alto en piedemonte/parques (más
#   gradiente de vegetación) y bajo en el centro denso (saturado).

# %% R^2 local
hex_validos = hex_geo.dropna(subset=["r2_local"])
fig, ax = plt.subplots(figsize=(8, 7))
choropleth_map(
    hex_validos,
    "r2_local",
    k=5,
    binning="fisher_jenks",
    palette="viridis",
    edgecolor="none",
    linewidth=0,
    ax=ax,
)
comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
ax.set_title(
    f"$R^2$ local GWR (mediana = {np.nanmedian(r2_local):.3f}, "
    f"Q1 = {np.nanquantile(r2_local, 0.25):.3f}, "
    f"Q3 = {np.nanquantile(r2_local, 0.75):.3f})"
)
fig.savefig(IMAGES / "05-ebird-gwr-r2-local.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer el R^2 local:
# - Mediana alta y rango angosto (Q1 y Q3 cercanos): el modelo funciona
#   parejo en todo el dominio.
# - Mediana baja con Q3 alto: el modelo funciona en algunas zonas y falla
#   en otras. Identificar las zonas malas en el mapa y discutir qué
#   predictor falta ahí (humedales, periurbano, áreas industriales).
# - Hexes enmascarados (NaN): vecindarios con varianza casi nula de y, no
#   se interpretan, son inestabilidad numérica.

# %%
# ========================================
# PARTE 11.Comparación final
# ========================================
comparacion_final = pd.DataFrame(
    {
        "Modelo": ["OLS", "Spatial Lag", "Spatial Error", "GWR"],
        "R^2": [ols.r2, lag.pr2, err.pr2, gwr.R2],
        "AIC / AICc": [ols.aic, lag.aic, err.aic, gwr.aicc],
    }
)
print("\n" + "=" * 60)
print("COMPARACIÓN FINAL")
print("=" * 60)
print(comparacion_final.round(4).to_string(index=False))

# Síntesis:
# - El AIC más bajo gana entre los modelos globales (OLS, SLM, SEM).
#   AICc del GWR tiene corrección por grados de libertad efectivos y se
#   compara con AIC de los demás (no es un AIC distinto, solo corregido).
# - Si SLM o SEM bajan AIC respecto a OLS, era necesario modelar el espacio.
# - Si GWR baja AICc respecto al mejor global, hay no-estacionariedad: el
#   efecto del predictor cambia geográficamente.
# - log_checklists positivo y alto en todos los modelos: el control de
#   esfuerzo funciona. Si desapareciera, revisar la preparación.
