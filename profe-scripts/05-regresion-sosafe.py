# %%
"""Regresión espacial sobre reportes SOSAFE 2024 en hexágonos H3-8.

Ajusta OLS, Spatial Lag, Spatial Error y GWR sobre tres hipótesis:

- H1 ambiental: reportes de basura, luminaria y árboles dependen de NDVI,
  densidad poblacional y educación.
- H2 disturbios: reportes de ruido, comercio ambulante y espacio público
  dependen de densidad, migración reciente y acceso a transporte.
- H3 delitos: reportes de robos dependen de acceso a transporte, densidad
  poblacional y educación.

Requiere ejecutar antes `profe-scripts/05-sosafe-dataset.py` para generar
`data/sosafe/h3-8-2024.parquet`.
"""

# %% Descarga de cartografía comunal (contexto en mapas)
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.maps import choropleth_map
from esda.moran import Moran
from libpysal.weights import Queen
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from spreg import OLS, ML_Error, ML_Lag

from gdsutils.geo import clip_geodataframe

setup_style(dpi=96)

# %% Configuración
RUTA_DATOS = Path("data") / "sosafe" / "h3-8-2024.parquet"
RUTA_CARTO = (
    Path("data") / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
BBOX = (-70.85, -33.65, -70.45, -33.30)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

# %%
# ========================================
# PARTE 1 — Carga del dataset
# ========================================
print(f"Cargando {RUTA_DATOS}")
hex_geo = gpd.read_parquet(RUTA_DATOS)
print(f"  Hexágonos cargados: {len(hex_geo)}")
print(f"  Índice: {hex_geo.index.name}, CRS: {hex_geo.crs}")

# Cartografía comunal para contexto en mapas
comunas = gpd.read_parquet(
    RUTA_CARTO, filters=[("COD_REGION", "=", 13)]
).to_crs(hex_geo.crs)
comunas = clip_geodataframe(comunas, BBOX)
print(f"  Comunas en bbox: {len(comunas)}")

# %%
# ========================================
# PARTE 2 — Transformaciones y filtrado
# ========================================
# Log-transformaciones para densidades muy asimétricas. Sumamos 1 para
# soportar valores cero (hexes sin paraderos) y para luminosidad, cuya
# distribución es muy sesgada.
hex_geo["log_densidad"] = np.log(hex_geo["densidad_hab_km2"] + 1)
hex_geo["log_densidad_paraderos"] = np.log(hex_geo["densidad_paraderos"] + 1)
hex_geo["log_luminosidad"] = np.log(hex_geo["luminosidad"].clip(lower=0) + 1)

# Filtramos hexes sin NDVI (bordes del raster) o con población mínima
n_antes = len(hex_geo)
hex_geo = hex_geo.dropna(subset=["ndvi", "luminosidad"]).reset_index(drop=True)
hex_geo = hex_geo[hex_geo["poblacion"] >= 10].reset_index(drop=True)
print(f"Filtrado: {n_antes} → {len(hex_geo)} hexágonos")

VARIABLES_RESUMEN = [
    "sqrt_n_ambiental", "sqrt_n_disturbios", "sqrt_n_delitos",
    "ndvi", "log_luminosidad", "log_densidad", "frac_edu_superior",
    "frac_migracion_reciente", "log_densidad_paraderos",
]
print(hex_geo[VARIABLES_RESUMEN].describe().round(3))

# %%
# ========================================
# PARTE 3 — Mapas de variables
# ========================================
config = [
    ("sqrt_n_ambiental", "Reportes ambientales (Anscombe)", "YlOrRd"),
    ("sqrt_n_disturbios", "Reportes de disturbios (Anscombe)", "YlOrRd"),
    ("sqrt_n_delitos", "Reportes de delitos (Anscombe)", "YlOrRd"),
    ("ndvi", "NDVI mediano (Sentinel-2)", "YlGn"),
    ("log_luminosidad", "log(1 + radiancia nocturna)", "cividis"),
]
fig, axes = small_multiples_from_geodataframe(
    comunas, len(config), col_wrap=3, height=5,
)
for ax, (col, titulo, paleta) in zip(axes, config):
    choropleth_map(
        hex_geo, col, k=5, binning="quantiles", palette=paleta,
        edgecolor="none", linewidth=0, ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(titulo)
fig.suptitle("Variables del análisis SOSAFE 2024")
fig.savefig(IMAGES / "05-sosafe-variables.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 4 — Exploración de la luminosidad nocturna
# ========================================
# Antes de incluir luminosidad como predictor conviene ver qué contiene:
# distribución de la radiancia, justificación del log, correlación con
# los predictores ya presentes (multicolinearidad) y con los conteos de
# reportes (signo esperado).

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.hist(hex_geo["luminosidad"], bins=40, color="#555")
ax1.set_xlabel("Radiancia nocturna (nW/cm²/sr)")
ax1.set_ylabel("Hexágonos")
ax1.set_title("Luminosidad cruda (VNP46A4 2023)")
ax2.hist(hex_geo["log_luminosidad"], bins=40, color="#555")
ax2.set_xlabel("log(1 + radiancia)")
ax2.set_title("Luminosidad log-transformada")
fig.suptitle("Distribución de la radiancia nocturna")
fig.savefig(IMAGES / "05-sosafe-lum-hist.png", dpi=192, bbox_inches="tight")
plt.show()

print("Radiancia nocturna — estadísticas por hexágono:")
print(hex_geo[["luminosidad", "log_luminosidad"]].describe().round(3))

cols_corr = [
    "log_luminosidad", "ndvi", "log_densidad", "frac_edu_superior",
    "frac_migracion_reciente", "log_densidad_paraderos",
    "sqrt_n_ambiental", "sqrt_n_disturbios", "sqrt_n_delitos",
]
corr = (
    hex_geo[cols_corr]
    .corr(method="spearman")["log_luminosidad"]
    .drop("log_luminosidad")
    .sort_values(ascending=False)
)
print("\nCorrelación Spearman entre log_luminosidad y el resto:")
print(corr.round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, grupo, color in zip(
    axes,
    ["ambiental", "disturbios", "delitos"],
    ["#2a9d8f", "#e9c46a", "#e76f51"],
):
    ax.scatter(
        hex_geo["log_luminosidad"], hex_geo[f"sqrt_n_{grupo}"],
        s=5, alpha=0.4, color=color,
    )
    ax.set_xlabel("log(1 + radiancia)")
    ax.set_ylabel(f"sqrt_n_{grupo}")
    rho = hex_geo[["log_luminosidad", f"sqrt_n_{grupo}"]].corr(
        method="spearman"
    ).iloc[0, 1]
    ax.set_title(f"Reportes {grupo} (ρ = {rho:.2f})")
fig.suptitle("Luminosidad vs conteos de reportes (Anscombe)")
fig.savefig(IMAGES / "05-sosafe-lum-scatter.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 5 — Pesos espaciales Queen
# ========================================
w = Queen.from_dataframe(hex_geo, use_index=True, silence_warnings=True)
w.transform = "r"
print(f"Pesos Queen: {w.n} celdas, {w.n_components} componente(s), "
      f"islas: {len(w.islands)}")

# %%
# ========================================
# PARTE 6 — Función para ajustar tres modelos por hipótesis
# ========================================


def ajustar_modelos(y, X, w, x_names, etiqueta, hex_geo, sufijo):
    """Ajusta OLS, Spatial Lag y Spatial Error e imprime resúmenes."""
    print(f"\n{'=' * 60}")
    print(f"{etiqueta}")
    print(f"{'=' * 60}")
    ols = OLS(y, X, w=w, spat_diag=True, moran=True,
              name_y=etiqueta, name_x=x_names)
    print(ols.summary)
    lag = ML_Lag(y, X, w=w, name_y=etiqueta, name_x=x_names)
    err = ML_Error(y, X, w=w, name_y=etiqueta, name_x=x_names)

    res_ols = ols.u.flatten()
    res_lag = lag.u.flatten()
    # Residuos filtrados del Spatial Error: (I − λW)u = ε, que sí deberían
    # ser iid si el modelo captura la dependencia. `err.u` es el error
    # compuesto y conserva autocorrelación por construcción.
    res_err = err.e_filtered.flatten()

    moran_ols = Moran(res_ols, w)
    moran_lag = Moran(res_lag, w)
    moran_err = Moran(res_err, w)

    hex_geo[f"residuos_ols_{sufijo}"] = res_ols
    hex_geo[f"residuos_lag_{sufijo}"] = res_lag
    hex_geo[f"residuos_err_{sufijo}"] = res_err

    comp = pd.DataFrame({
        "Modelo": ["OLS", "Spatial Lag", "Spatial Error"],
        "R²/pseudo-R²": [ols.r2, lag.pr2, err.pr2],
        "Log-likelihood": [ols.logll, lag.logll, err.logll],
        "AIC": [ols.aic, lag.aic, err.aic],
        "Moran I residuos": [moran_ols.I, moran_lag.I, moran_err.I],
        "p-Moran": [moran_ols.p_norm, moran_lag.p_norm, moran_err.p_norm],
    })
    print(f"\nComparación {etiqueta}:")
    print(comp.round(4).to_string(index=False))
    return ols, lag, err, comp


# %%
# ========================================
# PARTE 7 — H1: hipótesis ambiental
# ========================================
# La cantidad de reportes ambientales debería aumentar con NDVI (más
# vegetación que mantener) y con densidad poblacional (más demanda). La
# luminosidad nocturna entra como control: si los reportes de luminaria
# responden a fallas de alumbrado, esperamos signo negativo; si en cambio
# la luminancia refleja actividad urbana general, el signo será positivo.
# La educación capta sesgos de uso de la app.

X_AMB = ["ndvi", "log_luminosidad", "log_densidad", "frac_edu_superior"]
y_amb = hex_geo["sqrt_n_ambiental"].to_numpy(dtype=float).reshape(-1, 1)
X_amb = hex_geo[X_AMB].to_numpy(dtype=float)

ols_amb, lag_amb, err_amb, cmp_amb = ajustar_modelos(
    y_amb, X_amb, w, X_AMB, "H1: Reportes ambientales", hex_geo, "ambiental",
)

# %%
# ========================================
# PARTE 8 — H2: hipótesis de disturbios
# ========================================
# Disturbios y ruido deberían concentrarse donde hay densidad (más vecinos
# se molestan entre sí) y nodos de transporte (concentración de actividad).
# Migración reciente como variable a discutir críticamente: ¿incidencia o
# sesgo de reporte?

X_DIS = ["log_densidad", "frac_migracion_reciente", "log_densidad_paraderos"]
y_dis = hex_geo["sqrt_n_disturbios"].to_numpy(dtype=float).reshape(-1, 1)
X_dis = hex_geo[X_DIS].to_numpy(dtype=float)

ols_dis, lag_dis, err_dis, cmp_dis = ajustar_modelos(
    y_dis, X_dis, w, X_DIS, "H2: Reportes de disturbios", hex_geo, "disturbios",
)

# %%
# ========================================
# PARTE 9 — H3: hipótesis de delitos
# ========================================
# Reportes de delitos deberían depender del flujo de personas (paraderos),
# densidad y educación.

X_DEL = ["log_densidad_paraderos", "log_densidad", "frac_edu_superior"]
y_del = hex_geo["sqrt_n_delitos"].to_numpy(dtype=float).reshape(-1, 1)
X_del = hex_geo[X_DEL].to_numpy(dtype=float)

ols_del, lag_del, err_del, cmp_del = ajustar_modelos(
    y_del, X_del, w, X_DEL, "H3: Reportes de delitos", hex_geo, "delitos",
)

# %%
# ========================================
# PARTE 10 — Comparación espacial de residuos por modelo
# ========================================
# 3×3: filas = hipótesis, columnas = modelo. La paleta divergente RdBu_r
# resalta clústeres de sub/sobre-predicción. Si los modelos espaciales
# capturan la dependencia, el Moran I sobre los residuos debería caer cerca
# de cero al pasar de OLS a Lag/Error.

hipotesis_info = [
    ("ambiental", "H1: Ambiental"),
    ("disturbios", "H2: Disturbios"),
    ("delitos", "H3: Delitos"),
]
modelos_info = [
    ("ols", "OLS"),
    ("lag", "Spatial Lag"),
    ("err", "Spatial Error"),
]

fig, axes = small_multiples_from_geodataframe(
    comunas, 9, col_wrap=3, height=4,
)
for i, (sufijo, titulo_fila) in enumerate(hipotesis_info):
    for j, (mod, titulo_col) in enumerate(modelos_info):
        ax = axes[i * 3 + j]
        col = f"residuos_{mod}_{sufijo}"
        moran = Moran(hex_geo[col], w)
        choropleth_map(
            hex_geo, col, k=5, binning="quantiles",
            palette="RdBu_r", edgecolor="none", linewidth=0, ax=ax,
        )
        comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
        ax.set_title(
            f"{titulo_fila} — {titulo_col}\n"
            f"I = {moran.I:.3f}, p = {moran.p_norm:.3f}"
        )
fig.suptitle("Residuos por hipótesis y modelo")
fig.savefig(
    IMAGES / "05-sosafe-residuos-comparacion.png",
    dpi=192, bbox_inches="tight",
)
plt.show()

# Distribución de residuos por modelo (cada hipótesis en su panel): sirve
# para detectar cambios de escala y asimetrías entre OLS y los modelos ML.
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
colores_modelo = {
    "ols": "#3b8ea5", "lag": "#f4a261", "err": "#e76f51",
}
etiquetas_modelo = {"ols": "OLS", "lag": "Spatial Lag", "err": "Spatial Error"}
for ax, (sufijo, titulo) in zip(axes, hipotesis_info):
    for mod in ["ols", "lag", "err"]:
        col = f"residuos_{mod}_{sufijo}"
        ax.hist(
            hex_geo[col], bins=40, alpha=0.45, density=True,
            color=colores_modelo[mod], label=etiquetas_modelo[mod],
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(titulo)
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
axes[0].legend(loc="best", frameon=False)
fig.suptitle("Distribución de residuos por modelo")
fig.savefig(
    IMAGES / "05-sosafe-residuos-distribucion.png",
    dpi=192, bbox_inches="tight",
)
plt.show()

# %%
# ========================================
# PARTE 11 — GWR sobre H1 ambiental
# ========================================
# La hipótesis ambiental es la candidata más interesante para GWR: el efecto
# del NDVI sobre los reportes de mantención debería variar entre periferia
# (áreas verdes mal mantenidas) y centro (parques bien atendidos).

print("\n" + "=" * 60)
print("GWR — H1: Reportes ambientales")
print("=" * 60)

coords_utm = hex_geo.to_crs("EPSG:32719").geometry.centroid
coords = list(zip(coords_utm.x, coords_utm.y))

print("Buscando bandwidth óptimo (AICc)")
bw = Sel_BW(coords, y_amb, X_amb).search(criterion="AICc")
print(f"  bandwidth óptimo: {bw} vecinos")

print("Ajustando GWR")
gwr_amb = GWR(coords, y_amb, X_amb, bw).fit()
print(gwr_amb.summary())

# Coeficientes locales por hex (intercepto + 3 covariables). Saneamos
# inf/NaN: en vecindarios con varianza casi nula de y, TSS→0 y localR²
# diverge a −∞ (división por casi cero). Enmascaramos valores fuera de
# [-1, 1] como inestables: un R² local por debajo de −1 refleja ruido
# numérico, no un mal ajuste real. Reportamos la mediana, que es robusta
# a las colas que quedan.
params = np.where(np.isfinite(gwr_amb.params), gwr_amb.params, np.nan)
r2_local_raw = gwr_amb.localR2.flatten().astype(float)
n_inf = int((~np.isfinite(r2_local_raw)).sum())
r2_local = np.where(np.isfinite(r2_local_raw), r2_local_raw, np.nan)
n_extremos = int(np.sum((r2_local < -1) | (r2_local > 1)))
r2_local = np.where((r2_local >= -1) & (r2_local <= 1), r2_local, np.nan)

nombres_coef = ["intercepto"] + X_AMB
for i, nombre in enumerate(nombres_coef):
    hex_geo[f"beta_amb_{nombre}"] = params[:, i]
hex_geo["r2_local_amb"] = r2_local
print(
    f"  R² local — enmascarados: {n_inf} no finitos, "
    f"{n_extremos} fuera de [-1, 1] (inestabilidad numérica en "
    f"vecindarios con varianza casi nula)."
)

# %%
# Mapas de coeficientes locales
columnas_coef = [f"beta_amb_{c}" for c in X_AMB]
fig, axes = small_multiples_from_geodataframe(
    comunas, len(columnas_coef), col_wrap=3, height=5,
)
for ax, col in zip(axes, columnas_coef):
    hex_validos = hex_geo.dropna(subset=[col])
    choropleth_map(
        hex_validos, col, k=5, binning="quantiles", palette="RdBu_r",
        edgecolor="none", linewidth=0, ax=ax,
    )
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
    ax.set_title(col.replace("beta_amb_", "β "))
fig.suptitle("GWR H1 — coeficientes locales por hexágono")
fig.savefig(
    IMAGES / "05-sosafe-gwr-coeficientes.png", dpi=192, bbox_inches="tight"
)
plt.show()

# %% R² local
hex_validos = hex_geo.dropna(subset=["r2_local_amb"])
fig, ax = plt.subplots(figsize=(8, 7))
choropleth_map(
    hex_validos, "r2_local_amb", k=5, binning="quantiles", palette="viridis",
    edgecolor="none", linewidth=0, ax=ax,
)
comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.4)
ax.set_title(
    f"GWR H1 — R² local (mediana = {np.nanmedian(r2_local):.3f}, "
    f"Q1 = {np.nanquantile(r2_local, 0.25):.3f}, "
    f"Q3 = {np.nanquantile(r2_local, 0.75):.3f})"
)
fig.savefig(
    IMAGES / "05-sosafe-gwr-r2-local.png", dpi=192, bbox_inches="tight"
)
plt.show()

# %%
# ========================================
# PARTE 12 — Coeficientes con intervalos de confianza
# ========================================
# Forest plot con beta estimado y IC 95% de Wald por predictor, comparando OLS,
# Spatial Lag y Spatial Error en las tres hipótesis. En OLS, SE viene de
# sqrt(sigma^2 (X^T X)^{-1}); en ML, de la raíz de la diagonal de la inversa de
# la información de Fisher observada. Ambos usan z* ~ 1.96 para n grande (ver
# apunte). Si el intervalo cruza cero, el efecto no es distinguible de cero
# al 5%.


def graficar_coeficientes(modelos, x_names, ax, titulo):
    colores = {
        "OLS": "#3b8ea5",
        "Spatial Lag": "#f4a261",
        "Spatial Error": "#e76f51",
    }
    offsets = {"OLS": 0.22, "Spatial Lag": 0.0, "Spatial Error": -0.22}
    k = len(x_names)
    y_pos = np.arange(k)
    for nombre, modelo in modelos.items():
        b = np.asarray(modelo.betas).flatten()[1 : 1 + k]
        s = np.asarray(modelo.std_err).flatten()[1 : 1 + k]
        ax.errorbar(
            b, y_pos + offsets[nombre], xerr=1.96 * s,
            fmt="o", color=colores[nombre], label=nombre,
            capsize=3, markersize=5, linewidth=1.2,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(x_names)
    ax.invert_yaxis()
    ax.set_xlabel("Coeficiente (± IC 95%)")
    ax.set_title(titulo)


hipotesis = [
    ({"OLS": ols_amb, "Spatial Lag": lag_amb, "Spatial Error": err_amb},
     X_AMB, "H1: Ambientales"),
    ({"OLS": ols_dis, "Spatial Lag": lag_dis, "Spatial Error": err_dis},
     X_DIS, "H2: Disturbios"),
    ({"OLS": ols_del, "Spatial Lag": lag_del, "Spatial Error": err_del},
     X_DEL, "H3: Delitos"),
]
fig, axes = plt.subplots(3, 1, figsize=(9, 9))
for ax, (mods, xs, tit) in zip(axes, hipotesis):
    graficar_coeficientes(mods, xs, ax, tit)
axes[0].legend(loc="best", frameon=False, fontsize=9)
fig.suptitle("Coeficientes estimados por modelo (IC 95% Wald)")
fig.savefig(
    IMAGES / "05-sosafe-coeficientes.png", dpi=192, bbox_inches="tight",
)
plt.show()

# %%
# ========================================
# PARTE 13 — Comparación final
# ========================================
print("\n" + "=" * 60)
print("COMPARACIÓN FINAL")
print("=" * 60)
print("\nH1 — Ambiental:")
print(cmp_amb.round(4).to_string(index=False))
print("\nH2 — Disturbios:")
print(cmp_dis.round(4).to_string(index=False))
print("\nH3 — Delitos:")
print(cmp_del.round(4).to_string(index=False))
print(
    f"\nGWR sobre H1: R² = {gwr_amb.R2:.4f}, AICc = {gwr_amb.aicc:.2f} "
    f"(comparar con AIC OLS = {ols_amb.aic:.2f})"
)

# Notas de interpretación:
# - LM-lag vs LM-error en el resumen OLS: cuál sale más significativo decide
#   si se prefiere SLM (difusión) o SEM (heterogeneidad).
# - Cambios grandes de Moran I residual entre OLS y SLM/SEM indican que el
#   modelo espacial captura parte de la dependencia.
# - Si GWR mejora el AICc respecto a OLS y los mapas de β muestran patrón
#   coherente, hay no-estacionariedad: el efecto del predictor cambia en el
#   espacio. Si los mapas son ruidosos, el bandwidth puede estar sub-óptimo
#   o el predictor opera de forma global.
