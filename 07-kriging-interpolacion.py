# %%
"""Interpolación espacial: IDW y kriging sobre Santiago.

Dos casos en paralelo. Primero un caso pedagógico con ground truth: el
raster NDVI completo está disponible, fingimos que solo medimos NDVI en
N puntos aleatorios y reconstruimos la superficie. El error de
interpolación se compara contra el raster verdadero. Segundo, un caso
real sin ground truth: PM2.5 promedio diario de las estaciones SINCA de
la Región Metropolitana en un día de alta contaminación, donde el mapa
de varianza del kriging adquiere valor práctico.

Métodos cubiertos:

1. IDW (inverse distance weighting): baseline determinista, sin modelo
   estadístico de la dependencia espacial.
2. Variograma empírico y ajuste de modelo teórico (esférico,
   exponencial, gaussiano) con scikit-gstat.
3. Kriging ordinario con gstools, que produce predicción y varianza.
4. Validación cruzada leave-one-out comparando IDW y kriging.
5. Kriging con tendencia (regression kriging): el modelo lineal captura
   la tendencia global y el kriging modela la dependencia residual.
"""

# %% Descargas
from gdsutils.general import descargar_datos
from gdsutils.ndvi import (
    descargar_luminosidad_santiago,
    descargar_ndvi_santiago_precomputado,
)

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/sinca-santiago-2024.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

ruta_ndvi = descargar_ndvi_santiago_precomputado()
ruta_lum = descargar_luminosidad_santiago()

# %% Imports
from pathlib import Path

import geopandas as gpd
import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import skgstat as skg
from spreg import OLS
from chiricoca.config import setup_style
from chiricoca.geo.figures import (
    figure_from_geodataframe,
    small_multiples_from_geodataframe,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree

from gdsutils.geo import clip_geodataframe

setup_style(dpi=96)


def colorbar_horizontal(fig, im, ax, label, size="3%", pad=0.15):
    """Adjunta una colorbar horizontal debajo de ax sin deformar la
    grilla de small_multiples_from_geodataframe (que asume aspect
    fijo y figsize derivado del bbox del geodataframe)."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size=size, pad=pad)
    return fig.colorbar(im, cax=cax, orientation="horizontal", label=label)


def reservar_espacio_colorbar(ax, size="3%", pad=0.15):
    """Reserva el mismo espacio que colorbar_horizontal pero sin
    dibujar la colorbar. Necesario en grids donde solo algunos paneles
    llevan colorbar: si no se reserva, los paneles sin colorbar quedan
    con altura distinta a los que sí la tienen."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size=size, pad=pad)
    cax.set_visible(False)
    return cax

# %% Configuración
RUTA_CARTO = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
RUTA_SINCA_PM25 = Path("data") / "sinca-santiago-2024" / "pm25-diario.parquet"
RUTA_SINCA_EST = Path("data") / "sinca-santiago-2024" / "estaciones.parquet"

BBOX_WGS84 = (-70.85, -33.65, -70.45, -33.30)
CRS_UTM = "EPSG:32719"

IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)

SEMILLA = 42
N_MUESTRAS_NDVI = 80
RESOLUCION_GRILLA_M = 500
DIA_CRITICO = "2024-07-13"

# %%
# ========================================
# PARTE 1. Caso pedagógico: NDVI submuestreado
# ========================================
# El raster NDVI mediano sobre Sentinel-2 (verano austral 2023-2024) ya
# está reproyectado a UTM 19S, así que las distancias entre celdas son
# euclídeas en metros. Submuestreamos N puntos en zonas válidas (sin
# nodata) y los tratamos como las "observaciones" disponibles para
# reconstruir el resto de la superficie.

print("Cargando NDVI Santiago")
da = rioxarray.open_rasterio(ruta_ndvi).squeeze("band", drop=True)
print(f"  Dimensiones raster: {dict(da.sizes)}")
print(f"  CRS: {da.rio.crs}")
ndvi = da.values.astype(float)
ndvi[~np.isfinite(ndvi)] = np.nan

# Coordenadas del raster en UTM 19S.
xs = da.x.values  # eje x (este, m)
ys = da.y.values  # eje y (norte, m). En UTM, y decrece hacia el sur.
extent_utm = (xs.min(), xs.max(), ys.min(), ys.max())
print(f"  extent UTM: {extent_utm}")

# Comunas para contexto.
comunas = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(
    CRS_UTM
)
comunas = clip_geodataframe(comunas, list(da.rio.bounds()))
print(f"  Comunas en bbox: {len(comunas)}")

# Submuestreo: elegir índices aleatorios donde NDVI no es nan.
rng = np.random.default_rng(SEMILLA)
filas_validas, cols_validas = np.where(np.isfinite(ndvi))
idx = rng.choice(len(filas_validas), size=N_MUESTRAS_NDVI, replace=False)
filas_intro, cols_intro = filas_validas[idx], cols_validas[idx]
puntos_x_intro = xs[cols_intro]
puntos_y_intro = ys[filas_intro]
puntos_v_intro = ndvi[filas_intro, cols_intro]
print(
    f"  {N_MUESTRAS_NDVI} puntos muestrales: "
    f"NDVI min={puntos_v_intro.min():.2f} max={puntos_v_intro.max():.2f} "
    f"mean={puntos_v_intro.mean():.2f}"
)

# %% Mapa: ground truth + puntos muestrales
fig, axes = small_multiples_from_geodataframe(comunas, n_variables=2, height=6)
for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.5)

axes[0].imshow(
    ndvi,
    extent=extent_utm,
    origin="upper",
    cmap="YlGn",
    vmin=-0.2,
    vmax=0.8,
    aspect="auto",
)
axes[0].set_title("NDVI verdadero (ground truth)")

axes[1].imshow(
    ndvi,
    extent=extent_utm,
    origin="upper",
    cmap="Greys",
    alpha=0.25,
    aspect="auto",
)
axes[1].scatter(
    puntos_x_intro,
    puntos_y_intro,
    c=puntos_v_intro,
    cmap="YlGn",
    vmin=-0.2,
    vmax=0.8,
    s=40,
    edgecolor="black",
    linewidth=0.5,
)
axes[1].set_title(f"{N_MUESTRAS_NDVI} muestras aleatorias (entrada al kriging)")
fig.suptitle("Caso pedagógico: el ground truth está disponible")
fig.savefig(IMAGES / "07-ndvi-submuestreo.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 2. IDW como baseline determinista
# ========================================
# IDW asume que el valor en un punto no muestreado es un promedio
# ponderado por 1/d^p de los k vecinos más cercanos. No usa modelo
# estadístico de la dependencia espacial: no estima incertidumbre, no
# diferencia entre datos correlacionados a corta distancia y datos sin
# estructura. Sirve como referencia mínima.

# Grilla regular para visualizar la interpolación.
xs_grid = np.arange(xs.min(), xs.max(), RESOLUCION_GRILLA_M)
ys_grid = np.arange(ys.max(), ys.min(), -RESOLUCION_GRILLA_M)
xx, yy = np.meshgrid(xs_grid, ys_grid)
puntos_grid = np.column_stack([xx.ravel(), yy.ravel()])
print(f"Grilla de predicción: {len(xs_grid)} x {len(ys_grid)} = {len(puntos_grid):,} celdas")


def interpolar_idw(coords_obs, valores, coords_pred, k=8, potencia=2.0):
    """IDW con los k vecinos más cercanos.

    distancias d_i, peso w_i = 1 / d_i^p. Si un punto coincide con una
    observación (d=0), se devuelve el valor exacto.
    """
    arbol = cKDTree(coords_obs)
    dists, idxs = arbol.query(coords_pred, k=k)
    # Evitar división por cero: si un punto cae sobre observación.
    dists = np.where(dists < 1e-9, 1e-9, dists)
    pesos = 1.0 / dists**potencia
    return (pesos * valores[idxs]).sum(axis=1) / pesos.sum(axis=1)


coords_intro = np.column_stack([puntos_x_intro, puntos_y_intro])
ndvi_idw_intro = interpolar_idw(coords_intro, puntos_v_intro, puntos_grid).reshape(xx.shape)
print(
    f"IDW: predicción min={ndvi_idw_intro.min():.2f} "
    f"max={ndvi_idw_intro.max():.2f} mean={ndvi_idw_intro.mean():.2f}"
)

# %%
# ========================================
# PARTE 3. Variograma empírico
# ========================================
# El variograma describe cómo aumenta la varianza entre pares de puntos
# a medida que aumenta la distancia entre ellos. Si la variable tiene
# dependencia espacial, los pares cercanos tienen baja varianza
# (similares) y los lejanos tienden a la varianza total. La distancia
# donde el variograma se estabiliza es el rango (alcance de la
# dependencia espacial).
#
# scikit-gstat agrupa pares en bins de distancia y promedia la
# semivarianza dentro de cada bin. n_lags controla la resolución.

print("Calculando variograma empírico (NDVI)")
variograma_intro = skg.Variogram(
    coordinates=coords_intro,
    values=puntos_v_intro,
    n_lags=15,
    maxlag=20000,
    normalize=False,
)
print(variograma_intro)

# %%
# ========================================
# PARTE 4. Ajuste de modelos teóricos
# ========================================
# Cada modelo tiene tres parámetros estimados: nugget (varianza a
# distancia 0, ruido de medición), sill (varianza asintótica) y range
# (distancia donde se alcanza el sill). El modelo teórico permite
# evaluar la covarianza para cualquier distancia, no solo en los lags
# discretos del variograma empírico. Comparamos esférico, exponencial y
# gaussiano por RMSE entre la curva y la nube empírica.

modelos = ["spherical", "exponential", "gaussian"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    variograma_intro.bins,
    variograma_intro.experimental,
    color="black",
    s=40,
    label="empírico",
    zorder=3,
)
xs_var = np.linspace(0, variograma_intro.bins[-1], 300)
diagnosticos = []
for nombre in modelos:
    variograma_intro.model = nombre
    ax.plot(
        xs_var, variograma_intro.fitted_model(xs_var),
        label=f"{nombre} (RMSE={variograma_intro.rmse:.4f})", linewidth=2,
    )
    diagnosticos.append(
        {
            "modelo": nombre,
            "range_m": variograma_intro.parameters[0],
            "sill": variograma_intro.parameters[1],
            "nugget": variograma_intro.parameters[2] if len(variograma_intro.parameters) >= 3 else 0.0,
            "rmse": variograma_intro.rmse,
        }
    )
ax.set_xlabel("Distancia (m)")
ax.set_ylabel("Semivarianza")
ax.set_title("Variograma empírico y modelos teóricos (NDVI)")
ax.legend()
fig.savefig(IMAGES / "07-variograma-ndvi.png", dpi=192, bbox_inches="tight")
plt.show()

print("\nDiagnósticos por modelo:")
print(pd.DataFrame(diagnosticos).round(4).to_string(index=False))

# El mejor ajuste minimiza el RMSE entre semivarianza empírica y modelo.
mejor_intro = min(diagnosticos, key=lambda d: d["rmse"])
print(f"\nMejor ajuste: {mejor_intro['modelo']} (RMSE={mejor_intro['rmse']:.4f})")
variograma_intro.model = mejor_intro["modelo"]

# %%
# ========================================
# PARTE 5. Kriging ordinario con gstools
# ========================================
# El kriging ordinario asume:
# - media constante pero desconocida (sin tendencia explícita).
# - estructura de covarianza dada por el variograma.
# La predicción en un punto no muestreado es una combinación lineal de
# las observaciones, con pesos que minimizan el error cuadrático medio
# bajo la restricción de que sumen 1. Devuelve además la varianza de
# kriging, que mide la incertidumbre de la predicción en cada punto.

# scikit-gstat exporta directamente un objeto gstools.Krige con el modelo
# fitteado y las observaciones cargadas. unbiased=True es kriging
# ordinario (impone que los pesos sumen 1, equivalente a media constante
# desconocida).
kriging_intro = variograma_intro.to_gs_krige(unbiased=True)
print(f"Modelo de covarianza: {kriging_intro.model.name}")
print(
    f"  var={kriging_intro.model.var:.4f} len_scale={kriging_intro.model.len_scale:.0f} m "
    f"nugget={kriging_intro.model.nugget:.4f}"
)

print("Kriging ordinario sobre la grilla de predicción")
kriging_intro.set_pos((xx.ravel(), yy.ravel()))
kriging_intro(return_var=True)
ndvi_ok_intro = kriging_intro.field.reshape(xx.shape)
ndvi_var_intro = kriging_intro.krige_var.reshape(xx.shape)
print(
    f"  predicción min={ndvi_ok_intro.min():.2f} max={ndvi_ok_intro.max():.2f} "
    f"mean={ndvi_ok_intro.mean():.2f}"
)
print(
    f"  varianza kriging min={ndvi_var_intro.min():.3f} "
    f"max={ndvi_var_intro.max():.3f} mean={ndvi_var_intro.mean():.3f}"
)

# %% Comparación visual: ground truth, IDW, kriging, varianza kriging
fig, axes = small_multiples_from_geodataframe(
    comunas, n_variables=4, col_wrap=2, height=5
)

axes[0].imshow(
    ndvi, extent=extent_utm, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[0].set_title("Ground truth (raster Sentinel-2)")

extent_grid = (xs_grid.min(), xs_grid.max(), ys_grid.min(), ys_grid.max())
axes[1].imshow(
    ndvi_idw_intro, extent=extent_grid, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[1].scatter(puntos_x_intro, puntos_y_intro, c="black", s=4)
axes[1].set_title("IDW (k=8, p=2)")

axes[2].imshow(
    ndvi_ok_intro, extent=extent_grid, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[2].scatter(puntos_x_intro, puntos_y_intro, c="black", s=4)
axes[2].set_title(f"Kriging ordinario ({mejor_intro['modelo']})")

im = axes[3].imshow(
    ndvi_var_intro, extent=extent_grid, origin="upper", cmap="magma", aspect="auto",
)
axes[3].scatter(puntos_x_intro, puntos_y_intro, c="cyan", s=8, edgecolor="black", linewidth=0.3)
axes[3].set_title("Varianza del kriging (incertidumbre)")
colorbar_horizontal(fig, im, axes[3], r"Varianza del kriging ($\sigma^2$)")
for i in [0, 1, 2]:
    reservar_espacio_colorbar(axes[i])

for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4)

fig.suptitle("Reconstrucción de NDVI a partir de 80 muestras")
fig.savefig(IMAGES / "07-ndvi-comparacion.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - IDW da una superficie con artefactos en forma de "ojos de buey"
#   alrededor de cada punto: el peso 1/d^2 domina muy cerca de las
#   observaciones y se diluye lejos.
# - Kriging produce una superficie más suave porque pondera por la
#   estructura de covarianza del variograma, no solo por distancia.
# - La varianza del kriging es baja cerca de las observaciones y crece
#   lejos. Donde no hay puntos, el modelo dice "no sé". IDW no expresa
#   esa duda.

# %%
# ========================================
# PARTE 6. Validación cruzada leave-one-out
# ========================================
# Para cada observación: la quitamos del conjunto, predecimos su valor
# desde las restantes, comparamos con el real. RMSE y MAE en LOO miden
# qué tan bien generaliza cada método al espacio no muestreado.

print("Validación cruzada LOO")
preds_idw = np.zeros(N_MUESTRAS_NDVI)
preds_ok = np.zeros(N_MUESTRAS_NDVI)
for i in range(N_MUESTRAS_NDVI):
    mask = np.arange(N_MUESTRAS_NDVI) != i
    coords_train = coords_intro[mask]
    valores_train = puntos_v_intro[mask]
    pt = coords_intro[i : i + 1]

    preds_idw[i] = interpolar_idw(coords_train, valores_train, pt)[0]

    ok_loo = gs.krige.Krige(
        kriging_intro.model,
        cond_pos=(coords_train[:, 0], coords_train[:, 1]),
        cond_val=valores_train,
        unbiased=True,
    )
    ok_loo.set_pos((pt[:, 0], pt[:, 1]))
    ok_loo()
    preds_ok[i] = ok_loo.field[0]

resumen_loo = pd.DataFrame(
    {
        "metodo": ["IDW (k=8, p=2)", f"Kriging ordinario ({mejor_intro['modelo']})"],
        "RMSE": [
            np.sqrt(np.mean((preds_idw - puntos_v_intro) ** 2)),
            np.sqrt(np.mean((preds_ok - puntos_v_intro) ** 2)),
        ],
        "MAE": [
            np.mean(np.abs(preds_idw - puntos_v_intro)),
            np.mean(np.abs(preds_ok - puntos_v_intro)),
        ],
    }
)
print(resumen_loo.round(4).to_string(index=False))

# Si el variograma capturó estructura espacial real, kriging gana.
# Si no, IDW puede ser competitivo. NDVI tiene fuerte estructura
# espacial (vegetación se agrupa), así que kriging debería ganar.

# %%
# ========================================
# PARTE 7. ¿Cuándo conviene kriging? Error vs densidad de muestreo
# ========================================
# Hasta aquí mostramos cómo se calcula kriging. Con N=80, IDW y OK
# dieron RMSE parecidos en LOO. La pregunta operacional es a partir
# de qué densidad de muestreo el variograma se vuelve estable y
# kriging supera consistentemente a IDW. Como tenemos ground truth
# (raster completo), evaluamos sobre un set fijo de píxeles de test
# disjuntos del set de entrenamiento, para varios N y semillas.

print("\nEstudio de error vs N (NDVI)")

N_TEST = 2000
N_VALORES = [20, 40, 80, 160, 320, 640]
N_SEMILLAS = 5

rng_test = np.random.default_rng(SEMILLA + 1000)
idx_test = rng_test.choice(len(filas_validas), size=N_TEST, replace=False)
fi_test, ci_test = filas_validas[idx_test], cols_validas[idx_test]
test_x = xs[ci_test]
test_y = ys[fi_test]
test_v = ndvi[fi_test, ci_test]
test_coords = np.column_stack([test_x, test_y])
mask_disponibles = np.ones(len(filas_validas), dtype=bool)
mask_disponibles[idx_test] = False
idx_disponibles = np.where(mask_disponibles)[0]
print(f"  set de test: {N_TEST} píxeles, pool train: {len(idx_disponibles):,}")

resultados = []
for n_obs in N_VALORES:
    # n_lags chico cuando hay pocos puntos (variograma muy ruidoso si no).
    n_lags_n = min(15, max(5, n_obs // 4))
    for s in range(N_SEMILLAS):
        rng_local = np.random.default_rng(SEMILLA + 100 * s + n_obs)
        idx_obs = rng_local.choice(idx_disponibles, size=n_obs, replace=False)
        fi_obs = filas_validas[idx_obs]
        ci_obs = cols_validas[idx_obs]
        obs_coords = np.column_stack([xs[ci_obs], ys[fi_obs]])
        obs_v = ndvi[fi_obs, ci_obs]

        pred_idw = interpolar_idw(obs_coords, obs_v, test_coords)
        rmse_idw = np.sqrt(np.mean((pred_idw - test_v) ** 2))

        try:
            v_n = skg.Variogram(
                coordinates=obs_coords, values=obs_v,
                n_lags=n_lags_n, maxlag=20000, normalize=False,
            )
            mejor_modelo, mejor_modelo_rmse = None, np.inf
            for nm in modelos:
                v_n.model = nm
                if v_n.rmse < mejor_modelo_rmse:
                    mejor_modelo_rmse = v_n.rmse
                    mejor_modelo = nm
            v_n.model = mejor_modelo
            ok_n = v_n.to_gs_krige(unbiased=True)
            ok_n.set_pos((test_x, test_y))
            ok_n()
            rmse_ok = np.sqrt(np.mean((ok_n.field - test_v) ** 2))
        except Exception as e:
            print(f"  N={n_obs} s={s}: kriging falló ({e})")
            rmse_ok, mejor_modelo = np.nan, None

        resultados.append({
            "N": n_obs, "semilla": s,
            "rmse_idw": rmse_idw, "rmse_ok": rmse_ok,
            "modelo": mejor_modelo,
        })
        print(f"  N={n_obs:4d} s={s} idw={rmse_idw:.4f} kriging_intro={rmse_ok:.4f} ({mejor_modelo})")

df_resultados = pd.DataFrame(resultados)
resumen = df_resultados.groupby("N").agg(
    idw_mean=("rmse_idw", "mean"), idw_std=("rmse_idw", "std"),
    ok_mean=("rmse_ok", "mean"), ok_std=("rmse_ok", "std"),
).reset_index()
print("\nResumen RMSE vs N:")
print(resumen.round(4).to_string(index=False))

# %% Figura: RMSE vs N
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(resumen["N"], resumen["idw_mean"], marker="o", color="C0",
        label="IDW (k=8, p=2)")
ax.fill_between(
    resumen["N"],
    resumen["idw_mean"] - resumen["idw_std"],
    resumen["idw_mean"] + resumen["idw_std"],
    color="C0", alpha=0.2,
)
ax.plot(resumen["N"], resumen["ok_mean"], marker="o", color="C1",
        label="Kriging ordinario")
ax.fill_between(
    resumen["N"],
    resumen["ok_mean"] - resumen["ok_std"],
    resumen["ok_mean"] + resumen["ok_std"],
    color="C1", alpha=0.2,
)
ax.set_xscale("log")
ax.set_xlabel("Número de observaciones N")
ax.set_ylabel("RMSE contra ground truth")
ax.set_title(f"Error de interpolación vs densidad de muestreo ({N_SEMILLAS} semillas)")
ax.legend()
fig.savefig(IMAGES / "07-ndvi-error-vs-n.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - Con pocos puntos el variograma es ruidoso y kriging no tiene ventaja
#   sobre IDW; la elección entre uno y otro queda dentro del ruido de
#   muestreo. La banda de std refleja esa inestabilidad.
# - A partir de cierto N las bandas dejan de tocarse y kriging gana de
#   forma consistente. Ese es el umbral pragmático: por debajo de ahí,
#   pagar el costo del variograma no rinde.

# %%
# ========================================
# PARTE 7b. Reconstrucción con N elegido
# ========================================
# Elegimos el N más chico donde OK gana a IDW por más que la suma de
# las desviaciones estándar (criterio simple de no superposición de
# bandas). Si ningún N satisface, usamos el más grande del barrido.

gana_ok = (resumen["idw_mean"] - resumen["ok_mean"]) > (
    resumen["idw_std"] + resumen["ok_std"]
)
n_buenos = resumen.loc[gana_ok, "N"]
N_BUENO = int(n_buenos.iloc[0]) if len(n_buenos) > 0 else int(resumen["N"].max())
print(f"\nN elegido para reconstrucción: {N_BUENO}")

rng_obs = np.random.default_rng(SEMILLA)
idx_obs = rng_obs.choice(idx_disponibles, size=N_BUENO, replace=False)
filas_obs, cols_obs = filas_validas[idx_obs], cols_validas[idx_obs]
puntos_x_obs = xs[cols_obs]
puntos_y_obs = ys[filas_obs]
puntos_v_obs = ndvi[filas_obs, cols_obs]
coords_obs = np.column_stack([puntos_x_obs, puntos_y_obs])

variograma_obs = skg.Variogram(coordinates=coords_obs, values=puntos_v_obs,
                    n_lags=15, maxlag=20000, normalize=False)
mejor_modelo_obs, mejor_modelo_rmse_obs = None, np.inf
for nm in modelos:
    variograma_obs.model = nm
    if variograma_obs.rmse < mejor_modelo_rmse_obs:
        mejor_modelo_rmse_obs = variograma_obs.rmse
        mejor_modelo_obs = nm
variograma_obs.model = mejor_modelo_obs
print(f"  modelo de variograma: {mejor_modelo_obs} (RMSE={mejor_modelo_rmse_obs:.4f})")

kriging_obs = variograma_obs.to_gs_krige(unbiased=True)
kriging_obs.set_pos((xx.ravel(), yy.ravel()))
kriging_obs(return_var=True)
ndvi_ok_obs = kriging_obs.field.reshape(xx.shape)
ndvi_var_obs = kriging_obs.krige_var.reshape(xx.shape)
ndvi_idw_obs = interpolar_idw(coords_obs, puntos_v_obs, puntos_grid).reshape(xx.shape)

# %% Mapa final con N elegido
fig, axes = small_multiples_from_geodataframe(
    comunas, n_variables=4, col_wrap=2, height=5
)

axes[0].imshow(ndvi, extent=extent_utm, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[0].set_title("Ground truth")

axes[1].imshow(ndvi_idw_obs, extent=extent_grid, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[1].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[1].set_title(f"IDW con N={N_BUENO}")

axes[2].imshow(ndvi_ok_obs, extent=extent_grid, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[2].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[2].set_title(f"Kriging ordinario ({mejor_modelo_obs}) con N={N_BUENO}")

im = axes[3].imshow(ndvi_var_obs, extent=extent_grid, origin="upper",
                    cmap="magma", aspect="auto")
axes[3].scatter(puntos_x_obs, puntos_y_obs, c="cyan", s=8,
                edgecolor="black", linewidth=0.3)
axes[3].set_title("Varianza del kriging")
colorbar_horizontal(fig, im, axes[3], r"Varianza del kriging ($\sigma^2$)")
for i in [0, 1, 2]:
    reservar_espacio_colorbar(axes[i])

for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4)

fig.suptitle(f"Reconstrucción NDVI con N elegido = {N_BUENO}")
fig.savefig(IMAGES / "07-ndvi-reconstruccion-final.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 8. Kriging con tendencia (regression kriging) sobre NDVI
# ========================================
# Si una covariable explica parte de la variación, podemos quitarla
# antes de aplicar kriging. La luminosidad nocturna (urbanización)
# correlaciona negativamente con NDVI: zonas urbanas tienen poca
# vegetación. Modelamos:
#     NDVI(s) = a + b * luminosidad(s) + Z(s)
# donde Z(s) es un proceso espacial de media cero. Estimamos a y b por
# OLS sobre el sample N=N_BUENO de PARTE 7, kriging sobre los residuos
# y sumamos la tendencia ajustada en cada celda. Reusar N_BUENO permite
# comparar OK, RK y GPBoost (PARTE 9) con el mismo training set.

print("\nKriging con tendencia: NDVI ~ luminosidad")
da_lum = rioxarray.open_rasterio(ruta_lum).squeeze("band", drop=True)
print(f"  Luminosidad: dim={dict(da_lum.sizes)} CRS={da_lum.rio.crs}")

# Reproyectar luminosidad a la misma grilla del NDVI para indexar igual.
lum_aligned = da_lum.rio.reproject_match(da)
lum_arr = lum_aligned.values.astype(float)
lum_arr[~np.isfinite(lum_arr)] = np.nan

# Luminosidad en los puntos muestrales (N=N_BUENO) y en la grilla.
puntos_lum_obs = lum_arr[filas_obs, cols_obs]
log_lum_pts_obs = np.log1p(np.clip(puntos_lum_obs, 0, None))
mask_ok_obs = np.isfinite(log_lum_pts_obs) & np.isfinite(puntos_v_obs)
print(
    f"  N efectivo (con luminosidad válida): {int(mask_ok_obs.sum())} / {N_BUENO}"
)

# OLS NDVI = a + b * log(1 + luminosidad). Usamos spreg.OLS por
# consistencia con el script 05 (regresión sobre eBird). Para un
# único predictor el summary tiene el mismo contenido que polyfit
# pero incluye SE, t, p y R^2.
y_ols = puntos_v_obs[mask_ok_obs].reshape(-1, 1)
X_ols = log_lum_pts_obs[mask_ok_obs].reshape(-1, 1)
ols = OLS(y_ols, X_ols, name_y="ndvi", name_x=["log_luminosidad"])
print(ols.summary)
b0 = float(ols.betas[0, 0])
b1 = float(ols.betas[1, 0])

# Cómo leer el summary:
# - CONSTANT (b0): NDVI esperado para log_luminosidad = 0 (zonas oscuras).
# - log_luminosidad (b1): cambio en NDVI por aumento unitario en
#   log(1+luminosidad). Signo negativo esperado (más urbanización =
#   menos vegetación).
# - R^2 bajo (~0.08) indica que la tendencia lineal captura poca
#   varianza. Eso es exactamente lo que justifica el paso siguiente:
#   kriging sobre los residuos modela la estructura espacial que OLS
#   no explica.
# - Jarque-Bera rechaza normalidad (cola larga típica de NDVI) y
#   Breusch-Pagan está borderline. Los SE del OLS son aproximados,
#   pero no afectan el uso de las betas como estimador puntual de la
#   tendencia: lo que importa para regression kriging es que las
#   betas sean consistentes, y lo son aun bajo heterocedasticidad.

trend_pts_obs = b0 + b1 * log_lum_pts_obs
residuos_pts_obs = puntos_v_obs - trend_pts_obs
print(
    f"  residuos: mean={np.nanmean(residuos_pts_obs):.4f} "
    f"std={np.nanstd(residuos_pts_obs):.3f}"
)

# Variograma de los residuos. Si la tendencia capturó variación,
# el sill de los residuos es menor que el sill original.
v_res = skg.Variogram(
    coordinates=coords_obs[mask_ok_obs],
    values=residuos_pts_obs[mask_ok_obs],
    n_lags=15,
    maxlag=20000,
    normalize=False,
    model=mejor_modelo_obs,
)
print(
    f"  variograma residuos: sill={v_res.parameters[1]:.4f} "
    f"(NDVI N_BUENO sill={variograma_obs.parameters[1]:.4f})"
)

# Kriging sobre los residuos en la grilla.
ok_res = v_res.to_gs_krige(unbiased=True)
ok_res.set_pos((xx.ravel(), yy.ravel()))
ok_res()
res_field = ok_res.field.reshape(xx.shape)

# Tendencia en la grilla: muestrear luminosidad sobre xx, yy.
filas_grid = ((ys.max() - yy) / (ys.max() - ys.min()) * (lum_arr.shape[0] - 1)).astype(int)
cols_grid = ((xx - xs.min()) / (xs.max() - xs.min()) * (lum_arr.shape[1] - 1)).astype(int)
filas_grid = np.clip(filas_grid, 0, lum_arr.shape[0] - 1)
cols_grid = np.clip(cols_grid, 0, lum_arr.shape[1] - 1)
lum_grid = lum_arr[filas_grid, cols_grid]
log_lum_grid = np.log1p(np.clip(lum_grid, 0, None))
trend_grid = b0 + b1 * log_lum_grid

ndvi_rk = trend_grid + res_field
print(
    f"Regression kriging: min={np.nanmin(ndvi_rk):.2f} "
    f"max={np.nanmax(ndvi_rk):.2f} mean={np.nanmean(ndvi_rk):.2f}"
)

# %% Comparación: ground truth, OK, RK
fig, axes = small_multiples_from_geodataframe(
    comunas, n_variables=3, col_wrap=3, height=6
)
for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4)

axes[0].imshow(ndvi, extent=extent_utm, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[0].set_title("Ground truth")

axes[1].imshow(ndvi_ok_obs, extent=extent_grid, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[1].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[1].set_title(f"Kriging ordinario (N={N_BUENO})")

axes[2].imshow(ndvi_rk, extent=extent_grid, origin="upper", cmap="YlGn",
               vmin=-0.2, vmax=0.8, aspect="auto")
axes[2].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[2].set_title(f"Regression kriging (luminosidad, N={N_BUENO})")

fig.suptitle("Kriging ordinario vs regression kriging")
fig.savefig(IMAGES / "07-rk-comparacion.png", dpi=192, bbox_inches="tight")
plt.show()

# RMSE contra ground truth en la grilla.
filas_t = ((ys.max() - yy) / (ys.max() - ys.min()) * (ndvi.shape[0] - 1)).astype(int)
cols_t = ((xx - xs.min()) / (xs.max() - xs.min()) * (ndvi.shape[1] - 1)).astype(int)
filas_t = np.clip(filas_t, 0, ndvi.shape[0] - 1)
cols_t = np.clip(cols_t, 0, ndvi.shape[1] - 1)
ndvi_truth_grid = ndvi[filas_t, cols_t]

mask_eval = np.isfinite(ndvi_truth_grid) & np.isfinite(ndvi_ok_obs) & np.isfinite(ndvi_rk)
rmse_ok_g = np.sqrt(np.mean((ndvi_ok_obs[mask_eval] - ndvi_truth_grid[mask_eval]) ** 2))
rmse_rk_g = np.sqrt(np.mean((ndvi_rk[mask_eval] - ndvi_truth_grid[mask_eval]) ** 2))
print(f"\nRMSE contra ground truth en la grilla:")
print(f"  Kriging ordinario:  {rmse_ok_g:.4f}")
print(f"  Regression kriging: {rmse_rk_g:.4f}")
print(f"  Reducción: {(rmse_ok_g - rmse_rk_g) / rmse_ok_g * 100:.1f}%")

# Cómo leer:
# - Si la covariable explica variación residual, RK gana sobre OK.
# - Si NDVI ya está bien capturado por kriging puro (sample denso),
#   el beneficio es marginal. El gradiente urbano-periferia es real
#   pero ya está implícito en la dependencia espacial.

# %%
# ========================================
# PARTE 9. Tendencia no lineal con GPBoost
# ========================================
# Regression kriging asume tendencia lineal en las covariables. Si la
# relación es no lineal o hay interacciones entre covariables, OLS
# captura poco y deja todo el peso al variograma residual. GPBoost
# combina dos piezas que se ajustan en conjunto:
# 1. La tendencia F(X) la estima un booster de árboles (LightGBM bajo
#    el capó), que captura no linealidades sin especificar la forma
#    funcional.
# 2. La dependencia espacial residual la modela una GP con kernel
#    exponencial parametrizada por var, len_scale y nugget (los mismos
#    parámetros del variograma).
#
# Aquí el booster recibe luminosidad y coordenadas UTM, así puede
# aprender un patrón espacial general antes de que la GP modele la
# dependencia residual de corto alcance. Usamos el mismo sample
# N_BUENO de PARTE 7 para que el RMSE en el set de test sea
# directamente comparable con OK y RK.

import gpboost as gpb

print("\nGPBoost: NDVI ~ booster(log_lum) + GP(coords)")

# Solo usamos log_lum como feature del booster. Si pasamos también las
# coordenadas, hay redundancia con la GP: el booster memoriza la
# ubicación de cada punto y le quita estructura espacial al GP.
# Manteniendo solo log_lum, GPBoost es el análogo no lineal directo a
# regression kriging: si la relación NDVI ~ luminosidad es lineal,
# GPBoost colapsa a RK; si tiene no linealidades, las captura.
X_train = log_lum_pts_obs[mask_ok_obs].reshape(-1, 1)
y_train = puntos_v_obs[mask_ok_obs]
coords_train = coords_obs[mask_ok_obs]
print(f"  N train: {len(y_train)} (feature único: log_lum)")

gp_model = gpb.GPModel(gp_coords=coords_train, cov_function="exponential")
data_train = gpb.Dataset(X_train, y_train, feature_name=["log_lum"])
params = {
    "objective": "regression_l2",
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 5,
    "min_data_in_leaf": 30,
    "lambda_l2": 1.0,
    "verbose": -1,
}
booster = gpb.train(
    params=params,
    train_set=data_train,
    gp_model=gp_model,
    num_boost_round=100,
)
cov_pars = gp_model.get_cov_pars()
print("  parámetros GP estimados (Error_var=nugget, GP_var=sill, GP_range=len_scale):")
print(cov_pars.to_string())

# Predicción sobre la grilla.
X_grid = log_lum_grid.ravel().reshape(-1, 1)
coords_grid = np.column_stack([xx.ravel(), yy.ravel()])
print("  Prediciendo sobre grilla")
pred = booster.predict(
    data=X_grid,
    gp_coords_pred=coords_grid,
    predict_var=True,
    pred_latent=False,
)
ndvi_gpb = np.asarray(pred["response_mean"]).reshape(xx.shape)
ndvi_gpb_var = np.asarray(pred["response_var"]).reshape(xx.shape)
print(
    f"  predicción: min={np.nanmin(ndvi_gpb):.2f} "
    f"max={np.nanmax(ndvi_gpb):.2f} mean={np.nanmean(ndvi_gpb):.2f}"
)

# Evaluación en el set de test fijo de PARTE 7. Filtramos píxeles con
# luminosidad NaN para que las tres métricas (OK, RK, GPBoost)
# comparen la misma muestra.
log_lum_test = np.log1p(np.clip(lum_arr[fi_test, ci_test], 0, None))
mask_test_lum = np.isfinite(log_lum_test) & np.isfinite(test_v)
print(
    f"\nN test válido (luminosidad finita): {int(mask_test_lum.sum())} / {N_TEST}"
)

# OK sobre test set (refit con el sample N_BUENO).
ok_test = variograma_obs.to_gs_krige(unbiased=True)
ok_test.set_pos((test_x[mask_test_lum], test_y[mask_test_lum]))
ok_test()
rmse_ok_t = np.sqrt(np.mean((ok_test.field - test_v[mask_test_lum]) ** 2))

# RK sobre test set: tendencia OLS + krige de residuos.
ok_res_test = v_res.to_gs_krige(unbiased=True)
ok_res_test.set_pos((test_x[mask_test_lum], test_y[mask_test_lum]))
ok_res_test()
rk_test = (
    b0 + b1 * log_lum_test[mask_test_lum] + ok_res_test.field
)
rmse_rk_t = np.sqrt(np.mean((rk_test - test_v[mask_test_lum]) ** 2))

# GPBoost sobre test set.
X_test = log_lum_test[mask_test_lum].reshape(-1, 1)
coords_test_eval = np.column_stack([test_x[mask_test_lum], test_y[mask_test_lum]])
pred_test = booster.predict(
    data=X_test,
    gp_coords_pred=coords_test_eval,
    predict_var=False,
    pred_latent=False,
)
gpb_test = np.asarray(pred_test["response_mean"])
rmse_gpb_t = np.sqrt(np.mean((gpb_test - test_v[mask_test_lum]) ** 2))

resumen_metodos = pd.DataFrame(
    {
        "método": [
            "Kriging ordinario",
            "Regression kriging (OLS + GP)",
            "GPBoost (booster + GP)",
        ],
        "RMSE_test": [rmse_ok_t, rmse_rk_t, rmse_gpb_t],
    }
)
print(f"\nRMSE en el set de test (N={int(mask_test_lum.sum())}, mismo training N={N_BUENO}):")
print(resumen_metodos.round(4).to_string(index=False))

# %% Mapa: RK vs GPBoost
fig, axes = small_multiples_from_geodataframe(
    comunas, n_variables=4, col_wrap=2, height=5
)
for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.4)

axes[0].imshow(
    ndvi, extent=extent_utm, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[0].set_title("Ground truth")

axes[1].imshow(
    ndvi_rk, extent=extent_grid, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[1].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[1].set_title(f"Regression kriging (RMSE_test={rmse_rk_t:.3f})")

axes[2].imshow(
    ndvi_gpb, extent=extent_grid, origin="upper", cmap="YlGn",
    vmin=-0.2, vmax=0.8, aspect="auto",
)
axes[2].scatter(puntos_x_obs, puntos_y_obs, c="black", s=4)
axes[2].set_title(f"GPBoost (RMSE_test={rmse_gpb_t:.3f})")

im = axes[3].imshow(
    ndvi_gpb_var, extent=extent_grid, origin="upper", cmap="magma", aspect="auto",
)
axes[3].scatter(
    puntos_x_obs, puntos_y_obs, c="cyan", s=8, edgecolor="black", linewidth=0.3
)
axes[3].set_title("Varianza predictiva GPBoost")
colorbar_horizontal(fig, im, axes[3], r"Varianza del kriging ($\sigma^2$)")
for i in [0, 1, 2]:
    reservar_espacio_colorbar(axes[i])

fig.suptitle("Tendencia lineal (RK) vs tendencia boosteada (GPBoost)")
fig.savefig(IMAGES / "07-gpboost-comparacion.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - Si la tendencia lineal con luminosidad ya capturó el gradiente
#   urbano-periferia, GPBoost solo aporta marginalmente.
# - Si hay no linealidades (saturación de NDVI en zonas muy verdes,
#   transiciones bruscas) o el booster aprende un patrón espacial
#   no codificado por luminosidad, GPBoost reduce más el RMSE.
# - La varianza predictiva de GPBoost combina la incertidumbre del
#   booster con la de la GP, análoga a la varianza de kriging.

# %%
# ========================================
# PARTE 10. Caso real: PM2.5 SINCA en un día crítico
# ========================================
# El 2024-07-13 fue un día de alta contaminación por PM2.5 en Santiago
# (preemergencia ambiental). Las 10 estaciones SINCA de la RM dan una
# nube de mediciones puntuales y queremos un mapa continuo. No hay
# ground truth, así que la varianza del kriging adquiere valor: dice
# dónde el modelo extrapola con confianza y dónde no.

print(f"\nCargando SINCA PM2.5 día crítico {DIA_CRITICO}")
pm25 = pd.read_parquet(RUTA_SINCA_PM25)
estaciones = pd.read_parquet(RUTA_SINCA_EST)
print(f"  filas pm25: {len(pm25):,}, estaciones: {len(estaciones)}")

dia = pm25[pm25["fecha"] == DIA_CRITICO].merge(estaciones, on="codigo")
print(f"  estaciones activas el {DIA_CRITICO}: {len(dia)}")
print(dia[["codigo", "nombre", "pm25", "lat", "lon"]].to_string(index=False))

# Proyectar a UTM 19S para que las distancias estén en metros.
geo_dia = gpd.GeoDataFrame(
    dia, geometry=gpd.points_from_xy(dia["lon"], dia["lat"]), crs="EPSG:4326"
).to_crs(CRS_UTM)
sx = geo_dia.geometry.x.values
sy = geo_dia.geometry.y.values
sv = geo_dia["pm25"].values.astype(float)
print(f"  PM2.5: min={sv.min():.0f} max={sv.max():.0f} mean={sv.mean():.1f} ug/m^3")

# %% Variograma empírico SINCA y elección de modelo
# Con tan pocas estaciones (10) el variograma es ruidoso. Usamos pocos
# lags y el rango sugerido por la geometría del problema (~30 km es la
# extensión del Gran Santiago).

print("Variograma empírico PM2.5")
v_pm = skg.Variogram(
    coordinates=np.column_stack([sx, sy]),
    values=sv,
    n_lags=8,
    maxlag=30000,
    normalize=False,
)
print(v_pm)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(v_pm.bins, v_pm.experimental, color="black", s=40, zorder=3, label="empírico")
xs_var = np.linspace(0, v_pm.bins[-1], 300)
diag_pm = []
for nombre in modelos:
    v_pm.model = nombre
    ax.plot(
        xs_var, v_pm.fitted_model(xs_var),
        label=f"{nombre} (RMSE={v_pm.rmse:.2f})", linewidth=2,
    )
    diag_pm.append({"modelo": nombre, "rmse": v_pm.rmse})
ax.set_xlabel("Distancia (m)")
ax.set_ylabel("Semivarianza")
ax.set_title(f"Variograma PM2.5 ({DIA_CRITICO})")
ax.legend()
fig.savefig(IMAGES / "07-variograma-pm25.png", dpi=192, bbox_inches="tight")
plt.show()

mejor_pm = min(diag_pm, key=lambda d: d["rmse"])
v_pm.model = mejor_pm["modelo"]
print(f"Mejor modelo PM2.5: {mejor_pm['modelo']} (RMSE={mejor_pm['rmse']:.2f})")

# %% Kriging ordinario PM2.5 sobre grilla de Santiago
ok_pm = v_pm.to_gs_krige(unbiased=True)
print(f"Modelo PM2.5: {ok_pm.model.name} var={ok_pm.model.var:.1f} "
      f"len_scale={ok_pm.model.len_scale:.0f} m nugget={ok_pm.model.nugget:.1f}")
ok_pm.set_pos((xx.ravel(), yy.ravel()))
ok_pm(return_var=True)
pm_field = ok_pm.field.reshape(xx.shape)
pm_var = ok_pm.krige_var.reshape(xx.shape)

pm_idw = interpolar_idw(np.column_stack([sx, sy]), sv, puntos_grid).reshape(xx.shape)

# Recortar predicciones a comunas (fuera no tiene sentido).
from rasterio.features import geometry_mask
import rasterio.transform as rtransform

transform = rtransform.from_bounds(
    xs_grid.min(),
    ys_grid.min(),
    xs_grid.max(),
    ys_grid.max(),
    width=xx.shape[1],
    height=xx.shape[0],
)
mascara_comunas = ~geometry_mask(
    comunas.geometry,
    transform=transform,
    out_shape=xx.shape,
    invert=False,
)
pm_field_m = np.where(mascara_comunas, pm_field, np.nan)
pm_var_m = np.where(mascara_comunas, pm_var, np.nan)
pm_idw_m = np.where(mascara_comunas, pm_idw, np.nan)

# %% Mapa: IDW, kriging y varianza
fig, axes = small_multiples_from_geodataframe(
    comunas, n_variables=3, col_wrap=3, height=6
)
for ax in axes:
    comunas.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.6)

vmin, vmax = np.nanmin([pm_idw_m, pm_field_m]), np.nanmax([pm_idw_m, pm_field_m])

im0 = axes[0].imshow(
    pm_idw_m, extent=extent_grid, origin="upper", cmap="magma_r",
    vmin=vmin, vmax=vmax, aspect="auto",
)
axes[0].scatter(sx, sy, c=sv, cmap="magma_r", vmin=vmin, vmax=vmax,
                s=80, edgecolor="black", linewidth=0.8)
axes[0].set_title("IDW (k=8, p=2)")
colorbar_horizontal(fig, im0, axes[0], r"PM2.5 ($\mu$g/m$^3$)")

im1 = axes[1].imshow(
    pm_field_m, extent=extent_grid, origin="upper", cmap="magma_r",
    vmin=vmin, vmax=vmax, aspect="auto",
)
axes[1].scatter(sx, sy, c=sv, cmap="magma_r", vmin=vmin, vmax=vmax,
                s=80, edgecolor="black", linewidth=0.8)
axes[1].set_title(f"Kriging ordinario ({mejor_pm['modelo']})")
colorbar_horizontal(fig, im1, axes[1], r"PM2.5 ($\mu$g/m$^3$)")

im2 = axes[2].imshow(pm_var_m, extent=extent_grid, origin="upper",
                    cmap="cividis", aspect="auto")
axes[2].scatter(sx, sy, c="white", s=40, edgecolor="black", linewidth=0.8)
axes[2].set_title("Varianza del kriging")
colorbar_horizontal(fig, im2, axes[2], r"Varianza del kriging ($\sigma^2$)")

fig.suptitle(f"PM2.5 en Santiago el {DIA_CRITICO} (red SINCA, 10 estaciones)")
fig.savefig(IMAGES / "07-pm25-comparacion.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - El mapa de IDW se ve más "puntudo" cerca de cada estación porque
#   los pesos 1/d^2 dominan localmente.
# - El kriging suaviza más y respeta la estructura del variograma.
# - La varianza del kriging crece hacia el sur y este (Pirque,
#   Buin, San José de Maipo): allí no hay estaciones cerca y la
#   predicción es poco confiable. La política ambiental no debería
#   tomar decisiones a esa escala con esta red.

