# %%
"""Clasificación de riqueza de aves con XGBoost y SHAP en H3-8 (Santiago).

Target binario: `riqueza_alta = sqrt_riqueza > mediana`. Predictores:

- `ndvi`: vegetación (proxy de hábitat).
- `log_luminosidad`: urbanización nocturna.
- `log_densidad`: densidad poblacional comunal.
- `x_utm`, `y_utm`: centroide del hex en UTM 19S, como features espaciales
  explícitas. XGBoost puede aprender interacciones no lineales con la
  ubicación, en analogía a lo que hace GWR con coeficientes locales.

Para corregir el sesgo de muestreo de eBird (más observaciones donde hay
más acceso humano y donde se concentra la demografía de observadores),
ponderamos cada hex con `sample_weight` inverso al log de la población
flotante diurna estimada desde la EOD 2012. Hexes muy concurridos pesan
menos y hexes poco transitados pesan más.

Comparamos tres modelos en validación cruzada estratificada de 5-folds:

1. Base: solo predictores ambientales, sin pesos.
2. + Coordenadas: agrega x, y como features.
3. + Coordenadas + sample_weight: agrega corrección de sesgo.

Después interpretamos el modelo 3 con SHAP: importancia global,
distribución de impactos por feature, y mapas de SHAP por hex que
muestran dónde cada variable empuja la predicción.
"""

# %% Descargas
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/ebird-santiago-2024.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/eod2012.tgz")

# %% Imports
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from chiricoca.config import setup_style
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.maps import choropleth_map
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from gdsutils import eodscl as eod
from gdsutils.geo import clip_geodataframe

setup_style(dpi=96)

# %% Configuración
RUTA_DATOS = Path("data") / "ebird-santiago-2024" / "h3-8-2024.parquet"
RUTA_CARTO = (
    Path("data")
    / "censo2024-cartografia"
    / "Cartografia_censo2024_Pais_Comunal.parquet"
)
EOD_PATH = Path("data") / "eod2012" / "EOD_STGO"
BBOX = (-70.85, -33.65, -70.45, -33.30)
IMAGES = Path("images")
IMAGES.mkdir(exist_ok=True)
SEMILLA = 42

# %%
# ========================================
# PARTE 1.Carga del dataset
# ========================================
print(f"Cargando {RUTA_DATOS}")
hex_geo = gpd.read_parquet(RUTA_DATOS)
print(f"  Hexágonos: {len(hex_geo)}")
print(f"  CRS: {hex_geo.crs}")

comunas = gpd.read_parquet(RUTA_CARTO, filters=[("COD_REGION", "=", 13)]).to_crs(
    hex_geo.crs
)
comunas = clip_geodataframe(comunas, BBOX)
print(f"  Comunas de contexto: {len(comunas)}")

# %%
# ========================================
# PARTE 2.Población flotante diurna desde EOD 2012
# ========================================
# La población flotante mide cuántas personas están presentes en promedio
# en cada hex durante la franja diurna 7-19. Cada destino aporta en
# proporción al tiempo que la persona permanece allí (HoraIni del próximo
# viaje menos HoraFin actual). Excluimos viajes con destino residencial
# (volver a casa, visitar a alguien) porque no contribuyen a la población
# flotante "pública" del territorio.
print("Cargando EOD 2012")
viajes = eod.read_trips(EOD_PATH)
personas = eod.read_people(EOD_PATH)
viajes = viajes.merge(
    personas[["Persona", "FactorPersona"]], on="Persona", how="left"
)
viajes["Peso"] = viajes["FactorExpansion"] * viajes["FactorPersona"]
print(f"  Viajes válidos: {len(viajes):,}")

pob_flot = eod.calcular_poblacion_flotante(viajes, hex_geo, "h3_cell_id")
hex_geo["pob_flotante"] = hex_geo["h3_cell_id"].map(pob_flot).fillna(0)
print(
    f"  Hexes con flotante > 0: {(hex_geo['pob_flotante'] > 0).sum()} "
    f"de {len(hex_geo)}"
)
print(hex_geo["pob_flotante"].describe().round(1))

# %%
# ========================================
# PARTE 3.Variables y target binario
# ========================================
# Las transformaciones logarítmicas estabilizan la varianza de variables
# muy asimétricas (densidad, luminosidad, población flotante). Las
# coordenadas en UTM 19S (metros) entran como features para que XGBoost
# pueda partir el espacio en regiones según la ubicación.
hex_geo["log_densidad"] = np.log(hex_geo["densidad"])
hex_geo["log_luminosidad"] = np.log1p(hex_geo["luminosidad"].clip(lower=0))
hex_geo["log_pob_flotante"] = np.log1p(hex_geo["pob_flotante"])

centroides_utm = hex_geo.to_crs("EPSG:32719").geometry.centroid
hex_geo["x_utm"] = centroides_utm.x.values
hex_geo["y_utm"] = centroides_utm.y.values

mediana = hex_geo["sqrt_riqueza"].median()
hex_geo["riqueza_alta"] = (hex_geo["sqrt_riqueza"] > mediana).astype(int)
print(f"Mediana de sqrt_riqueza: {mediana:.3f}")
print(hex_geo["riqueza_alta"].value_counts().rename("hexes").to_string())

# %%
# ========================================
# PARTE 4.Sample weights para corregir sesgo de muestreo
# ========================================
# eBird sobre-muestrea hexes con alta población flotante (acceso fácil,
# observadores presentes). Usamos peso inverso a log(2 + pob_flotante)
# para reducir su influencia y elevar la de hexes poco transitados.
# Forma 1/log(2+x): positiva, acotada (no diverge en x=0), decreciente.
# Normalizamos a media 1 para que la escala sea comparable al caso
# uniforme (todos los pesos iguales a 1).
peso = 1.0 / np.log1p(hex_geo["pob_flotante"] + 1.0)
peso = peso * len(hex_geo) / peso.sum()
hex_geo["sample_weight"] = peso.values

print("Distribución de sample_weight:")
print(hex_geo["sample_weight"].describe().round(3))

# %%
# ========================================
# PARTE 5.Mapas exploratorios
# ========================================
# Antes de modelar, ver qué pasa con las variables clave. Comparar el
# mapa de target con el de población flotante revela el sesgo: si las
# zonas de "riqueza alta" coinciden con alta población flotante, el
# modelo sin pesos puede estar aprendiendo presencia humana en lugar de
# biodiversidad.
config_mapas = [
    ("ndvi", "NDVI mediano (Sentinel-2)", "YlGn"),
    ("log_luminosidad", "log(1 + luminosidad)", "cividis"),
    ("log_densidad", r"log(densidad hab/km$^2$)", "plasma"),
    ("log_pob_flotante", "log(1 + pob. flotante diurna)", "magma"),
    ("sample_weight", "Sample weight (1/log de pob. flotante)", "viridis"),
]
fig, axes = small_multiples_from_geodataframe(
    comunas,
    len(config_mapas) + 1,
    col_wrap=3,
    height=5,
)

# Mapa del target binario (categórico, dos clases)
hex_geo.plot(
    column="riqueza_alta",
    ax=axes[0],
    categorical=True,
    cmap="RdYlBu_r",
    edgecolor="none",
    legend=True,
    legend_kwds={"loc": "lower left", "title": "riqueza_alta"},
)
comunas.boundary.plot(ax=axes[0], color="black", linewidth=0.4, alpha=0.4)
axes[0].set_title("Target: riqueza alta vs baja")

for ax, (col, titulo, paleta) in zip(axes[1:], config_mapas):
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
fig.suptitle("Variables y target del clasificador de riqueza de aves")
fig.savefig(IMAGES / "06-variables.png", dpi=192, bbox_inches="tight")
plt.show()

# %%
# ========================================
# PARTE 6.Tres modelos XGBoost con validación cruzada
# ========================================
# Cada variante aísla una contribución:
# - Base: predictores ambientales puros (NDVI, luminosidad, densidad).
# - + Coordenadas: el modelo puede usar la ubicación como variable.
# - + Coordenadas + sample_weight: corrige el sesgo de eBird.
#
# Validación cruzada estratificada de 5-folds para tener estimaciones
# honestas de accuracy y AUC con un dataset chico (alrededor de 400
# hexes).

FEATURES_BASE = ["ndvi", "log_luminosidad", "log_densidad"]
FEATURES_ESP = FEATURES_BASE + ["x_utm", "y_utm"]

y = hex_geo["riqueza_alta"].astype(int).values
X_base = hex_geo[FEATURES_BASE].values
X_esp = hex_geo[FEATURES_ESP].values
w = hex_geo["sample_weight"].values


def hacer_modelo():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=SEMILLA,
        eval_metric="logloss",
    )


def cv_predict_proba(X, y, sample_weight=None, n_splits=5, seed=SEMILLA):
    """Predicción out-of-fold con sample_weight opcional."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        m = hacer_modelo()
        if sample_weight is not None:
            m.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
        else:
            m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba


print("Validación cruzada estratificada (5 folds)")
proba_base = cv_predict_proba(X_base, y)
proba_esp = cv_predict_proba(X_esp, y)
proba_esp_w = cv_predict_proba(X_esp, y, sample_weight=w)

resultados = pd.DataFrame(
    {
        "Modelo": [
            "Base (ambiental)",
            "+ Coordenadas",
            "+ Coords + sample_weight",
        ],
        "Accuracy": [
            accuracy_score(y, (p > 0.5).astype(int))
            for p in (proba_base, proba_esp, proba_esp_w)
        ],
        "AUC": [
            roc_auc_score(y, p)
            for p in (proba_base, proba_esp, proba_esp_w)
        ],
    }
)
print(resultados.round(3).to_string(index=False))

# Cómo leer la tabla:
# - El salto de "Base" a "+ Coordenadas" mide cuánta estructura espacial
#   no capturan los predictores ambientales. Si AUC sube fuerte, las
#   coordenadas aportan información residual (ubicación importa más allá
#   del NDVI o la urbanización).
# - El salto a "+ sample_weight" no necesariamente sube accuracy: la
#   ponderación inversa puede bajar el desempeño en hexes urbanos
#   sobrerrepresentados, pero hace que el modelo aprenda patrones más
#   generalizables a hexes con poca cobertura eBird.

# %%
# ========================================
# PARTE 7.Modelo final y SHAP
# ========================================
# Entrenamos el modelo 3 (con coordenadas y sample_weight) sobre todos
# los datos para tener un explicador estable. SHAP para árboles es
# exacto: cada predicción se descompone en aporte por feature, sumando
# log-odds.
modelo_final = hacer_modelo()
modelo_final.fit(X_esp, y, sample_weight=w)

explainer = shap.TreeExplainer(modelo_final)
shap_values = explainer.shap_values(X_esp)
print(f"shap_values shape: {shap_values.shape}")

# %% Importancia global (mean |SHAP|)
orden = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
nombres_ord = [FEATURES_ESP[i] for i in orden]
importancia = np.abs(shap_values).mean(axis=0)[orden]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(nombres_ord, importancia, color="#3b8ea5")
ax.invert_yaxis()
ax.set_xlabel("media |SHAP|")
ax.set_title("Importancia global por feature (modelo coords + sample_weight)")
fig.savefig(IMAGES / "06-shap-importancia.png", dpi=192, bbox_inches="tight")
plt.show()

# %% Strip plot manual del SHAP: cada hex es un punto, y = feature,
# x = SHAP, color = valor de la feature normalizado. Equivalente al
# beeswarm de shap pero sin chocar con el layout engine de matplotlib.
fig, ax = plt.subplots(figsize=(8, 0.6 * len(FEATURES_ESP) + 1.5))
for i_pos, i_feat in enumerate(orden):
    valores = X_esp[:, i_feat]
    if valores.max() > valores.min():
        color_norm = (valores - valores.min()) / (valores.max() - valores.min())
    else:
        color_norm = np.zeros_like(valores)
    jitter = (np.random.RandomState(SEMILLA + i_pos).uniform(-0.3, 0.3, len(valores)))
    ax.scatter(
        shap_values[:, i_feat],
        np.full(len(valores), i_pos) + jitter,
        c=color_norm,
        cmap="RdBu_r",
        s=14,
        alpha=0.7,
        edgecolor="none",
    )
ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
ax.set_yticks(range(len(orden)))
ax.set_yticklabels([FEATURES_ESP[i] for i in orden])
ax.invert_yaxis()
ax.set_xlabel("SHAP (impacto en log-odds)")
ax.set_title("Distribución de impactos por feature (color: valor normalizado)")
fig.savefig(IMAGES / "06-shap-summary.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - El bar plot ordena features por aporte promedio absoluto al log-odds.
#   x_utm e y_utm arriba significan que la ubicación es importante incluso
#   con NDVI y luminosidad presentes.
# - El beeswarm muestra cada hex como un punto: posición horizontal es
#   el SHAP, color es el valor de la feature. Si NDVI alto (rojo) está a
#   la derecha, NDVI alto sube la probabilidad de riqueza alta.

# %% Dependence plots: relación SHAP vs valor de feature
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
features_dep = ["ndvi", "log_luminosidad", "log_densidad", "x_utm"]
for ax, feat in zip(axes.flat, features_dep):
    idx = FEATURES_ESP.index(feat)
    sc = ax.scatter(
        X_esp[:, idx],
        shap_values[:, idx],
        c=X_esp[:, FEATURES_ESP.index("ndvi")],
        cmap="YlGn",
        s=18,
        alpha=0.8,
        edgecolor="none",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(feat)
    ax.set_ylabel(f"SHAP({feat})")
    ax.set_title(f"Dependencia parcial: {feat}")
    plt.colorbar(sc, ax=ax, label="ndvi (color)")
fig.suptitle("Cómo cada feature empuja el log-odds de riqueza alta")
fig.savefig(IMAGES / "06-shap-dependencia.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer los dependence plots:
# - Punto sobre la línea cero -> esa feature en ese hex no aporta a la
#   predicción. Por encima -> sube probabilidad de riqueza alta. Por
#   debajo -> la baja.
# - Patrón monótono creciente en NDVI: a más vegetación, más
#   probabilidad. Quiebres o curvas indican umbrales.
# - El color (NDVI) revela interacciones: si los puntos altos en SHAP
#   son sistemáticamente verdes, esa feature aporta más cuando hay
#   vegetación. Eso es información que GWR no explicita igual.

# %%
# ========================================
# PARTE 8.Mapas de SHAP por feature
# ========================================
# Cada hex tiene un SHAP value por feature. Mapearlos muestra dónde el
# modelo se apoya en cada variable. El mapa de "SHAP espacial" suma los
# aportes de x e y, indicando qué tanto la ubicación pura explica la
# predicción en cada hex (más allá de los predictores ambientales).
hex_geo["shap_ndvi"] = shap_values[:, FEATURES_ESP.index("ndvi")]
hex_geo["shap_luminosidad"] = shap_values[:, FEATURES_ESP.index("log_luminosidad")]
hex_geo["shap_densidad"] = shap_values[:, FEATURES_ESP.index("log_densidad")]
hex_geo["shap_espacial"] = (
    shap_values[:, FEATURES_ESP.index("x_utm")]
    + shap_values[:, FEATURES_ESP.index("y_utm")]
)

mapas_shap = [
    ("shap_ndvi", "SHAP NDVI"),
    ("shap_luminosidad", "SHAP log(1+luminosidad)"),
    ("shap_densidad", "SHAP log(densidad)"),
    ("shap_espacial", "SHAP coordenadas (x + y)"),
]
fig, axes = small_multiples_from_geodataframe(
    comunas,
    len(mapas_shap),
    col_wrap=2,
    height=5,
)
for ax, (col, titulo) in zip(axes, mapas_shap):
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
    ax.set_title(titulo)
fig.suptitle("Aporte espacial de cada feature al log-odds de riqueza alta")
fig.savefig(IMAGES / "06-shap-mapas.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer los mapas:
# - Rojo: la feature en ese hex empuja hacia "riqueza alta". Azul: hacia
#   "riqueza baja". El color refleja signo y magnitud del SHAP.
# - SHAP NDVI rojo en piedemonte y parques urbanos: la vegetación es la
#   evidencia que activa la predicción de riqueza alta ahí.
# - SHAP coordenadas rojo en zonas específicas: hay un patrón geográfico
#   residual que ni NDVI ni urbanización capturan. Esos son candidatos
#   para explicar con variables nuevas (humedales, fragmentación,
#   altitud).

# %%
# ========================================
# PARTE 9.Comparación de probabilidades predichas
# ========================================
# Mapear las probabilidades out-of-fold del modelo 3 muestra hexes
# donde la predicción es confiable (cerca de 0 o 1) versus inciertos
# (cerca de 0.5). Útil para identificar zonas donde más datos o
# variables nuevas ayudarían.
hex_geo["proba_riqueza_alta"] = proba_esp_w
hex_geo["error_oof"] = (proba_esp_w > 0.5).astype(int) - y

fig, axes = small_multiples_from_geodataframe(comunas, 2, col_wrap=2, height=5)

choropleth_map(
    hex_geo,
    "proba_riqueza_alta",
    k=5,
    binning="quantiles",
    palette="RdYlBu_r",
    edgecolor="none",
    linewidth=0,
    ax=axes[0],
)
comunas.boundary.plot(ax=axes[0], color="black", linewidth=0.4, alpha=0.4)
axes[0].set_title("Probabilidad predicha (out-of-fold)")

colores_error = {-1: "#3b8ea5", 0: "#cccccc", 1: "#e76f51"}
hex_geo.plot(
    column="error_oof",
    ax=axes[1],
    categorical=True,
    color=hex_geo["error_oof"].map(colores_error),
    edgecolor="none",
    legend=False,
)
comunas.boundary.plot(ax=axes[1], color="black", linewidth=0.4, alpha=0.4)
n_fn = int((hex_geo["error_oof"] == -1).sum())
n_ok = int((hex_geo["error_oof"] == 0).sum())
n_fp = int((hex_geo["error_oof"] == 1).sum())
axes[1].set_title(
    f"Error de clasificación: FN={n_fn} (azul), OK={n_ok} (gris), FP={n_fp} (naranja)"
)

fig.suptitle("Predicciones del modelo final")
fig.savefig(IMAGES / "06-predicciones.png", dpi=192, bbox_inches="tight")
plt.show()

# Cómo leer:
# - Hexes con probabilidad cerca de 0.5 son inciertos: el modelo ve
#   evidencia mixta. Cerca de los bordes urbano-periurbano es esperable.
# - El mapa de errores muestra clústeres de FP/FN: si están concentrados
#   espacialmente, hay variables que faltan. Si son aleatorios, el modelo
#   está cerca del límite con la información disponible.
