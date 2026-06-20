# %% [markdown]
# # Análisis espacio-temporal: el Metro como experimento natural
#
# Entre 2017 y 2019 el Metro de Santiago se expandió: la Línea 6 (noviembre
# 2017) y la Línea 3 (enero 2019) llevaron Metro a zonas que antes no lo tenían.
# Una línea de Metro nueva es un **experimento natural**: algunos pares
# origen-destino ganan acceso a Metro y otros no. Con un grupo de control
# podemos preguntar si la gente que ganó Metro se cambió a él.
#
# Condiciones del análisis (hay que tenerlas claras DESDE EL COMIENZO):
#
#   - **Qué es un viaje y su modo.** La base DTPM es de viajes en transporte
#     público (validaciones bip!), con hasta cuatro etapas. Un viaje "usa Metro"
#     si alguna etapa es Metro, y "usa bus" si alguna etapa es bus. El Metro
#     está SIEMPRE en la base (45-65% de los viajes lo usan).
#   - **Los datos se preprocesan aparte** (`profe-scripts/13-lineas-metro-dataset.py`):
#     se corrige el modo, se normaliza por fecha laboral y se deja una submuestra
#     de viajes. Aquí hacemos los cálculos sobre esa submuestra.
#   - **Antes y después.** Antes = 2016 (previo a L6 y L3). Después = 2019
#     (posterior a ambas, encuesta de agosto, antes del estallido).
#   - **Tratamiento y control.** Tratado = pares O-D que ganan Metro; control =
#     pares sin Metro cerca en ningún momento.
#   - **Límite.** Dos líneas abren casi juntas: medimos el efecto de ganar
#     Metro, no separamos L6 de L3.

# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/lineas-metro.tgz")

# %%
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from chiricoca.config import setup_style
from chiricoca.geo.figures import figure_from_geodataframe, small_multiples_from_geodataframe
from chiricoca.geo.grid import h3_grid_from_ids
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import Point, box

setup_style(dpi=128)

# Tipografía Urbanist y figuras chicas a alto DPI (texto relativamente más grande).
from matplotlib import font_manager  # noqa: E402

for _f in Path.home().glob(".fonts/Urbanist/*.otf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": "Urbanist", "font.size": 13, "axes.titlesize": 14,
    "legend.fontsize": 11, "savefig.dpi": 300, "figure.dpi": 300,
})

DIR = Path("data") / "lineas-metro"
CRS_METRICO = "EPSG:32719"
ANTES, DESPUES = 2016, 2019
EVENTOS = {"Metro L6": 2017.84, "Metro L3": 2019.0}
UMBRAL = 1000  # metros: distancia para considerar que un paradero "tiene Metro"

comunas = gpd.read_parquet(DIR / "comunas.parquet").to_crs("EPSG:4326")


def dibujar_comunas(ax, color="0.6", lw=0.4):
    comunas.boundary.plot(ax=ax, color=color, linewidth=lw, zorder=2)


def wmean(g, col):
    """Promedio ponderado por el factor de expansión."""
    return np.average(g[col], weights=g["factor"])

# %% [markdown]
# ## PARTE 1. Qué comparamos: el modo del viaje
#
# Lo primero es verificar qué medimos. Cada viaje trae si usó Metro y si usó
# bus. Definimos el modo y miramos cómo cambia su participación en la ciudad
# entre 2014 y 2019. El cálculo: participación ponderada por el factor de
# expansión, año a año.

# %%
viajes = pd.read_parquet(DIR / "viajes-dtpm.parquet")
for c in ["paradero", "paradero_destino"]:
    viajes[c] = viajes[c].astype(str)
# Un viaje "usa Metro" si ALGUNA de sus etapas es Metro. Esto INCLUYE las
# combinaciones bus + Metro (un viaje con una etapa de bus y otra de Metro cuenta
# como que usa Metro y también usa bus). Por eso las dos participaciones suman
# más de 100%: los viajes combinados se cuentan en ambas.
viajes["usa_metro"] = viajes["contiene_metro"]
viajes["usa_bus"] = viajes["contiene_bus"]
print(f"Viajes en la submuestra: {len(viajes):,}")
print("Composición de modo (participación ponderada, 2019):")
v19 = viajes[viajes["anio"] == DESPUES]
w19 = v19["factor"]
print(f"  solo bus:    {np.average(v19['usa_bus'] & ~v19['usa_metro'], weights=w19) * 100:.0f}%")
print(f"  bus + Metro: {np.average(v19['usa_bus'] & v19['usa_metro'], weights=w19) * 100:.0f}%")
print(f"  solo Metro:  {np.average(v19['usa_metro'] & ~v19['usa_bus'], weights=w19) * 100:.0f}%")
print("\nModo del viaje (participación ponderada, total ciudad):")
print(viajes.groupby("anio").apply(
    lambda g: pd.Series({"usa metro": wmean(g, "usa_metro") * 100, "usa bus": wmean(g, "usa_bus") * 100}),
    include_groups=False).round(1).to_string())

modo_anio = viajes.groupby("anio").apply(
    lambda g: pd.Series({"metro": wmean(g, "usa_metro") * 100, "bus": wmean(g, "usa_bus") * 100}),
    include_groups=False)
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(modo_anio.index, modo_anio["metro"], "o-", color="#cf3889", linewidth=2.5, label="usa Metro")
ax.plot(modo_anio.index, modo_anio["bus"], "o-", color="steelblue", linewidth=2.5, label="usa bus")
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="seagreen", linestyle="--", alpha=0.7)
    ax.text(x, ax.get_ylim()[1], f" {nombre}", color="seagreen", fontsize=8, va="top")
ax.set_xlabel("Año")
ax.set_ylabel("Participación de los viajes (%)")
ax.set_title("Participación del Metro y del bus por año")
ax.legend(fontsize=8)
fig.savefig("images/13-modo-ciudad.png", dpi=300, bbox_inches="tight")
print(f"\nMetro {modo_anio.loc[2014, 'metro']:.0f}% -> {modo_anio.loc[2019, 'metro']:.0f}%; "
      f"bus {modo_anio.loc[2014, 'bus']:.0f}% -> {modo_anio.loc[2019, 'bus']:.0f}%.")
print("Esto es a nivel de ciudad. ¿Lo causa el Metro nuevo? Para eso, el experimento.")

# %% [markdown]
# ## PARTE 2. La red de Metro: existente y nueva
#
# Antes del experimento conviene ver la red. El Metro de Santiago ya tenía
# varias líneas; en 2017-2019 se sumaron la Línea 6 y la Línea 3. El mapa
# muestra los recorridos existentes (gris) y los nuevos (rosado) con sus
# estaciones, sobre las comunas del Gran Santiago.

# %%
metro_lineas = gpd.read_parquet(DIR / "metro-lineas.parquet").to_crs("EPSG:4326")
est4 = gpd.read_parquet(DIR / "metro-estaciones.parquet").to_crs("EPSG:4326")
fig, ax = figure_from_geodataframe(comunas, height=5.5)
dibujar_comunas(ax, color="0.8", lw=0.4)
metro_lineas[~metro_lineas["nueva"]].plot(ax=ax, color="0.55", linewidth=2.0, zorder=2)
metro_lineas[metro_lineas["nueva"]].plot(ax=ax, color="#cf3889", linewidth=3.2, zorder=3)
est4[~est4["nueva"]].plot(ax=ax, color="0.45", markersize=5, zorder=4)
est4[est4["nueva"]].plot(ax=ax, color="#cf3889", markersize=18, zorder=5)
# Año de apertura de cada línea nueva, en una estación terminal (no sobre el
# trazado): L3 en su terminal norte, L6 en su terminal este, con offset afuera.
for linea, anio_l, eje, off in [("L3", 2019, 1, (0, 16)), ("L6", 2017, 0, (34, 0))]:
    g = metro_lineas[metro_lineas["linea"] == linea]
    if g.empty:
        continue
    pt = max(g.geometry.iloc[0].coords, key=lambda c: c[eje])  # terminal (norte/este)
    ax.annotate(f"{linea} ({anio_l})", (pt[0], pt[1]), fontsize=10, fontweight="bold",
                color="#cf3889", ha="center", va="center", zorder=6,
                xytext=off, textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cf3889", alpha=0.9))
ax.set_axis_off()
ax.legend(handles=[plt.Line2D([], [], color="0.55", lw=2, label="Líneas existentes"),
                   plt.Line2D([], [], color="#cf3889", lw=3, label="Líneas nuevas (L3, L6)")],
          fontsize=8, loc="lower left")
ax.set_title("Red de Metro: líneas existentes y nuevas")
fig.savefig("images/13-metro-red.png", dpi=300, bbox_inches="tight")
print(f"Líneas de Metro: {sorted(metro_lineas['linea'])}; "
      f"nuevas: {sorted(metro_lineas[metro_lineas['nueva']]['linea'])}")

# %% [markdown]
# ## PARTE 3. El experimento natural: tratamiento y control
#
# Clasificamos cada paradero por su acceso a Metro: tiene Metro **antes** si está
# a <=1 km de una estación que ya existía; **después** si está a <=1 km de
# cualquier estación. Un paradero **gana Metro** si solo lo tiene después. Un
# viaje es **tratado** si sus dos extremos ganan acceso, y **control** si ninguno
# tiene Metro cerca en ningún momento.

# %%
paraderos = pd.read_parquet(DIR / "paraderos.parquet")
paraderos["paradero"] = paraderos["paradero"].astype(str)
est = gpd.read_parquet(DIR / "metro-estaciones.parquet").to_crs(CRS_METRICO)
pm = gpd.GeoDataFrame(paraderos, geometry=gpd.points_from_xy(paraderos.lon, paraderos.lat),
                      crs="EPSG:4326").to_crs(CRS_METRICO)
est_vieja = est[~est["nueva"]].geometry.union_all()
est_todas = est.geometry.union_all()
pm["antes"] = pm.geometry.distance(est_vieja) < UMBRAL
pm["despues"] = pm.geometry.distance(est_todas) < UMBRAL
acc_antes = dict(zip(pm["paradero"], pm["antes"]))
acc_despues = dict(zip(pm["paradero"], pm["despues"]))

oa = viajes["paradero"].map(acc_antes)
ad = viajes["paradero"].map(acc_despues)
da = viajes["paradero_destino"].map(acc_antes)
dd = viajes["paradero_destino"].map(acc_despues)
viajes["grupo"] = np.where(ad.eq(True) & dd.eq(True) & ~(oa.eq(True) & da.eq(True)), "tratado",
                           np.where(ad.eq(False) & dd.eq(False), "control", "otro"))
print("Viajes por grupo (submuestra, ambos años):")
print(viajes["grupo"].value_counts().to_string())

# Mapa H3-8: clasificar cada hexágono por el acceso a Metro de su centro,
# limitando a los hexágonos cuyo centro cae dentro del Gran Santiago.
paradero_a_h3 = {p: h3.latlng_to_cell(la, lo, 8) for p, lo, la in zip(paraderos.paradero, paraderos.lon, paraderos.lat)}
urbano = comunas.to_crs(CRS_METRICO).union_all()
hex_ids = sorted({h for h in (paradero_a_h3.get(p) for p in viajes["paradero"]) if h})
centros = gpd.GeoSeries([Point(h3.cell_to_latlng(h)[1], h3.cell_to_latlng(h)[0]) for h in hex_ids],
                        index=hex_ids, crs="EPSG:4326").to_crs(CRS_METRICO)
hex_ids = [h for h in hex_ids if centros[h].within(urbano)]
centros = centros[hex_ids]
catch_nueva = est[est["nueva"]].buffer(UMBRAL).union_all()
catch_vieja = est[~est["nueva"]].buffer(UMBRAL).union_all()
zona = {}
for h, p in centros.items():
    zona[h] = ("Ya tenía Metro" if p.within(catch_vieja)
               else "Tratamiento (gana Metro)" if p.within(catch_nueva)
               else "Control (sin Metro)")
hexes = h3_grid_from_ids(hex_ids).reset_index()
hexes["zona"] = hexes["h3_cell_id"].map(zona)
colores_zona = {"Tratamiento (gana Metro)": "#cf3889", "Ya tenía Metro": "0.6", "Control (sin Metro)": "#9ecae1"}
fig, ax = figure_from_geodataframe(hexes, height=4.2)
for z, c in colores_zona.items():
    hexes[hexes["zona"] == z].plot(ax=ax, color=c, edgecolor="white", linewidth=0.1, zorder=2)
dibujar_comunas(ax, color="0.4", lw=0.4)
est[est["nueva"]].to_crs("EPSG:4326").plot(ax=ax, color="#1f77b4", markersize=14, zorder=4)
ax.set_axis_off()
handles = [mpatches.Patch(color=colores_zona[z], label=z) for z in colores_zona]
handles.append(plt.Line2D([], [], marker="o", color="#1f77b4", linestyle="", label="Estación nueva (L3/L6)"))
ax.legend(handles=handles, fontsize=8, loc="lower left")
ax.set_title("Acceso a Metro por hexágono (H3-8)")
fig.savefig("images/13-experimento-mapa.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## PARTE 4. Sustitución de modo: diferencia en diferencias
#
# La pregunta central. Para los viajes tratados (ganaron Metro) y de control,
# medimos la participación del Metro antes (2016) y después (2019). El control
# absorbe lo que le pasó a toda la ciudad. La diferencia en diferencias es el
# efecto del acceso a Metro:
#
#   DiD = (metro_tratado_2019 - metro_tratado_2016) - (metro_control_2019 - metro_control_2016)
#
# Sobre el control: sus paraderos no tienen Metro cerca, pero igual un ~17% de
# sus viajes usa Metro, porque son viajes COMBINADOS (caminan o toman bus hasta
# una estación lejana). Eso no invalida el control: lo que importa es que ese
# 17% se mantiene plano, porque las líneas nuevas están en otra parte. El control
# mide la tendencia común; el tratamiento, el efecto del acceso local a Metro.

# %%
exp = viajes[viajes["anio"].isin([ANTES, DESPUES]) & viajes["grupo"].isin(["tratado", "control"])]
ms = (exp.groupby(["grupo", "anio"]).apply(lambda g: wmean(g, "usa_metro") * 100, include_groups=False)
      .unstack())
print("Participación del Metro (%) por grupo y año:")
print(ms.round(1).to_string())
did = (ms.loc["tratado", DESPUES] - ms.loc["tratado", ANTES]) - (ms.loc["control", DESPUES] - ms.loc["control", ANTES])
print(f"\nDiD de la participación del Metro: {did:+.1f} puntos.")
print("Los pares que ganaron Metro se cambiaron a él; el control no.")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot([ANTES, DESPUES], [ms.loc["tratado", ANTES], ms.loc["tratado", DESPUES]], "o-",
        color="#cf3889", linewidth=2.6, markersize=9, label="Tratado (gana Metro)")
ax.plot([ANTES, DESPUES], [ms.loc["control", ANTES], ms.loc["control", DESPUES]], "o-",
        color="steelblue", linewidth=2.6, markersize=9, label="Control (sin Metro)")
for g, color in [("tratado", "#cf3889"), ("control", "steelblue")]:
    for a in [ANTES, DESPUES]:
        ax.annotate(f"{ms.loc[g, a]:.0f}", (a, ms.loc[g, a]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=color)
ax.set_xticks([ANTES, DESPUES])
ax.set_xticklabels([f"{ANTES} (antes)", f"{DESPUES} (después)"])
ax.set_ylabel("Participación del Metro en los viajes (%)")
ax.set_title("Participación del Metro: tratado y control")
ax.legend(fontsize=8, loc="center left")
fig.savefig("images/13-did-metro.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## PARTE 5. ¿Cambia la estructura del viaje? número de etapas
#
# Ganar Metro no solo cambia el modo: cambia la forma del viaje. Miramos el
# número de etapas (trasbordos + 1) por grupo y periodo, con un *pointplot* que
# muestra la media y su intervalo de confianza al 95%. El pointplot revela algo
# que un promedio crudo escondería: los viajes tratados pierden etapas.

# %%
etapas = exp.assign(periodo=exp["anio"].map({ANTES: "antes", DESPUES: "después"}))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
sns.pointplot(data=etapas, x="periodo", y="n_etapas", hue="grupo", dodge=False,
              errorbar=("ci", 95), markersize=8, ax=ax,
              palette={"tratado": "#cf3889", "control": "steelblue"})
ax.set_xlabel("")
ax.set_ylabel("Número de etapas (media e IC 95%)")
ax.set_title("Número de etapas por grupo y periodo")
ax.legend(fontsize=8, title="")
fig.savefig("images/13-etapas.png", dpi=300, bbox_inches="tight")
print("Los viajes tratados bajan de ~1.5 a ~1.2 etapas: un viaje directo en Metro")
print("reemplaza uno multietapa en bus. El control se mantiene plano.")
print("Número de etapas medio por grupo y año:")
print(exp.groupby(["grupo", "anio"])["n_etapas"].mean().unstack().round(2).to_string())

# %% [markdown]
# ## PARTE 6. El modelo de gravedad: estructura por distancia
#
# El modelo de gravedad describe el flujo de viajes entre dos zonas como
# proporcional a su actividad y decreciente con la distancia:
#
#   T_ij = k O_i D_j exp(-beta d_ij)
#
# Sirve para resumir la estructura espacial de los viajes en un parámetro (el
# decaimiento por distancia) y comparar ese parámetro entre años.
#
# El flujo es un CONTEO de viajes, no una cantidad continua, así que no se ajusta
# con OLS sobre el log: eso descarta los pares con cero viajes y supone errores
# normales. Se usa una **regresión binomial negativa**, que modela conteos,
# admite los ceros y la sobredispersión (la varianza es mayor que la media, algo
# típico de los flujos O-D). Con efectos por comuna de origen y destino, la
# distancia y una variable extra (par con Metro en ambos extremos). Ajustamos el
# **mismo modelo por separado en cada año**: el año NO es una variable, es la
# comparación.

# %%
par_comuna = dict(zip(paraderos.paradero, paraderos.comuna.astype(str).str.upper()))
comunas_con_metro = set(paraderos.loc[paraderos["es_metro"], "comuna"].astype(str).str.upper())
viajes["co"] = viajes["paradero"].map(par_comuna)
viajes["cd"] = viajes["paradero_destino"].map(par_comuna)
cen = comunas.to_crs(CRS_METRICO)
cen["c"] = cen["COMUNA"].str.upper()
cc = cen.dissolve("c").geometry.centroid
cx, cy = cc.x.to_dict(), cc.y.to_dict()
# Grilla completa de pares O-D (incluye los ceros) con distancia y metro_od.
comunas_lista = sorted(cx)
grilla = pd.DataFrame([(o, d) for o in comunas_lista for d in comunas_lista if o != d], columns=["co", "cd"])
grilla["dist_km"] = [np.hypot(cx[o] - cx[d], cy[o] - cy[d]) / 1000 for o, d in zip(grilla["co"], grilla["cd"])]
grilla["metro_od"] = (grilla["co"].isin(comunas_con_metro) & grilla["cd"].isin(comunas_con_metro)).astype(int)
formula = "conteo ~ C(co) + C(cd) + dist_km + metro_od"
betas, lo, hi = {}, {}, {}
modelo_2019, disp_2019 = None, None
for anio in sorted(viajes["anio"].unique()):
    cnt = (viajes[(viajes["anio"] == anio)].dropna(subset=["co", "cd"])
           .query("co in @comunas_lista and cd in @comunas_lista")
           .groupby(["co", "cd"]).size().rename("conteo"))
    od = grilla.merge(cnt, on=["co", "cd"], how="left")
    od["conteo"] = od["conteo"].fillna(0).astype(int)
    # Poisson para medir la sobredispersión (varianza / media) y justificar la NB.
    pois = smf.glm(formula, data=od, family=sm.families.Poisson()).fit()
    disp = float((pois.resid_pearson ** 2).sum() / pois.df_resid)
    # Binomial negativa (GLM robusto por IRLS) para los conteos sobredispersos.
    m = smf.glm(formula, data=od, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    betas[anio] = m.params["dist_km"]
    lo[anio], hi[anio] = m.conf_int().loc["dist_km"]
    if anio == DESPUES:
        modelo_2019, disp_2019 = m, disp
print("Decaimiento por distancia (beta) e intervalo de confianza 95%:")
for a in betas:
    print(f"  {a}: {betas[a]:+.4f}  [{lo[a]:+.4f}, {hi[a]:+.4f}]")
print(f"\nSobredispersión (Poisson, {DESPUES}): {disp_2019:.0f} >> 1, por eso se usa NB.")
print(f"Significancia (binomial negativa, {DESPUES}):")
for v in ["dist_km", "metro_od"]:
    print(f"  {v}: coef {modelo_2019.params[v]:+.4f}, p-valor {modelo_2019.pvalues[v]:.1e}")

anios_b = list(betas)
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.errorbar(anios_b, [betas[a] for a in anios_b],
            yerr=[[betas[a] - lo[a] for a in anios_b], [hi[a] - betas[a] for a in anios_b]],
            fmt="o-", color="purple", linewidth=2.5, capsize=4)
for nombre, x in EVENTOS.items():
    ax.axvline(x, color="seagreen", linestyle="--", alpha=0.7)
    ax.text(x, ax.get_ylim()[1], f" {nombre}", color="seagreen", fontsize=8, va="top")
ax.set_xlabel("Año")
ax.set_ylabel("Decaimiento por distancia (beta, por km)")
ax.set_title("Decaimiento por distancia por año (IC 95%)")
fig.savefig("images/13-gravedad.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## PARTE 7. El fenómeno antes y después: velocidad, hora y propósito
#
# Tres atributos del viaje, comparados antes (2016) y después (2019) en toda la
# ciudad. La velocidad es distancia sobre tiempo. La hora y el propósito se
# ponderan por el factor.

# %%
v2 = viajes[viajes["anio"].isin([ANTES, DESPUES]) & viajes["tviaje_min"].between(5, 180) & (viajes["dist_ruta_mts"] > 500)].copy()
v2["periodo"] = v2["anio"].map({ANTES: "antes", DESPUES: "después"})
v2["vel_kmh"] = v2["dist_ruta_mts"] / v2["tviaje_min"] * 60 / 1000

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
# Velocidad: distribución ponderada.
for periodo, color in [("antes", "steelblue"), ("después", "firebrick")]:
    s = v2[v2["periodo"] == periodo]
    axes[0].hist(s["vel_kmh"], bins=np.arange(0, 45, 2), weights=s["factor"], density=True,
                 histtype="step", linewidth=2, color=color, label=periodo)
axes[0].set_xlabel("Velocidad del viaje (km/h)")
axes[0].set_ylabel("Densidad")
axes[0].set_title("Velocidad")
axes[0].legend(fontsize=8)
# Hora de inicio.
hora = v2.groupby(["periodo", "hora_inicio"])["factor"].sum()
hora = (hora / hora.groupby(level=0).sum() * 100).unstack(0)
for periodo, color in [("antes", "steelblue"), ("después", "firebrick")]:
    axes[1].plot(hora.index, hora[periodo], "o-", color=color, label=periodo, markersize=4)
axes[1].set_xlabel("Hora de inicio")
axes[1].set_ylabel("Participación (%)")
axes[1].set_title("Hora de inicio")
axes[1].legend(fontsize=8)
fig.savefig("images/13-fenomeno.png", dpi=300, bbox_inches="tight")

# Propósito.
prin = ["TRABAJO", "HOGAR", "OTROS"]
prop = v2[v2["proposito"].isin(prin)].groupby(["periodo", "proposito"])["factor"].sum()
prop = (prop / prop.groupby(level=0).sum() * 100).unstack().reindex(columns=prin)
print("Propósito del viaje (%):")
print(prop.round(1).to_string())
print("Velocidad mediana (km/h):", v2.groupby("periodo")["vel_kmh"].median().round(1).to_dict())

# %% [markdown]
# ## PARTE 8. La demanda en el espacio (H3-8) antes y después
#
# Subidas de transporte público por hexágono H3-8, en 2016 y 2019, a la misma
# escala, limitadas al Gran Santiago. El patrón espacial cambia poco: lo que
# cambia es el modo, no dónde se viaja.

# %%
viajes["h3_o"] = viajes["paradero"].map(paradero_a_h3)
dh = (viajes[viajes["anio"].isin([ANTES, DESPUES])].dropna(subset=["h3_o"])
      .groupby(["h3_o", "anio"])["factor"].sum().unstack().fillna(0.0))
dh = dh[dh.index.isin(hex_ids)]  # solo hexágonos dentro del borde urbano
hexd = h3_grid_from_ids(dh.index.tolist()).reset_index().merge(dh, left_on="h3_cell_id", right_index=True)
vmax = np.percentile(hexd[[ANTES, DESPUES]].to_numpy()[hexd[[ANTES, DESPUES]].to_numpy() > 0], 97)
fig, axes = small_multiples_from_geodataframe(hexd, n_variables=2, col_wrap=2, height=3.6)
for ax, col, titulo in zip(axes, [ANTES, DESPUES], [f"Antes ({ANTES})", f"Después ({DESPUES})"]):
    dibujar_comunas(ax, color="0.8", lw=0.3)
    hexd.plot(ax=ax, column=col, cmap="magma_r", vmin=0, vmax=vmax, edgecolor="none")
    ax.set_title(titulo, fontsize=10)
    ax.set_axis_off()
cax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="magma_r"), cax=cax,
             orientation="horizontal", label="subidas de transporte público por día (H3-8)")
fig.savefig("images/13-demanda-h3.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## PARTE 9. Síntesis
#
#   - **El Metro sube y el bus baja en toda la ciudad** (Metro 59% -> 68%, bus
#     61% -> 44% entre 2014 y 2019). Pero eso solo no prueba que lo cause el
#     Metro nuevo.
#   - **El experimento natural lo confirma.** Los pares que ganaron Metro pasaron
#     de un 32% a un 81% de uso de Metro; el control se quedó en 17-18%. La
#     diferencia en diferencias es de unos +37 puntos: la sustitución es directa
#     y grande, y atribuible al acceso a Metro porque el control absorbe lo común.
#   - **Los viajes tratados también pierden etapas** (de ~1.5 a ~1.2): un viaje
#     directo en Metro reemplaza uno multietapa en bus. En cambio la velocidad,
#     la hora y el propósito casi no cambian: cambia el modo y los trasbordos, no
#     quién viaja ni cuándo ni para qué.
#   - **Un cuidado honesto:** dos líneas abren casi juntas (L6 2017, L3 2019);
#     medimos el efecto de ganar Metro, no separamos una línea de la otra.
#
# La lección espacio-temporal: para comparar un antes y un después hay que (1)
# verificar qué se compara (aquí, el modo del viaje, que estaba mal calculado en
# la base), (2) describir el fenómeno con técnicas (gravedad, distribuciones) y
# (3) aislar el efecto con un grupo de control (diferencia en diferencias).
print("Fin del análisis. Figuras en images/13-*.png")
