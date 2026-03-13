# %%
from gdsutils.general import descargar_datos

descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-cartografia.tgz")

# %%
descargar_datos("https://dcc.uchile.cl/~egraells/gds-data/censo2024-microdatos.tgz")
# %%
import dask.dataframe as dd
import geopandas as gpd
import seaborn as sns
from chiricoca.config import setup_style

setup_style(dpi=96)

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from chiricoca.base.weights import normalize_rows
from chiricoca.geo.figures import small_multiples_from_geodataframe
from chiricoca.maps import choropleth_map
from chiricoca.tables import marimekko

from gdsutils.censo2024 import reassign

# %%
df = dd.read_parquet(
    "data/censo2024-microdatos/personas",
    filters=[[("region", "==", 13)]],
)

# %%
carto = gpd.read_parquet(
    "data/censo2024-cartografia/Cartografia_censo2024_Pais_Comunal.parquet"
)

# %%
carto.plot()

# %%
carto.head()

# %%
manzanas_rm = (
    gpd.read_parquet(
        "data/censo2024-cartografia/Cartografia_censo2024_Pais_Manzanas.parquet",
        filters=[("COD_REGION", "=", 13)],
    )
    .assign(
        geometry=lambda x: x.clip_by_rect(
            *[-70.88006218, -33.67612715, -70.43015094, -33.31069169]
        ),
        MANZENT=lambda x: x["MANZENT"].astype(int),
    )
    .set_geometry("geometry")
    .pipe(lambda x: x[~x.is_empty])
    .drop("SHAPE", axis=1)
)


# %%
color = sns.color_palette("plasma", n_colors=1)[0]

# %%
fig, axes = small_multiples_from_geodataframe(manzanas_rm, 2)

carto.plot(ax=axes[0], edgecolor="white", facecolor=color, linewidth=0.5)
axes[0].set_title("Comunas (datos individuales)")
manzanas_rm.plot(ax=axes[1], edgecolor="none", facecolor=color)
axes[1].set_title("Manzanas (datos agregados)")

# %%
manzanas_rm.columns

# %%
df.columns

# %%
modo_transporte = (
    df.groupby(["comuna", "p45_medio_transporte"])
    .size()
    .compute()
    .rename("n")
    .reset_index()
    .pipe(lambda x: reassign(x, "p45_medio_transporte"))
    .dropna()
    .set_index(["comuna", "p45_medio_transporte"])["n"]
    .unstack(fill_value=0)
    .pipe(normalize_rows)
)

modo_transporte.head()

# %%
print(list(manzanas_rm.columns))

# %%
fig, axes = small_multiples_from_geodataframe(manzanas_rm, 5, col_wrap=3)

map_args = dict(
    k=5,
    binning="fisher_jenks",
    edgecolor="none",
    palette="inferno",
    alpha=0.9,
    cbar_args=dict(
        width="4%",
        height="50%",
        orientation="vertical",
        location="upper left",
        label_size="large",
        title_size="large",
        bbox_to_anchor=(0.01, 0.05, 0.98, 0.9),
    ),
)

columns = [
    "Auto particular",
    "Bicicleta (incluye scooter)",
    "Caminando",
    "Motocicleta",
    "Transporte público (bus, micro, metro, tren, taxi, colectivo)",
]

for ax, col in zip(axes.flatten(), columns):
    ax.set_title(col, loc="left")

    choropleth_map(
        carto[["CUT", "SHAPE"]].set_index("CUT").join(modo_transporte, how="inner"),
        col,
        ax=ax,
        **map_args,
    )


# %%
fig, axes = small_multiples_from_geodataframe(manzanas_rm, 5, col_wrap=3)

map_args = dict(
    k=5,
    binning="fisher_jenks",
    edgecolor="none",
    palette="inferno",
    alpha=0.9,
    cbar_args=dict(
        width="4%",
        height="50%",
        orientation="vertical",
        location="upper left",
        label_size="large",
        title_size="large",
        bbox_to_anchor=(0.01, 0.05, 0.98, 0.9),
    ),
)

column_names = [
    "Auto particular",
    "Bicicleta (incluye scooter)",
    "Caminando",
    "Motocicleta",
    "Transporte público (bus, micro, metro, tren, taxi, colectivo)",
]

columns = [
    "n_transporte_auto",
    "n_transporte_bicicleta",
    "n_transporte_camina",
    "n_transporte_motocicleta",
    "n_transporte_publico",
]

manzanas_rm_transporte = manzanas_rm[manzanas_rm["n_per"] > 10].copy()
manzanas_rm_transporte = gpd.GeoDataFrame(
    manzanas_rm_transporte[columns].pipe(normalize_rows),
    geometry=manzanas_rm_transporte["geometry"],
    crs=manzanas_rm_transporte.crs,
).dropna()
manzanas_rm_transporte = manzanas_rm_transporte[
    manzanas_rm_transporte.geom_type.isin(["Polygon", "MultiPolygon"])
]

for ax, col_name, col in zip(axes.flatten(), column_names, columns):
    ax.set_title(col_name, loc="left")

    choropleth_map(
        manzanas_rm_transporte,
        col,
        ax=ax,
        **map_args,
    )

# %%
modo_transporte_sexo = (
    df.groupby(["sexo", "comuna", "p45_medio_transporte", "p44_lug_trab"])
    .size()
    .compute()
    .rename("n")
    .reset_index()
    .pipe(lambda x: reassign(x, "p45_medio_transporte"))
    .pipe(lambda x: reassign(x, "sexo"))
    .pipe(lambda x: reassign(x, "p44_lug_trab"))
)

modo_transporte_sexo

# %%
fig, ax = plt.subplots(figsize=(7, 4))

ax = marimekko(
    modo_transporte_sexo.groupby(["sexo", "p45_medio_transporte"])["n"]
    .sum()
    .unstack()
    .drop(["No respuesta", "Caballo, lancha o bote", "Otro"], axis=1),
    sort_items=True,
    sort_categories=False,
    annotate=True,
    xlabel_levels=True,
    legend_args="right",
    ax=ax,
    width_transform=None,
)

# %%
particion = modo_transporte_sexo.groupby("p45_medio_transporte")["n"].sum()
particion / particion.sum()

# %%
fig, ax = plt.subplots(figsize=(7, 4))

ax = marimekko(
    modo_transporte_sexo.groupby(["p45_medio_transporte", "p44_lug_trab"])["n"]
    .sum()
    .unstack(fill_value=0)
    .drop(["No respuesta", "Caballo, lancha o bote", "Otro"], axis=0)
    .drop("En otro país", axis=1)
    .rename(
        {
            "Transporte público (bus, micro, metro, tren, taxi, colectivo)": "Transporte público"
        }
    ),
    annotate=True,
    ax=ax,
    sort_items=True,
    sort_categories=True,
    xlabel_levels=True,
    legend_args="right",
)
