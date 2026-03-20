import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def bubble_size_legend(
    ax,
    scale,
    values,
    n=4,
    title="",
    loc="lower left",
    inset_size=(1.2, 1.5),
    borderpad=1,
    fontsize=7,
    color="grey",
):
    """Leyenda de tamaño para bubble_map con círculos anidados que comparten base.

    El inset se calibra para que 1 unidad de datos = 1 point,
    así los Circle patches coinciden con el markersize de bubble_map (en points²).

    Parameters
    ----------
    ax : matplotlib Axes
        Axes principal donde se dibuja el mapa.
    scale : float
        Misma escala usada en bubble_map.
    values : array-like
        Valores de la variable de tamaño (se usa para calcular el rango).
    n : int
        Cantidad de círculos de referencia.
    title : str
        Título de la leyenda.
    loc : str
        Posición del inset (e.g., "lower left", "lower right").
    inset_size : tuple
        (ancho, alto) del inset en pulgadas.
    borderpad : float
        Padding del inset respecto al axes principal.
    fontsize : float
        Tamaño de fuente para las etiquetas.
    color : str
        Color de los círculos y texto.

    Returns
    -------
    ax_inset : matplotlib Axes
        El axes del inset creado.
    """
    v_max = np.max(values)
    size_refs = sorted(np.linspace(0, v_max, n + 1).astype(int)[1:], reverse=True)

    inset_w, inset_h = inset_size
    ax_inset = inset_axes(ax, width=inset_w, height=inset_h, loc=loc, borderpad=borderpad)
    ax_inset.set_facecolor("white")
    ax_inset.set_axis_off()

    # 72 points por pulgada → data range = dimensión en points
    w_pts = inset_w * 72
    h_pts = inset_h * 72
    ax_inset.set_xlim(0, w_pts)
    ax_inset.set_ylim(0, h_pts)

    x_center = w_pts * 0.35

    for v in size_refs:
        s = v * scale
        r = np.sqrt(s / np.pi)
        circle = Circle(
            (x_center, r), r, facecolor="none", edgecolor=color, linewidth=0.8
        )
        ax_inset.add_patch(circle)
        ax_inset.plot(
            [x_center, w_pts * 0.65], [2 * r, 2 * r],
            color=color, linewidth=0.5, linestyle="--",
        )
        ax_inset.text(
            w_pts * 0.68, 2 * r, f"{v:,}", va="center", fontsize=fontsize, color=color
        )

    if title:
        ax_inset.set_title(title, fontsize=fontsize, color=color, pad=2)

    return ax_inset


def categorical_color_legend(ax, palette, title="", loc="upper left", markersize=10):
    """Leyenda categórica de colores con círculos de tamaño uniforme.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes donde agregar la leyenda.
    palette : dict
        Mapeo {categoría: color}.
    title : str
        Título de la leyenda.
    loc : str
        Posición de la leyenda.
    markersize : float
        Tamaño de los marcadores en la leyenda.

    Returns
    -------
    legend : matplotlib Legend
        La leyenda creada (ya agregada al axes con add_artist).
    """
    handles = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=color, markersize=markersize,
            markeredgecolor="white", label=cat,
        )
        for cat, color in palette.items()
    ]
    legend = ax.legend(handles=handles, title=title, loc=loc)
    ax.add_artist(legend)
    return legend
