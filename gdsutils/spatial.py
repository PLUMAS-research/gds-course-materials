"""Wrappers en español sobre chiricoca.maps.lisa."""

from chiricoca.maps.lisa import LISA_COLORS, compute_lisa, lisa_map

# Re-exportar con etiqueta "No significativo" para uso en castellano
COLORES_LISA = {
    k if k != "Not significant" else "No significativo": v
    for k, v in LISA_COLORS.items()
}


def calcular_lisa(geodf, variable, w=None, alpha=0.05):
    """Wrapper de compute_lisa con etiqueta en español para no significativos."""
    return compute_lisa(geodf, variable, w=w, alpha=alpha, ns_label="No significativo")


def graficar_lisa(geodf_lisa, ax, titulo=None, contexto=None, colores=None):
    """Wrapper de lisa_map con parámetros en español."""
    if colores is None:
        colores = COLORES_LISA
    return lisa_map(
        geodf_lisa, ax, title=titulo, context=contexto,
        colors=colores, ns_label="No significativo",
    )
