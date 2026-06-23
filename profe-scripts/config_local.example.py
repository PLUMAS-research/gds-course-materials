"""Plantilla de configuración local de los profe-scripts.

Copiar a `config_local.py` (gitignored) y completar con los valores de tu
máquina. Solo definir constantes en MAYÚSCULAS; `config.py` las usa para
sobreescribir sus defaults (que son `None`). Alternativa: exportar las
variables de entorno `GDS_*` equivalentes. Precedencia: env var > config_local
> default.
"""

from pathlib import Path

# --- Publicación ---
# Destino scp y URL pública del servidor del curso. Sin esto, los scripts
# empaquetan el .tgz pero no pueden subirlo.
# DESTINO_SCP = "usuario@host:~/ruta/publicacion/"
# URL_BASE = "https://host/ruta/publicacion"

# --- Fuentes crudas ---
# eBird Basic Dataset descomprimido (05-ebird-dataset.py).
# EBD_DIR = Path.home() / "Descargas" / "ebd_CL_smp_relMar-2026"

# Reportes SOSAFE crudos y grid H3-8 censal (05-sosafe-dataset.py,
# 11-clustering-dataset.py).
# SOSAFE_RAW = Path.home() / "repositories" / "<repo-sosafe>" / "data" / "raw"
# SOSAFE_H3_GRID = (
#     Path.home() / "repositories" / "<repo-sosafe>" / "data" / "census_2024"
#     / "microdatos-h3-8.parquet"
# )

# Shapefile de ciclovías MINVU (09-redes-santiago.py).
# MINVU_ZIP = Path.home() / "Descargas" / "CICLOVchile_shp.zip"

# Dataset global de check-ins de Foursquare, Yang et al. WWW'19
# (14-foursquare-dataset.py). Carpeta con las subcarpetas CHECKINS/ y POIS/.
# FOURSQUARE_DIR = Path.home() / "datos" / "4sq_2019"

# Asignaciones censales precomputadas a importar (08-asignacion-rm.py).
# LEGADO_ASIGNACION = Path.home() / "repositories" / "<repo-censo>" / "results"
