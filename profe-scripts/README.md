# profe-scripts

Scripts de preparación de los datasets del curso. Se corren desde la raíz del
repositorio con `uv run python profe-scripts/<script>.py`. Cada uno lee fuentes
crudas o pesadas, las reduce a un dataset manejable y lo publica.

La configuración compartida vive en `config.py` (destino de publicación, helpers
y rutas a fuentes crudas). Los valores específicos de cada máquina no se
versionan: `config.py` los lee de `config_local.py` (ignorado por git) o de
variables de entorno. Para crear el archivo local, copiar `config_local.example.py`
a `config_local.py` y completar los valores.

## Parámetros de config_local.py

Todos tienen default `None` en `config.py`. La precedencia es variable de
entorno sobre `config_local.py` sobre el default. Definir en `config_local.py`
solo constantes en MAYÚSCULAS.

| Parámetro | Variable de entorno | Para qué sirve | Lo usan |
|---|---|---|---|
| `DESTINO_SCP` | `GDS_DESTINO_SCP` | Destino scp del servidor de publicación (`usuario@host:~/ruta/`). Sin esto, los scripts empaquetan el `.tgz` pero no pueden subirlo. | Todos los que publican |
| `URL_BASE` | `GDS_URL_BASE` | URL pública base desde donde se descargan los datasets. Solo se usa para imprimir el enlace tras subir. | Todos los que publican |
| `EBD_DIR` | `GDS_EBD_DIR` | Carpeta con el eBird Basic Dataset descomprimido. | `05-ebird-dataset.py` |
| `SOSAFE_RAW` | `GDS_SOSAFE_RAW` | Carpeta con los reportes SOSAFE crudos (JSON diarios). | `05-sosafe-dataset.py`, `11-clustering-dataset.py` |
| `SOSAFE_H3_GRID` | `GDS_SOSAFE_H3_GRID` | Parquet del grid H3-8 con perfil censal. | `05-sosafe-dataset.py` |
| `MINVU_ZIP` | `GDS_MINVU_ZIP` | Zip del shapefile nacional de ciclovías del MINVU. | `09-redes-santiago.py` |
| `LEGADO_ASIGNACION` | `GDS_LEGADO_ASIGNACION` | Carpeta con asignaciones censales precomputadas a reusar (opcional; si falta, se regeneran). | `08-asignacion-rm.py` |
| `FOURSQUARE_DIR` | `GDS_FOURSQUARE_DIR` | Carpeta del dataset global de check-ins de Foursquare (Yang et al., WWW'19), con subcarpetas `CHECKINS/` y `POIS/`. | `14-foursquare-dataset.py` |

## Subida al servidor

La subida no ocurre por defecto: los scripts empaquetan el `.tgz` y muestran el
comando manual. Para subir en la misma corrida, exportar `GDS_SUBIR=1` (requiere
`DESTINO_SCP` configurado y la ssh-key del servidor).

Cada script valida las rutas que necesita al arrancar: si falta una, falla con un
mensaje que indica qué parámetro o variable de entorno configurar.
