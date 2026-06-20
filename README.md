
# CC5216 -- Ciencia de datos geográficos

El material del curso se ejecuta con `uv`. [Aquí puedes ver sus instrucciones de instalación](https://docs.astral.sh/uv/getting-started/installation/).

Si tienes `uv` instalado, el comando `uv sync` instala todo lo necesario para que ejecutes el código del curso. 

Además, en cada clase es recomendable ejecutar `uv lock --upgrade-package chiricoca` y luego `uv sync`. Esto actualizará el código del repositorio `chiricoca` (base de visualización) en caso de que haya cambiado entre una clase y otra (¡es probable que lo haga!).

La carpeta `gdsutils` contiene funciones utilitarias para trabajar con los datos que veremos en el curso. La carpeta `data` contiene datos que se descargan de manera automática en cada script.

## Clases

* `01-carga-datos-geograficos.py`: cómo cargar datos con `geopandas` en formato Parquet. Ejemplo con datos del Censo 2024 de Chile a nivel de comunas y manzanas.
* `02-geografía-y-espacialidad.py`: diferencias de proporciones según la proyección, mapa de burbujas para Santiago con el censo, red de vecindad comunal.
* `03-definir-proyecto.py`: datos de transporte público (DTPM): descarga, consolidación y análisis exploratorio de viajes 2014–2025 y paraderos del Gran Santiago.
* `04-exploracion-espacial-lisa.py`: análisis exploratorio (incluyendo componente espacial y LISA) de la Encuesta Origen-Destino 2012 de Santiago.
* `05-regresion-ebird.py`: regresión clásica y espacial (OLS, spatial lag, spatial error, GWR) sobre riqueza de aves por hexágono H3-8 en Santiago, con datos de eBird, NDVI de Sentinel-2, luminosidad nocturna VIIRS Black Marble y densidad poblacional del Censo 2024. 
* `06-clasificacion-xgboost-shap.py`: clasificación binaria de riqueza alta o baja por hexágono H3-8 con XGBoost (gradient boosting con regularización) e interpretación con SHAP. Reusa el dataset de la clase 05 e incorpora dos elementos nuevos: coordenadas UTM como features (alternativa no paramétrica a GWR) y `sample_weight` derivado de la población flotante diurna estimada desde la EOD 2012, para corregir el sesgo de muestreo de eBird.
* `07-kriging-interpolacion.py`: interpolación espacial con IDW y kriging ordinario, comparados con validación cruzada leave-one-out. Caso pedagógico submuestreando el raster NDVI (ground truth disponible) y caso real con PM2.5 promedio diario de las estaciones SINCA de la RM en un día de alta contaminación de invierno 2024. Cierra con kriging con tendencia (regression kriging) usando luminosidad nocturna como covariable.
* `08-asignacion-censal.py`: cambio de soporte espacial. Compara areal weighting (reparto proporcional al área) con asignación de microdatos a celdas H3 vía simulated annealing (preserva correlaciones intra-vivienda). Ejemplo en vivo sobre La Pintana y resultado precomputado para todo el Gran Santiago a H3-8, listo para cruzar con eBird, NDVI o cualquier otra capa de las clases anteriores.
* `09-redes-urbanas.py`: redes urbanas con `networkx` y `quackosm`. Tres casos en el Gran Santiago: red de calles vehiculares (descriptivos topológicos, distribución del grado, caminos más cortos, centralidades de grado, cercanía, intermediación y PageRank, curva de resiliencia, edge betweenness), red de ciclovías (contraste entre el catálogo oficial MINVU y `highway=cycleway` de OSM, fragmentación, sensibilidad a la tolerancia de snap, edge betweenness sobre la componente gigante), y red de paraderos del Transantiago desde GTFS con geometrías shape-based.
* `10-redes-od.py`: redes origen-destino y diferencias de género en viajes de cuidado. Replica el análisis de `Viajes_Encadenados` sobre la EOD Santiago 2012: matriz O-D en H3-7 por sexo y propósito agregado (Cuidado, Empleo/Estudio, Personal), Pointwise Mutual Information para identificar pares O-D sobre-representados por categoría, subgrafos inducidos por PMI > 0, métricas topológicas (clustering, triángulos), y modelo nulo por permutación de sexo (200 réplicas) con p-values bilaterales.
* `11-clustering.py`: clustering espacial en tres problemas distintos. Detección de hotspots de puntos (k-means, DBSCAN, HDBSCAN) sobre reportes ciudadanos de SOSAFE; regionalización con restricción de contigüidad (SKATER y Ward) sobre hexágonos H3-8 con perfil censal; y el trade-off entre homogeneidad interna y fragmentación. Distancias en UTM 19S, caracterización de hotspots con alpha shapes y ritmos horario y semanal.
* `12-demostraciones-interactivas.py`: cómo construir visualizaciones interactivas con ayuda de un LLM. No es una técnica de análisis: cubre cuándo conviene lo interactivo, el patrón de HTML autónomo (un template con los datos embebidos, sin build), dónde corre el cómputo, y cómo verificar el código generado. El ejemplo trabajado corre DBSCAN en vivo en el navegador sobre los reportes SOSAFE de la clase 11 y comprueba que coincide con scikit-learn.
* `13-analisis-espacio-temporal.py`: análisis espacio-temporal usando la expansión del Metro de Santiago (Línea 6 en 2017, Línea 3 en 2019) como experimento natural. Compara pares origen-destino que ganaron acceso a Metro (tratamiento) con los que nunca lo tuvieron (control) sobre los viajes DTPM de 2016 y 2019. Incluye diferencias en diferencias para medir la sustitución de modo y un modelo de interacción espacial (gravedad) ajustado con regresión binomial negativa.

## Preparación de los datos

Los datasets que cada clase descarga de forma automática se preparan con los scripts de `profe-scripts/`, que se corren desde la raíz del repositorio con `uv run python profe-scripts/<script>.py`. Estos scripts leen fuentes crudas o pesadas, las reducen a un dataset manejable y lo publican. Cada script hace `import config`, el módulo compartido `profe-scripts/config.py` que centraliza el destino de publicación, los helpers y las rutas a las fuentes crudas.

Para correr un profe-script hay que indicarle dónde están sus fuentes crudas. Esas rutas privadas no se versionan: `config.py` las lee de un archivo local `profe-scripts/config_local.py` (ignorado por git) o de las variables de entorno `GDS_*`. Para crearlo, copiar `profe-scripts/config_local.example.py` a `profe-scripts/config_local.py` y completar las rutas. La precedencia es variable de entorno sobre `config_local.py` sobre el default. La subida al servidor del curso no ocurre por defecto: se activa con `GDS_SUBIR=1`.

Dependencias de ejecución por script:

* `05-preparar-luminosidad.py`: cuenta gratuita en NASA Earthdata con la aplicación "LAADS DAAC Cumulus (PROD)" autorizada y `earthaccess` (`uv add earthaccess h5py`). Produce el GeoTIFF de luminosidad nocturna (VNP46A4, VIIRS Black Marble) que consumen las clases 05 y 06.
* `05-ebird-dataset.py`: el eBird Basic Dataset descargado (`GDS_EBD_DIR`), el NDVI de Sentinel-2 (Planetary Computer, descarga automática) y el GeoTIFF de luminosidad de `05-preparar-luminosidad.py`.
* `05-sosafe-dataset.py`: los reportes SOSAFE crudos (`GDS_SOSAFE_RAW`) y el grid H3-8 con perfil censal (`GDS_SOSAFE_H3_GRID`).
* `07-sinca-dataset.py`: sin credenciales; descarga PM2.5 directamente del sitio del SINCA, así que requiere conexión a internet.
* `08-asignacion-rm.py`: microdatos y cartografía del Censo 2024 (se bajan del servidor del curso). Opcionalmente reusa asignaciones precomputadas con `GDS_LEGADO_ASIGNACION`.
* `09-redes-santiago.py`: calles y ciclovías de OpenStreetMap vía `quackosm` (descarga automática) y el shapefile nacional de ciclovías del MINVU (`GDS_MINVU_ZIP`).
* `11-clustering-dataset.py`: los reportes SOSAFE crudos (`GDS_SOSAFE_RAW`).
* `13-lineas-metro-dataset.py`: la base DTPM consolidada (se construye con `gdsutils.dtpm`) y el GTFS del Transantiago (URL en `gdsutils.dtpm`).

La clase 12 no tiene profe-script: reutiliza el dataset preparado para la clase 11.
