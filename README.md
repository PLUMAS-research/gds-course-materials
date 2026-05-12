
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
