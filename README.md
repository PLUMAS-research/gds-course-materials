
# CC5216 -- Ciencia de datos geográficos

El material del curso se ejecuta con `uv`. [Aquí puedes ver sus instrucciones de instalación](https://docs.astral.sh/uv/getting-started/installation/).

Si tienes `uv` instalado, el comando `uv sync` instala todo lo necesario para que ejecutes el código del curso. 

Además, en cada clase es recomendable ejecutar `uv lock --upgrade-package chiricoca`. Esto actualizará el código del repositorio `chiricoca` (base de visualización) en caso de que haya cambiado entre una clase y otra (¡es probable que lo haga!).

La carpeta `gdsutils` contiene funciones utilitarias para trabajar con los datos que veremos en el curso. La carpeta `data` contiene datos que se descargan de manera automática en cada script.

## Clases

* `01-carga-datos-geograficos.py`: cómo cargar datos con `geopandas` en formato Parquet. Ejemplo con datos del Censo 2024 de Chile a nivel de comunas y manzanas.
* `02-geografía-y-espacialidad.py`: diferencias de proporciones según la proyección, mapa de burbujas para Santiago con el censo, red de vecindad comunal.
* `03-definir-proyecto.py`: datos de transporte público (DTPM): descarga, consolidación y análisis exploratorio de viajes 2014–2025 y paraderos del Gran Santiago.
