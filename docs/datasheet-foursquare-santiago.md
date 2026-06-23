---
title: "Datasheet: Foursquare Santiago (subset de check-ins)"
subtitle: "Ejemplo de documentación de dataset (Gebru et al., 2021)"
lang: es
---

Datasheet de ejemplo para el subset de Santiago del dataset global de check-ins de Foursquare, usado en la clase de ética y sesgos como contraste de la EOD 2012.

## Motivación

- **¿Para qué se creó el dataset original?** Investigación académica sobre movilidad y redes sociales en LBSN (location-based social networks). No fue diseñado para representar la movilidad de una población.
- **¿Quién lo creó?** Yang, Qu, Yang y Cudre-Mauroux (WWW'19). El subset de Santiago lo deriva el curso con `profe-scripts/14-foursquare-dataset.py`.

## Composición

- **¿Qué representa cada instancia?** Un check-in: un usuario que marca su presencia en un venue (lugar) en un instante. Cada venue trae coordenadas y una categoría de Foursquare.
- **¿Cuántas instancias hay (subset Santiago)?** 1.304.059 check-ins de 41.034 usuarios sobre 91.954 venues, entre abril de 2012 y enero de 2014 (22 meses, multi-día).
- **¿Qué granularidad espacial y temporal?** Coordenadas de venue (precisas) y hora local (UTC más offset). Multi-día, a diferencia de la EOD.
- **Sesgos conocidos de cobertura (autoselección).** El check-in es voluntario, así que sobre-representa a usuarios urbanos, jóvenes y con smartphone, y a lugares "marcables" (comercio, ocio). Medido en la clase 14:
  - La actividad por zona concentra en el centro-oriente; correlación espacial con la EOD de solo 0,58.
  - El "hogar" inferido (check-ins nocturnos) correlaciona apenas 0,23 con el hogar declarado de la EOD: la app borra la periferia residencial.
  - En la red O-D, comunas periféricas completas quedan sin flujo observado.
- **¿Permite identificar personas?** El `user_id` viene anonimizado, pero las trayectorias son altamente únicas: a resolución de celda y hora, el 92% de los usuarios es único con cuatro puntos (clase 14). Es un caso de que "anonimizado" no equivale a "anónimo".

## Proceso de recolección

- **¿Cómo se obtuvo?** Recolectado por los autores desde la API/actividad pública de Foursquare durante 2012-2014.
- **¿Consentimiento?** Los usuarios aceptaron los términos de la plataforma, no un uso de investigación específico. Conviene tratarlo con cautela de dato personal.

## Preprocesamiento y limpieza (subset del curso)

- `profe-scripts/14-foursquare-dataset.py` filtra los venues de Chile dentro del bbox del Gran Santiago, calcula la hora local y deja `checkins.parquet` (usuario, venue, fecha local) y `venues.parquet` (venue, lat, lon, categoría).
- No se submuestrea: el subset conserva todos los check-ins de Santiago.

## Usos

- **Usos previstos en el curso:** ilustrar el sesgo de cobertura de una fuente pasiva y el riesgo de reidentificación por unicidad.
- **Usos no recomendados:** estimar demanda de movilidad representativa; medir brechas demográficas (la autoselección contamina); cruzar con el Censo como si fuera una muestra poblacional.

## Distribución

- El dataset global exige citar a Yang et al. (WWW'19) en cualquier material derivado. El subset de Santiago se distribuye para uso docente (`foursquare-santiago.tgz`) con esa cita.

## Mantenimiento

- El dataset original es estático (cubre 2012-2014). No hay actualización.

## Referencias

- Gebru et al. (2021). Datasheets for Datasets. Communications of the ACM.
- Yang, Qu, Yang y Cudre-Mauroux (2019). Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding Approach. WWW'19.
