---
title: "Datasheet: Encuesta Origen-Destino de Santiago 2012"
subtitle: "Ejemplo de documentación de dataset (Gebru et al., 2021)"
lang: es
---

Para datasets se usa una *datasheet* (Gebru et al., 2021); para modelos, una *model card* (Mitchell et al., 2019). Este documento es una datasheet de ejemplo para la clase de ética y sesgos. Las secciones siguen las preguntas de Gebru et al.

## Motivación

- **¿Para qué se creó?** Medir la demanda de viajes del Gran Santiago para planificación de transporte. La realiza el Estado (SECTRA, hoy bajo el Ministerio de Transportes y Telecomunicaciones).
- **¿Quién la financió?** Fondos públicos. La medición de 2012 tuvo un costo cercano a 600 millones de pesos de 2011.
- **¿Cada cuánto se levanta?** Aproximadamente cada diez años (1991, 2001, 2012). La Encuesta de Movilidad de Santiago 2024 (CEDEUS) cubre parte del vacío, pero no reemplaza formalmente a la EOD.

## Composición

- **¿Qué representa cada instancia?** El dataset es relacional: hogares, personas, viajes y etapas. Un viaje tiene hasta cuatro etapas y atributos de origen, destino, propósito, modo, hora y distancia.
- **¿Cuántas instancias hay?** Cerca de 18.000 hogares y del orden de 90 mil viajes válidos tras filtrar registros inválidos o imputados.
- **¿Qué granularidad espacial?** 866 zonas EOD para el Gran Santiago. Las direcciones se llevan al **centroide de la manzana**: una decisión que ya es agregación, con efectos en precisión y privacidad.
- **¿Tiene factores de expansión?** Sí. Cada persona y viaje trae factores por tipo de día (laboral, sábado, domingo; normal y estival). El análisis debe ponderar por el factor que corresponda al día.
- **Sesgos conocidos de cobertura.** La muestra está diseñada para ser representativa de la población vía factores, pero:
  - Es un retrato de **un solo día** por persona. No captura la variación entre días ni rutinas multi-día.
  - Depende de la **declaración** del encuestado: hay sesgo de memoria y de subreporte de viajes cortos, a pie o de cuidado.
  - Los viajes de cuidado quedan dispersos en categorías como "compras" o "acompañar", lo que invisibiliza por diseño la movilidad del cuidado (Sánchez de Madariaga, 2013).
- **¿Contiene datos que permitan identificar personas?** No trae nombres, pero sí coordenadas de hogar y secuencias de viajes. Una trayectoria de pocos puntos es altamente identificable (ver clase 14): tratar como dato sensible aunque la letra de la ley no lo liste.

## Proceso de recolección

- **¿Cómo se obtuvo?** Encuesta presencial a hogares, con un día asignado de registro de viajes.
- **¿Sobre qué población?** Hogares del Gran Santiago. Quedan fuera por construcción quienes no residen en el área de estudio.
- **Limitaciones del instrumento.** El costo y la baja frecuencia hacen que el retrato envejezca: para 2024 la estructura de la ciudad ya cambió respecto a 2012.

## Preprocesamiento y limpieza

- En el curso se lee con `gdsutils.eodscl` (`read_trips`, `read_people`, `read_homes`), que decodifica las columnas con las tablas de parámetros.
- `read_trips` filtra por defecto viajes sin hora o imputados. Las coordenadas vienen en UTM 19S (EPSG:32719).
- La edad está en un archivo aparte (`Edadpersonas.csv`) y la zona de hogar se obtiene cruzando `personas` con `Hogares`.

## Usos

- **Usos previstos:** análisis de demanda, partición modal, generación y atracción de viajes por zona, modelos de interacción espacial.
- **Usos no recomendados:** inferir comportamiento individual (falacia ecológica); comparar en niveles absolutos con fuentes pasivas (CDR, check-ins) sin alinear la unidad de observación y normalizar; tratar las trayectorias como anónimas.

## Distribución

- Dato público del Estado. En el curso se distribuye empaquetado (`eod2012.tgz`) para uso docente.

## Mantenimiento

- La mantiene el organismo público de transporte. No hay actualización continua: el dataset es una foto fija de 2012.

## Referencias

- Gebru et al. (2021). Datasheets for Datasets. Communications of the ACM.
- Sánchez de Madariaga (2013). The mobility of care.
