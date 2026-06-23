---
title: "Model card: clasificador de riqueza de aves (clase 06)"
subtitle: "Ejemplo de documentación de modelo (Mitchell et al., 2019)"
lang: es
---

Para modelos se usa una *model card* (Mitchell et al., 2019); para datasets, una *datasheet*. Este documento es una model card de ejemplo para el clasificador XGBoost de la clase 06. Las secciones siguen a Mitchell et al.

## Detalles del modelo

- Clasificador de *gradient boosting* (XGBoost) entrenado en la clase 06 con fines pedagógicos.
- Tarea: clasificación binaria de **riqueza alta** de aves por hexágono H3-8 en Santiago, definida como `sqrt_riqueza > mediana`.
- Interpretación con SHAP (importancia y dependencia por variable).

## Uso previsto

- **Previsto**: enseñar clasificación no lineal e interpretabilidad espacial. Comparar el aporte de variables ambientales y de la ubicación.
- **No previsto**: decisiones de conservación o asignación de recursos sin validación de campo; predecir biodiversidad real (la etiqueta es un corte por mediana, no un umbral ecológico).

## Variables (factores)

- Predictores: NDVI, log de luminosidad nocturna, log de densidad poblacional y coordenadas UTM (`x_utm`, `y_utm`) como alternativa no paramétrica a GWR.
- El esfuerzo de muestreo NO entra como variable: se corrige de forma exógena con `sample_weight = 1 / log(2 + poblacion_flotante)` (de la EOD), normalizado a media 1, para mitigar el sesgo de muestreo de eBird.

## Métricas

- Clasificación binaria: se reporta exactitud y AUC sobre datos de prueba, además de la lectura SHAP. Los valores los produce `06-clasificacion-xgboost-shap.py` al ejecutarse.
- La componente espacial del modelo se resume sumando las contribuciones SHAP de `x_utm` y `y_utm`.

## Datos de evaluación y entrenamiento

- Dataset de la clase 05: eBird Santiago agregado a H3-8 (`ebird-santiago-2024`), con NDVI (Sentinel-2), luminosidad (VIIRS) y densidad (Censo 2024).
- La población flotante diurna proviene de la EOD 2012, con sus propios sesgos (ver su datasheet).

## Consideraciones éticas

- **Sesgo de la fuente**: eBird es de ciencia ciudadana, con esfuerzo desigual en el espacio. La corrección por `sample_weight` lo mitiga, no lo elimina.
- **La ubicación como variable**: usar coordenadas como predictores captura autocorrelación espacial, pero puede codificar de forma implícita atributos del territorio (ver la discusión de equidad espacial).
- **Doble uso**: un modelo de "dónde hay más aves" podría orientar tanto conservación como presión inmobiliaria.

## Advertencias y recomendaciones

- La etiqueta es un corte por mediana: cambiar el umbral cambia el problema.
- No extrapolar fuera de la bbox del Gran Santiago ni del periodo de los datos.
- Acompañar siempre la predicción con la lectura SHAP y con la datasheet de los datos de entrada.

## Referencias

- Mitchell et al. (2019). Model Cards for Model Reporting. FAT*.
