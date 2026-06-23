---
title: "Data statement: descripciones de reportes SOSAFE"
subtitle: "Ejemplo de documentación de datos de texto (Bender y Friedman, 2018)"
lang: es
---

Un *data statement* documenta datos de lenguaje: quién escribe, en qué variedad, en qué situación y con qué características. Aplica al texto libre (`description`) de los reportes ciudadanos de SOSAFE que usamos en las clases de clustering e interactividad. Las secciones siguen a Bender y Friedman (2018).

## Curación: por qué existe este texto

Reportes ciudadanos de seguridad y entorno urbano, escritos por vecinos en la aplicación SOSAFE. El texto acompaña a una categoría (delitos, disturbios, ambiental) y a una geolocalización. No se recolectó para análisis de lenguaje, sino como registro operativo.

## Variedad lingüística

Español de Chile, registro informal: abreviaturas, jerga local, errores de tipeo, mayúsculas de énfasis. Predomina la oración corta y descriptiva.

## Demografía de quienes escriben

Usuarios de la aplicación SOSAFE: población urbana con smartphone y disposición a reportar. La demografía detallada (edad, sexo, ingreso) no está en los datos y no debe inferirse. Hay autoselección: quien no usa la app no aparece.

## Demografía de anotadores

No hay anotación humana posterior. La categoría la asigna quien reporta (o la app), no un equipo de anotación, así que no hay un esquema de anotación controlado ni acuerdo entre anotadores.

## Situación del habla

Texto escrito en el momento del evento o poco después, en contexto de alarma o molestia, de forma espontánea y asincrónica. No es habla editada ni revisada.

## Características del texto

Descripciones breves. Contienen información personal identificable (correos, teléfonos, a veces nombres). **Antes de salir del computador del profesor se anonimiza** con `gdsutils.sosafe` (`PATRON_CORREO`, `PATRON_TELEFONO`, `anonimizar_texto`), que reemplaza esos patrones por marcas fijas.

## Procedencia y calidad

Origen: JSON diarios de la app. La calidad del texto es heterogénea (ruido, duplicados). La geolocalización puede tener error de GPS urbano.

## Advertencias de uso

- No tratar el texto como muestra representativa de la percepción de seguridad de la ciudad: refleja a quienes usan la app.
- No reidentificar a las personas a partir del texto ni de la combinación texto + ubicación + hora.

## Referencias

- Bender y Friedman (2018). Data Statements for Natural Language Processing. TACL.
