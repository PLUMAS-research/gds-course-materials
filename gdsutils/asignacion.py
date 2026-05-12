"""
Asignacion de viviendas censales a celdas H3.

El problema: los microdatos del Censo 2024 estan disponibles a nivel de
vivienda, pero el detalle geografico solo llega a comuna. Las cartografias
publican totales por manzana. Queremos producir, para cada vivienda
individual, una celda H3 que respete los totales agregados por manzana
y preserve las correlaciones intra-vivienda.

El algoritmo opera sobre tres fases:

  Fase 0. Construye objetivos por celda escalando los totales del grid a
          los totales reales de las viviendas, calcula una matriz de
          distancia vivienda x celda con pesos por prioridad y aplica
          softmax con la capacidad como prior.

  Fase 1. Asignacion estratificada por tipo de vivienda (restriccion dura)
          seguida de un equilibrado iterativo de capacidades.

  Fase 2. Simulated annealing con swaps restringidos al mismo tipo,
          dirigidos por el residuo del atributo mas critico de la celda.

API publica: diagnosticar_atributos, asignar_viviendas, reportar_errores,
visualizar_errores.
"""

import numpy as np
import pandas as pd


def obtener_atributos(config):
    """Extrae listas planas de atributos desde la config."""
    attrs_persona = (
        config.get("persona", {}).get("prioridad_alta", [])
        + config.get("persona", {}).get("prioridad_media", [])
        + config.get("persona", {}).get("prioridad_baja", [])
    )
    attrs_vivienda = (
        config.get("vivienda", {}).get("prioridad_alta", [])
        + config.get("vivienda", {}).get("prioridad_media", [])
        + config.get("vivienda", {}).get("prioridad_baja", [])
    )
    tipos_vivienda = config.get("tipo_vivienda", [])
    return attrs_persona, attrs_vivienda, tipos_vivienda


def construir_pesos(config, peso_alta, peso_media, peso_baja):
    pesos = {}
    for tipo in ["persona", "vivienda"]:
        for attr in config.get(tipo, {}).get("prioridad_alta", []):
            pesos[attr] = peso_alta
        for attr in config.get(tipo, {}).get("prioridad_media", []):
            pesos[attr] = peso_media
        for attr in config.get(tipo, {}).get("prioridad_baja", []):
            pesos[attr] = peso_baja
    return pesos


def clasificar_viviendas(viviendas, tipos_vivienda):
    """Asigna a cada vivienda la columna `tipo_vivienda` que vale 1, o `_otros`."""
    n_viv = len(viviendas)
    tipos = np.full(n_viv, "_otros", dtype=object)
    for col in tipos_vivienda:
        mask = viviendas[col].values == 1
        tipos[mask] = col
    return tipos


def calcular_objetivos_escalados(grid_variables, viviendas, attrs_persona, attrs_vivienda, tipos_vivienda):
    """Reescala los conteos del grid para que coincidan con los totales reales."""
    grid = grid_variables.set_index("h3_cell_id").copy()

    total_viv_real = len(viviendas)
    total_per_real = (viviendas["n_hombres"] + viviendas["n_mujeres"]).sum()

    prop_viv = grid["n_vp_ocupada"] / grid["n_vp_ocupada"].sum()
    prop_per = grid["n_per"] / grid["n_per"].sum()

    grid["objetivo_viv"] = prop_viv * total_viv_real
    grid["objetivo_per"] = prop_per * total_per_real

    todos_attrs = attrs_persona + attrs_vivienda + tipos_vivienda
    for attr in todos_attrs:
        total_attr_real = viviendas[attr].sum()
        suma_grid = grid[attr].sum()
        if suma_grid > 0:
            prop_attr = grid[attr] / suma_grid
        else:
            prop_attr = 0
        grid[f"objetivo_{attr}"] = prop_attr * total_attr_real

    return grid


def calcular_proporciones_celdas(grid_objetivos, attrs_persona, attrs_vivienda, tipos_vivienda):
    df = grid_objetivos.copy()

    n_personas = df["objetivo_per"].values
    n_personas_safe = np.where(n_personas == 0, 1, n_personas)

    n_viviendas = df["objetivo_viv"].values
    n_viviendas_safe = np.where(n_viviendas == 0, 1, n_viviendas)

    for attr in attrs_persona:
        df[f"p_{attr}"] = df[f"objetivo_{attr}"].values / n_personas_safe

    for attr in attrs_vivienda + tipos_vivienda:
        df[f"p_{attr}"] = df[f"objetivo_{attr}"].values / n_viviendas_safe

    df["tamano_hogar"] = df["objetivo_per"] / n_viviendas_safe
    return df


def calcular_proporciones_viviendas(viviendas, attrs_persona, attrs_vivienda, tipos_vivienda):
    df = viviendas.set_index("id_vivienda").copy()

    df["n_personas"] = df["n_hombres"] + df["n_mujeres"]
    n_personas_safe = np.where(df["n_personas"] == 0, 1, df["n_personas"])

    for attr in attrs_persona:
        df[f"p_{attr}"] = df[attr].values / n_personas_safe

    for attr in attrs_vivienda + tipos_vivienda:
        df[f"p_{attr}"] = df[attr].values.astype(float)

    return df


def calcular_distancias(
    prop_viviendas,
    prop_celdas,
    attrs_persona,
    attrs_vivienda,
    tipos_vivienda,
    pesos,
    peso_tamano_hogar,
    peso_tipo,
):
    """Distancia ponderada vivienda x celda en el espacio de proporciones."""
    n_viv = len(prop_viviendas)
    n_celdas = len(prop_celdas)
    d = np.zeros((n_viv, n_celdas))

    for attr in attrs_persona + attrs_vivienda:
        w = pesos.get(attr, 1.0)
        v = prop_viviendas[f"p_{attr}"].values[:, np.newaxis]
        c = prop_celdas[f"p_{attr}"].values[np.newaxis, :]
        d += w * np.abs(v - c)

    for attr in tipos_vivienda:
        v = prop_viviendas[f"p_{attr}"].values[:, np.newaxis]
        c = prop_celdas[f"p_{attr}"].values[np.newaxis, :]
        d += peso_tipo * np.abs(v - c)

    v_tamano = prop_viviendas["n_personas"].values[:, np.newaxis]
    c_tamano = prop_celdas["tamano_hogar"].values[np.newaxis, :]
    c_tamano_safe = np.where(c_tamano == 0, 1, c_tamano)
    d += peso_tamano_hogar * (np.abs(v_tamano - c_tamano) / c_tamano_safe)

    return d


def calcular_probabilidades(distancias, capacidades, temperatura):
    """Softmax con log-capacidad como prior."""
    with np.errstate(divide="ignore"):
        log_capacidad = np.log(capacidades)
    log_capacidad = np.nan_to_num(log_capacidad, neginf=-1e20)

    log_p = log_capacidad[np.newaxis, :] - (distancias / temperatura)
    log_p -= log_p.max(axis=1, keepdims=True)

    p = np.exp(log_p)
    suma = p.sum(axis=1, keepdims=True)
    suma[suma == 0] = 1.0
    p /= suma
    return p


def asignar_estratificado(viviendas, grid_objetivos, probabilidades, tipos_viv, tipos_vivienda, rng, verbose):
    """Asignacion inicial respetando el tipo de vivienda como restriccion dura."""
    n_celdas = len(grid_objetivos)
    n_viv = len(viviendas)
    asignaciones = np.zeros(n_viv, dtype=int)

    tipos_unicos = np.unique(tipos_viv)

    if verbose:
        print("  Asignacion estratificada por tipo:")
        for tipo in tipos_unicos:
            n = (tipos_viv == tipo).sum()
            print(f"    {tipo}: {n:,} viviendas")

    capacidades = {}
    for tipo in tipos_unicos:
        if tipo in tipos_vivienda:
            capacidades[tipo] = grid_objetivos[f"objetivo_{tipo}"].values.copy()
        else:
            cap_total = grid_objetivos["objetivo_viv"].values.copy()
            for t in tipos_vivienda:
                cap_total = cap_total - grid_objetivos[f"objetivo_{t}"].values
            capacidades[tipo] = np.maximum(cap_total, 0)

    for tipo in tipos_unicos:
        indices = np.where(tipos_viv == tipo)[0]
        cap = capacidades[tipo].copy()

        prob_max = probabilidades[indices].max(axis=1)
        orden = indices[np.argsort(-prob_max)]

        for v in orden:
            prob_v = probabilidades[v] * cap
            suma = prob_v.sum()

            if suma > 1e-10:
                prob_v = prob_v / suma
                celda = rng.choice(n_celdas, p=prob_v)
            else:
                celda = np.argmax(cap)

            asignaciones[v] = celda
            cap[celda] = max(0, cap[celda] - 1)

    return asignaciones


def calcular_energia(asignaciones, viviendas, grid_objetivos, attrs_persona, attrs_vivienda, tipos_vivienda):
    """Energia agregada (suma de errores absolutos por atributo)."""
    n_celdas = len(grid_objetivos)
    viv = viviendas.set_index("id_vivienda")
    v_personas = viv["n_hombres"].values + viv["n_mujeres"].values

    energia = {}
    asignado = {}
    objetivo = {}

    asignado["n_vp"] = np.bincount(asignaciones, minlength=n_celdas).astype(float)
    objetivo["n_vp"] = grid_objetivos["objetivo_viv"].values
    energia["n_vp"] = np.abs(asignado["n_vp"] - objetivo["n_vp"]).sum()

    asignado["n_per"] = np.bincount(asignaciones, weights=v_personas, minlength=n_celdas)
    objetivo["n_per"] = grid_objetivos["objetivo_per"].values
    energia["n_per"] = np.abs(asignado["n_per"] - objetivo["n_per"]).sum()

    for attr in attrs_persona + attrs_vivienda + tipos_vivienda:
        weights = viv[attr].values.astype(float)
        asignado[attr] = np.bincount(asignaciones, weights=weights, minlength=n_celdas)
        objetivo[attr] = grid_objetivos[f"objetivo_{attr}"].values
        energia[attr] = np.abs(asignado[attr] - objetivo[attr]).sum()

    return energia, asignado, objetivo


def equilibrar_capacidades(
    asignaciones, viviendas, grid_objetivos, probabilidades, tipos_viv, tipos_vivienda, max_iter, tolerancia, rng, verbose
):
    """Reasigna desde celdas con exceso a celdas con deficit hasta `tolerancia`."""
    n_celdas = len(grid_objetivos)
    objetivo_viv = grid_objetivos["objetivo_viv"].values

    for i in range(max_iter):
        conteo = np.bincount(asignaciones, minlength=n_celdas).astype(float)
        residuo = objetivo_viv - conteo

        error_rel = np.abs(residuo).sum() / len(viviendas)
        if verbose:
            print(f"  Equilibrado iter {i}: error = {error_rel:.4f}")

        if error_rel < tolerancia:
            break

        exceso = np.where(residuo < -1)[0]
        deficit = np.where(residuo > 1)[0]

        if len(exceso) == 0 or len(deficit) == 0:
            break

        candidatas = np.where(np.isin(asignaciones, exceso))[0]
        n_mover = max(1, int(len(candidatas) * 0.1))
        a_mover = rng.choice(candidatas, size=min(n_mover, len(candidatas)), replace=False)

        pesos_deficit = np.maximum(residuo[deficit], 0)
        pesos_deficit = pesos_deficit / pesos_deficit.sum()

        for v in a_mover:
            p_afinidad = probabilidades[v, deficit]
            suma = p_afinidad.sum()
            if suma > 1e-10:
                p_afinidad = p_afinidad / suma
                p_final = pesos_deficit * p_afinidad
                p_final = p_final / p_final.sum()
            else:
                p_final = pesos_deficit

            asignaciones[v] = rng.choice(deficit, p=p_final)

    return asignaciones


def simulated_annealing(
    asignaciones,
    viviendas,
    grid_objetivos,
    attrs_persona,
    attrs_vivienda,
    tipos_viv,
    temperatura_inicial,
    temperatura_final,
    n_iteraciones,
    verbose,
):
    """SA con swaps restringidos al mismo tipo de vivienda."""
    if verbose:
        print(f"  SA: T={temperatura_inicial} -> {temperatura_final}, {n_iteraciones} iter")

    todos_attrs = attrs_persona + attrs_vivienda
    matriz_attr = viviendas[todos_attrs].values
    objetivos = np.array([grid_objetivos[f"objetivo_{a}"].values for a in todos_attrs]).T
    n_celdas = len(grid_objetivos)
    n_attrs = len(todos_attrs)

    estado = np.zeros((n_celdas, n_attrs))
    for i, attr in enumerate(todos_attrs):
        estado[:, i] = np.bincount(asignaciones, weights=matriz_attr[:, i], minlength=n_celdas)
    residuos = objetivos - estado

    tipos_unicos = np.unique(tipos_viv)
    vecindario = {}
    for c in range(n_celdas):
        vecindario[c] = {}
        for tipo in tipos_unicos:
            mask = (asignaciones == c) & (tipos_viv == tipo)
            vecindario[c][tipo] = np.where(mask)[0].tolist()

    factor_enfriamiento = (temperatura_final / temperatura_inicial) ** (1.0 / n_iteraciones)
    temperatura = temperatura_inicial

    energia_inicial = np.abs(residuos).sum()
    aceptados = 0
    mejoras = 0

    for k in range(n_iteraciones):
        energia_por_celda = np.abs(residuos).sum(axis=1)
        suma_energia = energia_por_celda.sum()
        if suma_energia == 0:
            break
        prob = energia_por_celda / suma_energia
        c1 = np.random.choice(n_celdas, p=prob)

        tipos_disponibles = [t for t in tipos_unicos if len(vecindario[c1][t]) > 0]
        if len(tipos_disponibles) == 0:
            temperatura *= factor_enfriamiento
            continue

        tipo = np.random.choice(tipos_disponibles)

        attr_critico = np.argmax(np.abs(residuos[c1]))
        necesita_mas = residuos[c1, attr_critico] > 0

        if necesita_mas:
            candidatas = np.where(residuos[:, attr_critico] < -0.1)[0]
        else:
            candidatas = np.where(residuos[:, attr_critico] > 0.1)[0]

        candidatas = np.array([c for c in candidatas if c != c1 and len(vecindario[c][tipo]) > 0])

        if len(candidatas) == 0:
            candidatas = np.array([c for c in range(n_celdas) if c != c1 and len(vecindario[c][tipo]) > 0])
            if len(candidatas) == 0:
                temperatura *= factor_enfriamiento
                continue

        c2 = np.random.choice(candidatas)

        v1 = np.random.choice(vecindario[c1][tipo])
        v2 = np.random.choice(vecindario[c2][tipo])

        attrs_v1 = matriz_attr[v1]
        attrs_v2 = matriz_attr[v2]

        energia_antes = np.abs(residuos[c1]).sum() + np.abs(residuos[c2]).sum()

        residuos_nuevos_c1 = residuos[c1] + attrs_v1 - attrs_v2
        residuos_nuevos_c2 = residuos[c2] + attrs_v2 - attrs_v1

        energia_despues = np.abs(residuos_nuevos_c1).sum() + np.abs(residuos_nuevos_c2).sum()
        delta_energia = energia_despues - energia_antes

        if delta_energia < 0:
            aceptar = True
            mejoras += 1
        else:
            probabilidad_aceptacion = np.exp(-delta_energia / temperatura)
            aceptar = np.random.random() < probabilidad_aceptacion

        if aceptar:
            asignaciones[v1] = c2
            asignaciones[v2] = c1

            residuos[c1] = residuos_nuevos_c1
            residuos[c2] = residuos_nuevos_c2

            vecindario[c1][tipo].remove(v1)
            vecindario[c1][tipo].append(v2)
            vecindario[c2][tipo].remove(v2)
            vecindario[c2][tipo].append(v1)

            aceptados += 1

        temperatura *= factor_enfriamiento

        if verbose and (k + 1) % 100000 == 0:
            energia_actual = np.abs(residuos).sum()
            print(f"    iter {k+1}: T={temperatura:.3f}, E={energia_actual:.0f}")

    energia_final = np.abs(residuos).sum()
    if verbose:
        reduccion = 100 * (energia_inicial - energia_final) / max(energia_inicial, 1)
        print(f"  SA completado: E={energia_inicial:.0f} -> {energia_final:.0f} ({reduccion:.1f}%)")
        print(f"    {aceptados} aceptados ({mejoras} mejoras)")

    return asignaciones


def asignar_viviendas(
    viviendas,
    grid_variables,
    config,
    peso_alta=10.0,
    peso_media=1.0,
    peso_baja=0.1,
    peso_tipo=20.0,
    peso_tamano_hogar=2.0,
    temperatura_inicial=0.01,
    max_iter_equilibrado=100,
    tolerancia_equilibrado=0.01,
    sa_temperatura_inicial=50.0,
    sa_temperatura_final=0.01,
    sa_iteraciones=500000,
    seed=42,
    verbose=False,
):
    """
    Asigna cada vivienda del DataFrame `viviendas` a una celda H3 del grid.

    Parametros principales:
        viviendas: DataFrame con columnas n_* y `id_vivienda`. Debe incluir
            todas las columnas listadas en config.
        grid_variables: GeoDataFrame indexado o con columna `h3_cell_id` y
            con conteos agregados por celda (n_vp_ocupada, n_per, atributos).
        config: dict con tres claves:
            persona: prioridad_alta/media/baja, atributos por persona
            vivienda: prioridad_alta/media/baja, atributos por vivienda
            tipo_vivienda: lista de columnas excluyentes (restriccion dura)
        peso_alta/media/baja: pesos en la distancia para cada prioridad.
        peso_tipo: peso del tipo de vivienda en la distancia (alto = adherencia
            estricta a la restriccion).
        peso_tamano_hogar: peso del tamano del hogar en la distancia.
        sa_iteraciones: numero de pasos del simulated annealing.

    Retorna:
        resultado: DataFrame con columnas id_vivienda y h3_cell_id.
        errores: dict atributo -> error absoluto agregado.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    attrs_persona, attrs_vivienda, tipos_vivienda = obtener_atributos(config)
    pesos = construir_pesos(config, peso_alta, peso_media, peso_baja)

    if verbose:
        print(f"Atributos persona: {len(attrs_persona)}")
        print(f"Atributos vivienda: {len(attrs_vivienda)}")
        print(f"Tipos vivienda (restriccion dura): {tipos_vivienda}")

    tipos_viv = clasificar_viviendas(viviendas, tipos_vivienda)

    if verbose:
        tipos_u, conteos = np.unique(tipos_viv, return_counts=True)
        for t, c in zip(tipos_u, conteos):
            print(f"  {t}: {c:,} ({100*c/len(viviendas):.1f}%)")

    if verbose:
        print("Fase 0: escalando objetivos...")
    grid_objetivos = calcular_objetivos_escalados(
        grid_variables, viviendas, attrs_persona, attrs_vivienda, tipos_vivienda
    )

    prop_celdas = calcular_proporciones_celdas(grid_objetivos, attrs_persona, attrs_vivienda, tipos_vivienda)
    prop_viviendas = calcular_proporciones_viviendas(viviendas, attrs_persona, attrs_vivienda, tipos_vivienda)

    distancias = calcular_distancias(
        prop_viviendas,
        prop_celdas,
        attrs_persona,
        attrs_vivienda,
        tipos_vivienda,
        pesos,
        peso_tamano_hogar,
        peso_tipo,
    )

    capacidades = grid_objetivos["objetivo_viv"].values
    probabilidades = calcular_probabilidades(distancias, capacidades, temperatura_inicial)

    if verbose:
        print("Fase 1: asignacion inicial...")

    if len(tipos_vivienda) > 0:
        asignaciones = asignar_estratificado(
            viviendas, grid_objetivos, probabilidades, tipos_viv, tipos_vivienda, rng, verbose
        )
    else:
        n_viv = probabilidades.shape[0]
        u = rng.random(n_viv)
        cumsum = np.cumsum(probabilidades, axis=1)
        asignaciones = np.array([np.searchsorted(cumsum[i], u[i]) for i in range(n_viv)])

    asignaciones = equilibrar_capacidades(
        asignaciones,
        viviendas,
        grid_objetivos,
        probabilidades,
        tipos_viv,
        tipos_vivienda,
        max_iter_equilibrado,
        tolerancia_equilibrado,
        rng,
        verbose,
    )

    if verbose:
        print("Fase 2: simulated annealing...")
    asignaciones = simulated_annealing(
        asignaciones,
        viviendas,
        grid_objetivos,
        attrs_persona,
        attrs_vivienda,
        tipos_viv,
        sa_temperatura_inicial,
        sa_temperatura_final,
        sa_iteraciones,
        verbose,
    )

    resultado = pd.DataFrame(
        {
            "id_vivienda": viviendas["id_vivienda"].values,
            "h3_cell_id": grid_objetivos.index[asignaciones],
        }
    )

    errores, _, _ = calcular_energia(
        asignaciones, viviendas, grid_objetivos, attrs_persona, attrs_vivienda, tipos_vivienda
    )

    return resultado, errores


def diagnosticar_atributos(viviendas, config, umbral_frecuencia=0.01, umbral_viviendas=0.05):
    """
    Identifica atributos demasiado raros para entrar al optimizador.

    Aplica dos umbrales: frecuencia (sobre el total de personas o viviendas) y
    fraccion de viviendas con valor mayor a cero. Retorna un DataFrame con el
    diagnostico y una version filtrada de la config.
    """
    attrs_persona, attrs_vivienda, tipos_vivienda = obtener_atributos(config)
    n_viviendas = len(viviendas)
    n_personas = (viviendas["n_hombres"] + viviendas["n_mujeres"]).sum()

    filas = []

    for attr in attrs_persona:
        total = viviendas[attr].sum()
        viv_con_valor = (viviendas[attr] > 0).sum()
        freq_personas = total / n_personas if n_personas > 0 else 0
        freq_viviendas = viv_con_valor / n_viviendas

        problemas = []
        if freq_personas < umbral_frecuencia:
            problemas.append(f"freq < {umbral_frecuencia:.0%}")
        if freq_viviendas < umbral_viviendas:
            problemas.append(f"viv < {umbral_viviendas:.0%}")

        filas.append(
            {
                "atributo": attr,
                "tipo": "persona",
                "total": total,
                "freq": freq_personas,
                "viviendas_con_valor": viv_con_valor,
                "freq_viviendas": freq_viviendas,
                "apto": len(problemas) == 0,
                "problemas": ", ".join(problemas) if problemas else "",
            }
        )

    for attr in attrs_vivienda + tipos_vivienda:
        total = viviendas[attr].sum()
        viv_con_valor = (viviendas[attr] > 0).sum()
        freq_viviendas = viv_con_valor / n_viviendas

        problemas = []
        if freq_viviendas < umbral_viviendas:
            problemas.append(f"viv < {umbral_viviendas:.0%}")

        filas.append(
            {
                "atributo": attr,
                "tipo": "vivienda" if attr in attrs_vivienda else "tipo_viv",
                "total": total,
                "freq": freq_viviendas,
                "viviendas_con_valor": viv_con_valor,
                "freq_viviendas": freq_viviendas,
                "apto": len(problemas) == 0,
                "problemas": ", ".join(problemas) if problemas else "",
            }
        )

    df = pd.DataFrame(filas).set_index("atributo")

    aptos = set(df[df["apto"]].index)
    config_filtrada = {
        "persona": {
            "prioridad_alta": [a for a in config.get("persona", {}).get("prioridad_alta", []) if a in aptos],
            "prioridad_media": [a for a in config.get("persona", {}).get("prioridad_media", []) if a in aptos],
            "prioridad_baja": [a for a in config.get("persona", {}).get("prioridad_baja", []) if a in aptos],
        },
        "vivienda": {
            "prioridad_alta": [a for a in config.get("vivienda", {}).get("prioridad_alta", []) if a in aptos],
            "prioridad_media": [a for a in config.get("vivienda", {}).get("prioridad_media", []) if a in aptos],
            "prioridad_baja": [a for a in config.get("vivienda", {}).get("prioridad_baja", []) if a in aptos],
        },
        "tipo_vivienda": [a for a in config.get("tipo_vivienda", []) if a in aptos],
    }

    return df, config_filtrada


def reportar_errores(errores, viviendas, config):
    """Tabla atributo x error_abs x error_rel ordenada por error relativo."""
    attrs_persona, attrs_vivienda, tipos_vivienda = obtener_atributos(config)

    totales = {
        "n_vp": len(viviendas),
        "n_per": (viviendas["n_hombres"] + viviendas["n_mujeres"]).sum(),
    }
    for attr in attrs_persona + attrs_vivienda + tipos_vivienda:
        totales[attr] = viviendas[attr].sum()

    filas = []
    for attr, error_abs in errores.items():
        total = totales.get(attr, 1)
        error_rel = error_abs / total if total > 0 else 0
        filas.append(
            {
                "atributo": attr,
                "total": total,
                "error_abs": error_abs,
                "error_rel": error_rel,
            }
        )

    df = pd.DataFrame(filas).set_index("atributo")
    df = df.sort_values("error_rel", ascending=False)
    return df


def visualizar_errores(errores, viviendas, config, figsize=(10, 6)):
    """Barplot horizontal de errores relativos por atributo, coloreado por prioridad."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = reportar_errores(errores, viviendas, config)
    attrs_persona, attrs_vivienda, tipos_vivienda = obtener_atributos(config)

    alta = config.get("persona", {}).get("prioridad_alta", [])
    media = config.get("persona", {}).get("prioridad_media", [])
    baja = config.get("persona", {}).get("prioridad_baja", [])

    colores = []
    for attr in df.index:
        if attr in alta:
            colores.append("#e63946")
        elif attr in media:
            colores.append("#f4a261")
        elif attr in baja:
            colores.append("#2a9d8f")
        elif attr in attrs_vivienda:
            colores.append("#457b9d")
        elif attr in tipos_vivienda:
            colores.append("#6d597a")
        else:
            colores.append("#adb5bd")

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(df))
    ax.barh(y_pos, df["error_rel"] * 100, color=colores)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Error relativo (%)")
    ax.set_title("Error por atributo")
    ax.invert_yaxis()

    ax.axvline(x=5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=10, color="gray", linestyle=":", alpha=0.5)

    legend_elements = [
        Patch(facecolor="#e63946", label="Persona: alta"),
        Patch(facecolor="#f4a261", label="Persona: media"),
        Patch(facecolor="#2a9d8f", label="Persona: baja"),
        Patch(facecolor="#457b9d", label="Vivienda"),
        Patch(facecolor="#6d597a", label="Tipo vivienda"),
        Patch(facecolor="#adb5bd", label="Totales"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    return fig, ax
