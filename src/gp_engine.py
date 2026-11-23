import operator
import random
from functools import partial
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, gp, tools


# ==============================================================================
# 1. OPERADORES ANALÍTICOS PROTEGIDOS (VECTORIZADOS CON NUMPY)
# ==============================================================================
def p_div(left, right):
    """División protegida para evitar singularidades cercanas a cero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(right) > 1e-6, np.divide(left, right), 1.0)


def p_sqrt(x):
    """Raíz cuadrada protegida sobre el valor absoluto."""
    return np.sqrt(np.abs(x))


def p_log(x):
    """Logaritmo natural protegido desplazado para evitar log(0)."""
    return np.log(np.abs(x) + 1e-6)


# ==============================================================================
# 2. CONFIGURACIÓN DEL CONJUNTO DE PRIMITIVAS (ÁRBOL DE SINTAXIS ABSTRACTA)
# ==============================================================================
def build_primitive_set():
    """
    Define los nodos funcionales y los 5 nodos terminales de carga:
      t_cc  : Duración de la etapa de corriente constante
      t_cv  : Duración de la etapa de voltaje constante
      t_max : Temperatura máxima registrada en el ciclo
      v_fix : Voltaje en un instante fijo de carga
      cycle : Número de ciclo acumulado
    """
    pset = gp.PrimitiveSet("MAIN", arity=5)
    pset.renameArguments(
        ARG0="t_cc",
        ARG1="t_cv",
        ARG2="t_max",
        ARG3="v_fix",
        ARG4="cycle",
    )

    # Conjunto de funciones F = {+, -, *, div_p, sqrt_p, log_p}
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(p_div, 2, name="pdiv")
    pset.addPrimitive(p_sqrt, 1, name="psqrt")
    pset.addPrimitive(p_log, 1, name="plog")

    # Constantes efímeras aleatorias en [-5, 5]
    pset.addEphemeralConstant("const", partial(random.uniform, -5.0, 5.0))
    return pset


# ==============================================================================
# 3. FUNCIÓN DE APTITUD CON PRESIÓN DE PARSIMONIA Y MÉTRICAS
# ==============================================================================
def evaluate_individual(individual, pset, X, y, lam=0.005, max_nodes=50):
    """
    Evalúa f(T) = RMSE(T) + lambda * C(T), penalizando árboles que excedan
    la cota operativa de nodos o generen inestabilidad numérica.
    """
    n_nodes = len(individual)
    if n_nodes > max_nodes:
        return (1e6,)

    func = gp.compile(expr=individual, pset=pset)
    try:
        y_pred = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
        # Si el árbol degenera en una constante escalar, expandir al tamaño de y
        if np.ndim(y_pred) == 0:
            y_pred = np.full_like(y, float(y_pred), dtype=np.float64)

        if not np.all(np.isfinite(y_pred)):
            return (1e6,)

        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        fitness = rmse + lam * n_nodes
        return (fitness,)
    except Exception:
        return (1e6,)


def compute_metrics(individual, pset, X, y):
    """Calcula RMSE, MAE, MAPE, R2 y complejidad estructural sobre un conjunto."""
    func = gp.compile(expr=individual, pset=pset)
    y_pred = func(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    if np.ndim(y_pred) == 0:
        y_pred = np.full_like(y, float(y_pred), dtype=np.float64)

    # Acotamiento físico del SoH al rango [0, 100]%
    y_pred_clipped = np.clip(y_pred, 0.0, 100.0)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred_clipped)))
    mae = float(mean_absolute_error(y, y_pred_clipped))
    mape = float(np.mean(np.abs((y - y_pred_clipped) / np.maximum(y, 1e-6))) * 100.0)
    r2 = float(r2_score(y, y_pred_clipped))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Nodes": len(individual),
    }


def extract_pareto_front(population, pset, X, y):
    """
    Extrae el conjunto de individuos no dominados en el espacio (RMSE, Nodos).
    A domina a B si RMSE(A) <= RMSE(B) y C(A) <= C(B) con al menos una
    desigualdad estricta.
    """
    candidates = []
    seen_exprs = set()

    for ind in population:
        expr_str = str(ind)
        if expr_str in seen_exprs:
            continue
        seen_exprs.add(expr_str)
        metrics = compute_metrics(ind, pset, X, y)
        candidates.append((ind, metrics["RMSE"], metrics["Nodes"]))

    pareto = []
    for i, (ind_a, err_a, comp_a) in enumerate(candidates):
        dominated = False
        for j, (_, err_b, comp_b) in enumerate(candidates):
            if i == j:
                continue
            if (err_b <= err_a and comp_b <= comp_a) and (err_b < err_a or comp_b < comp_a):
                dominated = True
                break
        if not dominated:
            pareto.append((ind_a, err_a, comp_a))

    return sorted(pareto, key=lambda item: item[2])  # Ordenado por complejidad


# ==============================================================================
# 4. BUCLE EVOLUTIVO CON OPERADORES ADAPTADOS Y ELITISMO
# ==============================================================================
def run_gp_evolution(
    X_train,
    y_train,
    pop_size=1500,
    n_gen=80,
    tournsize=10,
    cxpb=0.70,
    mut_subtree_pb=0.10,
    mut_node_pb=0.10,
    mut_ephemeral_pb=0.05,
    lam=0.005,
    max_depth=6,
    max_nodes=50,
    elitism=5,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    pset = build_primitive_set()

    # Registro seguro de tipos en DEAP (evita advertencias en corridas múltiples)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_individual,
        pset=pset,
        X=X_train,
        y=y_train,
        lam=lam,
        max_nodes=max_nodes,
    )
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mut_subtree", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mut_node", gp.mutNodeReplacement, pset=pset)
    toolbox.register("mut_eph", gp.mutEphemeral, mode="one")

    # Límites estrictos de altura para prevenir hipertrofia (bloat)
    limit_height = gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth)
    toolbox.decorate("mate", limit_height)
    toolbox.decorate("mut_subtree", limit_height)

    pop = toolbox.population(n=pop_size)

    # Evaluación inicial
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(10)
    hof.update(pop)

    for gen in range(1, n_gen + 1):
        # Preservación de élite
        elites = [toolbox.clone(ind) for ind in tools.selBest(pop, elitism)]

        # Selección por torneo
        offspring = toolbox.select(pop, len(pop) - elitism)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Cruce de subárboles (cxpb = 0.70)
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutaciones diferenciadas (0.10 subárbol / 0.10 nodo / 0.05 constante)
        for i in range(len(offspring)):
            r = random.random()
            if r < mut_subtree_pb:
                (offspring[i],) = toolbox.mut_subtree(offspring[i])
                del offspring[i].fitness.values
            elif r < mut_subtree_pb + mut_node_pb:
                (offspring[i],) = toolbox.mut_node(offspring[i])
                del offspring[i].fitness.values
            elif r < mut_subtree_pb + mut_node_pb + mut_ephemeral_pb:
                (offspring[i],) = toolbox.mut_eph(offspring[i])
                del offspring[i].fitness.values

        # Reevaluación únicamente de individuos modificados
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring + elites
        hof.update(pop)

    pareto_front = extract_pareto_front(pop + list(hof), pset, X_train, y_train)
    return hof[0], pareto_front, pset


# ==============================================================================
# 5. PROTOCOLO DE VALIDACIÓN LEAVE-ONE-BATTERY-OUT (LOBO)
# ==============================================================================
def run_lobo_validation(df: pd.DataFrame, feature_cols, target_col="SoH", seed=42):
    """
    Ejecuta validación cruzada por batería completa.
    Normaliza exclusivamente sobre el subconjunto de entrenamiento para evitar
    fuga de datos (data leakage).
    """
    batteries = df["battery_id"].unique()
    results = []

    for test_bat in batteries:
        train_df = df[df["battery_id"] != test_bat]
        test_df = df[df["battery_id"] == test_bat]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        y_train = train_df[target_col].values

        X_test = scaler.transform(test_df[feature_cols].values)
        y_test = test_df[target_col].values

        best_ind, pareto, pset = run_gp_evolution(X_train, y_train, seed=seed)

        train_metrics = compute_metrics(best_ind, pset, X_train, y_train)
        test_metrics = compute_metrics(best_ind, pset, X_test, y_test)

        results.append({
            "test_battery": test_bat,
            "expression": str(best_ind),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "pareto_size": len(pareto),
            "scaler_min": scaler.data_min_,
            "scaler_max": scaler.data_max_,
        })

    return results
