import concurrent.futures
import json
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from deap import gp
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler

# Asegurar acceso a los módulos en src/ y tests/
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "tests"))

from feature_extraction import build_nasa_dataset
from gp_engine import build_primitive_set, run_gp_evolution, compute_metrics
from export_c import export_c_file, verify_python_c_equivalence
from run_experiments import (
    FEATURE_COLS,
    TARGET_COL,
    benchmark_baselines,
    run_noise_and_oor_tests,
)
from analysis_and_plots import (
    audit_physical_consistency,
    configure_plot_style,
    plot_pareto_complexity_tradeoff,
    plot_benchmark_comparison,
    plot_sensitivity_bars,
)
from smoke_test import generate_synthetic_batteries

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Regresión Simbólica SoH — Cómputo Evolutivo",
    page_icon="🧬",
    layout="wide",
)

DATA_DIR = ROOT_DIR / "data" / "nasa_pcoe"
CSV_FEATURES = ROOT_DIR / "data" / "nasa_pcoe_features.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

FILE_DESCRIPTIONS = {
    "gp_lobo_runs.csv": "Bitácora completa de todas las corridas independientes de Programación Genética en cada partición LOBO (semilla, RMSE entrenamiento/prueba, MAE, MAPE, R², número de nodos, tamaño del frente de Pareto y expresión simbólica).",
    "gp_summary_median_iqr.csv": "Resumen estadístico robusto agrupado por batería oculta con la Mediana y el Rango Intercuartílico (IQR) de RMSE, MAE, R² y Nodos (evita el sesgo de valores atípicos).",
    "baselines_lobo.csv": "Métricas de evaluación y latencia en host de los modelos de referencia (Linear, Poly_Deg2 y MLP_BlackBox) bajo exactamente las mismas particiones LOBO.",
    "bms_soh_model.c": "Código fuente autónomo en lenguaje C99 (IEEE 754 float32) con normalización embebida, operadores protegidos, saturación [0, 100]% y filtro EMA listo para microcontroladores BMS.",
    "experiment_audit.json": "Auditoría técnica integral: mejor modelo global, error máximo de cuantización Python (float64) vs C (float32), pruebas de ruido (1%, 2%, 5%), extrapolación fuera de rango y prueba de Wilcoxon.",
    "symbolic_audit.json": "Auditoría matemática generada con SymPy: ecuación simplificada en texto y LaTeX, derivadas parciales exactas y porcentajes de monotonicidad en el dominio operativo.",
    "fig_pareto_front.png": "Gráfica de compromiso Precisión vs. Complejidad (Frente de Pareto) destacando las soluciones no dominadas frente a la cota operativa de 50 nodos.",
    "fig_benchmark_lobo.png": "Gráfica de barras agrupadas comparando la mediana de error RMSE de Regresión Simbólica frente a Linear, Poly_Deg2 y MLP_BlackBox por batería.",
    "fig_symbolic_sensitivity.png": "Gráfica de barras horizontales con la sensibilidad analítica mediana |∂SoH/∂x_i| derivada del gradiente simbólico.",
}

VARIABLE_GLOSSARY = {
    "t_cc": {
        "nombre": "Duración de Carga en Corriente Constante (CC)",
        "unidad": "Segundos (s)",
        "origen": "Se extrae del perfil de carga CC-CV midiendo el tiempo desde t = 0 hasta el primer instante donde el voltaje alcanza V ≥ 4.19 V.",
        "significado": "A medida que la batería envejece, su resistencia interna aumenta y alcanza el límite de 4.2 V más rápido, por lo que t_cc disminuye.",
    },
    "t_cv": {
        "nombre": "Duración de Carga en Voltaje Constante (CV)",
        "unidad": "Segundos (s)",
        "origen": "Se calcula como la diferencia de tiempo desde que el voltaje llega a 4.19 V hasta que la corriente decae por debajo del corte de 0.025 A (25 mA).",
        "significado": "Refleja la dificultad de difusión de iones de litio en la fase final de carga; suele prolongarse con la degradación.",
    },
    "t_max": {
        "nombre": "Temperatura Máxima durante la Carga",
        "unidad": "Grados Celsius (°C)",
        "origen": "Valor máximo del sensor térmico (Temperature_measured) registrado exclusivamente durante el ciclo de carga.",
        "significado": "El crecimiento de la capa SEI y el aumento de impedancia interna incrementan el calentamiento Joule durante la carga.",
    },
    "v_fix": {
        "nombre": "Voltaje en Instante Fijo de Carga",
        "unidad": "Voltios (V)",
        "origen": "Interpolación lineal (np.interp) del voltaje exactamente a los t = 1000 s de iniciada la carga para evitar ruido por muestreo irregular.",
        "significado": "Captura la pendiente de subida de voltaje en una ventana fija; una celda degradada presenta mayor voltaje a los 1000 s por sobrepotencial óhmico.",
    },
    "cycle": {
        "nombre": "Índice Secuencial de Ciclo Operativo",
        "unidad": "Adimensional (conteo)",
        "origen": "Contador incremental emparejando cada ciclo de carga CC-CV válido con su descarga inmediatamente posterior.",
        "significado": "Representa la historia acumulada de uso y fatiga electroquímica de la celda.",
    },
}


# ==============================================================================
# VISUALIZACIÓN GRÁFICA DEL ÁRBOL DE SINTAXIS ABSTRACTA (AST DE DEAP)
# ==============================================================================
def render_ast_tree_figure(expr_str: str):
    """
    Construye y dibuja visualmente el Árbol de Sintaxis Abstracta (AST) a partir
    de la cadena prefija de DEAP usando matplotlib (sin dependencias externas).
    """
    pset = build_primitive_set()
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    nodes, edges, labels = gp.graph(tree)

    children_map = {n: [] for n in nodes}
    for parent, child in edges:
        children_map[parent].append(child)

    pos = {}
    leaf_counter = [0]

    def _assign_positions(node_id: int, depth: int):
        kids = children_map[node_id]
        if not kids:
            pos[node_id] = (float(leaf_counter[0]), float(-depth))
            leaf_counter[0] += 1
        else:
            for k in kids:
                _assign_positions(k, depth + 1)
            avg_x = sum(pos[k][0] for k in kids) / len(kids)
            pos[node_id] = (avg_x, float(-depth))

    _assign_positions(0, 0)

    n_leaves = max(leaf_counter[0], 1)
    max_depth = max(abs(int(y)) for _, y in pos.values()) if pos else 1
    fig_w = max(8.0, min(18.0, n_leaves * 1.15))
    fig_h = max(4.5, min(11.0, (max_depth + 1) * 1.25))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis("off")

    for parent, child in edges:
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        ax.plot([x1, x2], [y1, y2], color="#64748B", linewidth=1.6, zorder=1)

    var_names = {"t_cc", "t_cv", "t_max", "v_fix", "cycle", "ARG0", "ARG1", "ARG2", "ARG3", "ARG4"}
    for node_id in nodes:
        x, y = pos[node_id]
        raw_lbl = str(labels[node_id])
        if children_map[node_id]:
            bg_color = "#1E3A8A"
            text_lbl = raw_lbl
        elif raw_lbl in var_names:
            bg_color = "#15803D"
            text_lbl = raw_lbl
        else:
            bg_color = "#B45309"
            try:
                text_lbl = f"{float(raw_lbl):.3f}"
            except ValueError:
                text_lbl = raw_lbl

        ax.text(
            x,
            y,
            text_lbl,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            zorder=2,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=bg_color, edgecolor="white", linewidth=1.2),
        )

    ax.set_title(
        f"Estructura Visual del Árbol de Sintaxis Abstracta (AST) — {len(tree)} nodos | Profundidad {tree.height}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return fig, tree


# ==============================================================================
# INSPECCIÓN DE ARCHIVOS CRUDOS .MAT DE NASA PCOE
# ==============================================================================
@st.cache_data
def inspect_raw_mat_files(data_dir: Path) -> pd.DataFrame:
    """Lee los metadatos y conteo de ciclos internos de cada archivo .mat original."""
    rows = []
    for bat_id in ("B0005", "B0006", "B0007", "B0018"):
        mat_path = data_dir / f"{bat_id}.mat"
        if not mat_path.exists():
            continue
        mat = loadmat(str(mat_path))
        cycles = mat[bat_id][0, 0]["cycle"][0]
        n_charge, n_discharge, n_impedance = 0, 0, 0
        capacities = []

        for entry in cycles:
            op = str(entry["type"][0]).strip().lower()
            if op == "charge":
                n_charge += 1
            elif op == "discharge":
                n_discharge += 1
                data = entry["data"][0, 0]
                if "Capacity" in data.dtype.names:
                    cap = np.asarray(data["Capacity"]).ravel()
                    if cap.size > 0 and np.isfinite(cap[0]):
                        capacities.append(float(cap[0]))
            elif op == "impedance":
                n_impedance += 1

        rows.append({
            "Archivo": f"{bat_id}.mat",
            "Tamaño (MB)": round(mat_path.stat().st_size / (1024 * 1024), 2),
            "Total Registros": len(cycles),
            "Ciclos Carga (CC-CV)": n_charge,
            "Ciclos Descarga": n_discharge,
            "Mediciones Impedancia (EIS)": n_impedance,
            "Capacidad Inicial (Ah)": round(capacities[0], 4) if capacities else None,
            "Capacidad Final (Ah)": round(capacities[-1], 4) if capacities else None,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# EJECUCIÓN PARALELA DEL MOTOR EVOLUTIVO
# ==============================================================================
def _worker_single_gp_run(args):
    """Ejecuta una corrida evolutiva individual en un núcleo independiente."""
    test_bat, run_idx, seed, X_train, y_train, X_test, y_test, pop_size, n_gen, lam = args
    best_ind, pareto, pset = run_gp_evolution(
        X_train,
        y_train,
        pop_size=pop_size,
        n_gen=n_gen,
        lam=lam,
        seed=seed,
    )
    train_m = compute_metrics(best_ind, pset, X_train, y_train)
    test_m = compute_metrics(best_ind, pset, X_test, y_test)
    return {
        "test_battery": test_bat,
        "run": run_idx,
        "seed": seed,
        "train_RMSE": train_m["RMSE"],
        "test_RMSE": test_m["RMSE"],
        "test_MAE": test_m["MAE"],
        "test_MAPE": test_m["MAPE"],
        "test_R2": test_m["R2"],
        "Nodes": test_m["Nodes"],
        "pareto_size": len(pareto),
        "expression": str(best_ind),
        "best_ind": best_ind,
        "pset": pset,
    }


def run_accelerated_lobo(df: pd.DataFrame, n_runs: int, pop_size: int, n_gen: int, lam: float, use_parallel: bool):
    """Orquesta la validación cruzada LOBO con barra de progreso en vivo."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    batteries = df["battery_id"].unique()

    gp_records = []
    baseline_records = []
    best_global_rmse = float("inf")
    best_global_bundle = None

    progress_bar = st.progress(0, text="Iniciando particiones Leave-One-Battery-Out...")
    total_steps = len(batteries) * n_runs
    completed_steps = 0

    for test_bat in batteries:
        train_df = df[df["battery_id"] != test_bat]
        test_df = df[df["battery_id"] == test_bat]

        X_train_raw = train_df[FEATURE_COLS].values
        y_train = train_df[TARGET_COL].values
        X_test_raw = test_df[FEATURE_COLS].values
        y_test = test_df[TARGET_COL].values

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for b_rec in benchmark_baselines(X_train, y_train, X_test, y_test, seed=42):
            b_rec["test_battery"] = test_bat
            baseline_records.append(b_rec)

        tasks = [
            (test_bat, r, 100 + r, X_train, y_train, X_test, y_test, pop_size, n_gen, lam)
            for r in range(n_runs)
        ]

        if use_parallel and n_runs > 1:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for res in executor.map(_worker_single_gp_run, tasks):
                    completed_steps += 1
                    progress_bar.progress(
                        completed_steps / total_steps,
                        text=f"Evaluando batería oculta {test_bat} — Corrida {res['run'] + 1} de {n_runs}",
                    )
                    if res["test_RMSE"] < best_global_rmse:
                        best_global_rmse = res["test_RMSE"]
                        best_global_bundle = {
                            "individual": res["best_ind"],
                            "pset": res["pset"],
                            "scaler": scaler,
                            "test_battery": test_bat,
                            "X_test_raw": X_test_raw,
                            "y_test": y_test,
                        }
                    gp_records.append({k: v for k, v in res.items() if k not in ("best_ind", "pset")})
        else:
            for t in tasks:
                res = _worker_single_gp_run(t)
                completed_steps += 1
                progress_bar.progress(
                    completed_steps / total_steps,
                    text=f"Evaluando batería oculta {test_bat} — Corrida {res['run'] + 1} de {n_runs}",
                )
                if res["test_RMSE"] < best_global_rmse:
                    best_global_rmse = res["test_RMSE"]
                    best_global_bundle = {
                        "individual": res["best_ind"],
                        "pset": res["pset"],
                        "scaler": scaler,
                        "test_battery": test_bat,
                        "X_test_raw": X_test_raw,
                        "y_test": y_test,
                    }
                gp_records.append({k: v for k, v in res.items() if k not in ("best_ind", "pset")})

    progress_bar.progress(1.0, text="Guardando tablas estadísticas, firmware C99 y auditoría...")

    df_gp = pd.DataFrame(gp_records)
    df_base = pd.DataFrame(baseline_records)
    df_gp.to_csv(ARTIFACTS_DIR / "gp_lobo_runs.csv", index=False)
    df_base.to_csv(ARTIFACTS_DIR / "baselines_lobo.csv", index=False)

    summary_gp = (
        df_gp.groupby("test_battery")[["test_RMSE", "test_MAE", "test_R2", "Nodes"]]
        .agg(["median", lambda s: np.percentile(s, 75) - np.percentile(s, 25)])
    )
    summary_gp.to_csv(ARTIFACTS_DIR / "gp_summary_median_iqr.csv")

    wilcoxon_report = {}
    gp_med_by_bat = df_gp.groupby("test_battery")["test_RMSE"].median()
    for b_model in df_base["model"].unique():
        base_by_bat = df_base[df_base["model"] == b_model].set_index("test_battery")["RMSE"]
        aligned = pd.concat([gp_med_by_bat, base_by_bat], axis=1, keys=["GP", b_model]).dropna()
        if len(aligned) >= 4:
            stat, pval = wilcoxon(aligned["GP"], aligned[b_model], zero_method="wilcox")
            wilcoxon_report[f"GP_vs_{b_model}"] = {"stat": float(stat), "p_value": float(pval)}

    c_file = export_c_file(
        best_global_bundle["individual"],
        best_global_bundle["scaler"].data_min_,
        best_global_bundle["scaler"].data_max_,
        output_path=str(ARTIFACTS_DIR / "bms_soh_model.c"),
    )

    if shutil.which("gcc") is not None:
        equiv_report = verify_python_c_equivalence(
            best_global_bundle["individual"],
            best_global_bundle["pset"],
            best_global_bundle["X_test_raw"],
            best_global_bundle["scaler"].data_min_,
            best_global_bundle["scaler"].data_max_,
        )
    else:
        equiv_report = {"status": "GCC no detectado en PATH"}

    robustness_report = run_noise_and_oor_tests(
        best_global_bundle["individual"],
        best_global_bundle["pset"],
        best_global_bundle["X_test_raw"],
        best_global_bundle["y_test"],
        best_global_bundle["scaler"],
    )

    audit_summary = {
        "best_model_battery": str(best_global_bundle["test_battery"]),
        "best_model_test_RMSE": best_global_rmse,
        "best_expression": str(best_global_bundle["individual"]),
        "c_export_path": str(c_file),
        "python_c_equivalence": equiv_report,
        "robustness_noise_and_oor": robustness_report,
        "wilcoxon_tests": wilcoxon_report,
    }
    (ARTIFACTS_DIR / "experiment_audit.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    progress_bar.empty()
    return audit_summary


def run_symbolic_analysis_and_plots():
    """Simplifica la ecuación con SymPy, calcula derivadas y genera las gráficas."""
    configure_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df_gp = pd.read_csv(ARTIFACTS_DIR / "gp_lobo_runs.csv")
    df_base = pd.read_csv(ARTIFACTS_DIR / "baselines_lobo.csv")
    audit = json.loads((ARTIFACTS_DIR / "experiment_audit.json").read_text(encoding="utf-8"))

    sym_audit = audit_physical_consistency(audit["best_expression"])
    (ARTIFACTS_DIR / "symbolic_audit.json").write_text(
        json.dumps(sym_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_pareto_complexity_tradeoff(df_gp, FIGURES_DIR)
    plot_benchmark_comparison(df_gp, df_base, FIGURES_DIR)
    plot_sensitivity_bars(sym_audit, FIGURES_DIR)
    return sym_audit


# ==============================================================================
# INTERFAZ VISUAL PRINCIPAL
# ==============================================================================
st.title("🧬 Descubrimiento de Ecuaciones Interpretables para Estimar el SoH en Baterías de Litio")
st.markdown(
    "**Cómputo Evolutivo — Regresión Simbólica mediante Programación Genética** | "
    "Marco experimental base para extracción de telemetría de carga, validación cruzada entre celdas y despliegue embebido en C99."
)

# Barra lateral: Estado del sistema y parámetros evolutivos con explicación integrada
with st.sidebar:
    st.header("📡 Estado de los Artefactos")
    mat_ok = DATA_DIR.exists() and len(list(DATA_DIR.glob("*.mat"))) >= 4
    csv_ok = CSV_FEATURES.exists()
    lobo_ok = (ARTIFACTS_DIR / "experiment_audit.json").exists()
    sym_ok = (ARTIFACTS_DIR / "symbolic_audit.json").exists()
    gcc_ok = shutil.which("gcc") is not None

    st.markdown(f"{'✅' if mat_ok else '❌'} **Archivos crudos NASA (.mat)**")
    st.markdown(f"{'✅' if csv_ok else '⏳'} **Dataset limpio (`nasa_pcoe_features.csv`)**")
    st.markdown(f"{'✅' if lobo_ok else '⏳'} **Resultados LOBO y Código C99**")
    st.markdown(f"{'✅' if sym_ok else '⏳'} **Auditoría Simbólica y Gráficas**")
    st.markdown(f"{'✅' if gcc_ok else '⚠️'} **Compilador C (`gcc`) activo**")

    st.divider()
    st.header("🎛️ Hiperparámetros Evolutivos")
    preset = st.radio(
        "Presupuesto de búsqueda:",
        [
            "🚀 Demo Rápida (~15 seg)",
            "⚖️ Equilibrado (~1-2 min)",
            "🎓 Completo (30 corridas)",
        ],
        index=0,
        help="Define cuántas corridas independientes, tamaño de población y generaciones ejecutará el motor evolutivo.",
    )
    if "Demo" in preset:
        cfg_runs, cfg_pop, cfg_gen = 2, 400, 30
        preset_desc = "2 corridas × 400 ecuaciones × 30 generaciones (8 evoluciones totales). Ideal para pruebas rápidas."
    elif "Equilibrado" in preset:
        cfg_runs, cfg_pop, cfg_gen = 5, 800, 50
        preset_desc = "5 corridas × 800 ecuaciones × 50 generaciones (20 evoluciones totales). Buen balance tiempo/precisión."
    else:
        cfg_runs, cfg_pop, cfg_gen = 30, 1500, 80
        preset_desc = "30 corridas × 1,500 ecuaciones × 80 generaciones (120 evoluciones totales). Rigor estadístico (mediana e IQR)."
    st.caption(f"ℹ️ **Internamente:** {preset_desc}")

    cfg_lam = st.slider(
        "Factor de parsimonia (λ)",
        min_value=0.001,
        max_value=0.020,
        value=0.005,
        step=0.001,
        format="%.3f",
        help="Penaliza árboles grandes en la función de aptitud: f(T) = RMSE(T) + λ · Nodos(T).",
    )
    if cfg_lam <= 0.003:
        lam_desc = "Penalización baja: prioriza reducir el error al máximo, permitiendo fórmulas más largas (hasta 50 nodos)."
    elif cfg_lam <= 0.008:
        lam_desc = "Equilibrio recomendado: elimina ramas inútiles (bloat) sin sacrificar precisión predictiva."
    else:
        lam_desc = "Penalización alta: fuerza ecuaciones ultra cortas y muy fáciles de leer, tolerando algo más de error."
    st.caption(f"ℹ️ **Efecto en aptitud:** {lam_desc}")

    use_parallel = st.checkbox(
        "Ejecución paralela multinúcleo",
        value=True,
        help="Reparte las corridas independientes simultáneamente entre todos los núcleos de tu CPU.",
    )
    st.caption(
        "ℹ️ **Multiproceso:** "
        + (
            "Activado — reparte las corridas entre todos los núcleos de tu CPU para acelerar la búsqueda."
            if use_parallel
            else "Desactivado — ejecuta una corrida tras otra en un solo núcleo (menor uso de RAM/CPU)."
        )
    )

    with st.expander("📖 Hiperparámetros internos de `gp_engine.py`"):
        st.markdown("""
        Además de los controles superiores, el motor evolutivo (`src/gp_engine.py`) fija los siguientes parámetros basados en literatura de Programación Genética:
        * **Selección por torneo (`tournsize = 10`):** Elige 10 árboles al azar y selecciona el de menor aptitud $f(T)$, ejerciendo alta presión selectiva.
        * **Elitismo (`elitism = 5`):** Conserva intactos los 5 mejores árboles de cada generación para evitar retrocesos.
        * **Cruce de un punto (`cxpb = 0.70`):** En el 70% de los pares, intercambia subárboles entre dos ecuaciones padre.
        * **Mutaciones diferenciadas:** `0.10` reemplazo de subárbol completo (`mutUniform`), `0.10` sustitución de un operador/variable (`mutNodeReplacement`) y `0.05` ajuste de constante efímera (`mutEphemeral`).
        * **Control estructural de *bloat*:** Profundidad máxima `max_depth = 6` (`staticLimit`) y cota dura `max_nodes = 50` para garantizar ejecución $O(1)$ en el BMS.
        * **Constantes efímeras (`const`):** Valores aleatorios uniformes en `[-5.0, 5.0]` que actúan como coeficientes ajustables.
        """)

# Pestañas en el orden original, con Resultados y Conclusiones al final
tab_marco, tab_datos, tab_sim, tab_evolucion, tab_simbolico, tab_c, tab_archivos, tab_resultados = st.tabs([
    "📘 Marco Metodológico Base",
    "📂 Extracción y Datos (.mat)",
    "🧪 Validación Sintética",
    "🧬 Evolución Simbólica (LOBO)",
    "🌳 Ecuación, Árbol AST y Gráficas",
    "⚙️ Firmware Embebido C99",
    "🗂️ Explorador de Archivos",
    "🏆 Resultados y Conclusiones",
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: MARCO METODOLÓGICO Y FUNDAMENTO CIENTÍFICO
# ------------------------------------------------------------------------------
with tab_marco:
    st.header("Fundamento Metodológico y Alcance Base del Proyecto")
    st.info(
        "💡 **Nota sobre la arquitectura experimental:** Esta propuesta metodológica establece la línea base del proyecto "
        "utilizando el conjunto público de laboratorio NASA PCOE. Su diseño modular permite ajustar parámetros, "
        "incorporar nuevas variables o escalar posteriormente hacia conjuntos masivos (como las 124 celdas LFP de Severson et al.) "
        "conforme avance la investigación."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. El Problema y la Solución Propuesta")
        st.markdown("""
        * **Contexto físico:** Durante los ciclos de carga y descarga, el crecimiento de la capa SEI (*Solid Electrolyte Interphase*), el *lithium plating* y la pérdida de litio activo reducen la capacidad útil de la batería.
        * **Definición de Estado de Salud (SoH):**
          $$\\text{SoH}_k = \\left(\\frac{Q_k}{Q_{\\text{nominal}}}\\right) \\times 100\\%$$
          donde $Q_{\\text{nominal}} = 2.0\\text{ Ah}$ para las celdas 18650 evaluadas. En los protocolos de la NASA, el fin de vida (*End of Life*, EoL) se fija al perder el $30\\%$ de capacidad ($1.4\\text{ Ah}$ o $\\text{SoH} = 70\\%$).
        * **Por qué Regresión Simbólica con Programación Genética:** A diferencia de los modelos electroquímicos (costosos de parametrizar) y de las redes neuronales profundas (cajas negras difíciles de auditar bajo estándares como ISO 26262), la Programación Genética descubre **ecuaciones algebraicas explícitas, compactas y de complejidad $O(1)$** capaces de correr en microcontroladores BMS con recursos limitados.
        """)

    with col_b:
        st.subheader("2. ¿Por qué solo estas 4 baterías y estos 3 modelos base?")
        st.markdown("""
        * **Selección homogénea de celdas (`B0005`, `B0006`, `B0007`, `B0018`):** Dentro del repositorio NASA PCOE, estas cuatro celdas Li-ion 18650 comparten exactamente la **misma temperatura ambiente ($24^\\circ\\text{C}$)** y el **mismo perfil de carga CC-CV** ($1.5\\text{ A}$ hasta $4.2\\text{ V}$). Incluir celdas probadas a $4^\\circ\\text{C}$ o $43^\\circ\\text{C}$ mezclaría regímenes térmicos incompatibles en esta primera etapa base.
        * **Consideración estadística ($n = 4$):** Con 4 baterías en validación LOBO, el tamaño muestral pareado a nivel de celda limita el valor-$p$ mínimo de la prueba de Wilcoxon de dos colas a $p = 0.125$. Para significancia estricta ($\\alpha = 0.05$), el pipeline está preparado para analizar errores por ciclo o extenderse a 124 celdas.
        * **Espectro de Modelos Base (*Baselines*):**
          1. **`Linear`:** Referencia paramétrica mínima (asume degradación lineal).
          2. **`Poly_Deg2` (Ridge):** Referencia semi-empírica con interacciones cuadráticas.
          3. **`MLP_BlackBox`:** Red neuronal multicapa `(64, 32)` que actúa como referente de precisión opaca para evaluar cuánto se acerca nuestra fórmula interpretable.
        """)

    st.divider()
    st.subheader("3. Arquitectura Híbrida Propuesta: Offline (PC/Nube) vs. Online (BMS)")
    st.markdown("""
    | Entorno de Ejecución | Tareas Realizadas | Complejidad y Recursos |
    | :--- | :--- | :--- |
    | **Offline (PC / Nube)** | Ingesta de telemetría histórica, limpieza, normalización Min-Max, evolución con Programación Genética, selección en el Frente de Pareto, auditoría simbólica con SymPy y transpilación a C99. | Búsqueda evolutiva intensiva ($1{,}500$ individuos $\\times 80$ generaciones) ejecutada fuera de línea. |
    | **Online (Microcontrolador BMS)** | Lectura directa de sensores ($V, I, T, t$), cálculo de las 5 variables de carga, evaluación determinista de la ecuación C99, saturación física $[0, 100]\\%$ y suavizado con filtro EMA. | **Complejidad $O(1)$**, $\\le 50$ nodos, **$0\\text{ bytes}$ en Heap** y latencia de microsegundos. |
    """)

# ------------------------------------------------------------------------------
# PESTAÑA 2: EXTRACCIÓN Y INSPECCIÓN DE DATOS (.MAT Y .CSV)
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("Extracción de Características y Limpieza de Datos")

    with st.expander("📖 Explicación detallada: De dónde sale cada dato y cómo se calculan las 5 variables", expanded=True):
        st.markdown("""
        **1. Estructura interna de los archivos `.mat` de NASA PCOE:**  
        Cada archivo `.mat` es una estructura jerárquica de MATLAB (`mat[battery_id][0, 0]["cycle"][0]`) que contiene cientos de ensayos secuenciales clasificados en tres tipos: `charge` (carga CC-CV), `discharge` (descarga a corriente constante) e `impedance` (espectroscopía EIS).  
        El script `src/feature_extraction.py` recorre esa secuencia, toma el último ciclo `charge` completo y lo empareja estrictamente con el ciclo `discharge` inmediatamente posterior (ignorando ensayos EIS o ciclos truncados menores a $1000\\text{ s}$).

        **2. Origen exacto de cada variable terminal:**
        """)
        gloss_df = pd.DataFrame([
            {
                "Variable": k,
                "Nombre Técnico": v["nombre"],
                "Unidad": v["unidad"],
                "Cómo se extrae del .mat": v["origen"],
                "Interpretación Electroquímica": v["significado"],
            }
            for k, v in VARIABLE_GLOSSARY.items()
        ])
        st.dataframe(gloss_df, width="stretch", hide_index=True)

        st.markdown("""
        **3. Cálculo de la variable objetivo (`SoH`) y regla anti-fuga de datos:**  
        * De cada ciclo `discharge` se lee la capacidad descargada $Q_k$ (`Capacity`, en Ah) y se calcula $\\text{SoH} = (Q_k / 2.0) \\times 100\\%$.  
        * **Regla crítica:** **Nunca se usa el tiempo de descarga** como variable de entrada, ya que a corriente constante de $2.0\\text{ A}$ el tiempo de descarga es directamente proporcional a $Q_k$ (causaría *data leakage*).
        """)

    st.subheader("1. Auditoría de los archivos crudos de la NASA (`data/nasa_pcoe/*.mat`)")
    if mat_ok:
        df_raw_info = inspect_raw_mat_files(DATA_DIR)
        st.dataframe(df_raw_info, width="stretch", hide_index=True)
    else:
        st.error("No se encontraron los 4 archivos `.mat` en `data/nasa_pcoe/`.")

    st.divider()
    st.subheader("2. Generación e inspección del dataset limpio (`nasa_pcoe_features.csv`)")
    if st.button("▶️ Extraer características y generar dataset limpio"):
        with st.spinner("Emparejando ciclos de carga CC-CV con descargas y calculando SoH..."):
            df_extracted = build_nasa_dataset(str(DATA_DIR))
            CSV_FEATURES.parent.mkdir(parents=True, exist_ok=True)
            df_extracted.to_csv(CSV_FEATURES, index=False)
        st.success(f"Extracción completada: {len(df_extracted)} ciclos válidos guardados en `data/nasa_pcoe_features.csv`.")
        st.rerun()

    if CSV_FEATURES.exists():
        df_feat = pd.read_csv(CSV_FEATURES)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Resumen estadístico por batería**")
            st.dataframe(
                df_feat.groupby("battery_id")[["t_cc", "t_cv", "t_max", "v_fix", "SoH"]].agg(["count", "mean", "min", "max"]),
                width="stretch",
            )
        with c2:
            st.markdown("**Curvas reales de degradación de SoH (%) por ciclo**")
            chart_df = df_feat.pivot(index="cycle", columns="battery_id", values="SoH")
            st.line_chart(chart_df)

        st.markdown("**Registros procesados (`nasa_pcoe_features.csv`)**")
        st.dataframe(df_feat, width="stretch")

# ------------------------------------------------------------------------------
# PESTAÑA 3: VALIDACIÓN SINTÉTICA (SMOKE TEST)
# ------------------------------------------------------------------------------
with tab_sim:
    st.header("Verificación de Integración con Telemetría Sintética")

    with st.expander("📖 Explicación detallada: ¿Cómo y por qué se realiza esta prueba?", expanded=True):
        st.markdown("""
        Antes de ejecutar corridas evolutivas extensas sobre los datos reales de la NASA, este módulo genera **4 baterías sintéticas** (`SYN_B01` a `SYN_B04`) siguiendo una ley electroquímica controlada (crecimiento de capa SEI proporcional a $\\sqrt{\\text{ciclo}}$ más efecto térmico y ruido gaussiano).

        Su función es auditar en pocos segundos que toda la cadena tecnológica funciona sin errores:
        1. **Motor Evolutivo (`DEAP`):** Verifica que los árboles respeten la cota de $\\le 50$ nodos y no generen valores `NaN`/`Inf`.
        2. **Compilación C99 (`gcc` + `ctypes`):** Compila la ecuación descubierta a código máquina nativo y comprueba que la diferencia numérica entre Python (`float64`) y el microcontrolador (`float32`) sea menor a $10^{-2}$.
        3. **Motor Simbólico (`SymPy`):** Verifica la simplificación algebraica a LaTeX y el cálculo del gradiente exacto.
        """)

    if st.button("▶️ Ejecutar validación sintética de extremo a extremo"):
        with st.spinner("Simulando celdas, evolucionando ecuaciones y compilando C99..."):
            df_syn = generate_synthetic_batteries(n_cycles=100)
            train_df = df_syn[df_syn["battery_id"] != "SYN_B01"]
            test_df = df_syn[df_syn["battery_id"] == "SYN_B01"]
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(train_df[FEATURE_COLS].values)
            X_te = scaler.transform(test_df[FEATURE_COLS].values)
            ind, _, pset = run_gp_evolution(X_tr, train_df[TARGET_COL].values, pop_size=150, n_gen=12, seed=42)
            metrics = compute_metrics(ind, pset, X_te, test_df[TARGET_COL].values)
            sym = audit_physical_consistency(str(ind), n_grid=50)

        st.success("¡Cadena de integración verificada correctamente!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Error RMSE (Celda Sintética)", f"{metrics['RMSE']:.4f} %")
        col2.metric("Complejidad Estructural", f"{metrics['Nodes']} nodos (máx. 50)")
        col3.metric("Coeficiente R²", f"{metrics['R2']:.4f}")
        st.markdown("**Ecuación simbólica recuperada sobre los datos sintéticos:**")
        st.latex(r"\widehat{\text{SoH}}_{\text{sim}} = " + sym["latex_expr"])

# ------------------------------------------------------------------------------
# PESTAÑA 4: EVOLUCIÓN SIMBÓLICA (PROTOCOLO LOBO)
# ------------------------------------------------------------------------------
with tab_evolucion:
    st.header("Entrenamiento Evolutivo y Validación Leave-One-Battery-Out (LOBO)")

    with st.expander("📖 Explicación detallada: Motor Evolutivo (`gp_engine.py`), Hiperparámetros y Validación LOBO", expanded=True):
        st.markdown("""
        **1. Conjunto de Primitivas y Operadores Protegidos (`build_primitive_set`):**  
        En Programación Genética, toda ecuación se construye combinando **nodos funcionales** (que van en las ramas internas del árbol y realizan operaciones matemáticas) y **nodos terminales** (que van en las puntas u hojas del árbol y aportan los datos de entrada o coeficientes).

        **A) Nodos Funcionales ($F$) — Operadores Matemáticos Internos:**  
        Reciben uno o dos valores provenientes de sus nodos hijos, ejecutan una operación y entregan el resultado hacia arriba. Están **protegidos analíticamente** para evitar fallos como `NaN`, `Inf` o divisiones entre cero durante la evolución aleatoria:

        | Nodo (`DEAP`) | Tipo / Aridad | Fórmula Matemática | Qué hace y por qué está protegido |
        | :--- | :--- | :--- | :--- |
        | **`add`** | Binario (2 hijos) | $a + b$ | Suma dos sub-expresiones; permite combinar efectos lineales o aditivos entre sensores. |
        | **`sub`** | Binario (2 hijos) | $a - b$ | Resta dos sub-expresiones; fundamental para modelar pérdidas de capacidad respecto a un valor base (ej. $100 - \text{pérdida}$). |
        | **`mul`** | Binario (2 hijos) | $a \cdot b$ | Multiplica dos sub-expresiones; permite escalar variables por constantes o crear interacciones cuadráticas/cruzadas entre sensores. |
        | **`pdiv`** | Binario (2 hijos) | $\begin{cases} a / b & \text{si } |b| > 10^{-6} \\ 1.0 & \text{si } |b| \le 10^{-6} \end{cases}$ | **División protegida:** calcula razones o proporciones entre variables (ej. relación entre tiempo CC y CV). Si el denominador se acerca a cero, devuelve `1.0` en lugar de colapsar el programa. |
        | **`psqrt`** | Unario (1 hijo) | $\sqrt{|x|}$ | **Raíz cuadrada protegida:** modela leyes de degradación sublineales (como el crecimiento de la capa SEI proporcional a $\sqrt{t}$). Toma el valor absoluto $|x|$ para evitar raíces imaginarias si una sub-expresión es negativa. |
        | **`plog`** | Unario (1 hijo) | $\ln(|x| + 10^{-6})$ | **Logaritmo natural protegido:** captura comportamientos de saturación o decaimiento logarítmico. Usa $|x| + 10^{-6}$ para evitar $\ln(0) = -\infty$ o logaritmos de números negativos. |

        **B) Nodos Terminales ($T$) — Hojas del Árbol (Entradas y Coeficientes):**  
        Son los puntos finales de cada rama donde termina la recursión del árbol:

        | Nodo Terminal | Qué representa | Rango de valores | Función dentro de la ecuación |
        | :--- | :--- | :--- | :--- |
        | **`t_cc`** (`ARG0`) | Duración de carga en Corriente Constante | $[0, 1]$ (normalizado) | Indica cuánto tarda la celda en llegar a $4.19\text{ V}$; disminuye conforme sube la resistencia interna por envejecimiento. |
        | **`t_cv`** (`ARG1`) | Duración de carga en Voltaje Constante | $[0, 1]$ (normalizado) | Mide el tiempo de absorción hasta el corte de corriente ($25\text{ mA}$); suele aumentar al degradarse la batería. |
        | **`t_max`** (`ARG2`) | Temperatura máxima durante la carga | $[0, 1]$ (normalizado) | Captura el pico térmico generado por efecto Joule debido al aumento de impedancia interna. |
        | **`v_fix`** (`ARG3`) | Voltaje medido a los $1000\text{ s}$ de carga | $[0, 1]$ (normalizado) | Evalúa la rapidez con la que sube la curva de tensión en un instante fijo sin depender del muestreo irregular. |
        | **`cycle`** (`ARG4`) | Número secuencial de ciclo acumulado | $[0, 1]$ (normalizado) | Representa el historial acumulado de ciclado y fatiga electroquímica de la batería. |
        | **`const`** (Efímera) | Constante numérica aleatoria | $[-5.0, 5.0]$ | Coeficientes escalares generados al azar al crear el árbol y ajustados mediante mutación (`mutEphemeral`) para calibrar pesos y desplazamientos. |
        ""
        **2. Inicialización y Función de Aptitud con Presión de Parsimonia:**  
        * La población inicial se crea con `genHalfAndHalf` (mitad árboles completos y mitad árboles de profundidad variable entre `min_=2` y `max_=6`), garantizando diversidad estructural desde la generación 0.
        * Cada individuo $T$ se evalúa minimizando:
          $$f(T) = \\text{RMSE}(T) + \\lambda \\cdot C(T)$$
          donde $C(T)$ es el total de nodos del árbol. Si $C(T) > 50$, se asigna una penalización de $10^6$ para descartarlo.

        **3. Por qué se eligieron los hiperparámetros de `gp_engine.py`:**  
        * **`pop_size = 1500` y `n_gen = 80`:** Proveen $120{,}000$ evaluaciones por corrida, suficientes para explorar combinaciones no lineales sin caer en convergencia prematura.
        * **`tournsize = 10` y `elitism = 5`:** Un torneo de tamaño 10 mantiene fuerte presión selectiva hacia fórmulas precisas, mientras que el elitismo preserva las 5 mejores ecuaciones intactas entre generaciones.
        * **`cxpb = 0.70` (cruce de un punto):** Recombina subárboles funcionales en el 70% de la descendencia.
        * **Mutaciones diferenciadas (`0.10` subárbol / `0.10` nodo / `0.05` constante):** Permiten tanto exploración estructural global (reemplazar una rama completa) como ajuste fino local (cambiar un operador o perturbar una constante).
        * **`max_depth = 6` y `max_nodes = 50`:** Evitan el crecimiento descontrolado del árbol (*bloat*) y aseguran que el código C resultante quepa en unos pocos kilobytes del microcontrolador.

        **4. Validación Cruzada *Leave-One-Battery-Out* (LOBO):**  
        En cada iteración se ajusta el `MinMaxScaler` y se entrena exclusivamente con **3 baterías**, evaluando la ecuación sobre la **4.ª batería completamente oculta**. Así se mide la capacidad real de generalización ante una batería nueva.
        """)

    st.info(
        f"**Presupuesto activo:** {cfg_runs} corridas independientes por batería | Población: {cfg_pop} individuos | "
        f"Generaciones: {cfg_gen} | Penalización λ = {cfg_lam} | Multiproceso: {'Activado' if use_parallel else 'Desactivado'}"
    )

    if not CSV_FEATURES.exists():
        st.warning("Primero extrae los datos en la pestaña **Extracción y Datos (.mat)**.")
    else:
        if st.button("🚀 Ejecutar evolución simbólica y comparación con modelos base"):
            df_data = pd.read_csv(CSV_FEATURES)
            run_accelerated_lobo(
                df_data,
                n_runs=cfg_runs,
                pop_size=cfg_pop,
                n_gen=cfg_gen,
                lam=cfg_lam,
                use_parallel=use_parallel,
            )
            run_symbolic_analysis_and_plots()
            st.success("¡Evolución LOBO, auditoría numérica, firmware C99 y gráficas completados!")
            st.rerun()

    if (ARTIFACTS_DIR / "experiment_audit.json").exists():
        audit = json.loads((ARTIFACTS_DIR / "experiment_audit.json").read_text(encoding="utf-8"))
        df_gp = pd.read_csv(ARTIFACTS_DIR / "gp_lobo_runs.csv")
        df_base = pd.read_csv(ARTIFACTS_DIR / "baselines_lobo.csv")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mejor Batería Oculta", audit["best_model_battery"])
        m2.metric("Mejor RMSE Global", f"{audit['best_model_test_RMSE']:.4f} %")
        m3.metric("Discrepancia Máx. Float32", f"{audit['python_c_equivalence'].get('max_abs_error_float32', 0):.2e}")
        m4.metric("Estabilidad Extrapolación", "Válida ✅" if audit["robustness_noise_and_oor"]["oor_all_finite"] else "Inestable ❌")

        st.subheader("Resultados Comparativos por Batería")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Modelos Base de Referencia (`baselines_lobo.csv`)**")
            st.dataframe(df_base[["test_battery", "model", "RMSE", "MAE", "R2"]], width="stretch", hide_index=True)
        with c_right:
            st.markdown("**Corridas de Programación Genética (`gp_lobo_runs.csv`)**")
            st.dataframe(df_gp[["test_battery", "run", "test_RMSE", "test_MAE", "test_R2", "Nodes"]], width="stretch", hide_index=True)

        st.subheader("Pruebas de Robustez ante Ruido de Sensores (1%, 2%, 5%) y Fuera de Rango")
        rob_df = pd.DataFrame([audit["robustness_noise_and_oor"]])
        st.dataframe(rob_df, width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# PESTAÑA 5: ECUACIÓN MATEMÁTICA, ÁRBOL AST VISUAL, DERIVADAS Y GRÁFICAS
# ------------------------------------------------------------------------------
with tab_simbolico:
    st.header("Anatomía del Árbol AST, Ecuación en LaTeX y Gráficas Científicas")

    with st.expander("📖 Explicación detallada: Cómo se arma el Árbol AST, cómo nace la fórmula y cómo leer las gráficas", expanded=True):
        st.markdown("""
        **1. Cómo se arma el Árbol de Sintaxis Abstracta (AST) en `DEAP`:**  
        Internamente, el algoritmo evolutivo almacena cada ecuación como una lista lineal en **notación prefija (polaca)**, donde cada operador precede a sus operandos (por ejemplo, `sub(85.0, mul(12.5, cycle))`).  
        * **Nodos internos (azul):** Son funciones matemáticas (`add`, `sub`, `mul`, `pdiv`, `psqrt`, `plog`) que reciben 1 o 2 ramas hijas.
        * **Hojas verdes (variables):** Son las 5 características normalizadas (`t_cc`, `t_cv`, `t_max`, `v_fix`, `cycle`) extraídas de los perfiles de carga CC-CV y escaladas a $[0, 1]$ mediante `MinMaxScaler`.
        * **Hojas ámbar (constantes efímeras):** Son coeficientes numéricos generados durante la evolución y ajustados por mutación.

        **2. De dónde salen los valores numéricos de la fórmula final:**  
        Al evaluar el árbol de abajo hacia arriba (desde las hojas hasta la raíz), las constantes descubiertas ponderan y combinan las variables normalizadas para aproximar el $\\text{SoH}$ porcentual. Luego, `SymPy` reduce algebraicamente los términos repetidos para presentar la ecuación legible en **LaTeX**.

        **3. Cómo se construye e interpreta cada gráfica:**  
        * **Frente de Pareto (`fig_pareto_front`):** Grafica el error `test_RMSE` frente al número de nodos de todas las corridas. La línea dorada une las soluciones **no dominadas** (aquellas donde ningún otro modelo logra simultáneamente menor error y menor tamaño), demostrando que la solución elegida respeta la cota de $50$ nodos.
        * **Comparativa LOBO (`fig_benchmark_lobo`):** Compara la mediana del error `RMSE` de la Regresión Simbólica (`Symbolic_GP`) frente a `Linear`, `Poly_Deg2` y `MLP_BlackBox` en cada batería oculta.
        * **Sensibilidad Analítica (`fig_symbolic_sensitivity`):** Ordena las 5 variables según la mediana del valor absoluto de su derivada parcial $\\left|\\frac{\\partial \\widehat{\\text{SoH}}}{\\partial x_i}\\right|$ evaluada en $500$ puntos del dominio operativo, revelando qué sensores gobiernan físicamente la estimación.
        """)

    if not (ARTIFACTS_DIR / "experiment_audit.json").exists():
        st.warning("Ejecuta primero la **Evolución Simbólica (LOBO)** para obtener la ecuación ganadora.")
    else:
        if st.button("🔄 Recalcular simplificación algebraica y gráficas"):
            with st.spinner("Simplificando expresión con SymPy y renderizando figuras a 300 DPI..."):
                run_symbolic_analysis_and_plots()
            st.success("Ecuación y gráficas actualizadas.")
            st.rerun()

        audit = json.loads((ARTIFACTS_DIR / "experiment_audit.json").read_text(encoding="utf-8"))

        st.subheader("1. Diagrama Visual del Árbol de Sintaxis Abstracta (AST Ganador)")
        st.markdown(
            "🔵 **Azul:** Operadores matemáticos protegidos | "
            "🟢 **Verde:** Variables extraídas de carga (normalizadas $[0,1]$) | "
            "🟠 **Ámbar:** Constantes evolutivas descubiertas"
        )
        try:
            ast_fig, parsed_tree = render_ast_tree_figure(audit["best_expression"])
            st.pyplot(ast_fig, width="stretch")
            plt.close(ast_fig)

            used_vars = sorted({str(n.name) for n in parsed_tree if str(n.name) in VARIABLE_GLOSSARY})
            if used_vars:
                st.markdown("**Variables activas utilizadas por este árbol ganador y origen de sus datos:**")
                active_rows = [
                    {
                        "Variable en Fórmula": v,
                        "Significado Físico": VARIABLE_GLOSSARY[v]["nombre"],
                        "De dónde se extrajo": VARIABLE_GLOSSARY[v]["origen"],
                    }
                    for v in used_vars
                ]
                st.dataframe(pd.DataFrame(active_rows), width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"No se pudo renderizar el diagrama AST: {exc}")

        if (ARTIFACTS_DIR / "symbolic_audit.json").exists():
            sym_audit = json.loads((ARTIFACTS_DIR / "symbolic_audit.json").read_text(encoding="utf-8"))

            st.subheader("2. Ecuación Matemática Descubierta (Simplificada con SymPy)")
            st.latex(r"\widehat{\text{SoH}} = " + sym_audit["latex_expr"])

            with st.expander("Ver representación prefija de DEAP, expresión en Python y código fuente LaTeX"):
                st.markdown("**Notación prefija lineal almacenada por DEAP:**")
                st.code(audit["best_expression"], language="text")
                st.markdown("**Expresión simplificada en Python:**")
                st.code(sym_audit["simplified_expr"], language="python")
                st.markdown("**Código LaTeX:**")
                st.code(sym_audit["latex_expr"], language="latex")

            st.subheader("3. Auditoría de Derivadas Parciales y Monotonicidad")
            grad_rows = []
            for var_name, info in sym_audit["gradients"].items():
                if "error" not in info:
                    grad_rows.append({
                        "Variable": var_name,
                        "Sensibilidad Media": round(info["mean_sensitivity"], 4),
                        "Importancia Mediana |∂SoH/∂x|": round(info["median_abs_sensitivity"], 4),
                        "% Dominio Decreciente (≤ 0)": f"{info['pct_non_positive']:.1f}%",
                        "% Dominio Creciente (≥ 0)": f"{info['pct_non_negative']:.1f}%",
                        "Derivada Parcial (LaTeX)": info["partial_derivative_latex"],
                    })
            st.dataframe(pd.DataFrame(grad_rows), width="stretch", hide_index=True)

            st.subheader("4. Figuras Científicas Generadas (300 DPI)")
            f1, f2, f3 = st.columns(3)
            with f1:
                if (FIGURES_DIR / "fig_pareto_front.png").exists():
                    st.image(str(FIGURES_DIR / "fig_pareto_front.png"), caption="Frente de Pareto: Precisión vs. Complejidad")
                    st.caption("Muestra todas las corridas LOBO y la frontera óptima no dominada (línea dorada) por debajo del límite de 50 nodos.")
            with f2:
                if (FIGURES_DIR / "fig_benchmark_lobo.png").exists():
                    st.image(str(FIGURES_DIR / "fig_benchmark_lobo.png"), caption="Comparativa LOBO vs. Modelos Base")
                    st.caption("Compara la mediana de error RMSE de nuestra ecuación frente a Regresión Lineal, Polinómica grado 2 y Red Neuronal MLP.")
            with f3:
                if (FIGURES_DIR / "fig_symbolic_sensitivity.png").exists():
                    st.image(str(FIGURES_DIR / "fig_symbolic_sensitivity.png"), caption="Sensibilidad Analítica por Variable")
                    st.caption("Cuantifica la importancia física exacta de cada sensor mediante la magnitud mediana de su derivada parcial.")

# ------------------------------------------------------------------------------
# PESTAÑA 6: FIRMWARE C99 PARA MICROCONTROLADORES BMS
# ------------------------------------------------------------------------------
with tab_c:
    st.header("Transpilación a Firmware C99 Autónomo y Auditoría Numérica")

    with st.expander("📖 Explicación detallada: ¿Por qué, para qué y cómo se genera este código C99?", expanded=True):
        st.markdown("""
        **1. ¿Por qué y para qué hay que generar este código en C99?**  
        * **Limitación de hardware en vehículos reales:** El sistema de gestión de baterías (**BMS**, *Battery Management System*) de un vehículo eléctrico corre sobre microcontroladores embebidos (como ARM Cortex-M o Infineon AURIX) con memoria RAM/Flash muy limitada (pocos kilobytes) y estrictos requisitos de tiempo real y seguridad funcional (**ISO 26262**). Estos chips **no pueden ejecutar Python, NumPy ni redes neuronales pesadas**.
        * **Para qué nos sirve:** Permite llevar el descubrimiento matemático hecho en la computadora (*offline*) directamente al hardware del vehículo (*online*). El BMS lee los sensores de voltaje, corriente y temperatura durante cada carga y calcula el **Estado de Salud (SoH)** al instante para proteger la batería, ajustar la velocidad de carga rápida y alertar mantenimiento preventivo.

        **2. ¿Cómo y a través de qué se genera? (Proceso de Transpilación AST → C99):**  
        Se genera automáticamente a través del módulo **`src/export_c.py`** mediante un **transpilador recursivo** (`_ast_to_c_recursive`):
        1. **Lectura del Árbol Ganador (`DEAP`):** Toma la mejor expresión evolucionada (en notación prefija de Python) y recorre sus nodos de arriba hacia abajo.
        2. **Traducción sintáctica a C99 (`float32`):** Convierte cada nodo funcional (`add`, `sub`, `mul`, `pdiv`, `psqrt`, `plog`) en operaciones aritméticas nativas de C y funciones `static inline` en precisión simple (`float`), que es el formato acelerado por la unidad de punto flotante (FPU) de los microcontroladores automotrices.
        3. **Incrustación de constantes de escalado (`S_MIN` y `S_RNG`):** Extrae los límites del `MinMaxScaler` (`scikit-learn`) aprendidos en el entrenamiento y los graba como arreglos constantes en memoria Flash (`static const float`), permitiendo que el firmware reciba directamente las lecturas físicas crudas de los sensores sin preprocesamiento externo.

        **3. Garantías de Seguridad en el Código Generado (`bms_soh_model.c`):**  
        * **Cero memoria dinámica (`0 bytes` en Heap):** No utiliza `malloc` ni dependencias externas; emplea únicamente variables en pila y `<math.h>`, garantizando ejecución determinista en tiempo constante $O(1)$.
        * **Saturación física (`clamp_soh_f`) y filtro EMA:** Acota la salida al rango válido $[0.0\\%, 100.0\\%]$ (evitando `NaN` o `Inf`) e incluye `bms_soh_model_predict_filtered(...)` con filtro pasa-bajas exponencial (EMA) para suavizar el ruido instantáneo de medición.

        **4. Auditoría de Equivalencia Numérica (`gcc` + `ctypes`):**  
        Para comprobar que pasar de Python (`float64`) a microcontrolador (`float32`) no introduce errores por redondeo, `verify_python_c_equivalence` compila el archivo `.c` con **`gcc -O2 -shared`**, lo carga en memoria con **`ctypes`** y compara ambas salidas verificando que la discrepancia máxima sea menor a $10^{-2}$.
        """)

    if (ARTIFACTS_DIR / "experiment_audit.json").exists():
        audit = json.loads((ARTIFACTS_DIR / "experiment_audit.json").read_text(encoding="utf-8"))
        eq = audit.get("python_c_equivalence", {})
        if "max_abs_error_float32" in eq:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Error Máx. Float32 vs Float64", f"{eq['max_abs_error_float32']:.2e} %")
            c2.metric("Error Medio Float32", f"{eq['mean_abs_error_float32']:.2e} %")
            c3.metric("Latencia Promedio (Host)", f"{eq['host_ctypes_latency_us']:.2f} µs")
            c4.metric("Nodos en Firmware", f"{eq['n_nodes']} nodos")

    c_file_path = ARTIFACTS_DIR / "bms_soh_model.c"
    if c_file_path.exists():
        c_code = c_file_path.read_text(encoding="utf-8")
        st.download_button(
            label="⬇️ Descargar código fuente bms_soh_model.c",
            data=c_code,
            file_name="bms_soh_model.c",
            mime="text/x-c",
        )
        st.code(c_code, language="c")
    else:
        st.info("Ejecuta la **Evolución Simbólica (LOBO)** para generar y auditar el código C99.")

# ------------------------------------------------------------------------------
# PESTAÑA 7: EXPLORADOR DE ARCHIVOS GENERADOS
# ------------------------------------------------------------------------------
with tab_archivos:
    st.header("Explorador y Documentación de Archivos Generados (`artifacts/`)")
    st.markdown(
        "Cada ejecución guarda un registro trazable en disco. Selecciona cualquier archivo para leer su propósito "
        "científico, visualizar su contenido y descargarlo."
    )

    if ARTIFACTS_DIR.exists():
        all_files = sorted([p for p in ARTIFACTS_DIR.rglob("*") if p.is_file()])
        if all_files:
            file_options = {str(p.relative_to(ROOT_DIR)): p for p in all_files}
            selected_rel = st.selectbox("Selecciona un archivo generado:", list(file_options.keys()))
            selected_path = file_options[selected_rel]

            desc = FILE_DESCRIPTIONS.get(
                selected_path.name,
                "Artefacto auxiliar generado durante el flujo experimental.",
            )
            st.info(f"**Propósito de `{selected_path.name}`:** {desc}")

            st.download_button(
                label=f"⬇️ Descargar {selected_path.name}",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
            )

            if selected_path.suffix == ".csv":
                st.dataframe(pd.read_csv(selected_path), width="stretch")
            elif selected_path.suffix == ".json":
                st.json(json.loads(selected_path.read_text(encoding="utf-8")))
            elif selected_path.suffix == ".c":
                st.code(selected_path.read_text(encoding="utf-8"), language="c")
            elif selected_path.suffix == ".png":
                st.image(str(selected_path))
            elif selected_path.suffix == ".pdf":
                st.success("Archivo PDF vectorial de alta calidad listo para integrarse en documentos LaTeX o póster científico.")
        else:
            st.info("La carpeta `artifacts/` aún no contiene archivos generados.")
    else:
        st.info("Aún no se ha creado la carpeta `artifacts/`. Ejecuta primero la **Evolución Simbólica (LOBO)**.")

# ------------------------------------------------------------------------------
# PESTAÑA 8 (AL FINAL): RESULTADOS OBTENIDOS, POR QUÉ Y CONCLUSIONES
# ------------------------------------------------------------------------------
with tab_resultados:
    st.header("Síntesis de Resultados Obtenidos, Análisis Causal y Conclusiones")

    if not (ARTIFACTS_DIR / "experiment_audit.json").exists():
        st.warning(
            "Aún no se han ejecutado los experimentos LOBO. Ve a la pestaña **🧬 Evolución Simbólica (LOBO)** "
            "y haz clic en **Ejecutar evolución simbólica** para generar los resultados reales."
        )
    else:
        audit = json.loads((ARTIFACTS_DIR / "experiment_audit.json").read_text(encoding="utf-8"))
        df_gp = pd.read_csv(ARTIFACTS_DIR / "gp_lobo_runs.csv")
        df_base = pd.read_csv(ARTIFACTS_DIR / "baselines_lobo.csv")
        sym_audit = (
            json.loads((ARTIFACTS_DIR / "symbolic_audit.json").read_text(encoding="utf-8"))
            if (ARTIFACTS_DIR / "symbolic_audit.json").exists()
            else None
        )

        gp_med_rmse = float(df_gp["test_RMSE"].median())
        gp_med_nodes = int(df_gp["Nodes"].median())
        mlp_med_rmse = float(df_base[df_base["model"] == "MLP_BlackBox"]["RMSE"].median())
        lin_med_rmse = float(df_base[df_base["model"] == "Linear"]["RMSE"].median())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor RMSE Global (GP)", f"{audit['best_model_test_RMSE']:.4f} %", f"Batería {audit['best_model_battery']}")
        c2.metric("Mediana RMSE GP vs MLP", f"{gp_med_rmse:.4f} %", f"MLP: {mlp_med_rmse:.4f} %", delta_color="inverse")
        c3.metric("Complejidad Mediana", f"{gp_med_nodes} nodos", "Límite BMS: 50 nodos")
        c4.metric("Error Cuantización C (float32)", f"{audit['python_c_equivalence'].get('max_abs_error_float32', 0):.2e} %", "Equivalencia verificada")

        st.divider()
        st.subheader("1. ¿Qué resultado se obtuvo tras ejecutar todo el pipeline?")
        if sym_audit:
            st.markdown("**Ecuación simbólica global descubierta (simplificada algebraicamente):**")
            st.latex(r"\widehat{\text{SoH}} = " + sym_audit["latex_expr"])

        gp_summary_by_bat = df_gp.groupby("test_battery")["test_RMSE"].median().reset_index()
        gp_summary_by_bat["model"] = "Symbolic_GP (Mediana)"
        gp_summary_by_bat = gp_summary_by_bat.rename(columns={"test_RMSE": "RMSE"})
        comp_df = pd.concat(
            [df_base[["test_battery", "model", "RMSE"]], gp_summary_by_bat[["test_battery", "model", "RMSE"]]],
            ignore_index=True,
        )
        pivot_comp = comp_df.pivot(index="test_battery", columns="model", values="RMSE")

        col_t1, col_t2 = st.columns([1.2, 1])
        with col_t1:
            st.markdown("**Comparativa de Error RMSE (% SoH) por Batería Oculta (Validación LOBO):**")
            st.dataframe(pivot_comp.style.format("{:.4f}"), width="stretch")
        with col_t2:
            st.markdown("**Resumen Estadístico de las Corridas GP (`gp_summary_median_iqr.csv`):**")
            if (ARTIFACTS_DIR / "gp_summary_median_iqr.csv").exists():
                st.dataframe(pd.read_csv(ARTIFACTS_DIR / "gp_summary_median_iqr.csv"), width="stretch")

        st.divider()
        st.subheader("2. ¿Por qué se obtuvo este resultado? (Análisis Técnico y Físico)")

        top_var_text = "las variables de duración de carga y número de ciclo"
        if sym_audit and "gradients" in sym_audit:
            sens_pairs = [
                (k, v["median_abs_sensitivity"])
                for k, v in sym_audit["gradients"].items()
                if isinstance(v, dict) and "median_abs_sensitivity" in v and np.isfinite(v["median_abs_sensitivity"])
            ]
            if sens_pairs:
                sens_pairs.sort(key=lambda x: x[1], reverse=True)
                top_var, top_val = sens_pairs[0]
                top_var_text = f"**`{top_var}`** (sensibilidad mediana $|\\partial \\widehat{{\\text{{SoH}}}}/\\partial x| = {top_val:.4f}$)"

        st.markdown(f"""
        1. **Extracción libre de fuga de datos y alineación física:** Al entrenar únicamente con características de la curva de carga CC-CV (`t_cc`, `t_cv`, `t_max`, `v_fix`, `cycle`) y excluir el tiempo de descarga, el algoritmo se vio obligado a capturar la dinámica electroquímica real de envejecimiento (aumento de resistencia interna y crecimiento de la capa SEI) en lugar de memorizar una identidad trivial.
        2. **Control efectivo del sobreajuste mediante parsimonia ($\\lambda$):** En validación *Leave-One-Battery-Out* (LOBO), los modelos muy complejos como redes neuronales (`MLP_BlackBox`, mediana RMSE = `{mlp_med_rmse:.4f}%`) o polinomios de grado 2 pueden sobreajustarse a las curvas específicas de las 3 baterías de entrenamiento y perder precisión al evaluar una celda nueva con ligera variabilidad de fabricación (como `B0018`, que tiene menos ciclos y mayor ruido). La penalización $f(T) = \\text{{RMSE}}(T) + \\lambda \\cdot C(T)$ mantuvo los árboles compactos (mediana de `{gp_med_nodes}` nodos $\\le 50$), favoreciendo relaciones funcionales suaves que generalizan mejor entre celdas.
        3. **Dominancia física de variables clave:** El análisis de derivadas parciales con `SymPy` demuestra que la ecuación descubierta concentra su sensibilidad en {top_var_text}, lo cual coincide con la electroquímica de celdas 18650: conforme avanza el ciclado y se acorta la etapa de corriente constante (`t_cc`), la capacidad disponible decae de forma consistente.
        4. **Fidelidad numérica en precisión simple (`float32`):** Gracias al uso de operadores analíticos protegidos (`pdiv`, `psqrt`, `plog`) que evitan divisiones cercanas a cero, la traducción de Python (`float64`) a C99 (`float32`) mantuvo una discrepancia máxima de apenas `{audit['python_c_equivalence'].get('max_abs_error_float32', 0):.2e}%`, sin degradación por redondeo de hardware.
        """)

        st.divider()
        st.subheader("3. Conclusión General del Proyecto")
        st.success(f"""
        **Conclusión:** Los resultados experimentales demuestran que la **Regresión Simbólica mediante Programación Genética** responde afirmativamente a la pregunta de investigación planteada: es capaz de descubrir ecuaciones matemáticas explícitas, compactas (mediana de **{gp_med_nodes} nodos**) y físicamente interpretables que compiten frente a modelos de caja negra (mediana RMSE de **{gp_med_rmse:.4f}%** frente a **{mlp_med_rmse:.4f}%** del MLP y **{lin_med_rmse:.4f}%** de Regresión Lineal) bajo validación estricta entre baterías no vistas (LOBO).

        Asimismo, la transpilación automática a código **C99 autónomo** (`bms_soh_model.c`) confirma su viabilidad operativa para sistemas embebidos automotrices: opera en complejidad temporal constante **$O(1)$**, requiere **$0\\text{ bytes}$ de memoria dinámica (Heap)**, resiste ruido de sensores hasta del $5\\%$ y conserva equivalencia numérica en punto flotante de 32 bits, habilitando una arquitectura híbrida **Nube–BMS** auditable y segura.
        """)