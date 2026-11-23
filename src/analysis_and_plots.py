import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp


FEATURE_NAMES = ["t_cc", "t_cv", "t_max", "v_fix", "cycle"]


# ==============================================================================
# 1. AUDITORÍA SIMBÓLICA Y ANÁLISIS DE SENSIBILIDAD ANALÍTICA (SYMPY)
# ==============================================================================
def deap_expr_to_sympy(expr_str: str):
    """
    Convierte la cadena en notación funcional de DEAP a una expresión simbólica
    de SymPy para simplificación algebraica y diferenciación exacta.
    """
    t_cc, t_cv, t_max, v_fix, cycle = sp.symbols("t_cc t_cv t_max v_fix cycle", real=True)

    # Mapeo de operadores protegidos a su forma analítica en el dominio operativo
    local_dict = {
        "t_cc": t_cc,
        "t_cv": t_cv,
        "t_max": t_max,
        "v_fix": v_fix,
        "cycle": cycle,
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "pdiv": lambda a, b: a / b,
        "psqrt": lambda x: sp.sqrt(sp.Abs(x)),
        "plog": lambda x: sp.log(sp.Abs(x) + sp.Float("1e-6")),
    }

    raw_sym = sp.sympify(expr_str, locals=local_dict)
    simplified = sp.simplify(raw_sym)
    return simplified, [t_cc, t_cv, t_max, v_fix, cycle]


def audit_physical_consistency(expr_str: str, n_grid: int = 500, seed: int = 42) -> dict:
    """
    Calcula el gradiente exacto df/dx_i y evalúa la sensibilidad media y la
    fracción de dominio donde se cumple la monotonicidad esperada en [0, 1]^5.
    """
    sym_expr, symbols = deap_expr_to_sympy(expr_str)
    rng = np.random.default_rng(seed)

    # Muestreo uniforme dentro del dominio normalizado operativo [0.05, 0.95]^5
    grid = rng.uniform(0.05, 0.95, size=(n_grid, len(symbols)))

    gradient_report = {}
    for idx, sym in enumerate(symbols):
        deriv = sp.diff(sym_expr, sym)
        deriv_fn = sp.lambdify(symbols, deriv, modules="numpy")

        try:
            vals = deriv_fn(grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3], grid[:, 4])
            if np.ndim(vals) == 0:
                vals = np.full(n_grid, float(vals), dtype=np.float64)
            vals = np.asarray(vals, dtype=np.float64)
            finite_vals = vals[np.isfinite(vals)]

            gradient_report[str(sym)] = {
                "partial_derivative_latex": sp.latex(sp.simplify(deriv)),
                "mean_sensitivity": float(np.mean(finite_vals)) if finite_vals.size else np.nan,
                "median_abs_sensitivity": float(np.median(np.abs(finite_vals))) if finite_vals.size else np.nan,
                "pct_non_positive": float(np.mean(finite_vals <= 0.0) * 100.0) if finite_vals.size else np.nan,
                "pct_non_negative": float(np.mean(finite_vals >= 0.0) * 100.0) if finite_vals.size else np.nan,
            }
        except Exception as exc:
            gradient_report[str(sym)] = {"error": str(exc)}

    return {
        "simplified_expr": str(sym_expr),
        "latex_expr": sp.latex(sym_expr),
        "gradients": gradient_report,
    }


# ==============================================================================
# 2. GENERACIÓN DE FIGURAS CIENTÍFICAS (300 DPI) PARA TESIS Y PÓSTER
# ==============================================================================
def configure_plot_style():
    """Configuración tipográfica sobria compatible con publicaciones académicas."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_pareto_complexity_tradeoff(df_gp: pd.DataFrame, out_dir: Path):
    """
    Grafica la dispersión RMSE vs. Número de nodos de todas las corridas LOBO
    y destaca la envolvente del frente de Pareto no dominado.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.scatter(
        df_gp["Nodes"],
        df_gp["test_RMSE"],
        alpha=0.55,
        s=32,
        color="#2E5E4E",
        label="Corridas GP (Validación LOBO)",
    )

    sorted_df = df_gp.sort_values(by=["Nodes", "test_RMSE"])
    pareto_pts = []
    best_err = float("inf")
    for _, row in sorted_df.iterrows():
        if row["test_RMSE"] < best_err:
            pareto_pts.append((row["Nodes"], row["test_RMSE"]))
            best_err = row["test_RMSE"]

    if pareto_pts:
        px, py = zip(*pareto_pts)
        ax.plot(px, py, color="#B8860B", linewidth=2.0, marker="o", markersize=5, label="Frente de Pareto")

    ax.axvline(50, color="#8B0000", linestyle=":", linewidth=1.2, label="Cota BMS (50 nodos)")
    ax.set_xlabel("Complejidad Estructural (Número de Nodos)")
    ax.set_ylabel("Error de Prueba LOBO — RMSE (% SoH)")
    ax.set_title("Compromiso Precisión–Complejidad (Frente de Pareto)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pareto_front.png", dpi=300)
    fig.savefig(out_dir / "fig_pareto_front.pdf")
    plt.close(fig)


def plot_benchmark_comparison(df_gp: pd.DataFrame, df_base: pd.DataFrame, out_dir: Path):
    """
    Genera gráfica de barras agrupadas comparando la mediana de RMSE de GP
    frente a Linear, Poly_Deg2 y MLP_BlackBox por cada batería de prueba LOBO.
    """
    gp_med = (
        df_gp.groupby("test_battery")["test_RMSE"]
        .median()
        .reset_index()
        .rename(columns={"test_RMSE": "RMSE"})
    )
    gp_med["model"] = "Symbolic_GP"

    combined = pd.concat(
        [df_base[["test_battery", "model", "RMSE"]], gp_med[["test_battery", "model", "RMSE"]]],
        ignore_index=True,
    )
    pivot = combined.pivot(index="test_battery", columns="model", values="RMSE")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = ["#4A6B5D", "#8C9E96", "#C59B27", "#1F3B2F"]
    pivot.plot(kind="bar", ax=ax, color=colors[: len(pivot.columns)], edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Batería Oculta en Validación LOBO")
    ax.set_ylabel("RMSE (% SoH)")
    ax.set_title("Comparativa de Error Predictivo: Regresión Simbólica vs. Modelos Base")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Modelo", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_lobo.png", dpi=300)
    fig.savefig(out_dir / "fig_benchmark_lobo.pdf")
    plt.close(fig)


def plot_sensitivity_bars(audit_sym: dict, out_dir: Path):
    """Grafica la sensibilidad analítica mediana |df/dx_i| por variable terminal."""
    grads = audit_sym["gradients"]
    vars_list = [v for v in FEATURE_NAMES if v in grads and "median_abs_sensitivity" in grads[v]]
    vals = [grads[v]["median_abs_sensitivity"] for v in vars_list]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.barh(vars_list, vals, color="#2E5E4E", edgecolor="black", linewidth=0.6)
    ax.set_xlabel(r"Sensibilidad Analítica Mediana $|\partial \widehat{SoH} / \partial x_i|$")
    ax.set_ylabel("Variable Terminal Normalizada")
    ax.set_title("Importancia Física Derivada del Gradiente Simbólico")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_symbolic_sensitivity.png", dpi=300)
    fig.savefig(out_dir / "fig_symbolic_sensitivity.pdf")
    plt.close(fig)


# ==============================================================================
# 3. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def run_post_analysis(artifacts_dir: str = "artifacts"):
    configure_plot_style()
    art_path = Path(artifacts_dir)
    fig_path = art_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    df_gp = pd.read_csv(art_path / "gp_lobo_runs.csv")
    df_base = pd.read_csv(art_path / "baselines_lobo.csv")
    audit = json.loads((art_path / "experiment_audit.json").read_text(encoding="utf-8"))

    # 1. Auditoría simbólica de la ecuación ganadora
    sym_audit = audit_physical_consistency(audit["best_expression"])
    (art_path / "symbolic_audit.json").write_text(
        json.dumps(sym_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Generación de gráficas en PNG (300 DPI) y PDF vectorial
    plot_pareto_complexity_tradeoff(df_gp, fig_path)
    plot_benchmark_comparison(df_gp, df_base, fig_path)
    plot_sensitivity_bars(sym_audit, fig_path)

    print("Ecuación simplificada (LaTeX):", sym_audit["latex_expr"])
    print("Figuras y auditoría simbólica guardadas en:", fig_path)


if __name__ == "__main__":
    run_post_analysis("artifacts")
