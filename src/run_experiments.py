import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import gp

from feature_extraction import build_nasa_dataset
from gp_engine import run_gp_evolution, compute_metrics
from export_c import export_c_file, verify_python_c_equivalence


FEATURE_COLS = ["t_cc", "t_cv", "t_max", "v_fix", "cycle"]
TARGET_COL = "SoH"


# ==============================================================================
# 1. MÉTRICAS UNIFICADAS Y EVALUACIÓN DE MODELOS BASE (BASELINES)
# ==============================================================================
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula RMSE, MAE, MAPE y R2 aplicando acotamiento físico [0, 100]%."""
    y_clip = np.clip(y_pred, 0.0, 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_clip)))
    mae = float(mean_absolute_error(y_true, y_clip))
    mape = float(np.mean(np.abs((y_true - y_clip) / np.maximum(y_true, 1e-6))) * 100.0)
    r2 = float(r2_score(y_true, y_clip))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def benchmark_baselines(X_train, y_train, X_test, y_test, seed: int = 42) -> list:
    """
    Entrena y evalúa los modelos de referencia sobre la misma partición LOBO:
      - Regresión Lineal (referencia paramétrica mínima)
      - Regresión Polinómica grado 2 con regularización L2 (Ridge)
      - Perceptrón Multicapa (MLP) como modelo de caja negra de ML
    """
    models = {
        "Linear": LinearRegression(),
        "Poly_Deg2": make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
        "MLP_BlackBox": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        ),
    }

    records = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Medición de tiempo de inferencia promedio en Python (host)
        t0 = time.perf_counter_ns()
        y_pred = model.predict(X_test)
        t1 = time.perf_counter_ns()
        inf_us = (t1 - t0) / (len(X_test) * 1000.0)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["model"] = name
        metrics["inference_us_host"] = float(inf_us)
        metrics["Nodes"] = np.nan
        records.append(metrics)

    return records


# ==============================================================================
# 2. PRUEBAS DE ROBUSTEZ: RUIDO DE SENSORES Y CONDICIONES FUERA DE RANGO
# ==============================================================================
def run_noise_and_oor_tests(
    individual,
    pset,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    noise_levels=(0.01, 0.02, 0.05),
    n_trials: int = 20,
    seed: int = 42,
) -> dict:
    """
    Evalúa la estabilidad de la ecuación ante:
      1. Ruido blanco gaussiano proporcional a la desviación estándar de cada sensor.
      2. Entradas sintéticas fuera del rango de entrenamiento (Out-of-Range).
    """
    rng = np.random.default_rng(seed)
    func = gp.compile(expr=individual, pset=pset)
    feature_std = np.std(X_test_raw, axis=0)

    noise_report = {}
    for sigma_rel in noise_levels:
        rmses = []
        for _ in range(n_trials):
            noise = rng.normal(0.0, sigma_rel * feature_std, size=X_test_raw.shape)
            X_noisy_norm = scaler.transform(X_test_raw + noise)
            y_pred = func(
                X_noisy_norm[:, 0],
                X_noisy_norm[:, 1],
                X_noisy_norm[:, 2],
                X_noisy_norm[:, 3],
                X_noisy_norm[:, 4],
            )
            if np.ndim(y_pred) == 0:
                y_pred = np.full_like(y_test, float(y_pred), dtype=np.float64)
            rmses.append(evaluate_predictions(y_test, y_pred)["RMSE"])

        noise_report[f"noise_{int(sigma_rel * 100)}pct_RMSE_median"] = float(np.median(rmses))
        noise_report[f"noise_{int(sigma_rel * 100)}pct_RMSE_iqr"] = float(
            np.percentile(rmses, 75) - np.percentile(rmses, 25)
        )

    # Prueba fuera de rango (extrapolación a +/- 30% del rango normalizado [0, 1])
    X_oor = np.array([
        [-0.3, -0.3, -0.3, -0.3, -0.3],
        [1.3, 1.3, 1.3, 1.3, 1.3],
        [0.0, 1.5, 1.4, -0.2, 1.5],
    ])
    y_oor = func(X_oor[:, 0], X_oor[:, 1], X_oor[:, 2], X_oor[:, 3], X_oor[:, 4])
    if np.ndim(y_oor) == 0:
        y_oor = np.full(len(X_oor), float(y_oor), dtype=np.float64)

    noise_report["oor_all_finite"] = bool(np.all(np.isfinite(y_oor)))
    noise_report["oor_raw_min"] = float(np.min(y_oor))
    noise_report["oor_raw_max"] = float(np.max(y_oor))
    return noise_report


# ==============================================================================
# 3. ORQUESTACIÓN COMPLETA DEL PROTOCOLO EXPERIMENTAL
# ==============================================================================
def run_full_benchmark(
    data_dir: str = "data/nasa_pcoe",
    output_dir: str = "artifacts",
    n_runs: int = 10,
    pop_size: int = 1500,
    n_gen: int = 80,
    lam: float = 0.005,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Carga o construcción del dataset limpio
    csv_cache = Path(data_dir).parent / "nasa_pcoe_features.csv"
    if csv_cache.exists():
        df = pd.read_csv(csv_cache)
    else:
        df = build_nasa_dataset(data_dir)
        df.to_csv(csv_cache, index=False)

    batteries = df["battery_id"].unique()
    gp_records = []
    baseline_records = []

    best_global_rmse = float("inf")
    best_global_bundle = None

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

        # Evaluación de modelos base en esta partición LOBO
        for b_rec in benchmark_baselines(X_train, y_train, X_test, y_test, seed=42):
            b_rec["test_battery"] = test_bat
            baseline_records.append(b_rec)

        # Corridas independientes de Programación Genética
        for run_idx in range(n_runs):
            seed = 100 + run_idx
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

            gp_records.append({
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
            })

            if test_m["RMSE"] < best_global_rmse:
                best_global_rmse = test_m["RMSE"]
                best_global_bundle = {
                    "individual": best_ind,
                    "pset": pset,
                    "scaler": scaler,
                    "test_battery": test_bat,
                    "X_test_raw": X_test_raw,
                    "y_test": y_test,
                }

    # 2. Exportación de resultados detallados y resumen estadístico (Mediana e IQR)
    df_gp = pd.DataFrame(gp_records)
    df_base = pd.DataFrame(baseline_records)
    df_gp.to_csv(out_path / "gp_lobo_runs.csv", index=False)
    df_base.to_csv(out_path / "baselines_lobo.csv", index=False)

    summary_gp = (
        df_gp.groupby("test_battery")[["test_RMSE", "test_MAE", "test_R2", "Nodes"]]
        .agg(["median", lambda s: np.percentile(s, 75) - np.percentile(s, 25)])
    )
    summary_gp.to_csv(out_path / "gp_summary_median_iqr.csv")

    # Prueba no paramétrica de Wilcoxon (GP mediana por batería vs baselines)
    wilcoxon_report = {}
    gp_med_by_bat = df_gp.groupby("test_battery")["test_RMSE"].median()
    for b_model in df_base["model"].unique():
        base_by_bat = df_base[df_base["model"] == b_model].set_index("test_battery")["RMSE"]
        aligned = pd.concat([gp_med_by_bat, base_by_bat], axis=1, keys=["GP", b_model]).dropna()
        if len(aligned) >= 4:
            stat, pval = wilcoxon(aligned["GP"], aligned[b_model], zero_method="wilcox")
            wilcoxon_report[f"GP_vs_{b_model}"] = {"stat": float(stat), "p_value": float(pval)}

    # 3. Exportación a C, verificación Python/C y pruebas de robustez del mejor modelo
    c_file = export_c_file(
        best_global_bundle["individual"],
        best_global_bundle["scaler"].data_min_,
        best_global_bundle["scaler"].data_max_,
        output_path=str(out_path / "bms_soh_model.c"),
    )

    equiv_report = verify_python_c_equivalence(
        best_global_bundle["individual"],
        best_global_bundle["pset"],
        best_global_bundle["X_test_raw"],
        best_global_bundle["scaler"].data_min_,
        best_global_bundle["scaler"].data_max_,
    )

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

    (out_path / "experiment_audit.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    # Para pruebas rápidas de verificación usar n_runs=5; para tesis usar n_runs=30
    run_full_benchmark(n_runs=10)
