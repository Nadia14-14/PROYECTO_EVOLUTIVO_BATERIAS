import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Asegurar resolución de módulos en src/ al ejecutar desde la raíz del repositorio
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from gp_engine import run_gp_evolution, compute_metrics
from export_c import export_c_file, verify_python_c_equivalence
from run_experiments import FEATURE_COLS, TARGET_COL, benchmark_baselines, run_noise_and_oor_tests
from analysis_and_plots import audit_physical_consistency, run_post_analysis


# ==============================================================================
# 1. GENERADOR DE TELEMETRÍA SINTÉTICA CON LEY ELECTROQUÍMICA CONOCIDA
# ==============================================================================
def generate_synthetic_batteries(n_cycles: int = 140, seed: int = 42) -> pd.DataFrame:
    """
    Simula 4 celdas 18650 con heterogeneidad de fabricación y ruido de sensor.
    Ley subyacente: degradación sublineal (SEI) + pérdida por estrés térmico.
    """
    rng = np.random.default_rng(seed)
    profiles = [
        ("SYN_B01", 24.0, 1.00),
        ("SYN_B02", 26.5, 1.08),
        ("SYN_B03", 23.0, 0.94),
        ("SYN_B04", 28.0, 1.12),
    ]

    records = []
    for bat_id, base_temp, aging_factor in profiles:
        cycles = np.arange(1, n_cycles + 1, dtype=np.float64)

        # Dinámica física simulada: al envejecer cae t_cc, crece t_cv y sube T_max
        t_cc = 2400.0 - 6.5 * aging_factor * cycles + rng.normal(0.0, 12.0, n_cycles)
        t_cv = 1200.0 + 4.8 * aging_factor * cycles + rng.normal(0.0, 10.0, n_cycles)
        t_max = base_temp + 0.03 * cycles + rng.normal(0.0, 0.35, n_cycles)
        v_fix = 3.82 + 0.0014 * aging_factor * cycles + rng.normal(0.0, 0.005, n_cycles)

        # Ley real de SoH (%) con decaimiento tipo SEI (raíz cuadrada) + efecto térmico
        soh_true = (
            100.0
            - 1.85 * aging_factor * np.sqrt(cycles)
            - 0.18 * (t_max - 23.0)
            + 0.0025 * (t_cc - 2400.0)
            + rng.normal(0.0, 0.25, n_cycles)
        )
        soh_true = np.clip(soh_true, 65.0, 100.0)

        for i in range(n_cycles):
            records.append({
                "battery_id": bat_id,
                "t_cc": float(t_cc[i]),
                "t_cv": float(t_cv[i]),
                "t_max": float(t_max[i]),
                "v_fix": float(v_fix[i]),
                "cycle": int(cycles[i]),
                "SoH": float(soh_true[i]),
            })

    return pd.DataFrame(records)


# ==============================================================================
# 2. PRUEBA DE INTEGRACIÓN DE EXTREMO A EXTREMO (SMOKE TEST)
# ==============================================================================
def run_smoke_test():
    df = generate_synthetic_batteries()
    assert len(df) == 560, "Tamaño de dataset sintético inesperado."

    with tempfile.TemporaryDirectory(prefix="soh_smoke_") as tmpdir:
        art_dir = Path(tmpdir)
        gp_rows = []
        base_rows = []
        best_bundle = None
        best_rmse = float("inf")

        # Validación LOBO reducida (población y generaciones ligeras para ejecución rápida)
        for test_bat in df["battery_id"].unique():
            train_df = df[df["battery_id"] != test_bat]
            test_df = df[df["battery_id"] == test_bat]

            X_train_raw = train_df[FEATURE_COLS].values
            y_train = train_df[TARGET_COL].values
            X_test_raw = test_df[FEATURE_COLS].values
            y_test = test_df[TARGET_COL].values

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            for b in benchmark_baselines(X_train, y_train, X_test, y_test, seed=42):
                b["test_battery"] = test_bat
                base_rows.append(b)

            best_ind, pareto, pset = run_gp_evolution(
                X_train,
                y_train,
                pop_size=200,
                n_gen=15,
                tournsize=5,
                lam=0.005,
                max_nodes=50,
                seed=42,
            )

            tr_m = compute_metrics(best_ind, pset, X_train, y_train)
            te_m = compute_metrics(best_ind, pset, X_test, y_test)
            assert te_m["Nodes"] <= 50, "Violación de cota estructural de nodos."
            assert np.isfinite(te_m["RMSE"]), "RMSE no finito en prueba LOBO."

            gp_rows.append({
                "test_battery": test_bat,
                "run": 0,
                "seed": 42,
                "train_RMSE": tr_m["RMSE"],
                "test_RMSE": te_m["RMSE"],
                "test_MAE": te_m["MAE"],
                "test_MAPE": te_m["MAPE"],
                "test_R2": te_m["R2"],
                "Nodes": te_m["Nodes"],
                "pareto_size": len(pareto),
                "expression": str(best_ind),
            })

            if te_m["RMSE"] < best_rmse:
                best_rmse = te_m["RMSE"]
                best_bundle = (best_ind, pset, scaler, test_bat, X_test_raw, y_test)

        # Persistencia de tablas intermedias
        pd.DataFrame(gp_rows).to_csv(art_dir / "gp_lobo_runs.csv", index=False)
        pd.DataFrame(base_rows).to_csv(art_dir / "baselines_lobo.csv", index=False)

        best_ind, pset, scaler, test_bat, X_test_raw, y_test = best_bundle

        # Exportación a C99
        c_path = export_c_file(
            best_ind,
            scaler.data_min_,
            scaler.data_max_,
            output_path=str(art_dir / "bms_soh_model.c"),
        )
        assert c_path.exists() and c_path.stat().st_size > 200, "Archivo C no generado correctamente."

        # Verificación de equivalencia numérica Python/C (si gcc está disponible en PATH)
        if shutil.which("gcc") is not None:
            equiv = verify_python_c_equivalence(
                best_ind, pset, X_test_raw, scaler.data_min_, scaler.data_max_, n_timing_repeats=200
            )
            assert equiv["max_abs_error_float32"] < 1e-2, (
                f"Discrepancia float32 excesiva: {equiv['max_abs_error_float32']}"
            )
        else:
            equiv = {"status": "skipped (gcc no encontrado en PATH)"}

        # Pruebas de ruido y extrapolación fuera de rango
        rob = run_noise_and_oor_tests(best_ind, pset, X_test_raw, y_test, scaler, n_trials=5)
        assert rob["oor_all_finite"], "La ecuación generó NaN/Inf en entradas fuera de rango."

        # Guardar auditoría experimental y ejecutar post-análisis simbólico + gráficas
        audit = {
            "best_model_battery": test_bat,
            "best_model_test_RMSE": best_rmse,
            "best_expression": str(best_ind),
            "python_c_equivalence": equiv,
            "robustness_noise_and_oor": rob,
        }
        (art_dir / "experiment_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

        sym_audit = audit_physical_consistency(str(best_ind), n_grid=100)
        assert "latex_expr" in sym_audit and len(sym_audit["gradients"]) == 5

        run_post_analysis(str(art_dir))
        for fig_name in ("fig_pareto_front.png", "fig_benchmark_lobo.png", "fig_symbolic_sensitivity.png"):
            assert (art_dir / "figures" / fig_name).exists(), f"Falta la figura: {fig_name}"

        print("[OK] Smoke test superado de extremo a extremo.")
        print(f"     Mejor batería LOBO: {test_bat} | RMSE={best_rmse:.4f}% | Nodos={len(best_ind)}")
        print(f"     Equivalencia C (float32): {equiv}")
        print(f"     Expresión LaTeX: {sym_audit['latex_expr']}")


if __name__ == "__main__":
    run_smoke_test()
