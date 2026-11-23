import os
import subprocess
import tempfile
import time
import ctypes
from pathlib import Path
import numpy as np
from deap import gp


# ==============================================================================
# 1. TRADUCCIÓN RECURSIVA DEL AST DE DEAP A EXPRESIÓN C (FLOAT32)
# ==============================================================================
def _ast_to_c_recursive(individual, index: int, arg_map: dict):
    """
    Recorre el árbol en notación prefija de DEAP y construye la expresión
    algebraica equivalente en C99 usando aritmética float32.
    """
    node = individual[index]

    if isinstance(node, gp.Primitive):
        args_c = []
        next_idx = index + 1
        for _ in range(node.arity):
            arg_str, next_idx = _ast_to_c_recursive(individual, next_idx, arg_map)
            args_c.append(arg_str)

        name = node.name
        if name == "add":
            return f"({args_c[0]} + {args_c[1]})", next_idx
        elif name == "sub":
            return f"({args_c[0]} - {args_c[1]})", next_idx
        elif name == "mul":
            return f"({args_c[0]} * {args_c[1]})", next_idx
        elif name == "pdiv":
            return f"pdiv_f({args_c[0]}, {args_c[1]})", next_idx
        elif name == "psqrt":
            return f"psqrt_f({args_c[0]})", next_idx
        elif name == "plog":
            return f"plog_f({args_c[0]})", next_idx
        else:
            raise ValueError(f"Operador no soportado en transpilación C: {name}")

    # Nodo terminal: variable de entrada o constante efímera numérica
    if isinstance(node.value, str):
        var_name = arg_map.get(node.name, arg_map.get(node.value, node.name))
        return var_name, index + 1

    const_val = float(node.value)
    return f"{const_val:.8e}f", index + 1


def transpile_tree_to_c_expr(individual) -> str:
    """Convierte un individuo de DEAP en una cadena de expresión C válida."""
    arg_map = {
        "t_cc": "t_cc_n",
        "t_cv": "t_cv_n",
        "t_max": "t_max_n",
        "v_fix": "v_fix_n",
        "cycle": "cycle_n",
        "ARG0": "t_cc_n",
        "ARG1": "t_cv_n",
        "ARG2": "t_max_n",
        "ARG3": "v_fix_n",
        "ARG4": "cycle_n",
    }
    expr_c, end_idx = _ast_to_c_recursive(individual, 0, arg_map)
    if end_idx != len(individual):
        raise RuntimeError("Error estructural: el recorrido AST no consumió todos los nodos.")
    return expr_c


# ==============================================================================
# 2. GENERADOR DE CÓDIGO FUENTE C EMBEBIDO (SIN MEMORIA DINÁMICA)
# ==============================================================================
def generate_bms_c_code(
    individual,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
    model_name: str = "bms_soh_model",
) -> str:
    """
    Genera un archivo fuente C99 autónomo listo para integrarse en un firmware
    BMS. Incluye normalización Min-Max embebida, operadores protegidos,
    saturación [0, 100]% y filtro EMA opcional.
    """
    expr_c = transpile_tree_to_c_expr(individual)
    ranges = np.where((scaler_max - scaler_min) > 1e-12, scaler_max - scaler_min, 1.0)

    c_source = f"""/*
 * Modelo subrogado de Estado de Salud (SoH) generado por Regresión Simbólica (PG)
 * Complejidad estructural: {len(individual)} nodos
 * Complejidad temporal: O(1) | Asignación en Heap: 0 bytes
 * Estándar: C99 (IEEE 754 float32)
 */
#include <math.h>
#include <stddef.h>

/* Parámetros de normalización Min-Max fijados en entrenamiento (LOBO) */
static const float S_MIN[5] = {{
    {scaler_min[0]:.8e}f, {scaler_min[1]:.8e}f, {scaler_min[2]:.8e}f,
    {scaler_min[3]:.8e}f, {scaler_min[4]:.8e}f
}};
static const float S_RNG[5] = {{
    {ranges[0]:.8e}f, {ranges[1]:.8e}f, {ranges[2]:.8e}f,
    {ranges[3]:.8e}f, {ranges[4]:.8e}f
}};

/* Operadores analíticos protegidos equivalentes al motor evolutivo */
static inline float pdiv_f(float a, float b) {{
    return (fabsf(b) > 1.0e-6f) ? (a / b) : 1.0f;
}}

static inline float psqrt_f(float x) {{
    return sqrtf(fabsf(x));
}}

static inline float plog_f(float x) {{
    return logf(fabsf(x) + 1.0e-6f);
}}

static inline float clamp_soh_f(float val) {{
    if (!isfinite(val)) return 0.0f;
    if (val < 0.0f) return 0.0f;
    if (val > 100.0f) return 100.0f;
    return val;
}}

/* Inferencia puntual determinista en línea (entradas físicas sin escalar) */
float {model_name}_predict(float t_cc, float t_cv, float t_max, float v_fix, float cycle) {{
    const float t_cc_n = (t_cc - S_MIN[0]) / S_RNG[0];
    const float t_cv_n = (t_cv - S_MIN[1]) / S_RNG[1];
    const float t_max_n = (t_max - S_MIN[2]) / S_RNG[2];
    const float v_fix_n = (v_fix - S_MIN[3]) / S_RNG[3];
    const float cycle_n = (cycle - S_MIN[4]) / S_RNG[4];

    const float raw_soh = {expr_c};
    return clamp_soh_f(raw_soh);
}}

/* Inferencia con filtro pasa-bajas exponencial (EMA) para operación en BMS */
float {model_name}_predict_filtered(
    float t_cc, float t_cv, float t_max, float v_fix, float cycle,
    float *prev_soh, float alpha
) {{
    const float current = {model_name}_predict(t_cc, t_cv, t_max, v_fix, cycle);
    if (prev_soh == NULL) {{
        return current;
    }}
    const float filtered = alpha * current + (1.0f - alpha) * (*prev_soh);
    *prev_soh = clamp_soh_f(filtered);
    return *prev_soh;
}}
"""
    return c_source


def export_c_file(individual, scaler_min, scaler_max, output_path: str, model_name: str = "bms_soh_model"):
    """Escribe el archivo fuente C en disco y retorna la ruta generada."""
    code = generate_bms_c_code(individual, scaler_min, scaler_max, model_name=model_name)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8")
    return out


# ==============================================================================
# 3. VERIFICACIÓN DE EQUIVALENCIA NUMÉRICA PYTHON (FLOAT64) VS C (FLOAT32)
# ==============================================================================
def verify_python_c_equivalence(
    individual,
    pset,
    X_raw: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
    model_name: str = "bms_soh_model",
    n_timing_repeats: int = 1000,
) -> dict:
    """
    Compila el código C generado en una librería compartida temporal, evalúa
    todo el conjunto X_raw en ambos entornos y cuantifica la discrepancia
    máxima de precisión IEEE 754 (float64 vs float32) y la latencia promedio.
    """
    ranges = np.where((scaler_max - scaler_min) > 1e-12, scaler_max - scaler_min, 1.0)
    X_norm = (X_raw - scaler_min) / ranges

    # 1. Predicción de referencia en Python (float64)
    py_func = gp.compile(expr=individual, pset=pset)
    y_py = py_func(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], X_norm[:, 3], X_norm[:, 4])
    if np.ndim(y_py) == 0:
        y_py = np.full(X_raw.shape[0], float(y_py), dtype=np.float64)
    y_py = np.clip(y_py, 0.0, 100.0)

    is_windows = os.name == "nt"
    lib_ext = ".dll" if is_windows else ".so"

    # 2. Compilación y evaluación en C (float32)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        c_path = Path(tmpdir) / f"{model_name}.c"
        lib_path = Path(tmpdir) / f"{model_name}{lib_ext}"
        export_c_file(individual, scaler_min, scaler_max, str(c_path), model_name=model_name)

        if is_windows:
            cmd = ["gcc", "-O2", "-shared", str(c_path), "-o", str(lib_path), "-lm"]
        else:
            cmd = ["gcc", "-O2", "-shared", "-fPIC", str(c_path), "-o", str(lib_path), "-lm"]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Error de compilación GCC:\n{proc.stderr}")

        lib = ctypes.CDLL(str(lib_path))
        try:
            c_predict = getattr(lib, f"{model_name}_predict")
            c_predict.argtypes = [ctypes.c_float] * 5
            c_predict.restype = ctypes.c_float

            y_c = np.empty(X_raw.shape[0], dtype=np.float64)
            for i in range(X_raw.shape[0]):
                y_c[i] = c_predict(
                    float(X_raw[i, 0]),
                    float(X_raw[i, 1]),
                    float(X_raw[i, 2]),
                    float(X_raw[i, 3]),
                    float(X_raw[i, 4]),
                )

            # Medición de latencia (incluye overhead de llamada ctypes en host)
            sample = [float(v) for v in X_raw[0]]
            t0 = time.perf_counter_ns()
            for _ in range(n_timing_repeats):
                c_predict(*sample)
            t1 = time.perf_counter_ns()
            avg_latency_us = (t1 - t0) / (n_timing_repeats * 1000.0)
        finally:
            # Liberar el handle de la DLL en Windows para permitir su eliminación
            if is_windows:
                import _ctypes
                _ctypes.FreeLibrary(lib._handle)

    abs_diff = np.abs(y_py - y_c)
    return {
        "max_abs_error_float32": float(np.max(abs_diff)),
        "mean_abs_error_float32": float(np.mean(abs_diff)),
        "host_ctypes_latency_us": float(avg_latency_us),
        "n_nodes": len(individual),
    }