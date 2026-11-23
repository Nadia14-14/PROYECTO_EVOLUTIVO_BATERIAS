/*
 * Modelo subrogado de Estado de Salud (SoH) generado por Regresión Simbólica (PG)
 * Complejidad estructural: 34 nodos
 * Complejidad temporal: O(1) | Asignación en Heap: 0 bytes
 * Estándar: C99 (IEEE 754 float32)
 */
#include <math.h>
#include <stddef.h>

/* Parámetros de normalización Min-Max fijados en entrenamiento (LOBO) */
static const float S_MIN[5] = {
    6.85900000e+00f, 6.28140000e+02f, 2.31132054e+01f,
    3.93287445e+00f, 1.00000000e+00f
};
static const float S_RNG[5] = {
    3.52828200e+03f, 8.19776600e+03f, 1.30734459e+01f,
    2.74877385e-01f, 1.67000000e+02f
};

/* Operadores analíticos protegidos equivalentes al motor evolutivo */
static inline float pdiv_f(float a, float b) {
    return (fabsf(b) > 1.0e-6f) ? (a / b) : 1.0f;
}

static inline float psqrt_f(float x) {
    return sqrtf(fabsf(x));
}

static inline float plog_f(float x) {
    return logf(fabsf(x) + 1.0e-6f);
}

static inline float clamp_soh_f(float val) {
    if (!isfinite(val)) return 0.0f;
    if (val < 0.0f) return 0.0f;
    if (val > 100.0f) return 100.0f;
    return val;
}

/* Inferencia puntual determinista en línea (entradas físicas sin escalar) */
float bms_soh_model_predict(float t_cc, float t_cv, float t_max, float v_fix, float cycle) {
    const float t_cc_n = (t_cc - S_MIN[0]) / S_RNG[0];
    const float t_cv_n = (t_cv - S_MIN[1]) / S_RNG[1];
    const float t_max_n = (t_max - S_MIN[2]) / S_RNG[2];
    const float v_fix_n = (v_fix - S_MIN[3]) / S_RNG[3];
    const float cycle_n = (cycle - S_MIN[4]) / S_RNG[4];

    const float raw_soh = (psqrt_f((((-4.30953074e+00f * t_cc_n) - ((4.11535495e+00f - cycle_n) - cycle_n)) * (-1.60142313e+00f * (t_cc_n + (2.35503593e+00f + v_fix_n))))) * ((4.92050013e+00f + (t_cc_n + (2.35503593e+00f + v_fix_n))) + (1.32707601e+00f * ((4.11535495e+00f - t_cv_n) * 1.32707601e+00f))));
    return clamp_soh_f(raw_soh);
}

/* Inferencia con filtro pasa-bajas exponencial (EMA) para operación en BMS */
float bms_soh_model_predict_filtered(
    float t_cc, float t_cv, float t_max, float v_fix, float cycle,
    float *prev_soh, float alpha
) {
    const float current = bms_soh_model_predict(t_cc, t_cv, t_max, v_fix, cycle);
    if (prev_soh == NULL) {
        return current;
    }
    const float filtered = alpha * current + (1.0f - alpha) * (*prev_soh);
    *prev_soh = clamp_soh_f(filtered);
    return *prev_soh;
}
