from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ==============================================================================
# 1. EXTRACCIÓN DE CARACTERÍSTICAS DE UN CICLO DE CARGA (CC-CV)
# ==============================================================================
def extract_charge_features(
    charge_data,
    cycle_idx: int,
    v_cv_threshold: float = 4.19,
    i_cutoff_cv: float = 0.025,
    fixed_time_s: float = 1000.0,
):
    """
    Extrae las 5 variables terminales del perfil de carga CC-CV:
      - t_cc  : Duración de la etapa de corriente constante (s).
      - t_cv  : Duración de la etapa de voltaje constante (s).
      - t_max : Temperatura máxima registrada durante la carga (°C).
      - v_fix : Voltaje interpolado en un instante fijo de carga (V).
      - cycle : Índice secuencial del ciclo operativo.

    Retorna None si el registro de carga está truncado o corrupto.
    """
    time = np.asarray(charge_data["Time"][0], dtype=np.float64)
    voltage = np.asarray(charge_data["Voltage_measured"][0], dtype=np.float64)
    current = np.asarray(charge_data["Current_measured"][0], dtype=np.float64)
    temp = np.asarray(charge_data["Temperature_measured"][0], dtype=np.float64)

    # Descartar perfiles demasiado cortos o con valores no finitos
    valid_mask = np.isfinite(time) & np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temp)
    time, voltage, current, temp = (
        time[valid_mask],
        voltage[valid_mask],
        current[valid_mask],
        temp[valid_mask],
    )
    if len(time) < 20:
        return None

    # Referenciar el tiempo al inicio del ciclo de carga
    t_rel = time - time[0]

    # Verificar que el ciclo haya alcanzado la ventana de tiempo fijo
    if t_rel[-1] < fixed_time_s:
        return None

    # 1. Transición CC -> CV: primer instante donde el voltaje alcanza la meseta de 4.2 V
    cv_candidates = np.where(voltage >= v_cv_threshold)[0]
    if len(cv_candidates) == 0:
        return None
    idx_cv_start = int(cv_candidates[0])
    t_cc = float(t_rel[idx_cv_start])

    # 2. Fin de la etapa CV: cuando la corriente cae por debajo del corte (~20 mA)
    post_cv_current = current[idx_cv_start:]
    cv_end_candidates = np.where(post_cv_current <= i_cutoff_cv)[0]
    if len(cv_end_candidates) > 0:
        idx_cv_end = idx_cv_start + int(cv_end_candidates[0])
    else:
        idx_cv_end = len(t_rel) - 1

    t_cv = float(t_rel[idx_cv_end] - t_rel[idx_cv_start])

    # Filtrar anomalías de sensor (etapas sin duración física válida)
    if t_cc <= 0.0 or t_cv <= 0.0:
        return None

    # 3. Temperatura máxima durante la carga
    t_max = float(np.max(temp))

    # 4. Voltaje en instante fijo (interpolación lineal robusta a muestreo irregular)
    v_fix = float(np.interp(fixed_time_s, t_rel, voltage))

    return {
        "t_cc": t_cc,
        "t_cv": t_cv,
        "t_max": t_max,
        "v_fix": v_fix,
        "cycle": int(cycle_idx),
    }


# ==============================================================================
# 2. CÁLCULO DE CAPACIDAD Y SoH EN CICLO DE DESCARGA
# ==============================================================================
def extract_discharge_soh(discharge_data, q_nominal: float = 2.0, use_integration: bool = False):
    """
    Calcula la capacidad descargada Q_k (Ah) y el SoH porcentual.
    Por defecto emplea la capacidad oficial reportada por NASA PCOE (integrada
    hasta el voltaje de corte de cada celda); opcionalmente calcula la integral
    trapezoidal directa Q_k = integral(|I(t)| dt) / 3600.
    """
    if not use_integration and "Capacity" in discharge_data.dtype.names:
        cap_raw = np.asarray(discharge_data["Capacity"]).ravel()
        if cap_raw.size > 0 and np.isfinite(cap_raw[0]) and cap_raw[0] > 0:
            q_k = float(cap_raw[0])
            soh_k = (q_k / q_nominal) * 100.0
            return q_k, soh_k

    # Integración numérica trapezoidal como respaldo o verificación física
    time = np.asarray(discharge_data["Time"][0], dtype=np.float64)
    current = np.asarray(discharge_data["Current_measured"][0], dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(current)
    if np.sum(valid) < 5:
        return None, None

    q_k = float(np.trapezoid(np.abs(current[valid]), time[valid]) / 3600.0)
    soh_k = (q_k / q_nominal) * 100.0
    return q_k, soh_k


# ==============================================================================
# 3. PARSER COMPLETO POR BATERÍA Y CONSTRUCCIÓN DEL DATASET UNIFICADO
# ==============================================================================
def parse_nasa_battery(mat_path: Path, q_nominal: float = 2.0, fixed_time_s: float = 1000.0) -> pd.DataFrame:
    """
    Lee un archivo .mat de NASA PCOE y empareja cada carga CC-CV con su
    descarga inmediatamente posterior (ignorando ciclos de impedancia EIS).
    """
    battery_id = mat_path.stem  # Ej. "B0005"
    mat = loadmat(str(mat_path))
    cycles = mat[battery_id][0, 0]["cycle"][0]

    records = []
    last_charge_data = None
    cycle_counter = 0

    for entry in cycles:
        op_type = str(entry["type"][0]).strip().lower()
        data = entry["data"][0, 0]

        if op_type == "charge":
            last_charge_data = data

        elif op_type == "discharge":
            cycle_counter += 1
            if last_charge_data is None:
                continue

            features = extract_charge_features(
                last_charge_data,
                cycle_idx=cycle_counter,
                fixed_time_s=fixed_time_s,
            )
            q_k, soh_k = extract_discharge_soh(data, q_nominal=q_nominal)

            # Resetear para evitar reutilizar una misma carga en dos descargas
            last_charge_data = None

            if features is None or soh_k is None or not (0.0 < soh_k <= 110.0):
                continue

            features["battery_id"] = battery_id
            features["capacity_ah"] = q_k
            features["SoH"] = soh_k
            records.append(features)

    return pd.DataFrame(records)


def build_nasa_dataset(data_dir: str, battery_ids=("B0005", "B0006", "B0007", "B0018")) -> pd.DataFrame:
    """Construye y limpia el DataFrame consolidado listo para validación LOBO."""
    base_path = Path(data_dir)
    frames = []

    for bat_id in battery_ids:
        mat_file = base_path / f"{bat_id}.mat"
        if not mat_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {mat_file}")
        df_bat = parse_nasa_battery(mat_file)
        frames.append(df_bat)

    df_all = pd.concat(frames, ignore_index=True)

    # Limpieza final de valores faltantes y verificación de tipos
    feature_cols = ["t_cc", "t_cv", "t_max", "v_fix", "cycle"]
    df_all = df_all.dropna(subset=feature_cols + ["SoH"]).reset_index(drop=True)
    return df_all


if __name__ == "__main__":
    dataset = build_nasa_dataset("data/nasa_pcoe")
    print(dataset.groupby("battery_id")[["t_cc", "t_cv", "SoH"]].agg(["count", "min", "max"]))
    dataset.to_csv("data/nasa_pcoe_features.csv", index=False)
