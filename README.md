#  Predicción de SoH en Baterías de Iones de Litio mediante Regresión Simbólica

Proyecto de **Cómputo Evolutivo** que utiliza Programación Genética (`DEAP`) para descubrir ecuaciones matemáticas compactas e interpretables que estiman el Estado de Salud (*State of Health*, SoH) de baterías de iones de litio a partir de variables de carga CC-CV (dataset NASA PCOE), con transpilación automática a código embebido **C99 (`float32`)** e interfaz interactiva en **Streamlit**.

---

1. Instalación y Configuración del Entorno Virtual
* **En Windows (PowerShell o CMD):**
```powershell
python -m venv .venv
py -m venv .venv

```
* **En Linux / macOS:**
```bash
python3 -m venv .venv

```

2. Activar el entorno virtual
* **En Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1

```
*(Nota: Si PowerShell bloquea la ejecución de scripts, ejecuta primero `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` y vuelve a intentar).*
* **En Windows (Símbolo del sistema / CMD):**
```cmd
.venv\Scripts\activate.bat

```

* **En Linux / macOS:**
```bash
source .venv/bin/activate

```

3. Instalar las dependencias del proyecto

Con el entorno virtual activo, instala todas las librerías necesarias:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

4. Ejecución del Proyecto

### Opción A: Interfaz Gráfica Interactiva (Recomendada)

Lanza la aplicación web local en Streamlit para ejecutar cada etapa visualmente, inspeccionar el árbol AST, la ecuación en LaTeX, las gráficas y el código C99:

```bash
streamlit run app.py

```

### Opción B: Ejecución por Consola (Paso a Paso)

```bash
# 1. Validar integración con datos sintéticos
python tests/smoke_test.py

# 2. Extraer variables de carga de los archivos .mat de NASA PCOE
python src/feature_extraction.py

# 3. Ejecutar validación LOBO, modelos base y exportación a C99
python src/run_experiments.py

# 4. Generar auditoría simbólica (SymPy) y figuras científicas a 300 DPI
python src/analysis_and_plots.py

```

---

## 📁 Estructura del Repositorio

```text
soh_symbolic_regression/
├── app.py                     # Interfaz gráfica interactiva (Streamlit)
├── requirements.txt           # Dependencias de Python
├── .gitignore                 # Exclusión de .venv y archivos temporales
├── data/
│   ├── nasa_pcoe/             # Archivos .mat originales (B0005, B0006, B0007, B0018)
│   └── nasa_pcoe_features.csv # Dataset limpio de características de carga CC-CV
├── src/
│   ├── feature_extraction.py  # Ingesta de .mat y extracción de 5 variables
│   ├── gp_engine.py           # Motor evolutivo (DEAP) y operadores protegidos
│   ├── export_c.py            # Transpilador AST -> C99 y auditoría GCC
│   ├── run_experiments.py     # Protocolo LOBO, baselines y pruebas de ruido
│   └── analysis_and_plots.py  # Simplificación SymPy, gradientes y gráficas
├── tests/
│   └── smoke_test.py          # Prueba sintética de extremo a extremo
└── artifacts/                 # Resultados, métricas CSV/JSON, código C y figuras

```

```


