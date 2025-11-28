# Manual de Usuario - Proyecto CRISP-DM Intrusión

Este manual proporciona una guía rápida y clara para configurar y ejecutar el proyecto de detección de intrusiones basado en la metodología CRISP-DM.

## 1. Requisitos Previos

Se recomienda tener instalado **Python 3.10** o superior.

## 2. Instalación de Dependencias

Antes de ejecutar cualquier script, es necesario instalar las librerías requeridas. Ejecute el siguiente comando en su terminal:

```bash
pip install -r requirements.txt
```

**Librerías principales:**
- `streamlit`: Para la interfaz web.
- `scikit-learn`: Para los modelos de Machine Learning.
- `pandas`: Para manipulación de datos.
- `SQLAlchemy` & `psycopg2-binary`: Para conexión a base de datos.
- `plotly` & `altair`: Para visualizaciones.

## 3. Descripción y Ejecución de los Componentes

A continuación se describen los scripts principales y cómo ejecutarlos.

### A. Entrenamiento del Modelo
**Archivo:** `model/train_and_save_model.py`
**Contexto:** Este script se encarga de entrenar el modelo de Machine Learning (Random Forest). Toma los datos procesados, entrena el algoritmo y guarda los archivos necesarios (`intrusion_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) en la carpeta `model/` para su uso posterior.
**Comando:**
```bash
python model/train_and_save_model.py
```

### B. Análisis Completo CRISP-DM
**Archivo:** `analysis_notebooks/analisis_completo_crisp_dm.py`
**Contexto:** Ejecuta el ciclo completo de la metodología CRISP-DM. Realiza la comprensión de datos, genera visualizaciones estadísticas, entrena y compara 7 modelos diferentes (como XGBoost, LightGBM, etc.) y genera un reporte de resultados.
**Comando:**
```bash
python analysis_notebooks/analisis_completo_crisp_dm.py
```

## 4. Iniciar la Interfaz Web

**Archivo:** `app/main.py`
**Contexto:** Despliega el Dashboard interactivo para la visualización de datos y detección de intrusiones. Permite filtrar tráfico histórico y analizar nuevos registros.

**Comando para prender la interfaz:**

```bash
streamlit run app/main.py
```
