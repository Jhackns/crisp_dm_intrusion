## Visión General

* Transformar el dashboard histórico en una herramienta de análisis predictivo interactiva.

* Añadir entrenamiento y artefactos del modelo, refactorizar a app multi‑página, crear el analizador de lotes, visualizar resultados y soportar extracción desde .pcap.

* Mantener estética “hacker neón” y buenas prácticas (caching, responsividad, manejo de errores).

## Fase 3.A: Artefactos del Modelo

* **Script**: `model/train_and_save_model.py`

* **Datos**: cargar `data/processed/train_processed.csv`

* **Features**:

  * `categorical_cols`: `['protocol_type', 'service', 'flag']` (ajustable si existen otras categóricas)

  * `numeric_cols`: usar las 25 métricas del datamart (FACT\_NUMERIC\_COLS) cuando estén presentes: `duration, src_bytes, dst_bytes, land, wrong_fragment, urgent, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate`.

* **Transformadores**:

  * Ajustar (`fit`) `StandardScaler` sobre `numeric_cols`.

  * Ajustar (`fit`) `LabelEncoder` por columna en `categorical_cols` y guardar dict `{col: encoder}`.

  * Función de transformación segura: para valores no vistos, mapear a una categoría “desconocida” o al modo más frecuente.

* **Modelo**: `RandomForestClassifier` (alternativa: `LightGBM` si ya

