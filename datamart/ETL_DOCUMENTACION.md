# Documentación del Proceso ETL: Datamart de Intrusiones

Este documento detalla, paso a paso, el proceso de Extracción, Transformación y Carga (ETL) para construir el datamart `datamart_intrusion` a partir de los archivos de datos procesados.

---

## PASO 1: Importar Librerías

### 1. Explicación del Paso
Importamos las librerías necesarias para gestionar rutas, manipular datos, y conectarnos a PostgreSQL. Estas dependencias sostienen todo el flujo ETL: lectura de CSV con pandas, conexión y operaciones en la base con psycopg2, y utilidades extra para cargas masivas.

### 2. Código de Ejecución
```python
import os
import sys
from pathlib import Path
import pandas as pd
import psycopg2
import psycopg2.extras as pg_extras
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Librerías importadas: os, sys, pathlib, pandas(2.x), psycopg2(2.x)
```

---

## PASO 2: Configuración de Rutas

### 1. Explicación del Paso
Definimos rutas a recursos clave: el archivo SQL que crea el esquema tipo Estrella (Star Schema) y el archivo .env en la raíz del proyecto con credenciales de conexión. Esta configuración centraliza el acceso a archivos requeridos por el ETL.

### 2. Código de Ejecución
```python
SQL_PATH = Path(__file__).resolve().parent / "create_schema.sql"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] SQL_PATH: f:\Python\crisp_dm_intrusion\datamart\create_schema.sql
[OK] ENV_PATH: f:\Python\crisp_dm_intrusion\.env
```

---

## PASO 3: Cargar Variables de Entorno (.env)

### 1. Explicación del Paso
Leemos el archivo .env para poblar variables de entorno (host, puerto, usuario, contraseña, base) que alimentan la conexión. Esto permite mantener credenciales fuera del código y ajustarlas por entorno.

### 2. Código de Ejecución
```python
def load_env_file():
    p = ENV_PATH
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v:
                    os.environ.setdefault(k, v)
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Variables cargadas: PGHOST=localhost, PGPORT=5432, PGUSER=postgres, PGDATABASE=datamart_intrusion
```

---

## PASO 4: Conexión a PostgreSQL

### 1. Explicación del Paso
Establecemos la conexión a PostgreSQL usando las variables de entorno. Si existe `DATABASE_URL`, se usa directamente; de lo contrario, se construye el DSN con los campos PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE.

### 2. Código de Ejecución
```python
def get_conn():
    load_env_file()
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD")
    dbname = os.environ.get("PGDATABASE", "datamart_intrusion")
    return psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Conexión establecida a: host=localhost port=5432 db=datamart_intrusion user=postgres
```

---

## PASO 5: Crear Esquema de Estrella

### 1. Explicación del Paso
Ejecutamos el SQL que define el Star Schema: dimensiones (`dim_protocol`, `dim_service`, `dim_flag`, `dim_attack`) y la tabla de hechos (`fact_network_traffic`). Esto asegura estructura consistente para consultas de BI.

### 2. Código de Ejecución
```python
def run_schema_sql(conn):
    with conn.cursor() as cur:
        with open(SQL_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
        cur.execute(sql)
    conn.commit()
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Esquema aplicado: dim_protocol, dim_service, dim_flag, dim_attack, fact_network_traffic
```

---

## PASO 6: Función de Categorías de Ataques

### 1. Explicación del Paso
Estandarizamos la categoría del ataque (Normal, DoS, Probe, R2L, U2R) a partir del nombre del ataque. Esto alimenta la dimensión `dim_attack` y facilita análisis por tipo.

### 2. Código de Ejecución
```python
def categorize_attack(name: str) -> str:
    n = (name or "").lower()
    if n == "normal":
        return "Normal"
    dos = {
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "apache2",
        "udpstorm",
        "processtable",
        "worm",
    }
    probe = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
    r2l = {
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "spy",
        "warezclient",
        "warezmaster",
        "xlock",
        "xsnoop",
        "snmpgetattack",
        "named",
        "sendmail",
        "snmpguess",
        "worm",
    }
    u2r = {
        "buffer_overflow",
        "loadmodule",
        "perl",
        "rootkit",
        "sqlattack",
        "xterm",
        "ps",
    }
    if n in dos:
        return "DoS"
    if n in probe:
        return "Probe"
    if n in r2l:
        return "R2L"
    if n in u2r:
        return "U2R"
    return "Unknown"
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] categorize_attack("smurf") -> "DoS"
[OK] categorize_attack("normal") -> "Normal"
```

---

## PASO 7: Normalización del DataFrame

### 1. Explicación del Paso
Aseguramos columnas clave: `attack_type`, `attack_category` y `is_attack`. Si el dataset trae `label`, lo usamos para derivar `attack_type`. Esta normalización uniformiza la carga en las dimensiones y hechos.

### 2. Código de Ejecución
```python
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "attack_type" not in df.columns:
        if "label" in df.columns:
            df["attack_type"] = df["label"].astype(str).str.strip()
        else:
            raise KeyError("No se encontró columna 'attack_type' ni 'label' en el dataset procesado")
    if "attack_category" not in df.columns:
        df["attack_category"] = df["attack_type"].map(categorize_attack)
    if "is_attack" in df.columns:
        df["is_attack"] = df["is_attack"].astype(bool)
    else:
        df["is_attack"] = df["attack_type"].astype(str).str.lower() != "normal"
    return df
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] df_normalizado columnas: [..., 'attack_type', 'attack_category', 'is_attack']
```

---

## PASO 8: Cargar DataFrames (Train/Test) desde CSV

### 1. Explicación del Paso
Leemos `train_processed.csv` y `test_processed.csv`, normalizamos ambos, marcamos `is_test_data` y combinamos en un único DataFrame listo para poblar el datamart.

### 2. Código de Ejecución
```python
def load_dataframes(base_dir: Path):
    train_path = base_dir / "data" / "processed" / "train_processed.csv"
    test_path = base_dir / "data" / "processed" / "test_processed.csv"
    df_train = pd.read_csv(train_path)
    df_train = _normalize_df(df_train)
    df_train["is_test_data"] = False
    df_test = pd.read_csv(test_path)
    df_test = _normalize_df(df_test)
    df_test["is_test_data"] = True
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] train_processed.csv leído: 125973 filas
[OK] test_processed.csv leído: 22544 filas
[OK] df combinado: 148517 filas, 45 columnas
```

---

## PASO 9: Carga/Upsert de Dimensiones

### 1. Explicación del Paso
Insertamos valores únicos en tablas de dimensión. Usamos `execute_values` para inserciones por lotes y `ON CONFLICT DO NOTHING` para idempotencia. Esto garantiza catálogo consistente de protocolos, servicios, flags y ataques.

### 2. Código de Ejecución
```python
def upsert_dimensions(conn, df: pd.DataFrame):
    with conn.cursor() as cur:
        protocols = sorted({str(x) for x in df["protocol_type"].unique()})
        services = sorted({str(x) for x in df["service"].unique()})
        flags = sorted({str(x) for x in df["flag"].unique()})
        attacks = sorted({str(x) for x in df["attack_type"].unique()})

        pg_extras.execute_values(
            cur,
            "INSERT INTO dim_protocol(protocol_name) VALUES %s ON CONFLICT (protocol_name) DO NOTHING",
            [(p,) for p in protocols],
        )
        pg_extras.execute_values(
            cur,
            "INSERT INTO dim_service(service_name) VALUES %s ON CONFLICT (service_name) DO NOTHING",
            [(s,) for s in services],
        )
        pg_extras.execute_values(
            cur,
            "INSERT INTO dim_flag(flag_name) VALUES %s ON CONFLICT (flag_name) DO NOTHING",
            [(f,) for f in flags],
        )
        # Preferir categoría provista en el dataset si existe
        if "attack_category" in df.columns:
            cat_map = {row[0]: row[1] for row in df[["attack_type", "attack_category"]].drop_duplicates().itertuples(index=False, name=None)}
            pg_extras.execute_values(
                cur,
                "INSERT INTO dim_attack(attack_name, attack_category) VALUES %s ON CONFLICT (attack_name) DO NOTHING",
                [(a, cat_map.get(a, categorize_attack(a))) for a in attacks],
            )
        else:
            pg_extras.execute_values(
                cur,
                "INSERT INTO dim_attack(attack_name, attack_category) VALUES %s ON CONFLICT (attack_name) DO NOTHING",
                [(a, categorize_attack(a)) for a in attacks],
            )
    conn.commit()
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] dim_protocol: 3 valores insertados (tcp, udp, icmp)
[OK] dim_service: 70 valores insertados
[OK] dim_flag: 11 valores insertados
[OK] dim_attack: 39 valores insertados
```

---

## PASO 10: Construcción de Lookups (Mapeos ID)

### 1. Explicación del Paso
Consultamos las dimensiones para construir diccionarios de mapeo nombre→id. Estos lookups permiten transformar el DataFrame a claves foráneas numéricas para la tabla de hechos.

### 2. Código de Ejecución
```python
def build_lookups(conn):
    lookups = {}
    with conn.cursor(cursor_factory=pg_extras.DictCursor) as cur:
        cur.execute("SELECT protocol_id, protocol_name FROM dim_protocol")
        lookups["protocol"] = {row[1]: row[0] for row in cur.fetchall()}
        cur.execute("SELECT service_id, service_name FROM dim_service")
        lookups["service"] = {row[1]: row[0] for row in cur.fetchall()}
        cur.execute("SELECT flag_id, flag_name FROM dim_flag")
        lookups["flag"] = {row[1]: row[0] for row in cur.fetchall()}
        cur.execute("SELECT attack_id, attack_name FROM dim_attack")
        lookups["attack"] = {row[1]: row[0] for row in cur.fetchall()}
    return lookups
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] lookups construidos: keys=['protocol','service','flag','attack']
[OK] ejemplo: protocol['tcp']=1, service['http']=12, flag['SF']=3, attack['smurf']=8
```

---

## PASO 11: Columnas Numéricas de Hechos

### 1. Explicación del Paso
Definimos el conjunto de métricas que se insertarán en la tabla de hechos. Estas columnas cuantitativas describen el tráfico y se combinan con las dimensiones para análisis OLAP.

### 2. Código de Ejecución
```python
FACT_NUMERIC_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] FACT_NUMERIC_COLS: 25 columnas numéricas
```

---

## PASO 12: Transformación a Tabla de Hechos

### 1. Explicación del Paso
Convertimos columnas semánticas a ids (FK) usando los lookups y seleccionamos únicamente las columnas requeridas por `fact_network_traffic`. Este paso prepara el DataFrame para la carga masiva.

### 2. Código de Ejecución
```python
def transform_facts(df: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    df = df.copy()
    df["protocol_id"] = df["protocol_type"].map(lookups["protocol"]).astype(int)
    df["service_id"] = df["service"].map(lookups["service"]).astype(int)
    df["flag_id"] = df["flag"].map(lookups["flag"]).astype(int)
    df["attack_id"] = df["attack_type"].map(lookups["attack"]).astype(int)
    cols = [
        "protocol_id",
        "service_id",
        "flag_id",
        "attack_id",
        *FACT_NUMERIC_COLS,
        "is_attack",
        "is_test_data",
    ]
    return df[cols]
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] facts_df.head():
   protocol_id  service_id  flag_id  attack_id  duration  src_bytes  ...
0            1          12        3          8         0        181  ...
1            2          34        5          2         0          0  ...
```

---

## PASO 13: Inserción Masiva de Hechos

### 1. Explicación del Paso
Insertamos los registros de hechos en `fact_network_traffic` usando `execute_values` para eficiencia. Se envían lotes grandes para acelerar la carga.

### 2. Código de Ejecución
```python
def insert_facts(conn, facts_df: pd.DataFrame):
    records = [tuple(x) for x in facts_df.to_numpy()]
    cols = [
        "protocol_id",
        "service_id",
        "flag_id",
        "attack_id",
        *FACT_NUMERIC_COLS,
        "is_attack",
        "is_test_data",
    ]
    insert_sql = (
        "INSERT INTO fact_network_traffic (" + ", ".join(cols) + ") VALUES %s"
    )
    with conn.cursor() as cur:
        pg_extras.execute_values(cur, insert_sql, records, page_size=10000)
    conn.commit()
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Insertados 148517 registros en fact_network_traffic (page_size=10000)
```

---

## PASO 14: Orquestación del ETL

### 1. Explicación del Paso
Coordinamos todo el flujo: conexión, creación de esquema, carga de CSV, upsert de dimensiones, construcción de lookups, transformación y carga de hechos. Finalizamos liberando la conexión.

### 2. Código de Ejecución
```python
def etl():
    base_dir = Path(__file__).resolve().parents[1]
    conn = get_conn()
    try:
        run_schema_sql(conn)
        df = load_dataframes(base_dir)
        upsert_dimensions(conn, df)
        lookups = build_lookups(conn)
        facts_df = transform_facts(df, lookups)
        insert_facts(conn, facts_df)
        print("ETL completado. Datamart 'datamart_intrusion' poblado.")
    finally:
        conn.close()


if __name__ == "__main__":
    etl()
```

### 3. Resultado / Evidencia (Simulado)
```
[OK] Esquema creado/aplicado
[OK] Datos leídos y normalizados
[OK] Dimensiones pobladas
[OK] Lookups construidos
[OK] Hechos transformados e insertados
ETL completado. Datamart 'datamart_intrusion' poblado.
```

---

## Conclusión
Con este proceso, el datamart `datamart_intrusion` queda listo para ser consultado por herramientas de BI o por la aplicación Streamlit que actuará como cuadro de mando interactivo. El esquema de estrella facilita análisis OLAP eficientes sobre intrusiones de red.