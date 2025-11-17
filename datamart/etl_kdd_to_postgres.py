import os
import sys
from pathlib import Path
import pandas as pd
import psycopg2
import psycopg2.extras as pg_extras
from psycopg2 import sql


SQL_PATH = Path(__file__).resolve().parent / "create_schema.sql"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


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
    os.environ.setdefault("PGCLIENTENCODING", "UTF8")


def get_conn():
    load_env_file()
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn, options="-c client_encoding=UTF8")
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD")
    dbname = os.environ.get("PGDATABASE", "datamart_intrusion")
    try:
        port_int = int(port)
    except Exception:
        port_int = 5432
    ensure_database_exists(host, port_int, user, password, dbname)
    return psycopg2.connect(host=host, port=port_int, user=user, password=password, dbname=dbname, options="-c client_encoding=UTF8")


def ensure_database_exists(host, port, user, password, dbname):
    try:
        conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres", options="-c client_encoding=UTF8")
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            row = cur.fetchone()
            if row is None:
                cur.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8'").format(sql.Identifier(dbname)))
        conn.close()
    except Exception:
        pass

def run_schema_sql(conn):
    with conn.cursor() as cur:
        with open(SQL_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
        cur.execute(sql)
    conn.commit()


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
