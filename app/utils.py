import os
from pathlib import Path
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, bindparam
import plotly.express as px
import plotly.io as pio
import numpy as np
import pickle

def load_env_file():
    p = Path(__file__).resolve().parents[1] / ".env"
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

def get_engine():
    load_env_file()
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return create_engine(dsn)
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    password = os.environ.get("PGPASSWORD")
    dbname = os.environ.get("PGDATABASE", "datamart_intrusion")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")

def load_css():
    css_path = Path(__file__).resolve().parent / "styles.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def fetch_options(engine):
    with engine.connect() as conn:
        protocols = pd.read_sql(text("SELECT protocol_name FROM dim_protocol ORDER BY protocol_name"), conn)["protocol_name"].tolist()
        categories = pd.read_sql(text("SELECT DISTINCT attack_category FROM dim_attack ORDER BY attack_category"), conn)["attack_category"].tolist()
    return protocols, categories

def fetch_base_df(engine, selected_protocols, selected_categories, contexto):
    base_sql = text(
        """
        SELECT f.is_attack, p.protocol_name, a.attack_category, a.attack_name, s.service_name, f.src_bytes, f.dst_bytes, f.count
        FROM fact_network_traffic f
        JOIN dim_protocol p ON f.protocol_id = p.protocol_id
        JOIN dim_service s ON f.service_id = s.service_id
        JOIN dim_flag fl ON f.flag_id = fl.flag_id
        JOIN dim_attack a ON f.attack_id = a.attack_id
        WHERE (:use_protos = false OR p.protocol_name IN :protos)
          AND (:use_cats = false OR a.attack_category IN :cats)
          AND (:use_ctx = false OR f.is_test_data = :ctx_value)
        """
    ).bindparams(
        bindparam("protos", expanding=True),
        bindparam("cats", expanding=True),
    )
    params = {
        "use_protos": bool(selected_protocols),
        "use_cats": bool(selected_categories),
        "protos": selected_protocols or [""],
        "cats": selected_categories or [""],
        "use_ctx": contexto in ("Solo Entrenamiento", "Solo Prueba"),
        "ctx_value": True if contexto == "Solo Prueba" else False,
    }
    with get_engine().connect() as conn:
        df = pd.read_sql(base_sql, conn, params=params)
    return df

def kpis(df: pd.DataFrame):
    total = int(len(df))
    ataques = int(df["is_attack"].sum())
    tasa = (ataques / total * 100) if total else 0.0
    dom = df["protocol_name"].mode()
    dominante = dom.iloc[0] if not dom.empty else "-"
    return total, ataques, tasa, dominante

def charts_row_1(df: pd.DataFrame):
    c1, c2 = st.columns(2)
    with c1:
        pio.templates.default = "plotly_dark"
        neon_map = {"Normal": "#06b6d4", "Ataque": "#a21caf"}
        vc = df["is_attack"].value_counts()
        labels = ["Normal" if k == 0 or k is False else "Ataque" for k in vc.index.tolist()]
        values = vc.values.tolist()
        fig = px.pie(names=labels, values=values)
        fig.update_traces(marker=dict(colors=[neon_map.get(l, "#06b6d4") for l in labels]), textinfo="percent+label", pull=[0.02]*len(labels))
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="#0f172a",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(size=12, color="#e2e8f0"),
            ),
        )
        with st.container():
            st.markdown("<div class='chart-header'>Distribución: Normal vs Ataque</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')
    with c2:
        pio.templates.default = "plotly_dark"
        cat_counts = df["attack_category"].value_counts().reset_index()
        cat_counts.columns = ["attack_category", "count"]
        fig = px.bar(
            cat_counts,
            x="attack_category",
            y="count",
            color="attack_category",
            color_discrete_sequence=["#06b6d4", "#22c55e", "#a21caf", "#fde047", "#0ea5e9"],
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(
                title="Categoría",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="#0f172a",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(size=12, color="#e2e8f0"),
            ),
        )
        with st.container():
            st.markdown("<div class='chart-header'>Categorías de Ataque (DoS, Probe, R2L, U2R, Normal)</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')

def charts_row_2(df: pd.DataFrame):
    c1, c2 = st.columns(2)
    with c1:
        top_attacks = df["attack_name"].value_counts().head(10).reset_index()
        top_attacks.columns = ["attack_name", "count"]
        pio.templates.default = "plotly_dark"
        fig = px.bar(
            top_attacks,
            x="count",
            y="attack_name",
            orientation="h",
            color="attack_name",
            color_discrete_sequence=["#06b6d4", "#22c55e", "#a21caf", "#fde047", "#0ea5e9"],
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=20, b=40),
            showlegend=False,
        )
        with st.container():
            st.markdown("<div class='chart-header'>Top 10 tipos de ataque</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, width='stretch')
    with c2:
        gp = df.groupby(["service_name", "protocol_name"]).size().reset_index(name="count")
        fig = px.bar(
            gp,
            x="service_name",
            y="count",
            color="protocol_name",
            color_discrete_sequence=["#06b6d4", "#22c55e", "#a21caf"],
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(
                title="Protocolo",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="#0f172a",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(size=12, color="#e2e8f0"),
            ),
        )
        with st.container():
            st.markdown("<div class='chart-header'>Tráfico por servicio y protocolo</div>", unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')

def charts_row_3(df: pd.DataFrame):
    pio.templates.default = "plotly_dark"
    hover_cols = [c for c in ["attack_name", "service_name", "protocol_name", "prediccion", "confianza_ataque_%"] if c in df.columns]
    fig = px.scatter(
        df,
        x="src_bytes",
        y="dst_bytes",
        color="attack_category",
        size="count",
        hover_data=hover_cols,
        log_x=True,
        log_y=True,
        color_discrete_sequence=["#06b6d4", "#22c55e", "#a21caf", "#fde047", "#0ea5e9"],
    )
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#e2e8f0",
        margin=dict(l=20, r=20, t=10, b=20),
        title=dict(text="Relación de Bytes (Fuente vs. Destino)", x=0.0, font=dict(size=18, color="#e2e8f0")),
        legend=dict(
            title="Categoría",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="#0f172a",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=12, color="#e2e8f0"),
        ),
    )
    st.plotly_chart(fig, width='stretch')

@st.cache_resource
def load_artifacts_cached(model_dir: Path):
    with open(model_dir / "intrusion_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler_obj = pickle.load(f)
    with open(model_dir / "label_encoders.pkl", "rb") as f:
        encoders_obj = pickle.load(f)
    return model, scaler_obj, encoders_obj

def validate_columns(df, numeric_cols, categorical_cols):
    req = set(numeric_cols + categorical_cols)
    missing = [c for c in req if c not in df.columns]
    return missing

def transform_features(df, scaler_obj, encoders_obj):
    numeric_cols = scaler_obj["cols"]
    categorical_cols = encoders_obj["cols"]
    encoders = encoders_obj["encoders"]
    X_num = df[numeric_cols].astype(float).values
    X_num_scaled = scaler_obj["scaler"].transform(X_num)
    X_cat = []
    for col in categorical_cols:
        mapping = encoders[col]["mapping"]
        unk = mapping.get("__unknown__", max(mapping.values()) + 1)
        X_cat.append(df[col].astype(str).map(lambda v: mapping.get(v, unk)).astype(int).values.reshape(-1, 1))
    X_cat_arr = np.hstack(X_cat) if X_cat else np.empty((len(df), 0))
    X = np.hstack([X_num_scaled, X_cat_arr])
    return X

def render_batch_analyzer():
    st.title("Analizador de Lotes: Identificar Nuevo Tráfico")
    base = Path(__file__).resolve().parents[1] / "model"
    try:
        model, scaler_obj, encoders_obj = load_artifacts_cached(base)
    except FileNotFoundError:
        st.error("Faltan artefactos del modelo en 'model/'.")
        st.info("Ejecuta: python model/train_and_save_model.py para generar intrusion_model.pkl, scaler.pkl y label_encoders.pkl")
        return
    uploader = st.file_uploader("Subir lote", type=["csv", "pcap", "pcapng"])
    if uploader is not None:
        fname = str(getattr(uploader, "name", "")).lower()
        if fname.endswith(".csv"):
            df_in = pd.read_csv(uploader)
            missing = validate_columns(df_in, scaler_obj["cols"], encoders_obj["cols"])
            if missing:
                st.error(f"Columnas faltantes: {', '.join(missing)}")
            else:
                X = transform_features(df_in, scaler_obj, encoders_obj)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                df_out = df_in.copy()
                df_out["prediccion"] = np.where(y_pred == 1, "Ataque", "Normal")
                df_out["confianza_ataque_%"] = (y_proba[:, 1] * 100).round(2)
                df_charts = df_out.copy()
                df_charts["is_attack"] = (y_pred == 1).astype(int)
                df_charts["attack_category"] = np.where(df_charts["is_attack"] == 1, "Ataque", "Normal")
                if "service_name" not in df_charts.columns and "service" in df_charts.columns:
                    df_charts["service_name"] = df_charts["service"].astype(str)
                if "protocol_name" not in df_charts.columns and "protocol_type" in df_charts.columns:
                    df_charts["protocol_name"] = df_charts["protocol_type"].astype(str)
                df_charts["count"] = df_charts.get("count", pd.Series([1]*len(df_charts)))
                ataques = int(df_charts["is_attack"].sum())
                tasa = (ataques / len(df_charts) * 100) if len(df_charts) else 0.0
                if tasa > 10:
                    st.error("¡ADVERTENCIA! Alta tasa de ataque en el lote analizado")
                else:
                    st.success("Análisis completado. Riesgo bajo en el lote analizado")
                st.subheader("Resultados del Lote Analizado")
                charts_row_1(df_charts)
                charts_row_3(df_charts)
                with st.container():
                    st.markdown("<div class='chart-header'>Sugerencias de Prevención</div>", unsafe_allow_html=True)
                    cats = set(df_out.get("attack_category", pd.Series([])).astype(str).tolist())
                    if any("DoS" in c for c in cats):
                        st.write("Aplicar rate limiting, mitigación DDoS y endurecer servicios expuestos")
                    if any("Probe" in c for c in cats):
                        st.write("Revisar reglas IDS/IPS, ajustar firewall y segmentar la red")
                    if any("R2L" in c or "U2R" in c for c in cats):
                        st.write("Fortalecer autenticación, auditoría de privilegios y detección de escalamiento")
                st.subheader("Tabla del Lote Analizado")
                st.dataframe(df_out, width='stretch')
        elif fname.endswith(".pcap") or fname.endswith(".pcapng"):
            with st.spinner("Analizando .pcap..."):
                try:
                    from app.core.pcap_extractor import extract_features_from_pcap
                    df_pcap = extract_features_from_pcap(uploader)
                except Exception as e:
                    st.error(f"Error al extraer características del PCAP: {e}")
                    df_pcap = None
            if df_pcap is not None:
                missing = validate_columns(df_pcap, scaler_obj["cols"], encoders_obj["cols"])
                if missing:
                    st.error(f"Columnas faltantes tras extracción: {', '.join(missing)}")
                else:
                    X = transform_features(df_pcap, scaler_obj, encoders_obj)
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X)
                    df_out = df_pcap.copy()
                    df_out["prediccion"] = np.where(y_pred == 1, "Ataque", "Normal")
                    df_out["confianza_ataque_%"] = (y_proba[:, 1] * 100).round(2)
                    df_charts = df_out.copy()
                    df_charts["is_attack"] = (y_pred == 1).astype(int)
                    df_charts["attack_category"] = np.where(df_charts["is_attack"] == 1, "Ataque", "Normal")
                    if "service_name" not in df_charts.columns and "service" in df_charts.columns:
                        df_charts["service_name"] = df_charts["service"].astype(str)
                    if "protocol_name" not in df_charts.columns and "protocol_type" in df_charts.columns:
                        df_charts["protocol_name"] = df_charts["protocol_type"].astype(str)
                    df_charts["count"] = df_charts.get("count", pd.Series([1]*len(df_charts)))
                    ataques = int(df_charts["is_attack"].sum())
                    tasa = (ataques / len(df_charts) * 100) if len(df_charts) else 0.0
                    if tasa > 10:
                        st.error("¡ADVERTENCIA! Alta tasa de ataque en el lote analizado")
                    else:
                        st.success("Análisis completado. Riesgo bajo en el lote analizado")
                    st.subheader("Resultados del Lote Analizado")
                    charts_row_1(df_charts)
                    charts_row_3(df_charts)
                    with st.container():
                        st.markdown("<div class='chart-header'>Sugerencias de Prevención</div>", unsafe_allow_html=True)
                        st.write("Revisar tráfico y reforzar controles según resultados")
                    st.subheader("Tabla del Lote Analizado")
                    st.dataframe(df_out, width='stretch')
        else:
            st.error("Tipo de archivo no soportado. Use CSV, PCAP o PCAPNG.")