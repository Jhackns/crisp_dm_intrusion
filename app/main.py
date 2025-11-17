import streamlit as st
import pandas as pd
from sqlalchemy import text, bindparam
from app import utils
try:
    from app.utils import render_batch_analyzer as _render_batch
except Exception:
    _render_batch = None

st.set_page_config(page_title="Dashboard de Intrusiones", layout="wide")
utils.load_css()

st.sidebar.title("Análisis CRISP-DM")
view = st.session_state.get("view", "historico")
if view == "historico":
    if st.sidebar.button("Identificar Nuevo Tráfico"):
        st.session_state["view"] = "analizador"
else:
    if st.sidebar.button("Volver al Dashboard Histórico"):
        st.session_state["view"] = "historico"

if view == "historico":
    st.sidebar.caption("Sugerencia: Seleccione una categoría de ataque para filtrar")
    engine = utils.get_engine()
    protocols, categories = utils.fetch_options(engine)
    if "cat_selected" not in st.session_state:
        st.session_state["cat_selected"] = list(categories)
    if "proto_selected" not in st.session_state:
        st.session_state["proto_selected"] = list(protocols)

    st.sidebar.subheader("Categoría de Ataque")
    st.sidebar.caption("Pulse para activar/desactivar categorías")
    for c in categories:
        key = f"cat_{c}"
        if key in st.session_state:
            st.sidebar.checkbox(c, key=key)
        else:
            st.sidebar.checkbox(c, key=key, value=True)

    st.sidebar.header("Filtro de Contexto")
    contexto = st.sidebar.radio(
        "Seleccionar conjunto de datos",
        ("Completo", "Solo Entrenamiento", "Solo Prueba"),
        index=0,
    )

    st.sidebar.write("""
    <hr style='border-color:#06b6d4;'>
    """, unsafe_allow_html=True)

    st.sidebar.subheader("Protocolo")
    st.sidebar.caption("Pulse para activar/desactivar protocolos")
    for p in protocols:
        key = f"proto_{p}"
        if key in st.session_state:
            st.sidebar.checkbox(p, key=key)
        else:
            st.sidebar.checkbox(p, key=key, value=True)

    st.sidebar.write("""
    <hr style='border-color:#06b6d4;'>
    """, unsafe_allow_html=True)

    st.title("Reporte y Cuadro de Mando: Detección de Intrusiones")
    sel_categories = [c for c in categories if st.session_state.get(f"cat_{c}", False)]
    sel_protocols = [p for p in protocols if st.session_state.get(f"proto_{p}", False)]
    df = utils.fetch_base_df(engine, sel_protocols, sel_categories, contexto)
    k1, k2, k3, k4 = st.columns(4)
    total, ataques, tasa, dominante = utils.kpis(df)
    with k1:
        st.metric("Total Registros", f"{total:,}")
        st.caption("Estos son los registros totales")
    with k2:
        st.metric("Total Ataques", f"{ataques:,}")
        st.caption("Total de eventos clasificados como ataque")
    with k3:
        st.metric("Tasa de Ataque", f"{tasa:.2f}%")
        st.caption("Porcentaje de registros con ataque")
    with k4:
        st.metric("Protocolo Dominante", dominante)
        st.caption("Protocolo más frecuente en los registros")
    utils.charts_row_1(df)
    utils.charts_row_2(df)
    utils.charts_row_3(df)
    with st.expander("Ver Datos Crudos Filtrados (Mesa de Analista)"):
        st.write("Mostrando los primeros 1000 registros que coinciden con los filtros actuales...")
        filtered_sql = text(
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
            LIMIT 1000
            """
        ).bindparams(
            bindparam("protos", expanding=True),
            bindparam("cats", expanding=True),
        )
        params_table = {
            "use_protos": bool(sel_protocols),
            "use_cats": bool(sel_categories),
            "protos": sel_protocols or [""],
            "cats": sel_categories or [""],
            "use_ctx": contexto in ("Solo Entrenamiento", "Solo Prueba"),
            "ctx_value": True if contexto == "Solo Prueba" else False,
        }
        with engine.connect() as conn:
            df_tabla = pd.read_sql(filtered_sql, conn, params=params_table)
        st.dataframe(df_tabla, width='stretch')
else:
    if _render_batch is not None:
        _render_batch()
    else:
        utils.render_batch_analyzer()