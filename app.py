"""
FraudGuard AI — Interfaz Interactiva de Detección de Fraude y Simulación de Alertas
Metodología KDD + Machine Learning Híbrido + Motor de Reglas de Decisión
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import uuid
from datetime import datetime, timezone
import sqlalchemy
from pathlib import Path

# Importar el motor de inferencia
from src.fraud_engine import FraudInferenceEngine

# Configuración de página
st.set_page_config(
    page_title="FraudGuard AI | Simulador de Detección de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una estética Fintech profesional
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-box-critical {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        color: #991B1B;
        font-weight: 600;
        margin: 15px 0;
    }
    .alert-box-warning {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
        color: #92400E;
        font-weight: 600;
        margin: 15px 0;
    }
    .alert-box-success {
        background-color: #DCFCE7;
        border-left: 6px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        color: #065F46;
        font-weight: 600;
        margin: 15px 0;
    }
    .sms-mockup {
        background-color: #1E293B;
        color: #F8FAFC;
        border-radius: 12px;
        padding: 18px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Carga el motor de inferencia en memoria compartida de Streamlit."""
    return FraudInferenceEngine()


def ensure_predictions_schema(db_engine):
    """Asegura que la tabla raw.predictions contenga todas las columnas de auditoría requeridas."""
    with db_engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS raw.predictions (
                transaction_id TEXT,
                scoring_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                prediction_datetime TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                fraud_probability DOUBLE PRECISION DEFAULT 0.0,
                model_prediction SMALLINT DEFAULT 0,
                is_fraud SMALLINT DEFAULT 0,
                anomaly_score DOUBLE PRECISION,
                recommended_action VARCHAR(20) DEFAULT 'APPROVE'
            );
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS scoring_timestamp TIMESTAMPTZ;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS prediction_datetime TIMESTAMPTZ;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS fraud_probability DOUBLE PRECISION;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS model_prediction SMALLINT;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS is_fraud SMALLINT;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS anomaly_score DOUBLE PRECISION;
            ALTER TABLE raw.predictions ADD COLUMN IF NOT EXISTS recommended_action VARCHAR(20);
        """))
        # Rellenar valores nulos heredados si la tabla fue creada con esquema reducido de Airflow
        conn.execute(sqlalchemy.text("""
            UPDATE raw.predictions
            SET
                scoring_timestamp = COALESCE(scoring_timestamp, prediction_datetime, CURRENT_TIMESTAMP),
                model_prediction = COALESCE(model_prediction, is_fraud, 0),
                fraud_probability = COALESCE(fraud_probability, CASE WHEN is_fraud = 1 THEN 0.85 ELSE 0.05 END),
                recommended_action = COALESCE(
                    recommended_action,
                    CASE
                        WHEN is_fraud = 1 THEN 'DECLINE'
                        WHEN anomaly_score = -1 THEN 'MANUAL_REVIEW'
                        ELSE 'APPROVE'
                    END
                )
            WHERE scoring_timestamp IS NULL
               OR fraud_probability IS NULL
               OR recommended_action IS NULL;
        """))


def get_db_connection():
    """Conecta a la base de datos PostgreSQL local si está disponible."""
    try:
        user = os.getenv("PGUSER", "postgres")
        pwd = os.getenv("PGPASSWORD", "R0bb34l3x")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")
        db = os.getenv("PGDATABASE", "fraud_detection_kdd")
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}",
            connect_args={"connect_timeout": 3}
        )
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1;"))
        ensure_predictions_schema(engine)
        return engine
    except Exception:
        return None


# Cargar motor
try:
    engine = load_engine()
except Exception as e:
    st.error(f"❌ Error al inicializar el motor de modelos: {e}")
    st.stop()


# ---------------------------------------------------------
# BARRA LATERAL: Selección de Escenarios Predefinidos
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("🛡️ FraudGuard AI")
st.sidebar.markdown("**Demostrador de Detección de Fraude en Tiempo Real**")
st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Cargar Escenario de Prueba")
preset_choice = st.sidebar.radio(
    "Selecciona un perfil de transacción:",
    [
        "🟢 Transacción Legítima Habitual",
        "🔴 Fraude Crítico Internacional (Regla R1)",
        "🟡 Desviación Anómala de Comportamiento (Isolation Forest)",
        "🔵 Madrugada Sin Tarjeta Presente (Regla R3)",
        "⚙️ Personalizado (Ajuste Manual)"
    ]
)

# Definición de presets
if preset_choice == "🟢 Transacción Legítima Habitual":
    default_data = {
        "amount": 35.50,
        "hour": 14,
        "age": 38,
        "account_days": 850,
        "limit": 6000.0,
        "credit_score": 740,
        "cards": 1,
        "tx_1h": 1,
        "tx_24h": 2,
        "tx_7d": 9,
        "amt_1h": 35.50,
        "amt_24h": 70.0,
        "days_last_tx": 2,
        "dist_home": 3.2,
        "dist_last_tx": 1.0,
        "ratio_avg": 0.88,
        "avg_30d": 40.0,
        "std_30d": 12.0,
        "declined_30d": 0,
        "chargebacks": 0,
        "card_present": 1,
        "used_chip": 1,
        "used_pin": 1,
        "high_risk": 0,
        "international": 0,
        "category": "Supermarket",
        "country": "USA",
        "channel": "In-store",
        "device": "POS Terminal",
        "browser": "No_Aplica_Fisico"
    }
elif preset_choice == "🔴 Fraude Crítico Internacional (Regla R1)":
    default_data = {
        "amount": 2950.00,
        "hour": 3,
        "age": 28,
        "account_days": 40,
        "limit": 3000.0,
        "credit_score": 570,
        "cards": 3,
        "tx_1h": 4,
        "tx_24h": 7,
        "tx_7d": 15,
        "amt_1h": 4500.00,
        "amt_24h": 6800.00,
        "days_last_tx": 0,
        "dist_home": 9200.0,
        "dist_last_tx": 9100.0,
        "ratio_avg": 7.2,
        "avg_30d": 65.0,
        "std_30d": 18.0,
        "declined_30d": 3,
        "chargebacks": 2,
        "card_present": 0,
        "used_chip": 0,
        "used_pin": 0,
        "high_risk": 1,
        "international": 1,
        "category": "Online Retail",
        "country": "Nigeria",
        "channel": "Online",
        "device": "Desktop",
        "browser": "Chrome"
    }
elif preset_choice == "🟡 Desviación Anómala de Comportamiento (Isolation Forest)":
    default_data = {
        "amount": 1450.00,
        "hour": 16,
        "age": 45,
        "account_days": 1200,
        "limit": 8000.0,
        "credit_score": 710,
        "cards": 2,
        "tx_1h": 2,
        "tx_24h": 4,
        "tx_7d": 10,
        "amt_1h": 1450.00,
        "amt_24h": 1800.00,
        "days_last_tx": 1,
        "dist_home": 1250.0,
        "dist_last_tx": 1180.0,
        "ratio_avg": 5.4,
        "avg_30d": 55.0,
        "std_30d": 15.0,
        "declined_30d": 1,
        "chargebacks": 0,
        "card_present": 0,
        "used_chip": 0,
        "used_pin": 0,
        "high_risk": 0,
        "international": 1,
        "category": "Electronics",
        "country": "Romania",
        "channel": "Online",
        "device": "Mobile",
        "browser": "Safari"
    }
elif preset_choice == "🔵 Madrugada Sin Tarjeta Presente (Regla R3)":
    default_data = {
        "amount": 420.00,
        "hour": 2,
        "age": 31,
        "account_days": 210,
        "limit": 4500.0,
        "credit_score": 680,
        "cards": 1,
        "tx_1h": 1,
        "tx_24h": 2,
        "tx_7d": 6,
        "amt_1h": 420.00,
        "amt_24h": 450.00,
        "days_last_tx": 3,
        "dist_home": 3400.0,
        "dist_last_tx": 3200.0,
        "ratio_avg": 3.8,
        "avg_30d": 45.0,
        "std_30d": 14.0,
        "declined_30d": 0,
        "chargebacks": 0,
        "card_present": 0,
        "used_chip": 0,
        "used_pin": 0,
        "high_risk": 0,
        "international": 1,
        "category": "Travel",
        "country": "China",
        "channel": "Online",
        "device": "Desktop",
        "browser": "Firefox"
    }
else:
    default_data = {
        "amount": 100.0,
        "hour": 12,
        "age": 35,
        "account_days": 365,
        "limit": 5000.0,
        "credit_score": 700,
        "cards": 2,
        "tx_1h": 1,
        "tx_24h": 2,
        "tx_7d": 8,
        "amt_1h": 100.0,
        "amt_24h": 150.0,
        "days_last_tx": 1,
        "dist_home": 10.0,
        "dist_last_tx": 5.0,
        "ratio_avg": 1.5,
        "avg_30d": 50.0,
        "std_30d": 15.0,
        "declined_30d": 0,
        "chargebacks": 0,
        "card_present": 1,
        "used_chip": 1,
        "used_pin": 1,
        "high_risk": 0,
        "international": 0,
        "category": "Supermarket",
        "country": "USA",
        "channel": "In-store",
        "device": "POS Terminal",
        "browser": "No_Aplica_Fisico"
    }


# ---------------------------------------------------------
# CABECERA PRINCIPAL
# ---------------------------------------------------------
st.markdown('<div class="main-header">🛡️ FraudGuard AI: Pipeline de Detección de Fraude</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluación en tiempo real: Random Forest + Isolation Forest + Reglas de Decisión KDD + Alertas Multi-Canal</div>', unsafe_allow_html=True)

# Pestañas principales
tab_simulador, tab_bd, tab_arquitectura = st.tabs([
    "⚡ Simulador y Despacho de Alertas",
    "📊 Monitor de PostgreSQL (Auditoría)",
    "📐 Arquitectura Técnica KDD"
])


# =========================================================
# PESTAÑA 1: SIMULADOR DE TRANSACCIÓN Y EVALUACIÓN
# =========================================================
with tab_simulador:
    st.markdown("### 1. Parámetros de la Transacción Entrante")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Detalles Financieros**")
        amount = st.number_input("Monto de la Transacción ($ USD)", min_value=1.0, max_value=25000.0, value=float(default_data["amount"]), step=10.0)
        hour = st.slider("Hora de la Transacción (0-23h)", min_value=0, max_value=23, value=int(default_data["hour"]))
        credit_limit = st.number_input("Límite de Crédito del Cliente ($)", min_value=500.0, max_value=50000.0, value=float(default_data["limit"]), step=500.0)
        avg_30d = st.number_input("Monto Promedio Histórico 30d ($)", min_value=10.0, max_value=2000.0, value=float(default_data["avg_30d"]))
        ratio_avg = amount / (avg_30d + 1e-5)
        st.caption(f"ℹ️ Ratio Monto vs Promedio: **{ratio_avg:.2f}x**")

    with col2:
        st.markdown("**⚡ Ratios de Velocidad y Ubicación**")
        tx_1h = st.number_input("Transacciones en la última hora", min_value=0, max_value=20, value=int(default_data["tx_1h"]))
        tx_24h = st.number_input("Transacciones en últimas 24h", min_value=0, max_value=50, value=int(default_data["tx_24h"]))
        amt_1h = st.number_input("Monto total gastado en última hora ($)", min_value=0.0, max_value=50000.0, value=float(default_data["amt_1h"]))
        amt_24h = st.number_input("Monto total gastado en 24h ($)", min_value=0.0, max_value=100000.0, value=float(default_data["amt_24h"]))
        dist_home = st.number_input("Distancia desde el domicilio (km)", min_value=0.0, max_value=20000.0, value=float(default_data["dist_home"]))
        dist_last = st.number_input("Distancia desde la última tx (km)", min_value=0.0, max_value=20000.0, value=float(default_data["dist_last_tx"]))

    with col3:
        st.markdown("**🔒 Factores de Seguridad y Entorno**")
        category = st.selectbox("Categoría de Comercio", ["Supermarket", "Online Retail", "Electronics", "Travel", "Dining", "Fuel", "ATM"], index=["Supermarket", "Online Retail", "Electronics", "Travel", "Dining", "Fuel", "ATM"].index(default_data["category"]) if default_data["category"] in ["Supermarket", "Online Retail", "Electronics", "Travel", "Dining", "Fuel", "ATM"] else 0)
        country = st.selectbox("País de Comercio", ["USA", "Nigeria", "Romania", "China", "Russia", "UK", "Mexico"], index=["USA", "Nigeria", "Romania", "China", "Russia", "UK", "Mexico"].index(default_data["country"]) if default_data["country"] in ["USA", "Nigeria", "Romania", "China", "Russia", "UK", "Mexico"] else 0)
        channel = st.selectbox("Canal de Transacción", ["In-store", "Online", "ATM"], index=["In-store", "Online", "ATM"].index(default_data["channel"]))
        card_present = st.checkbox("Tarjeta Física Presente", value=bool(default_data["card_present"]))
        used_chip = st.checkbox("Uso de Chip EMV", value=bool(default_data["used_chip"]))
        used_pin = st.checkbox("Uso de PIN de Seguridad", value=bool(default_data["used_pin"]))
        high_risk_country = 1 if country in ["Nigeria", "Romania", "Russia"] else int(default_data["high_risk"])
        intl_tx = 1 if country != "USA" else int(default_data["international"])
        account_days = st.number_input("Antigüedad de la cuenta (días)", min_value=1, max_value=3650, value=int(default_data["account_days"]))
        chargebacks = st.number_input("Contracargos históricos (chargebacks)", min_value=0, max_value=10, value=int(default_data["chargebacks"]))

    # Construir objeto de transacción
    tx_input = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
        "transaction_amount": float(amount),
        "transaction_hour": int(hour),
        "customer_age": int(default_data["age"]),
        "account_age_days": int(account_days),
        "credit_limit": float(credit_limit),
        "credit_score": int(default_data["credit_score"]),
        "num_cards_held": int(default_data["cards"]),
        "transactions_last_1h": int(tx_1h),
        "transactions_last_24h": int(tx_24h),
        "transactions_last_7d": int(default_data["tx_7d"]),
        "amount_last_1h": float(amt_1h),
        "amount_last_24h": float(amt_24h),
        "days_since_last_transaction": int(default_data["days_last_tx"]),
        "distance_from_home_km": float(dist_home),
        "distance_from_last_tx_km": float(dist_last),
        "amount_vs_avg_ratio": float(ratio_avg),
        "avg_transaction_amount_30d": float(avg_30d),
        "std_transaction_amount_30d": float(default_data["std_30d"]),
        "num_declined_last_30d": int(default_data["declined_30d"]),
        "num_chargebacks_ever": int(chargebacks),
        "card_present": 1 if card_present else 0,
        "used_chip": 1 if used_chip else 0,
        "used_pin": 1 if used_pin else 0,
        "high_risk_country": int(high_risk_country),
        "international_transaction": int(intl_tx),
        "merchant_category": category,
        "merchant_country": country,
        "transaction_day_of_week": "Wednesday",
        "customer_gender": "F",
        "transaction_channel": channel,
        "device_type": default_data["device"],
        "browser_clean": default_data["browser"],
        "email_domain": "gmail.com"
    }

    st.markdown("---")
    eval_btn = st.button("🚀 Evaluar Transacción & Ejecutar Pipeline de Fraude", type="primary", use_container_width=True)

    if eval_btn:
        with st.spinner("Ejecutando feature engineering, aislamiento de anomalías y ensamble de modelos..."):
            st.session_state["last_eval_result"] = engine.predict(tx_input)
            st.session_state["last_tx_input"] = tx_input.copy()

    if "last_eval_result" in st.session_state:
        result = st.session_state["last_eval_result"]
        saved_tx_input = st.session_state["last_tx_input"]

        st.markdown("### 2. Veredicto del Sistema y Métricas de Inferencia")
        
        # Banner del veredicto
        rec_action = result["recommended_action"]
        prob = result["fraud_probability"]
        iso_score = result["anomaly_score"]
        is_anom = result["is_anomaly"]
        
        if rec_action == "DECLINE":
            st.markdown(f"""
            <div class="alert-box-critical">
                <span style="font-size: 1.5rem;">🛑 TRANSACCIÓN BLOQUEADA (DECLINE)</span><br>
                <span>Acción preventiva inmediata: La transacción presenta un riesgo inaceptable o violó una regla crítica de fraude.</span>
            </div>
            """, unsafe_allow_html=True)
        elif rec_action == "MANUAL_REVIEW":
            st.markdown(f"""
            <div class="alert-box-warning">
                <span style="font-size: 1.5rem;">⚠️ REVISIÓN MANUAL / RETENCIÓN (MANUAL_REVIEW)</span><br>
                <span>Acción preventiva: Se detectó comportamiento anómalo o puntuación de riesgo intermedia. Se sugiere autenticación 2FA.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box-success">
                <span style="font-size: 1.5rem;">✅ TRANSACCIÓN APROBADA (APPROVE)</span><br>
                <span>Acción: Transacción dentro de los parámetros habituales de uso y bajo riesgo predictivo.</span>
            </div>
            """, unsafe_allow_html=True)

        # Métricas KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Probabilidad de Fraude (RF)", f"{prob * 100:.2f}%", delta="Riesgo Alto" if prob >= 0.5 else "Bajo Control", delta_color="inverse")
        kpi2.metric("Score de Anomalía (Isolation Forest)", f"{iso_score:.4f}", delta="Atípico" if is_anom else "Típico", delta_color="inverse")
        kpi3.metric("Índice de Riesgo Compuesto", f"{result['engineered_features']['composite_risk_index']} señales", delta="Crítico" if result['engineered_features']['composite_risk_index'] >= 5 else "Normal", delta_color="inverse")
        kpi4.metric("Factores de Seguridad Activos", f"{result['engineered_features']['security_factors_count']} / 3", delta="Seguro" if result['engineered_features']['security_factors_count'] >= 2 else "Vulnerable", delta_color="normal")

        # Reglas de negocio evaluadas
        st.markdown("#### 📋 Evaluación de Reglas de Negocio")
        rules_df = pd.DataFrame([
            {
                "Regla": r["code"],
                "Nombre": r["name"],
                "Descripción": r["description"],
                "Severidad": r["severity"],
                "Estado": "🚨 ACTIVADA" if r["triggered"] else "✔️ Cumple"
            }
            for r in result["rules_evaluated"]
        ])
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # SIMULACIÓN DE ALERTA OPERATIVA
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 3. Simulación de Despacho de Alertas")
        
        col_sms, col_webhook = st.columns(2)
        
        with col_sms:
            st.markdown("**📱 Alerta SMS / Notificación Push (Cliente)**")
            st.markdown(f"""
            <div class="sms-mockup">
                <small style="color: #94A3B8;">MENSAJE DE SEGURIDAD BANCARIA • AHORA</small><br><br>
                {result['alert']['simulated_sms']}
            </div>
            """, unsafe_allow_html=True)

        with col_webhook:
            st.markdown("**🌐 Payload Webhook Enviado (Slack / SOC / Microservicio)**")
            st.json(result["alert"]["simulated_webhook"])

        # Opción para guardar en base de datos PostgreSQL
        st.markdown("---")
        st.markdown("### 4. Persistencia en Base de Datos (Auditoría SQL)")
        db_engine = get_db_connection()
        if db_engine:
            if st.button("💾 Registrar esta Transacción y su Predicción en PostgreSQL"):
                try:
                    now_ts = datetime.now(timezone.utc)
                    tx_db_row = saved_tx_input.copy()
                    tx_db_row["transaction_id"] = result["transaction_id"]
                    tx_db_row["customer_id"] = "CUST-SIM-001"
                    tx_db_row["card_id"] = "CARD-SIM-001"
                    tx_db_row["merchant_id"] = "MERCH-SIM-001"
                    tx_db_row["transaction_datetime"] = now_ts.replace(tzinfo=None)
                    tx_db_row["customer_state"] = "CA"
                    tx_db_row["is_weekend"] = 1 if now_ts.weekday() >= 5 else 0
                    browser_val = tx_db_row.pop("browser_clean", None)
                    tx_db_row["browser"] = None if browser_val in (None, "No_Aplica_Fisico") else browser_val

                    df_raw = pd.DataFrame([tx_db_row])
                    df_raw.to_sql("transactions", con=db_engine, schema="raw", if_exists="append", index=False)
                    
                    df_pred = pd.DataFrame([{
                        "transaction_id": result["transaction_id"],
                        "scoring_timestamp": now_ts,
                        "prediction_datetime": now_ts,
                        "fraud_probability": result["fraud_probability"],
                        "model_prediction": result["model_prediction"],
                        "is_fraud": result["model_prediction"],
                        "anomaly_score": result["anomaly_score"],
                        "recommended_action": result["recommended_action"]
                    }])
                    df_pred.to_sql("predictions", con=db_engine, schema="raw", if_exists="append", index=False)
                    st.success(f"✅ Transacción `{result['transaction_id']}` y predicción guardadas en PostgreSQL (`raw.transactions` y `raw.predictions`).")
                except Exception as ex:
                    st.error(f"Error al guardar en base de datos: {ex}")
        else:
            st.info("ℹ️ PostgreSQL local no está activo en el puerto 5432 o tiene credenciales distintas. La simulación opera en modo autónomo en memoria.")


# =========================================================
# PESTAÑA 2: MONITOR DE POSTGRESQL (AUDITORÍA HISTÓRICA)
# =========================================================
with tab_bd:
    st.markdown("### 📊 Monitor de Base de Datos PostgreSQL en Vivo")
    db_engine = get_db_connection()
    if db_engine:
        try:
            with db_engine.connect() as conn:
                count_tx = conn.execute(sqlalchemy.text("SELECT count(*) FROM raw.transactions;")).scalar()
                count_pred = conn.execute(sqlalchemy.text("SELECT count(*) FROM raw.predictions;")).scalar()
                
                m1, m2 = st.columns(2)
                m1.metric("Transacciones en `raw.transactions`", f"{count_tx:,}")
                m2.metric("Predicciones en `raw.predictions`", f"{count_pred:,}")

                st.markdown("#### Últimas 15 Transacciones Calificadas")
                query_recent = """
                    SELECT 
                        p.transaction_id,
                        COALESCE(p.scoring_timestamp, p.prediction_datetime) AS scoring_timestamp,
                        t.transaction_amount,
                        t.merchant_category,
                        t.merchant_country,
                        round(cast(COALESCE(p.fraud_probability, 0.0) as numeric), 4) as fraud_probability,
                        COALESCE(p.recommended_action, 'APPROVE') as recommended_action
                    FROM raw.predictions p
                    LEFT JOIN raw.transactions t ON p.transaction_id = t.transaction_id
                    ORDER BY COALESCE(p.scoring_timestamp, p.prediction_datetime) DESC NULLS LAST
                    LIMIT 15;
                """
                df_recent = pd.read_sql(query_recent, con=db_engine)
                st.dataframe(df_recent, use_container_width=True, hide_index=True)
                
                # Resumen de acciones recomendadas
                query_dist = """
                    SELECT COALESCE(recommended_action, 'APPROVE') AS recommended_action, count(*) as total
                    FROM raw.predictions
                    GROUP BY COALESCE(recommended_action, 'APPROVE');
                """
                df_dist = pd.read_sql(query_dist, con=db_engine)
                st.bar_chart(df_dist.set_index("recommended_action"))
        except Exception as ex:
            st.warning(f"No se pudieron leer las tablas de auditoría: {ex}")
    else:
        st.warning("⚠️ No se pudo establecer conexión con PostgreSQL (`localhost:5432`). Verifica que el servicio esté arriba.")


# =========================================================
# PESTAÑA 3: ARQUITECTURA TÉCNICA KDD
# =========================================================
with tab_arquitectura:
    st.markdown("### 📐 Arquitectura de Ingeniería y Metodología KDD")
    st.markdown("""
    Este proyecto implementa una solución de detección de fraude de nivel empresarial siguiendo las 8 etapas de la metodología **KDD (Knowledge Discovery in Databases)**:
    
    1. **Comprensión del Dominio:** Planteamiento del problema financiero con señales débiles y desbalance severo (2.05% de fraude real).
    2. **Selección de Datos & Data Leakage:** Eliminación estricta de variables posteriores al evento (`fraud_amount_lost`, `fraud_type`) y claves UUID.
    3. **Limpieza & Outliers:** Tratamiento de nulos estructurales en canales físicos y conservación de valores atípicos mediante compresión logarítmica.
    4. **Ingeniería de Características:** Generación de ratios de velocidad (1h vs 24h), z-scores de desviación sobre 30 días, factor de autenticación y el *Composite Risk Index*.
    5. **Minería de Datos Híbrida:** 
       - *No supervisada:* **Isolation Forest** para descubrir anomalías no catalogadas.
       - *Supervisada:* **Random Forest** balanceado con **SMOTE** para optimizar el Recall.
    6. **Evaluación de Modelos:** Optimización del umbral de corte orientada a métrica **F2-Score** para castigar los Falsos Negativos.
    7. **Extracción de Reglas de Negocio:** 5 reglas de decisión determinísticas de alta interpretabilidad validadas con Lift estadístico.
    8. **Despliegue y Orquestación:** Orquestación con **Apache Airflow 3** en Docker para scoring en micro-lotes y reentrenamiento semanal continuo.
    """)
    st.image("results/04_comparativa_modelos_curvas_roc_pr.png", caption="Curvas ROC y Precision-Recall de los modelos evaluados.")
