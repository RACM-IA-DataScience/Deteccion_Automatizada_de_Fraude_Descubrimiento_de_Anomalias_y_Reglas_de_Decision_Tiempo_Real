from kafka import KafkaConsumer
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine

# 1. Configurar conexión a PostgreSQL (Ajusta tus credenciales)
DATABASE_URL = "postgresql://usuario:contraseña@localhost:5432/fraud_detection_kdd"
engine = create_engine(DATABASE_URL)

# 2. Cargar modelos en memoria RAM una sola vez
modelo = joblib.load("models/fraud_classifier.joblib")
scaler = joblib.load("models/scaler.joblib")
isolator = joblib.load("models/isolation_forest.joblib")

consumer = KafkaConsumer(
    'credit-card-transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True
)

print("🚀 Sistema en tiempo real activo. Escuchando transacciones...")

def disparar_alerta_inmediata(tx_data):
    """Simula el envío de una alerta por Webhook, SMS o bloqueo automático."""
    print(f"🚨 [ALERTA ROJA] ¡Fraude detectado en la transacción {tx_data['transaction_id']}!")
    print(f"   - Monto: ${tx_data['transaction_amount']}")
    print(f"   - Acción: Notificación enviada vía Webhook / Bloqueo preventivo activado.")
    # Aquí puedes integrar solicitudes HTTP (requests.post) para un Webhook o una API de SMS (Twilio, etc.)

for message in consumer:
    tx_data = message.value
    df_tx = pd.DataFrame([tx_data])
    
    # --- PROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS ---
    if 'browser' in df_tx.columns:
        df_tx = df_tx.rename(columns={'browser': 'browser_clean'})
    elif 'browser_clean' not in df_tx.columns:
        df_tx['browser_clean'] = "No_Aplica_Fisico"
    df_tx["browser_clean"] = df_tx["browser_clean"].fillna("No_Aplica_Fisico")
    
    df_tx["velocity_tx_ratio_1h_24h"] = df_tx["transactions_last_1h"] / (df_tx["transactions_last_24h"] + 1.0)
    df_tx["velocity_amt_ratio_1h_24h"] = df_tx["amount_last_1h"] / (df_tx["amount_last_24h"] + 1.0)
    df_tx["credit_utilization_ratio"] = df_tx["transaction_amount"] / (df_tx["credit_limit"] + 1.0)
    df_tx["zscore_amount_30d"] = (df_tx["transaction_amount"] - df_tx["avg_transaction_amount_30d"]) / (df_tx["std_transaction_amount_30d"] + 1.0)
    df_tx["security_factors_count"] = df_tx["card_present"] + df_tx["used_chip"] + df_tx["used_pin"]
    df_tx["is_night_tx"] = (df_tx["transaction_hour"] <= 4).astype(int)
    df_tx["is_new_account"] = (df_tx["account_age_days"] <= 90).astype(int)

    df_tx["composite_risk_index"] = (
        df_tx["high_risk_country"] * 2
        + df_tx["international_transaction"]
        + (df_tx["card_present"] == 0).astype(int)
        + (df_tx["transactions_last_1h"] >= 3).astype(int)
        + (df_tx["num_chargebacks_ever"] >= 2).astype(int)
        + (df_tx["num_declined_last_30d"] >= 2).astype(int)
        + df_tx["is_night_tx"]
        + df_tx["is_new_account"]
        + (df_tx["amount_vs_avg_ratio"] > 4.8).astype(int)
        + (df_tx["distance_from_last_tx_km"] > 800).astype(int)
    )
    
    cat_cols = ["merchant_category", "merchant_country", "transaction_day_of_week", "customer_gender", "transaction_channel", "device_type", "browser_clean", "email_domain"]
    df_tx_encoded = pd.get_dummies(df_tx, columns=cat_cols, drop_first=True, dtype=int)
    
    log_cols = ["transaction_amount", "amount_last_1h", "amount_last_24h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio"]
    df_tx_encoded[log_cols] = np.log1p(np.maximum(0, df_tx_encoded[log_cols]))
    
    num_cols = scaler.feature_names_in_
    for col in num_cols:
        if col not in df_tx_encoded.columns:
            df_tx_encoded[col] = 0
    df_tx_encoded[num_cols] = scaler.transform(df_tx_encoded[num_cols])
    
    iso_features = ["transaction_amount", "amount_last_1h", "transactions_last_1h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio", "zscore_amount_30d", "composite_risk_index"]
    iso_predicciones = isolator.predict(df_tx_encoded[iso_features])
    df_tx_encoded["iso_anomaly_score"] = -isolator.score_samples(df_tx_encoded[iso_features])
    
    with open("models/model_columns.json", "r") as f:
        selected_features = json.load(f)
        
    features_final = df_tx_encoded.reindex(columns=selected_features, fill_value=0)
    
    # --- PREDICCIÓN DE LOS MODELOS ---
    fraud_prob = float(modelo.predict_proba(features_final)[:, 1][0])
    is_fraud = int(fraud_prob >= 0.50)
    anomaly_val = float(df_tx_encoded["iso_anomaly_score"].iloc[0])
    is_anomaly = bool(iso_predicciones[0] == -1)

    r1_triggered = bool(
        (df_tx["high_risk_country"].iloc[0] == 1)
        and (df_tx["card_present"].iloc[0] == 0)
        and (df_tx["transactions_last_1h"].iloc[0] >= 3)
    )
    r5_triggered = bool(df_tx["composite_risk_index"].iloc[0] >= 5)

    if r1_triggered or r5_triggered or fraud_prob >= 0.70:
        rec_action = "DECLINE"
    elif fraud_prob >= 0.35 or is_anomaly:
        rec_action = "MANUAL_REVIEW"
    else:
        rec_action = "APPROVE"
    
    # 3. EVALUAR ALERTA INMEDIATA SI ES FRAUDE
    if is_fraud == 1 or rec_action == "DECLINE":
        disparar_alerta_inmediata(tx_data)
    else:
        print(f"✅ Transacción legítima procesada: {tx_data['transaction_id']}")

    # 4. PERSISTENCIA SIMULTÁNEA EN POSTGRESQL (Auditoría)
    now_ts = datetime.now(timezone.utc)
    df_resultado = pd.DataFrame({
        'transaction_id': [tx_data['transaction_id']],
        'scoring_timestamp': [now_ts],
        'prediction_datetime': [now_ts],
        'fraud_probability': [fraud_prob],
        'model_prediction': [is_fraud],
        'is_fraud': [is_fraud],
        'anomaly_score': [anomaly_val],
        'recommended_action': [rec_action]
    })
    
    df_resultado.to_sql(
        name="predictions",
        schema="raw",
        con=engine,
        if_exists="append",
        index=False
    )
