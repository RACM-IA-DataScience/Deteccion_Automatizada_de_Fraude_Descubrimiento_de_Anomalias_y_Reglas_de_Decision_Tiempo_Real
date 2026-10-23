from airflow import DAG
import pandas as pd
import uuid
import random
from datetime import datetime, timezone
import joblib
import json
import numpy as np
import os
import sqlalchemy

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from airflow.providers.postgres.hooks.postgres import PostgresHook


def get_db_engine():
    """Obtiene el motor SQLAlchemy conectando a PostgreSQL."""
    conn_str = os.getenv("AIRFLOW_CONN_POSTGRES_DEFAULT")
    if conn_str:
        try:
            return sqlalchemy.create_engine(conn_str)
        except Exception:
            pass
    user = os.getenv("PGUSER", "postgres")
    pwd = os.getenv("PGPASSWORD", "R0bb34l3x")
    host = os.getenv("PGHOST", "host.docker.internal")
    port = os.getenv("PGPORT", "5432")
    db = os.getenv("PGDATABASE", "fraud_detection_kdd")
    return sqlalchemy.create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}")


def simulate_transactions_callable():
    engine = get_db_engine()
    
    batch_size = random.randint(10, 50)
    
    query = "SELECT * FROM raw.transactions LIMIT 500;"
    df_template = pd.read_sql(query, con=engine)
    
    if df_template.empty:
        raise ValueError("No hay datos en raw.transactions para usar como plantilla.")
        
    df_new = df_template.sample(n=batch_size, replace=True).copy()

    df_new["transaction_id"] = [f"TXN-{uuid.uuid4().hex[:8].upper()}" for _ in range(batch_size)]
    df_new["transaction_datetime"] = datetime.now(timezone.utc)

    df_new["transaction_amount"] = df_new["transaction_amount"] * [random.uniform(0.95, 1.05) for _ in range(batch_size)]
    df_new["transaction_amount"] = df_new["transaction_amount"].round(2)

    df_new.to_sql(
        name="transactions",
        schema="raw",
        con=engine,
        if_exists="append",
        index=False
    )
    print(f"Simulación exitosa: {batch_size} nuevas transacciones inyectadas.")


def detect_fraud_callable():
    engine = get_db_engine()
    
    try:
        query = """
            SELECT * FROM raw.transactions 
            WHERE transaction_id NOT IN (SELECT transaction_id FROM raw.predictions)
            LIMIT 2000;
        """
        df_nuevos = pd.read_sql(query, con=engine)
    except Exception as e:
        print(f"Nota en consulta diferencial: {e}. Procesando lote inicial...")
        query = "SELECT * FROM raw.transactions LIMIT 1000;"
        df_nuevos = pd.read_sql(query, con=engine)
    
    if df_nuevos.empty:
        print("No hay transacciones nuevas para procesar.")
        return
        
    modelo = joblib.load("/opt/airflow/models/fraud_classifier.joblib")
    scaler = joblib.load("/opt/airflow/models/scaler.joblib")
    isolator = joblib.load("/opt/airflow/models/isolation_forest.joblib")
    with open("/opt/airflow/models/model_columns.json", "r") as f:
        selected_features = json.load(f)
        
    features = df_nuevos.copy()
    
    if 'browser' in features.columns:
        features = features.rename(columns={'browser': 'browser_clean'})
    elif 'browser_clean' not in features.columns:
        features['browser_clean'] = "No_Aplica_Fisico"

    features["browser_clean"] = features["browser_clean"].fillna("No_Aplica_Fisico")
    
    features["velocity_tx_ratio_1h_24h"] = features["transactions_last_1h"] / (features["transactions_last_24h"] + 1.0)
    features["velocity_amt_ratio_1h_24h"] = features["amount_last_1h"] / (features["amount_last_24h"] + 1.0)
    features["credit_utilization_ratio"] = features["transaction_amount"] / (features["credit_limit"] + 1.0)
    features["zscore_amount_30d"] = (features["transaction_amount"] - features["avg_transaction_amount_30d"]) / (features["std_transaction_amount_30d"] + 1.0)
    features["security_factors_count"] = features["card_present"] + features["used_chip"] + features["used_pin"]
    features["is_night_tx"] = (features["transaction_hour"] <= 4).astype(int)
    features["is_new_account"] = (features["account_age_days"] <= 90).astype(int)

    features["composite_risk_index"] = (
        features["high_risk_country"] * 2
        + features["international_transaction"]
        + (features["card_present"] == 0).astype(int)
        + (features["transactions_last_1h"] >= 3).astype(int)
        + (features["num_chargebacks_ever"] >= 2).astype(int)
        + (features["num_declined_last_30d"] >= 2).astype(int)
        + features["is_night_tx"]
        + features["is_new_account"]
        + (features["amount_vs_avg_ratio"] > 4.8).astype(int)
        + (features["distance_from_last_tx_km"] > 800).astype(int)
    )
    
    cat_cols = ["merchant_category", "merchant_country", "transaction_day_of_week", "customer_gender", "transaction_channel", "device_type", "browser_clean", "email_domain"]
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True, dtype=int)
    
    log_cols = ["transaction_amount", "amount_last_1h", "amount_last_24h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio"]
    features[log_cols] = np.log1p(np.maximum(0, features[log_cols]))
    
    num_cols = getattr(scaler, "feature_names_in_", None)
    if num_cols is not None:
        for col in num_cols:
            if col not in features.columns:
                features[col] = 0
        features[num_cols] = scaler.transform(features[num_cols])
    
    iso_features = ["transaction_amount", "amount_last_1h", "transactions_last_1h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio", "zscore_amount_30d", "composite_risk_index"]
    iso_predicciones = isolator.predict(features[iso_features])
    features["iso_anomaly_score"] = -isolator.score_samples(features[iso_features])
    
    features_final = features.reindex(columns=selected_features, fill_value=0)
    rf_predicciones = modelo.predict(features_final)
    
    df_resultados = pd.DataFrame({
        'transaction_id': df_nuevos['transaction_id'],
        'is_fraud': rf_predicciones,
        'anomaly_score': iso_predicciones,
        'prediction_datetime': datetime.now(timezone.utc)
    })
    
    df_resultados.to_sql(
        name="predictions",
        schema="raw",
        con=engine,
        if_exists="append",
        index=False
    )
    print(f"Se evaluaron y guardaron {len(df_resultados)} transacciones con éxito.")


with DAG(
    dag_id="fraud_simulation_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
) as dag:

    tarea_simulacion = PythonOperator(
        task_id="Simular_Transacciones",
        python_callable=simulate_transactions_callable
    )
    
    tarea_scoring = PythonOperator(
        task_id="Detectar_Fraude",
        python_callable=detect_fraud_callable
    )

    tarea_simulacion >> tarea_scoring
