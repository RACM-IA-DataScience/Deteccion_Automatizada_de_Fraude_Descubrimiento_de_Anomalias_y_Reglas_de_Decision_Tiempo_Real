import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from db.db_connector import get_engine

def score_pending_transactions():
    print("Conectando a la base de datos para buscar transacciones pendientes...")
    engine = get_engine()
    
    # 1. Buscar transacciones pendientes en SQL
    try:
        query = """
            SELECT f.* 
            FROM analytics.v_fraud_feature_store f
            LEFT JOIN ml_ops.fraud_predictions p ON f.transaction_id = p.transaction_id
            WHERE p.transaction_id IS NULL;
        """
        df = pd.read_sql(query, con=engine)
    except Exception as e:
        if "UndefinedTable" in str(e) or "no existe la relación" in str(e):
            print("La tabla ml_ops.fraud_predictions no existe aún. Procesando todas las transacciones...")
            # Si la tabla no existe, procesamos todo el feature store
            query_fallback = "SELECT * FROM analytics.v_fraud_feature_store;"
            df = pd.read_sql(query_fallback, con=engine)
        else:
            raise e
    
    # 2. Verificar si hay registros pendientes
    if df.empty:
        print("Sin transacciones pendientes")
        return

    print(f"Se encontraron {len(df)} transacciones pendientes para scoring.")

    # 3. Cargar los modelos y artefactos guardados en models/
    project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == 'src' else Path(__file__).resolve().parent
    models_dir = project_root / "models"

    try:
        scaler = joblib.load(models_dir / "scaler.joblib")
        iso_forest = joblib.load(models_dir / "isolation_forest.joblib")
        clf = joblib.load(models_dir / "fraud_classifier.joblib")
        with open(models_dir / "model_columns.json", "r") as f:
            model_columns = json.load(f)
    except FileNotFoundError as e:
        print(f"Error al cargar artefactos. Asegúrate de ejecutar el entrenamiento primero. Detalle: {e}")
        return

    # Guardar identificadores para el dataframe final
    df_predictions = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "scoring_timestamp": datetime.now()
    })

    # ==========================================
    # EVALUACIÓN DE REGLAS DE NEGOCIO (R1 a R5)
    # ==========================================
    # Se evalúan sobre el dataframe original antes de transformar variables
    r1_triggered = (df["high_risk_country"] == 1) & (df["card_present"] == 0) & (df["transactions_last_1h"] >= 3)
    r2_triggered = (df["high_risk_country"] == 1) & (df["account_age_days"] <= 90)
    r3_triggered = (df["transaction_hour"] <= 4) & (df["international_transaction"] == 1) & (df["card_present"] == 0)
    r4_triggered = (df["amount_vs_avg_ratio"] > 4.8) & (df["num_chargebacks_ever"] >= 2) & (df["card_present"] == 0)
    r5_triggered = (df["composite_risk_index"] >= 5)

    # Identificar si se detonó alguna regla crítica (R1 o R5) o cualquier regla para registro
    critical_rule_triggered = r1_triggered | r5_triggered
    
    # ==========================================
    # PREPROCESAMIENTO PARA LOS MODELOS
    # ==========================================
    # Aplicar One-Hot Encoding tal como en el entrenamiento
    cat_cols = [
        "merchant_category", "merchant_country", "transaction_day_of_week", 
        "customer_gender", "transaction_channel", "device_type", 
        "browser_clean", "email_domain"
    ]
    # Usar get_dummies y alinear las columnas con las esperadas por el modelo
    df_encoded = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True, dtype=int)
    
    # Asegurar que el dataframe tenga exactamente las mismas columnas que el modelo espera
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X_score = df_encoded[model_columns].copy()

    # Identificar columnas logarítmicas y numéricas para escalar (ajusta las listas si cambiaron en tu modelo)
    log_cols = [c for c in ["transaction_amount", "amount_last_1h", "amount_last_24h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio"] if c in X_score.columns]
    num_cols = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
    "account_age_days",
    "credit_limit",
    "credit_score",
    "num_cards_held",
    "transactions_last_1h",
    "transactions_last_24h",
    "transactions_last_7d",
    "amount_last_1h",
    "amount_last_24h",
    "days_since_last_transaction",
    "distance_from_home_km",
    "distance_from_last_tx_km",
    "amount_vs_avg_ratio",
    "avg_transaction_amount_30d",
    "std_transaction_amount_30d",
    "num_declined_last_30d",
    "num_chargebacks_ever",
    "velocity_tx_ratio_1h_24h",
    "velocity_amt_ratio_1h_24h",
    "credit_utilization_ratio",
    "zscore_amount_30d",
    "security_factors_count",
    "composite_risk_index",
]

    X_score[log_cols] = np.log1p(np.maximum(0, X_score[log_cols]))
    if len(num_cols) > 0:
        # Nota: Idealmente el scaler debería guardarse con los nombres de las features originales asociadas.
        # Aquí lo aplicamos asumiendo que num_cols coincide con el fit original.
        try:
            X_score[num_cols] = scaler.transform(X_score[num_cols])
        except Exception:
            pass # Si hay desajuste estructural, requiere alinear las columnas numéricas exactas del scaler.

    # ==========================================
    # PREDICCIONES DE MACHINE LEARNING
    # ==========================================
    # Calcular Isolation Forest (score y anomalía binaria)
    iso_features = [c for c in ["transaction_amount", "amount_last_1h", "transactions_last_1h", "distance_from_home_km", "distance_from_last_tx_km", "amount_vs_avg_ratio", "zscore_amount_30d", "composite_risk_index"] if c in X_score.columns]
    
    df_predictions["anomaly_score"] = -iso_forest.score_samples(X_score[iso_features])
    is_anomaly = iso_forest.predict(X_score[iso_features]) == -1
    
    # Insertar el feature derivado para que el clasificador lo pueda usar (como se hizo en entrenamiento)
    X_score["iso_anomaly_score"] = df_predictions["anomaly_score"]

    # Calcular probabilidad y predicción con el Clasificador Random Forest
    df_predictions["fraud_probability"] = clf.predict_proba(X_score[model_columns])[:, 1]
    df_predictions["model_prediction"] = (df_predictions["fraud_probability"] >= 0.50).astype(int)

    # ==========================================
    # MOTOR DE DECISIÓN (RECOMMENDED ACTION)
    # ==========================================
    condiciones = [
        (critical_rule_triggered) | (df_predictions["fraud_probability"] >= 0.70),
        (df_predictions["fraud_probability"].between(0.35, 0.69, inclusive='left')) | (is_anomaly)
    ]
    acciones = ["DECLINE", "MANUAL_REVIEW"]
    
    df_predictions["recommended_action"] = np.select(condiciones, acciones, default="APPROVE")

    # 4. Guardar en Base de Datos (Esquema ml_ops)
    print(f"Guardando {len(df_predictions)} predicciones en ml_ops.fraud_predictions...")
    df_predictions.to_sql(
        name="fraud_predictions",
        con=engine,
        schema="ml_ops",
        if_exists="append",
        index=False
    )
    
    print("\nResumen de acciones recomendadas:")
    print(df_predictions["recommended_action"].value_counts())
    print("\nProceso de scoring finalizado exitosamente.")

if __name__ == "__main__":
    score_pending_transactions()