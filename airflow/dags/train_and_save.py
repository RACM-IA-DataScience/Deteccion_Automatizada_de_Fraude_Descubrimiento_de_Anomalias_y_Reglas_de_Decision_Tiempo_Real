import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import sqlalchemy
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

warnings.filterwarnings("ignore")

DEFAULT_SELECTED_FEATURES = [
    "transaction_amount", "transaction_hour", "customer_age", "account_age_days",
    "credit_limit", "credit_score", "num_cards_held", "transactions_last_1h",
    "transactions_last_24h", "transactions_last_7d", "amount_last_1h",
    "amount_last_24h", "days_since_last_transaction", "distance_from_home_km",
    "distance_from_last_tx_km", "amount_vs_avg_ratio", "avg_transaction_amount_30d",
    "std_transaction_amount_30d", "num_declined_last_30d", "num_chargebacks_ever",
    "velocity_tx_ratio_1h_24h", "velocity_amt_ratio_1h_24h", "credit_utilization_ratio",
    "zscore_amount_30d", "security_factors_count", "composite_risk_index",
    "iso_anomaly_score", "high_risk_country", "merchant_country_USA",
    "merchant_country_Romania", "merchant_country_China", "international_transaction",
    "merchant_country_Nigeria", "merchant_country_Russia", "is_new_account",
    "browser_clean_No_Aplica_Fisico", "card_present", "transaction_channel_In-store",
    "device_type_POS Terminal", "transaction_channel_Online", "used_chip",
    "is_night_tx", "used_pin", "device_type_Desktop", "merchant_category_Online Retail"
]


def get_engine():
    """Obtiene el motor SQLAlchemy directo a PostgreSQL."""
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


def load_data_from_db():
    """Carga el dataset de entrenamiento desde la vista de PostgreSQL."""
    print("Conectando a PostgreSQL...")
    engine = get_engine()
    query = "SELECT * FROM analytics.v_model_training_dataset;"
    df = pd.read_sql(query, con=engine)
    print(f"Dataset cargado: {df.shape[0]:,} transacciones y {df.shape[1]} columnas.")
    return df


def train_and_save_model():
    """Ejecuta el ciclo de reentrenamiento y serialización de modelos."""
    print("=" * 80)
    print("INICIANDO REENTRENAMIENTO DEL PIPELINE DE MODELOS DE FRAUDE (KDD)")
    print("=" * 80)
    
    df = load_data_from_db()
    
    # 1. Ingeniería de características conductuales
    print("Calculando 8 variables de comportamiento financiero...")
    df["velocity_tx_ratio_1h_24h"] = df["transactions_last_1h"] / (df["transactions_last_24h"] + 1.0)
    df["velocity_amt_ratio_1h_24h"] = df["amount_last_1h"] / (df["amount_last_24h"] + 1.0)
    df["credit_utilization_ratio"] = df["transaction_amount"] / (df["credit_limit"] + 1.0)
    df["zscore_amount_30d"] = (
        df["transaction_amount"] - df["avg_transaction_amount_30d"]
    ) / (df["std_transaction_amount_30d"] + 1.0)
    df["security_factors_count"] = df["card_present"] + df["used_chip"] + df["used_pin"]
    df["is_night_tx"] = (df["transaction_hour"] <= 4).astype(int)
    df["is_new_account"] = (df["account_age_days"] <= 90).astype(int)

    df["composite_risk_index"] = (
        df["high_risk_country"] * 2
        + df["international_transaction"]
        + (df["card_present"] == 0).astype(int)
        + (df["transactions_last_1h"] >= 3).astype(int)
        + (df["num_chargebacks_ever"] >= 2).astype(int)
        + (df["num_declined_last_30d"] >= 2).astype(int)
        + df["is_night_tx"]
        + df["is_new_account"]
        + (df["amount_vs_avg_ratio"] > 4.8).astype(int)
        + (df["distance_from_last_tx_km"] > 800).astype(int)
    )

    if 'browser' in df.columns and 'browser_clean' not in df.columns:
        df = df.rename(columns={'browser': 'browser_clean'})
    elif 'browser_clean' not in df.columns:
        df['browser_clean'] = "No_Aplica_Fisico"
    df["browser_clean"] = df["browser_clean"].fillna("No_Aplica_Fisico")

    # 2. Codificación One-Hot de categorías nominales
    cat_cols = [
        "merchant_category",
        "merchant_country",
        "transaction_day_of_week",
        "customer_gender",
        "transaction_channel",
        "device_type",
        "browser_clean",
        "email_domain",
    ]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # 3. Separación X e y con partición estratificada
    y = df_encoded["is_fraud"].copy()
    X = df_encoded.drop(columns=["is_fraud"]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Transformación Logarítmica y Escalamiento Robusto
    log_cols = [
        "transaction_amount",
        "amount_last_1h",
        "amount_last_24h",
        "distance_from_home_km",
        "distance_from_last_tx_km",
        "amount_vs_avg_ratio",
    ]
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
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[log_cols] = np.log1p(np.maximum(0, X_train[log_cols]))
    X_test[log_cols] = np.log1p(np.maximum(0, X_test[log_cols]))

    scaler = RobustScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 5. Detección no supervisada de anomalías (Isolation Forest)
    print("Entrenando Isolation Forest para anomalías...")
    iso_features = [
        "transaction_amount",
        "amount_last_1h",
        "transactions_last_1h",
        "distance_from_home_km",
        "distance_from_last_tx_km",
        "amount_vs_avg_ratio",
        "zscore_amount_30d",
        "composite_risk_index",
    ]
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.035, random_state=42, n_jobs=-1
    )
    iso_forest.fit(X_train[iso_features])
    
    # Meta-característica de anomalía
    X_train["iso_anomaly_score"] = -iso_forest.score_samples(X_train[iso_features])
    X_test["iso_anomaly_score"] = -iso_forest.score_samples(X_test[iso_features])

    # 6. Selección de características para el modelo Random Forest
    models_dir = Path("/opt/airflow/models") if Path("/opt/airflow/models").exists() else Path(__file__).resolve().parent.parent / "models"
    cols_file = models_dir / "model_columns.json"
    if cols_file.exists():
        with open(cols_file, "r") as f:
            selected_features = json.load(f)
    else:
        selected_features = DEFAULT_SELECTED_FEATURES

    X_train_model = X_train.reindex(columns=selected_features, fill_value=0)
    X_test_model = X_test.reindex(columns=selected_features, fill_value=0)

    # 7. Clasificador supervisado de fraude (Random Forest con balanceo de pesos)
    print("Entrenando clasificador supervisado (Random Forest)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_model, y_train)

    y_pred = rf.predict(X_test_model)
    y_proba = rf.predict_proba(X_test_model)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Rendimiento en Test -> ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # 8. Serialización y almacenamiento de artefactos
    models_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(iso_forest, models_dir / "isolation_forest.joblib")
    joblib.dump(rf, models_dir / "fraud_classifier.joblib")
    
    with open(cols_file, "w") as f:
        json.dump(selected_features, f)
        
    print(f"ARTEFACTOS ACTUALIZADOS EXITOSAMENTE EN: {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train_and_save_model()
