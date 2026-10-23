from kafka import KafkaProducer
import json
import uuid
import random
from datetime import datetime, timezone

# Conectar al broker de Kafka local
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
)

# Simular una transacción de prueba con datos que simulen alto riesgo
transaccion_prueba = {
    "transaction_id": f"TXN-TEST-{uuid.uuid4().hex[:8].upper()}",
    "transaction_datetime": datetime.now(timezone.utc),
    "transaction_amount": 1500.0,
    "customer_age": 25,
    "account_age_days": 10,  # Cuenta muy nueva
    "credit_limit": 2000.0,
    "credit_score": 600,
    "num_cards_held": 1,
    "transactions_last_1h": 4,  # Alta frecuencia
    "transactions_last_24h": 10,
    "transactions_last_7d": 15,
    "amount_last_1h": 1200.0,
    "amount_last_24h": 3000.0,
    "days_since_last_transaction": 0,
    "distance_from_home_km": 500.0,
    "distance_from_last_tx_km": 300.0,
    "amount_vs_avg_ratio": 5.2,
    "avg_transaction_amount_30d": 200.0,
    "std_transaction_amount_30d": 50.0,
    "num_declined_last_30d": 2,
    "num_chargebacks_ever": 1,
    "high_risk_country": 1,  # País de alto riesgo
    "international_transaction": 1,
    "browser": "Chrome",
    "card_present": 0,  # Tarjeta no presente
    "used_chip": 0,
    "used_pin": 0,
    "transaction_hour": 2  # Madrugada
}

# Enviar el evento al tópico
print("Enviando transacción de prueba a Kafka...")
producer.send('credit-card-transactions', value=transaccion_prueba)
producer.flush()
print("¡Transacción enviada con éxito!")