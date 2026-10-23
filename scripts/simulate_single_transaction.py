#!/usr/bin/env python
"""
Script CLI: Simulación y Evaluación Unitaria de Transacciones (FraudGuard CLI)
Permite inyectar transacciones de prueba directamente desde la terminal,
evaluarlas con el ensamble de modelos y mostrar la alerta correspondiente.

Uso:
    python scripts/simulate_single_transaction.py --preset normal
    python scripts/simulate_single_transaction.py --preset fraud
    python scripts/simulate_single_transaction.py --preset anomaly
    python scripts/simulate_single_transaction.py --amount 3000 --high-risk-country 1
"""

import sys
import argparse
import json
from pathlib import Path

# Configurar encoding UTF-8 para consola en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Agregar raíz al sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_engine import FraudInferenceEngine


PRESETS = {
    "normal": {
        "transaction_amount": 32.50, "transaction_hour": 14, "customer_age": 38,
        "account_age_days": 850, "credit_limit": 6000.0, "credit_score": 740,
        "num_cards_held": 1, "transactions_last_1h": 1, "transactions_last_24h": 2,
        "transactions_last_7d": 9, "amount_last_1h": 32.50, "amount_last_24h": 70.0,
        "days_since_last_transaction": 2, "distance_from_home_km": 3.2,
        "distance_from_last_tx_km": 1.0, "amount_vs_avg_ratio": 0.88,
        "avg_transaction_amount_30d": 40.0, "std_transaction_amount_30d": 12.0,
        "num_declined_last_30d": 0, "num_chargebacks_ever": 0,
        "card_present": 1, "used_chip": 1, "used_pin": 1,
        "high_risk_country": 0, "international_transaction": 0,
        "merchant_category": "Supermarket", "merchant_country": "USA",
        "transaction_day_of_week": "Wednesday", "customer_gender": "F",
        "transaction_channel": "In-store", "device_type": "POS Terminal",
        "browser_clean": "No_Aplica_Fisico", "email_domain": "gmail.com"
    },
    "fraud": {
        "transaction_amount": 2950.00, "transaction_hour": 3, "customer_age": 28,
        "account_age_days": 40, "credit_limit": 3000.0, "credit_score": 570,
        "num_cards_held": 3, "transactions_last_1h": 4, "transactions_last_24h": 7,
        "transactions_last_7d": 15, "amount_last_1h": 4500.00, "amount_last_24h": 6800.00,
        "days_since_last_transaction": 0, "distance_from_home_km": 9200.0,
        "distance_from_last_tx_km": 9100.0, "amount_vs_avg_ratio": 7.2,
        "avg_transaction_amount_30d": 65.0, "std_transaction_amount_30d": 18.0,
        "num_declined_last_30d": 3, "num_chargebacks_ever": 2,
        "card_present": 0, "used_chip": 0, "used_pin": 0,
        "high_risk_country": 1, "international_transaction": 1,
        "merchant_category": "Online Retail", "merchant_country": "Nigeria",
        "transaction_day_of_week": "Sunday", "customer_gender": "M",
        "transaction_channel": "Online", "device_type": "Desktop",
        "browser_clean": "Chrome", "email_domain": "tempmail.com"
    },
    "anomaly": {
        "transaction_amount": 1450.00, "transaction_hour": 16, "customer_age": 45,
        "account_age_days": 1200, "credit_limit": 8000.0, "credit_score": 710,
        "num_cards_held": 2, "transactions_last_1h": 2, "transactions_last_24h": 4,
        "transactions_last_7d": 10, "amount_last_1h": 1450.00, "amount_last_24h": 1800.00,
        "days_since_last_transaction": 1, "distance_from_home_km": 1250.0,
        "distance_from_last_tx_km": 1180.0, "amount_vs_avg_ratio": 5.4,
        "avg_transaction_amount_30d": 55.0, "std_transaction_amount_30d": 15.0,
        "num_declined_last_30d": 1, "num_chargebacks_ever": 0,
        "card_present": 0, "used_chip": 0, "used_pin": 0,
        "high_risk_country": 0, "international_transaction": 1,
        "merchant_category": "Electronics", "merchant_country": "Romania",
        "transaction_day_of_week": "Friday", "customer_gender": "F",
        "transaction_channel": "Online", "device_type": "Mobile",
        "browser_clean": "Safari", "email_domain": "yahoo.com"
    }
}


def main():
    parser = argparse.ArgumentParser(description="Simular e inferir riesgo de fraude en una transacción.")
    parser.add_argument("--preset", choices=["normal", "fraud", "anomaly"], default="fraud", help="Escenario predefinido")
    parser.add_argument("--amount", type=float, default=None, help="Monto de la transacción")
    parser.add_argument("--hour", type=int, default=None, help="Hora de la transacción (0-23)")
    parser.add_argument("--high-risk-country", type=int, choices=[0, 1], default=None, help="Indicador país de alto riesgo")
    parser.add_argument("--card-present", type=int, choices=[0, 1], default=None, help="Tarjeta presente")
    parser.add_argument("--tx-last-1h", type=int, default=None, help="Transacciones en última hora")
    parser.add_argument("--json", action="store_true", help="Salida exclusivamente en formato JSON")

    args = parser.parse_args()

    tx_data = PRESETS[args.preset].copy()
    if args.amount is not None:
        tx_data["transaction_amount"] = args.amount
    if args.hour is not None:
        tx_data["transaction_hour"] = args.hour
    if args.high_risk_country is not None:
        tx_data["high_risk_country"] = args.high_risk_country
    if args.card_present is not None:
        tx_data["card_present"] = args.card_present
    if args.tx_last_1h is not None:
        tx_data["transactions_last_1h"] = args.tx_last_1h

    engine = FraudInferenceEngine()
    result = engine.predict(tx_data)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 80)
    print(f"🛡️  EVALUACIÓN DE FRAUDE EN TIEMPO REAL — ESCENARIO: {args.preset.upper()}")
    print("=" * 80)
    print(f"ID Transacción:         {result['transaction_id']}")
    print(f"Monto:                  ${tx_data['transaction_amount']:,.2f} USD")
    print(f"Comercio / País:        {tx_data['merchant_category']} ({tx_data['merchant_country']})")
    print("-" * 80)
    print("MÉTRICAS DE INFERENCIA:")
    print(f"  • Probabilidad de Fraude (Random Forest): {result['fraud_probability'] * 100:.2f}%")
    print(f"  • Score de Anomalía (Isolation Forest):    {result['anomaly_score']:.4f} (¿Anomalía?: {'SÍ' if result['is_anomaly'] else 'NO'})")
    print(f"  • Índice de Riesgo Compuesto:              {result['engineered_features']['composite_risk_index']} señales")
    print(f"  • Factores de Autenticación Activos:       {result['engineered_features']['security_factors_count']} / 3")
    print("-" * 80)
    print("REGLAS DE NEGOCIO DETONADAS:")
    triggered_rules = [r for r in result["rules_evaluated"] if r["triggered"]]
    if triggered_rules:
        for r in triggered_rules:
            print(f"  🚨 [{r['code']}] {r['name']} — {r['description']}")
    else:
        print("  ✔️ Ninguna regla de negocio violada.")
    print("-" * 80)
    print(f"VEREDICTO FINAL:        >>> {result['recommended_action']} <<<")
    print(f"Descripción:            {result['status_description']}")
    print("-" * 80)
    print("SIMULACIÓN DE ALERTA OPERATIVA:")
    print(f"  📱 SMS al Cliente:  {result['alert']['simulated_sms']}")
    print(f"  🌐 Payload Webhook: {json.dumps(result['alert']['simulated_webhook'], indent=2)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
