-- ==============================================================================
-- 04_seed_and_migrate.sql
-- Distribución de datos desde Staging hacia raw.transactions y raw.fraud_labels
-- ==============================================================================

-- Conectarse a la base fraud_detection_kdd:
-- \c fraud_detection_kdd;

-- 1. Insertar en raw.transactions (40 variables predictoras pre-evento)
INSERT INTO raw.transactions (
    transaction_id,
    customer_id,
    card_id,
    merchant_id,
    transaction_datetime,
    transaction_amount,
    merchant_category,
    merchant_country,
    transaction_hour,
    transaction_day_of_week,
    card_present,
    customer_age,
    customer_gender,
    customer_state,
    account_age_days,
    credit_limit,
    credit_score,
    num_cards_held,
    transactions_last_1h,
    transactions_last_24h,
    transactions_last_7d,
    amount_last_1h,
    amount_last_24h,
    days_since_last_transaction,
    distance_from_home_km,
    distance_from_last_tx_km,
    international_transaction,
    high_risk_country,
    amount_vs_avg_ratio,
    used_chip,
    used_pin,
    transaction_channel,
    device_type,
    browser,
    is_weekend,
    avg_transaction_amount_30d,
    std_transaction_amount_30d,
    num_declined_last_30d,
    num_chargebacks_ever,
    email_domain
)
SELECT 
    transaction_id,
    customer_id,
    card_id,
    merchant_id,
    transaction_datetime,
    transaction_amount,
    merchant_category,
    merchant_country,
    transaction_hour,
    transaction_day_of_week,
    card_present,
    customer_age,
    customer_gender,
    customer_state,
    account_age_days,
    credit_limit,
    credit_score,
    num_cards_held,
    transactions_last_1h,
    transactions_last_24h,
    transactions_last_7d,
    amount_last_1h,
    amount_last_24h,
    days_since_last_transaction,
    distance_from_home_km,
    distance_from_last_tx_km,
    international_transaction,
    high_risk_country,
    amount_vs_avg_ratio,
    used_chip,
    used_pin,
    transaction_channel,
    device_type,
    NULLIF(browser, 'N/A'),
    is_weekend,
    avg_transaction_amount_30d,
    std_transaction_amount_30d,
    num_declined_last_30d,
    num_chargebacks_ever,
    email_domain
FROM raw.staging_seed_csv
ON CONFLICT (transaction_id) DO NOTHING;

-- 2. Insertar en raw.fraud_labels (Etiquetas de auditoría / Contracargos)
INSERT INTO raw.fraud_labels (
    transaction_id,
    is_fraud,
    fraud_type,
    fraud_amount_lost,
    label_source,
    labeled_at
)
SELECT 
    transaction_id,
    is_fraud,
    fraud_type,
    fraud_amount_lost,
    'HISTORICAL_KDD_SEED',
    transaction_datetime + INTERVAL '1 day' -- Simulación de reporte posterior
FROM raw.staging_seed_csv
ON CONFLICT (transaction_id) DO NOTHING;

-- 3. Limpiar tabla de paso para liberar espacio si se desea
-- TRUNCATE TABLE raw.staging_seed_csv;

-- 4. Verificación de conteo
SELECT 'Transacciones cargadas en raw.transactions:' AS metrica, COUNT(*) AS total FROM raw.transactions
UNION ALL
SELECT 'Etiquetas cargadas en raw.fraud_labels:', COUNT(*) FROM raw.fraud_labels
UNION ALL
SELECT 'Fraudes confirmados (is_fraud = 1):', COUNT(*) FROM raw.fraud_labels WHERE is_fraud = 1;
