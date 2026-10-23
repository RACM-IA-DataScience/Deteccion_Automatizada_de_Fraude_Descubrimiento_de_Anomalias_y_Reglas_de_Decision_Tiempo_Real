-- ==============================================================================
-- 03_feature_store_and_rules.sql
-- Vistas Analíticas: Feature Store y Motor de Reglas de Negocio en SQL
-- ==============================================================================

-- Conectarse a la base fraud_detection_kdd:
-- \c fraud_detection_kdd;

--------------------------------------------------------------------------------
-- 1. VISTA: Feature Store con Imputación Estructural e Ingeniería de Atributos
-- Replica exactamente las 8 variables derivadas del pipeline KDD Etapa 4
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW analytics.v_fraud_feature_store AS
SELECT 
    t.transaction_id,
    t.customer_id,
    t.card_id,
    t.merchant_id,
    t.transaction_datetime,
    
    -- Variables base
    t.transaction_amount,
    t.merchant_category,
    t.merchant_country,
    t.transaction_hour,
    t.transaction_day_of_week,
    t.card_present,
    t.customer_age,
    t.customer_gender,
    t.customer_state,
    t.account_age_days,
    t.credit_limit,
    t.credit_score,
    t.num_cards_held,
    t.transactions_last_1h,
    t.transactions_last_24h,
    t.transactions_last_7d,
    t.amount_last_1h,
    t.amount_last_24h,
    t.days_since_last_transaction,
    t.distance_from_home_km,
    t.distance_from_last_tx_km,
    t.international_transaction,
    t.high_risk_country,
    t.amount_vs_avg_ratio,
    t.used_chip,
    t.used_pin,
    t.transaction_channel,
    t.device_type,
    
    -- Imputación estructural Etapa 3 (Previene sesgo de selección)
    COALESCE(t.browser, 'No_Aplica_Fisico') AS browser_clean,
    
    t.is_weekend,
    t.avg_transaction_amount_30d,
    t.std_transaction_amount_30d,
    t.num_declined_last_30d,
    t.num_chargebacks_ever,
    t.email_domain,

    -- =========================================================================
    -- INGENIERÍA DE CARACTERÍSTICAS (Las 8 variables derivadas de Etapa 4)
    -- =========================================================================
    -- 1. Ratios de Velocidad
    ROUND((t.transactions_last_1h::NUMERIC / (t.transactions_last_24h + 1.0)), 4) AS velocity_tx_ratio_1h_24h,
    ROUND((t.amount_last_1h::NUMERIC / (t.amount_last_24h + 1.0)), 4) AS velocity_amt_ratio_1h_24h,
    
    -- 2. Utilización del crédito transaccional
    ROUND((t.transaction_amount::NUMERIC / (t.credit_limit + 1.0)), 4) AS credit_utilization_ratio,
    
    -- 3. Z-Score relativo de monto a 30 días
    ROUND(((t.transaction_amount - t.avg_transaction_amount_30d)::NUMERIC / (t.std_transaction_amount_30d + 1.0)), 4) AS zscore_amount_30d,
    
    -- 4. Factores de autenticación presentes (0 a 3)
    (t.card_present + t.used_chip + t.used_pin) AS security_factors_count,
    
    -- 5. Indicador horario nocturno (00:00 - 04:59 hrs)
    CASE WHEN t.transaction_hour <= 4 THEN 1 ELSE 0 END AS is_night_tx,
    
    -- 6. Indicador de cuenta nueva (<= 90 días)
    CASE WHEN t.account_age_days <= 90 THEN 1 ELSE 0 END AS is_new_account,
    
    -- 7. Índice Compuesto de Riesgo Multiseñal (0 a 11 puntos)
    (
        (t.high_risk_country * 2)
        + t.international_transaction
        + (CASE WHEN t.card_present = 0 THEN 1 ELSE 0 END)
        + (CASE WHEN t.transactions_last_1h >= 3 THEN 1 ELSE 0 END)
        + (CASE WHEN t.num_chargebacks_ever >= 2 THEN 1 ELSE 0 END)
        + (CASE WHEN t.num_declined_last_30d >= 2 THEN 1 ELSE 0 END)
        + (CASE WHEN t.transaction_hour <= 4 THEN 1 ELSE 0 END)
        + (CASE WHEN t.account_age_days <= 90 THEN 1 ELSE 0 END)
        + (CASE WHEN t.amount_vs_avg_ratio > 4.8 THEN 1 ELSE 0 END)
        + (CASE WHEN t.distance_from_last_tx_km > 800 THEN 1 ELSE 0 END)
    ) AS composite_risk_index

FROM raw.transactions t;

COMMENT ON VIEW analytics.v_fraud_feature_store IS 'Feature Store unificado con las 8 variables derivadas y limpieza estructural';

--------------------------------------------------------------------------------
-- 2. VISTA: Motor de Reglas de Negocio en Tiempo Real (Etapa 7 KDD)
-- Evalúa las 5 reglas Si-Entonces (IF-THEN) con Lift de hasta 16.46x
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW analytics.v_rule_engine_eval AS
SELECT 
    f.transaction_id,
    f.transaction_datetime,
    f.transaction_amount,
    f.customer_id,
    
    -- Regla R1: Riesgo Crítico Internacional + Velocidad (Lift: 16.46x, Precisión: 33.8%)
    (f.high_risk_country = 1 AND f.card_present = 0 AND f.transactions_last_1h >= 3) AS rule_r1_international_velocity,

    -- Regla R2: País de Alto Riesgo + Cuenta Nueva <= 90 días (Lift: 11.28x, Precisión: 23.2%)
    (f.high_risk_country = 1 AND f.account_age_days <= 90) AS rule_r2_high_risk_new_account,

    -- Regla R3: Madrugada 00-04h + Internacional + Sin Tarjeta (Lift: 6.15x, Precisión: 12.6%)
    (f.transaction_hour <= 4 AND f.international_transaction = 1 AND f.card_present = 0) AS rule_r3_night_international_cardless,

    -- Regla R4: Monto Desviado > 4.8x + Contracargos Previos >= 2 (Lift: 4.94x, Precisión: 10.2%)
    (f.amount_vs_avg_ratio > 4.8 AND f.num_chargebacks_ever >= 2 AND f.card_present = 0) AS rule_r4_amount_spike_chargebacks,

    -- Regla R5: Acumulación Multiseñal Crítica (composite_risk_index >= 5) (Lift: 9.79x, Precisión: 20.1%)
    (f.composite_risk_index >= 5) AS rule_r5_composite_risk_critical,

    -- Conteo total de reglas activadas
    (
        (CASE WHEN f.high_risk_country = 1 AND f.card_present = 0 AND f.transactions_last_1h >= 3 THEN 1 ELSE 0 END) +
        (CASE WHEN f.high_risk_country = 1 AND f.account_age_days <= 90 THEN 1 ELSE 0 END) +
        (CASE WHEN f.transaction_hour <= 4 AND f.international_transaction = 1 AND f.card_present = 0 THEN 1 ELSE 0 END) +
        (CASE WHEN f.amount_vs_avg_ratio > 4.8 AND f.num_chargebacks_ever >= 2 AND f.card_present = 0 THEN 1 ELSE 0 END) +
        (CASE WHEN f.composite_risk_index >= 5 THEN 1 ELSE 0 END)
    ) AS total_rules_triggered

FROM analytics.v_fraud_feature_store f;

COMMENT ON VIEW analytics.v_rule_engine_eval IS 'Evaluación instantánea de las 5 reglas de decisión validadas en KDD';

--------------------------------------------------------------------------------
-- 3. VISTA: Dataset de Entrenamiento Histórico
-- Cruza el Feature Store con las etiquetas auditadas para el reentrenamiento
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW analytics.v_model_training_dataset AS
SELECT 
    f.*,
    l.is_fraud,
    l.fraud_type,
    l.fraud_amount_lost
FROM analytics.v_fraud_feature_store f
INNER JOIN raw.fraud_labels l ON f.transaction_id = l.transaction_id;

COMMENT ON VIEW analytics.v_model_training_dataset IS 'Dataset de entrenamiento con etiquetas auditadas para reentrenamiento continuo';
