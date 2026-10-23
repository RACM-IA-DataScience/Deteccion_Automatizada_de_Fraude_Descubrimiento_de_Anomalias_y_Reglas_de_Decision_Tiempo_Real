-- ==============================================================================
-- 02_create_tables.sql
-- Definición de Tablas DDL (Raw, Labels, Predicciones y Auditoría)
-- ==============================================================================

-- Conectarse a la base fraud_detection_kdd antes de ejecutar:
-- \c fraud_detection_kdd;

--------------------------------------------------------------------------------
-- 1. TABLA STAGING (Para carga directa del CSV original de 100,000 registros)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS raw.staging_seed_csv CASCADE;

CREATE TABLE raw.staging_seed_csv (
    transaction_id                  VARCHAR(32),
    customer_id                     VARCHAR(32),
    card_id                         VARCHAR(32),
    merchant_id                     VARCHAR(32),
    transaction_datetime            TIMESTAMP,
    transaction_amount              NUMERIC(12, 2),
    merchant_category               VARCHAR(100),
    merchant_country                VARCHAR(100),
    transaction_hour                SMALLINT,
    transaction_day_of_week         VARCHAR(20),
    card_present                    SMALLINT,
    customer_age                    SMALLINT,
    customer_gender                 VARCHAR(20),
    customer_state                  VARCHAR(20),
    account_age_days                INTEGER,
    credit_limit                    NUMERIC(12, 2),
    credit_score                    SMALLINT,
    num_cards_held                  SMALLINT,
    transactions_last_1h            SMALLINT,
    transactions_last_24h           SMALLINT,
    transactions_last_7d            SMALLINT,
    amount_last_1h                  NUMERIC(12, 2),
    amount_last_24h                 NUMERIC(12, 2),
    days_since_last_transaction     NUMERIC(10, 2),
    distance_from_home_km           NUMERIC(12, 2),
    distance_from_last_tx_km        NUMERIC(12, 2),
    international_transaction       SMALLINT,
    high_risk_country               SMALLINT,
    amount_vs_avg_ratio             NUMERIC(10, 3),
    used_chip                       SMALLINT,
    used_pin                        SMALLINT,
    transaction_channel             VARCHAR(50),
    device_type                     VARCHAR(50),
    browser                         VARCHAR(50),
    is_weekend                      SMALLINT,
    avg_transaction_amount_30d      NUMERIC(12, 2),
    std_transaction_amount_30d      NUMERIC(12, 2),
    num_declined_last_30d           SMALLINT,
    num_chargebacks_ever            SMALLINT,
    email_domain                    VARCHAR(100),
    is_fraud                        SMALLINT,
    fraud_type                      VARCHAR(50),
    fraud_amount_lost               NUMERIC(12, 2)
);

COMMENT ON TABLE raw.staging_seed_csv IS 'Tabla de paso inicial para importar el CSV credit_card_fraud_detection_2026';

--------------------------------------------------------------------------------
-- 2. TABLA TRANSACCIONAL PRINCIPAL (Producción / OLTP)
-- Contiene las transacciones en tiempo de ocurrencia (sin datos post-evento)
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS raw.transactions CASCADE;

CREATE TABLE raw.transactions (
    transaction_id                  VARCHAR(32) PRIMARY KEY,
    customer_id                     VARCHAR(32) NOT NULL,
    card_id                         VARCHAR(32) NOT NULL,
    merchant_id                     VARCHAR(32) NOT NULL,
    transaction_datetime            TIMESTAMP NOT NULL,
    transaction_amount              NUMERIC(12, 2) NOT NULL CHECK (transaction_amount >= 0),
    merchant_category               VARCHAR(100) NOT NULL,
    merchant_country                VARCHAR(100) NOT NULL,
    transaction_hour                SMALLINT NOT NULL CHECK (transaction_hour BETWEEN 0 AND 23),
    transaction_day_of_week         VARCHAR(20) NOT NULL,
    card_present                    SMALLINT NOT NULL CHECK (card_present IN (0, 1)),
    customer_age                    SMALLINT NOT NULL,
    customer_gender                 VARCHAR(20) NOT NULL,
    customer_state                  VARCHAR(20) NOT NULL,
    account_age_days                INTEGER NOT NULL CHECK (account_age_days >= 0),
    credit_limit                    NUMERIC(12, 2) NOT NULL CHECK (credit_limit >= 0),
    credit_score                    SMALLINT NOT NULL,
    num_cards_held                  SMALLINT NOT NULL DEFAULT 1,
    transactions_last_1h            SMALLINT NOT NULL DEFAULT 0,
    transactions_last_24h           SMALLINT NOT NULL DEFAULT 0,
    transactions_last_7d            SMALLINT NOT NULL DEFAULT 0,
    amount_last_1h                  NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    amount_last_24h                 NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    days_since_last_transaction     NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    distance_from_home_km           NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    distance_from_last_tx_km        NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    international_transaction       SMALLINT NOT NULL CHECK (international_transaction IN (0, 1)),
    high_risk_country               SMALLINT NOT NULL CHECK (high_risk_country IN (0, 1)),
    amount_vs_avg_ratio             NUMERIC(10, 3) NOT NULL DEFAULT 1.0,
    used_chip                       SMALLINT NOT NULL CHECK (used_chip IN (0, 1)),
    used_pin                        SMALLINT NOT NULL CHECK (used_pin IN (0, 1)),
    transaction_channel             VARCHAR(50) NOT NULL,
    device_type                     VARCHAR(50) NOT NULL,
    browser                         VARCHAR(50), -- NULL para transacciones presenciales
    is_weekend                      SMALLINT NOT NULL CHECK (is_weekend IN (0, 1)),
    avg_transaction_amount_30d      NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    std_transaction_amount_30d      NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    num_declined_last_30d           SMALLINT NOT NULL DEFAULT 0,
    num_chargebacks_ever            SMALLINT NOT NULL DEFAULT 0,
    email_domain                    VARCHAR(100) NOT NULL,
    created_at                      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índices optimizados para consultas en ventanas de tiempo y por cliente/tarjeta
CREATE INDEX IF NOT EXISTS idx_transactions_datetime ON raw.transactions (transaction_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON raw.transactions (customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_card ON raw.transactions (card_id);

--------------------------------------------------------------------------------
-- 3. TABLA DE ETIQUETAS POST-INCIDENTE (Auditoría / Contracargos)
-- Previene Data Leakage: Solo se llena cuando se confirma el reporte bancario
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS raw.fraud_labels CASCADE;

CREATE TABLE raw.fraud_labels (
    transaction_id                  VARCHAR(32) PRIMARY KEY REFERENCES raw.transactions(transaction_id) ON DELETE CASCADE,
    is_fraud                        SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    fraud_type                      VARCHAR(50) NOT NULL,
    fraud_amount_lost               NUMERIC(12, 2) NOT NULL DEFAULT 0.0,
    label_source                    VARCHAR(50) DEFAULT 'CHARGEBACK_CONFIRMED',
    labeled_at                      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fraud_labels_type ON raw.fraud_labels (fraud_type);

--------------------------------------------------------------------------------
-- 4. TABLA DE PREDICCIONES Y MOTOR DE INFERENCIA (MLOps)
-- Guarda la evaluación del sistema antifraude para cada transacción
--------------------------------------------------------------------------------
DROP TABLE IF EXISTS ml_ops.fraud_predictions CASCADE;

CREATE TABLE ml_ops.fraud_predictions (
    prediction_id                   BIGSERIAL PRIMARY KEY,
    transaction_id                  VARCHAR(32) NOT NULL REFERENCES raw.transactions(transaction_id) ON DELETE CASCADE,
    
    -- Minería de anomalías no supervisada
    isolation_forest_anomaly_score  NUMERIC(8, 5),
    is_anomaly                      BOOLEAN,
    
    -- Modelo supervisado (Random Forest / Gradient Boosting)
    fraud_probability               NUMERIC(6, 4) NOT NULL,
    model_prediction                SMALLINT NOT NULL CHECK (model_prediction IN (0, 1)),
    model_version                   VARCHAR(50) NOT NULL DEFAULT 'v1.0.0-kdd2026',
    
    -- Reglas de decisión accionables (R1 - R5)
    r1_international_velocity       BOOLEAN DEFAULT FALSE,
    r2_high_risk_new_account        BOOLEAN DEFAULT FALSE,
    r3_night_international_cardless BOOLEAN DEFAULT FALSE,
    r4_amount_spike_chargebacks     BOOLEAN DEFAULT FALSE,
    r5_composite_risk_critical      BOOLEAN DEFAULT FALSE,
    active_rules_count              SMALLINT DEFAULT 0,
    
    -- Acción recomendada por el motor de decisión
    recommended_action              VARCHAR(20) NOT NULL CHECK (recommended_action IN ('APPROVE', 'MANUAL_REVIEW', 'DECLINE')),
    
    scored_at                       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_tx_id ON ml_ops.fraud_predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_pred_action ON ml_ops.fraud_predictions (recommended_action);
CREATE INDEX IF NOT EXISTS idx_pred_probability ON ml_ops.fraud_predictions (fraud_probability DESC);
