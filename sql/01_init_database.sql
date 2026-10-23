-- ==============================================================================
-- 01_init_database.sql
-- Creación de la base de datos y esquemas para Detección de Fraude KDD
-- ==============================================================================

-- 1. Crear base de datos (ejecutar conectado a la base 'postgres' por defecto)
-- NOTA: En psql puedes ejecutar: \i 01_init_database.sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'fraud_detection_kdd') THEN
        PERFORM dblink_exec('dbname=' || current_database(), 'CREATE DATABASE fraud_detection_kdd')
        WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'dblink');
    END IF;
END
$$;

-- Alternativamente en psql:
-- CREATE DATABASE fraud_detection_kdd;
-- \c fraud_detection_kdd;

-- 2. Esquemas de separación de responsabilidades:
--    - raw: Ingesta transaccional en bruto (datos de origen / producción).
--    - analytics: Feature store, vistas analíticas, métricas y evaluación de reglas.
--    - ml_ops: Almacenamiento de predicciones, alertas, auditoría e inferencia.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_ops;

COMMENT ON SCHEMA raw IS 'Datos crudos de transacciones bancarias y etiquetas de auditoría';
COMMENT ON SCHEMA analytics IS 'Feature Store SQL: variables derivadas, agregaciones y motor de reglas';
COMMENT ON SCHEMA ml_ops IS 'Predicciones de modelos de Machine Learning, anomalías y registro de alertas';
