"""
Script de Automatización: Inicialización y Carga de Semilla (Seed) en PostgreSQL
Ejecuta los scripts DDL y carga los 100,000 registros del CSV a la base de datos PostgreSQL.
Compatible tanto con PostgreSQL nativo en Windows como en contenedor Docker.
"""

import os
import sys
import argparse
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"
CSV_PATH = PROJECT_ROOT / "Datasets" / "credit_card_fraud_detection_2026.csv"


def get_connection(host, port, user, password, dbname):
    """Intenta establecer conexión utilizando psycopg2, psycopg (v3) o sqlalchemy."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
        conn.autocommit = True
        return conn, "psycopg2"
    except ImportError:
        pass

    try:
        import psycopg
        conn = psycopg.connect(
            host=host, port=port, user=user, password=password, dbname=dbname, autocommit=True
        )
        return conn, "psycopg"
    except ImportError:
        pass

    raise ImportError(
        "No se encontró un conector PostgreSQL instalado. Instala 'psycopg2-binary' o 'psycopg':\n"
        "    pip install psycopg2-binary"
    )


def ejecutar_archivo_sql(cursor, file_path: Path):
    """Lee y ejecuta un archivo de sentencias SQL."""
    print(f"  -> Ejecutando: {file_path.name}...")
    sql_text = file_path.read_text(encoding="utf-8")
    cursor.execute(sql_text)


def cargar_csv_a_staging(conn, connector_type: str, csv_path: Path):
    """Carga masiva ultra-rápida del CSV a la tabla raw.staging_seed_csv usando COPY."""
    print(f"  -> Cargando CSV con método COPY de alta velocidad ({csv_path.name})...")
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE raw.staging_seed_csv;")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        # Saltar la cabecera en el flujo
        if connector_type == "psycopg2":
            cursor.copy_expert(
                sql="""
                    COPY raw.staging_seed_csv FROM STDIN 
                    WITH (FORMAT csv, HEADER true, DELIMITER ',');
                """,
                file=f,
            )
        else:
            with cursor.copy(
                "COPY raw.staging_seed_csv FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER ',')"
            ) as copy:
                while data := f.read(8192):
                    copy.write(data)

    cursor.execute("SELECT COUNT(*) FROM raw.staging_seed_csv;")
    total = cursor.fetchone()[0]
    print(f"  [OK] Registros importados en staging: {total:,}")


def main():
    parser = argparse.ArgumentParser(description="Sembrador de datos KDD Fraude para PostgreSQL")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"), help="Host de Postgres (def: localhost)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", 5432)), help="Puerto de Postgres (def: 5432)")
    parser.add_argument("--user", default=os.getenv("PGUSER", "postgres"), help="Usuario (def: postgres)")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "postgres"), help="Contraseña")
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fraud_detection_kdd"), help="Nombre base de datos")
    parser.add_argument("--skip-csv", action="store_true", help="Solo crea tablas y vistas sin cargar el CSV")
    args = parser.parse_args()

    print("=" * 80)
    print("MIGRACIÓN Y SEEDING DE BASE DE DATOS POSTGRESQL (KDD FRAUDE 2026)")
    print("=" * 80)
    print(f"Destino: postgresql://{args.user}:***@{args.host}:{args.port}/{args.dbname}\n")

    # Paso 1: Asegurar que la base de datos existe
    try:
        conn_admin, _ = get_connection(args.host, args.port, args.user, args.password, "postgres")
        cur_admin = conn_admin.cursor()
        cur_admin.execute(f"SELECT 1 FROM pg_database WHERE datname = '{args.dbname}';")
        if not cur_admin.fetchone():
            print(f"[*] Creando base de datos '{args.dbname}'...")
            cur_admin.execute(f"CREATE DATABASE {args.dbname};")
        conn_admin.close()
    except Exception as e:
        print(f"[!] Advertencia conectando a 'postgres' por defecto: {e}")
        print("    Intentando conexión directa a la base de datos objetivo...")

    # Paso 2: Conexión a la base objetivo
    conn, conn_type = get_connection(args.host, args.port, args.user, args.password, args.dbname)
    cur = conn.cursor()
    print(f"[+] Conexión establecida con éxito usando: {conn_type}\n")

    # Paso 3: Ejecutar DDL
    print("[1/4] Creando esquemas...")
    ejecutar_archivo_sql(cur, SQL_DIR / "01_init_database.sql")

    print("\n[2/4] Creando tablas e índices...")
    ejecutar_archivo_sql(cur, SQL_DIR / "02_create_tables.sql")

    print("\n[3/4] Creando vistas del Feature Store y Motor de Reglas...")
    ejecutar_archivo_sql(cur, SQL_DIR / "03_feature_store_and_rules.sql")

    # Paso 4: Carga de Datos
    if not args.skip_csv:
        if not CSV_PATH.exists():
            print(f"[!] Archivo CSV no encontrado en: {CSV_PATH}")
            return
        print("\n[4/4] Ingestando datos del CSV y distribuyendo a tablas de producción...")
        cargar_csv_a_staging(conn, conn_type, CSV_PATH)
        ejecutar_archivo_sql(cur, SQL_DIR / "04_seed_and_migrate.sql")
        
        # Mostrar resumen de validación
        cur.execute("SELECT COUNT(*) FROM raw.transactions;")
        n_tx = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM raw.fraud_labels WHERE is_fraud = 1;")
        n_fraude = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM analytics.v_rule_engine_eval WHERE total_rules_triggered > 0;")
        n_reglas = cur.fetchone()[0]

        print("\n" + "=" * 80)
        print("RESUMEN DE MIGRACIÓN EXITOSA:")
        print(f"  - Total transacciones en raw.transactions:  {n_tx:,}")
        print(f"  - Total fraudes auditados en raw.fraud_labels: {n_fraude:,} ({n_fraude/n_tx*100:.2f}%)")
        print(f"  - Transacciones con alertas en motor de reglas: {n_reglas:,}")
        print("=" * 80)

    conn.close()


if __name__ == "__main__":
    main()
