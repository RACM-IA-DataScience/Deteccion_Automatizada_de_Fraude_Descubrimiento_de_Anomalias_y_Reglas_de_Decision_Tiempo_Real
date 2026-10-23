# Catálogo y Diccionario Técnico de Funciones del Proyecto FraudGuard AI (KDD 2026)

| Metadato | Detalle |
| :--- | :--- |
| **Proyecto** | Detección Automatizada de Fraude, Descubrimiento de Anomalías y Reglas de Decisión |
| **Metodología** | KDD (*Knowledge Discovery in Databases*) de 8 Etapas |
| **Tecnologías** | Python 3.12, PostgreSQL 16/18, Apache Airflow 3, Docker Compose, Scikit-Learn, Streamlit |
| **Autor** | Roberto Alejandro Chagra Martínez |
| **Materia** | Minería de Datos y Descubrimiento de Conocimiento (*Data Mining & Knowledge Discovery*) |
| **Institución** | Centro Universitario de Guadalajara (CUGDL) — Universidad de Guadalajara |

---

## Índice del Diccionario
1. [Módulo Central de Inferencia y Alertas (`src/fraud_engine.py`)](#1-módulo-central-de-inferencia-y-alertas-srcfraud_enginepy)
2. [Interfaz Interactiva y Visualización (`app.py`)](#2-interfaz-interactiva-y-visualización-apppy)
3. [Script CLI de Simulación y Alertas (`scripts/simulate_single_transaction.py`)](#3-script-cli-de-simulación-y-alertas-scriptssimulate_single_transactionpy)
4. [Orquestación y Pipelines de Apache Airflow (`airflow/dags/`)](#4-orquestación-y-pipelines-de-apache-airflow-airflowdags)
5. [Carga de Base de Datos y Automatización SQL (`sql/seed_database.py` y `src/db/`)](#5-carga-de-base-de-datos-y-automatización-sql-sqlseed_databasepy-y-srcdb)
6. [Pipeline de Entrenamiento y Scoring en Base de Datos (`src/train_and_save.py` y `src/score_transactions.py`)](#6-pipeline-de-entrenamiento-y-scoring-en-base-de-datos-srctrain_and_savepy-y-srcscore_transactionspy)
7. [Funciones de Librerías Científicas (`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`)](#7-funciones-de-librerías-científicas-pandas-numpy-scikit-learn-imbalanced-learn)

---

## 1. Módulo Central de Inferencia y Alertas (`src/fraud_engine.py`)

Clase y métodos del motor unificado que procesa transacciones unitarias o en micro-lotes, aplica reglas de negocio, aísla anomalías y simula el despacho de alertas multicanal.

| Nombre de la Función / Método | Descripción Funcional | Sintaxis y Firma | Parámetros y Retorno |
| :--- | :--- | :--- | :--- |
| `FraudInferenceEngine.__init__` | Constructor de la clase. Carga en memoria RAM el escalador robusto (`scaler.joblib`), el modelo no supervisado (`isolation_forest.joblib`), el clasificador (`fraud_classifier.joblib`) y la lista ordenada de columnas (`model_columns.json`). | `engine = FraudInferenceEngine(models_dir=None)` | **Parámetros:**<br>`models_dir` (*Path*, opcional): Ruta al directorio de artefactos serializados.<br>**Retorno:** Instancia lista para inferencia. |
| `preprocess_and_feature_engineering` | Limpia los datos de la transacción, imputa nulos estructurales (`browser_clean`), y computa en vuelo las 8 variables derivadas (ratios de velocidad 1h/24h, utilización de crédito, Z-Score 30d, factores de seguridad y el `composite_risk_index`). | `df_proc, meta = engine.preprocess_and_feature_engineering(raw_tx)` | **Parámetros:**<br>`raw_tx` (*Dict[str, Any]*): Diccionario con datos crudos de la transacción.<br>**Retorno:**<br>`Tuple[pd.DataFrame, Dict[str, Any]]`: DataFrame enriquecido y diccionario de metadatos de riesgo. |
| `evaluate_business_rules` | Evalúa de manera determinística las 5 reglas de negocio validadas en el proyecto KDD (R1 a R5), clasificándolas por severidad (CRITICAL o HIGH) según la combinación de condiciones lógicas. | `rules_list = engine.evaluate_business_rules(df)` | **Parámetros:**<br>`df` (*pd.DataFrame*): DataFrame con la transacción y sus variables derivadas.<br>**Retorno:**<br>`List[Dict[str, Any]]`: Lista de reglas con su código (`R1`–`R5`), descripción, bandera booleana (`triggered`) y severidad. |
| `predict` | Orquesta la inferencia completa: preprocesamiento, evaluación de reglas, transformación logarítmica y escalamiento, cálculo del score de anomalía con Isolation Forest, predicción de probabilidad con Random Forest y resolución del veredicto tripartito (`DECLINE`, `MANUAL_REVIEW`, `APPROVE`). | `result = engine.predict(raw_tx)` | **Parámetros:**<br>`raw_tx` (*Dict[str, Any]*): Datos de la transacción de entrada.<br>**Retorno:**<br>`Dict[str, Any]`: Resultado completo con veredicto, probabilidad, anomalía, reglas detonadas y payload de alertas. |
| `_build_alert_payload` | Genera los formatos de notificación para la simulación operativa: texto formateado para SMS/Push al titular de la tarjeta y estructura JSON estándar para Webhooks (Slack, SOC, SIEM, PagerDuty). | `alert = engine._build_alert_payload(tx_id, raw_tx, fraud_prob, iso_score, is_anomaly, rules, recommended_action, severity)` | **Parámetros:**<br>Métricas y decisiones del análisis.<br>**Retorno:**<br>`Dict[str, Any]`: Estructura JSON con `alert_id`, `simulated_sms`, `simulated_webhook`, severidad y timestamps. |

---

## 2. Interfaz Interactiva y Visualización (`app.py`)

Funciones auxiliares que sustentan la aplicación web interactiva en Streamlit.

| Nombre de la Función | Descripción Funcional | Sintaxis y Firma | Parámetros y Retorno |
| :--- | :--- | :--- | :--- |
| `load_engine` | Función decorada con `@st.cache_resource` que instancia y mantiene en caché de memoria el motor `FraudInferenceEngine`, evitando re-cargar los modelos de disco en cada interacción del usuario. | `engine = load_engine()` | **Parámetros:** Ninguno.<br>**Retorno:** Instancia de `FraudInferenceEngine`. |
| `get_db_connection` | Abre un pool de conexiones liviano y verifica la disponibilidad de la base de datos PostgreSQL local en el puerto 5432. Si la base no está activa, degrada elegantemente a modo memoria sin romper la UI. | `db_engine = get_db_connection()` | **Parámetros:** Ninguno.<br>**Retorno:** Instancia de `sqlalchemy.engine.Engine` o `None` si falla la conexión. |

---

## 3. Script CLI de Simulación y Alertas (`scripts/simulate_single_transaction.py`)

Funciones para la ejecución desde la línea de comandos (terminal / CI/CD).

| Nombre de la Función | Descripción Funcional | Sintaxis y Firma | Parámetros y Retorno |
| :--- | :--- | :--- | :--- |
| `main` | Parsea argumentos de línea de comandos (`--preset`, `--amount`, `--high-risk-country`, `--json`), configura la salida UTF-8 para consolas de Windows, invoca a `FraudInferenceEngine` y formatea la salida visual o en JSON crudo. | `python scripts/simulate_single_transaction.py --preset fraud` | **Parámetros:** Argumentos de terminal (`sys.argv`).<br>**Retorno:** Salida formateada a `stdout` (código de salida 0 en éxito). |

---

## 4. Orquestación y Pipelines de Apache Airflow (`airflow/dags/`)

Funciones ejecutadas dentro de los contenedores Docker por el Worker de Airflow 3.

| Nombre de la Función | Archivo | Descripción Funcional | Sintaxis y Firma |
| :--- | :--- | :--- | :--- |
| `get_db_engine` | `dag_fraud_pipeline.py` | Genera un motor SQLAlchemy apuntando al host PostgreSQL (`host.docker.internal` dentro del contenedor o `localhost` fuera). | `engine = get_db_engine()` |
| `simulate_transactions_callable` | `dag_fraud_pipeline.py` | Tarea de Airflow (`Simular_Transacciones`): Toma una muestra aleatoria de transacciones plantilla, les asigna nuevos UUIDs y montos con jitter gaussiano, e inserta un micro-lote en `raw.transactions`. | `simulate_transactions_callable()` |
| `detect_fraud_callable` | `dag_fraud_pipeline.py` | Tarea de Airflow (`Detectar_Fraude`): Identifica transacciones en `raw.transactions` que no han sido calificadas en `raw.predictions`, calcula variables derivadas, ejecuta los modelos y persiste los resultados. | `detect_fraud_callable()` |
| `get_engine` | `train_and_save.py` | Establece conexión directa hacia PostgreSQL para las tareas pesadas de entrenamiento. | `engine = get_engine()` |
| `load_data_from_db` | `train_and_save.py` | Consulta la vista `analytics.v_model_training_dataset` o `raw.transactions` y carga las observaciones históricas completas a un DataFrame. | `df = load_data_from_db()` |
| `train_and_save_model` | `train_and_save.py` | Tarea de Airflow (`Reentrenar_Y_Actualizar_Modelos`): Ejecuta el pipeline headless de reentrenamiento, ajusta `RobustScaler`, `IsolationForest` y `RandomForestClassifier`, evalúa métricas ROC-AUC y sobrescribe los artefactos en `/opt/airflow/models`. | `train_and_save_model()` |

---

## 5. Carga de Base de Datos y Automatización SQL (`sql/seed_database.py` y `src/db/`)

Funciones para la inicialización y migración DDL del Data Warehouse.

| Nombre de la Función | Archivo | Descripción Funcional | Sintaxis y Firma |
| :--- | :--- | :--- | :--- |
| `get_connection` | `sql/seed_database.py` | Intenta conectar a PostgreSQL mediante `psycopg2`, `psycopg` (v3) o SQLAlchemy, activando modo `autocommit`. | `conn, ctype = get_connection(host, port, user, pwd, db)` |
| `ejecutar_archivo_sql` | `sql/seed_database.py` | Lee el contenido de un script `.sql` en disco con codificación UTF-8 y lo ejecuta secuencialmente en el cursor. | `ejecutar_archivo_sql(cursor, file_path)` |
| `cargar_csv_a_staging` | `sql/seed_database.py` | Ejecuta la ingesta masiva de los 100,000 registros del CSV hacia `raw.staging_seed_csv` utilizando el protocolo nativo `COPY` de PostgreSQL (tiempo: < 3 segundos). | `cargar_csv_a_staging(conn, connector_type, csv_path)` |
| `migrar_a_tablas_finales` | `sql/seed_database.py` | Ejecuta el script `04_seed_and_migrate.sql` para normalizar los datos desde staging hacia `raw.customers`, `raw.cards`, `raw.merchants` y `raw.transactions`. | `migrar_a_tablas_finales(cursor)` |
| `get_engine` | `src/db/db_connector.py` | Retorna un objeto Engine de SQLAlchemy con la cadena de conexión configurada para el host local. | `engine = get_engine()` |
| `test_connection` | `src/db/db_connector.py` | Función de diagnóstico rápido para validar credenciales y conectividad con la base de datos PostgreSQL. | `test_connection()` |

---

## 6. Pipeline de Entrenamiento y Scoring en Base de Datos (`src/train_and_save.py` y `src/score_transactions.py`)

Funciones del pipeline de entrenamiento continuo y scoring masivo sobre la base de datos PostgreSQL:

| Nombre de la Función | Archivo | Descripción Funcional | Sintaxis y Firma |
| :--- | :--- | :--- | :--- |
| `train_and_save_model` | `src/train_and_save.py` | Función maestra de reentrenamiento. Carga el histórico desde PostgreSQL, aplica la ingeniería de 8 variables y compresión logarítmica, ajusta `IsolationForest` y `RandomForestClassifier` con balanceo sintético `SMOTE`, genera gráficos de desempeño y serializa los artefactos en `models/`. | `metrics = train_and_save_model(models_dir=None)` |
| `load_data_from_db` | `src/train_and_save.py` | Extrae transacciones confirmadas y auditadas desde la base de datos PostgreSQL utilizando el motor de SQLAlchemy. | `df = load_data_from_db(engine, limit=None)` |
| `score_unprocessed_transactions` | `src/score_transactions.py` | Proceso por lotes que consulta transacciones no evaluadas en PostgreSQL (`raw_transactions`), ejecuta la inferencia con `FraudInferenceEngine` e inserta los resultados en `fraud_predictions_and_alerts`. | `total_scored = score_unprocessed_transactions(batch_size=1000)` |

---

## 7. Funciones de Librerías Científicas (`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`)

Catálogo de funciones estándar utilizadas en el procesamiento matemático del proyecto.

| Nombre de la Función | Librería | Descripción de Uso en el Proyecto | Sintaxis en el Código |
| :--- | :--- | :--- | :--- |
| `pd.read_csv()` | `pandas` | Ingesta del archivo CSV de transacciones a DataFrame. | `df = pd.read_csv(csv_path)` |
| `Series.value_counts()` | `pandas` | Diagnóstico del desbalance de la variable objetivo `is_fraud`. | `df["is_fraud"].value_counts(normalize=True)` |
| `DataFrame.drop()` | `pandas` | Purgado de variables con fuga de datos o identificadores. | `df.drop(columns=cols_to_drop)` |
| `Series.fillna()` | `pandas` | Imputación estructural en `browser_clean`. | `df["browser"].fillna("No_Aplica_Fisico")` |
| `Series.quantile()` | `pandas` | Cálculo de percentiles $Q_1$, Mediana y $Q_3$ para el IQR de Tukey. | `q1, q3 = df[col].quantile([0.25, 0.75])` |
| `np.log1p()` | `numpy` | Compresión logarítmica $\ln(1 + x)$ para montos asimétricos con ceros. | `np.log1p(np.maximum(0, df[cols]))` |
| `train_test_split()` | `scikit-learn` | Partición 80/20 con conservación de proporción de fraude (`stratify=y`). | `train_test_split(X, y, test_size=0.20, stratify=y)` |
| `RobustScaler()` | `scikit-learn` | Escalamiento insensible a valores atípicos usando mediana e IQR. | `scaler.fit_transform(X_train[num_cols])` |
| `chi2()` | `scikit-learn` | Prueba de independencia Chi-Cuadrado para selección de variables categóricas. | `chi_vals, p_vals = chi2(X[bin_cols], y)` |
| `IsolationForest()` | `scikit-learn` | Detección no supervisada de transacciones anómalas multidimensionales. | `iso.fit(X_train[iso_features])` |
| `iso.score_samples()` | `scikit-learn` | Extracción continua del puntaje de anomalía (-score). | `-iso.score_samples(X[iso_features])` |
| `SMOTE()` | `imbalanced-learn` | Sobremuestreo de minorías sintéticas exclusivamente en `X_train`. | `smote.fit_resample(X_train, y_train)` |
| `RandomForestClassifier()` | `scikit-learn` | Clasificador supervisado principal con ponderación balanceada. | `RandomForestClassifier(n_estimators=200)` |
| `GradientBoostingClassifier()` | `scikit-learn` | Clasificador secuencial que optimizó el máximo `ROC-AUC = 0.8379`. | `GradientBoostingClassifier(max_depth=4)` |
| `roc_auc_score()` | `scikit-learn` | Cálculo del área bajo la curva ROC en el conjunto de prueba. | `roc_auc_score(y_test, probs)` |
| `average_precision_score()` | `scikit-learn` | Cálculo del área bajo la curva Precision-Recall (`PR-AUC`). | `average_precision_score(y_test, probs)` |
| `confusion_matrix()` | `scikit-learn` | Auditoría de aciertos y falsas alarmas ($TN, FP, FN, TP$). | `confusion_matrix(y_test, preds)` |
| `export_text()` | `scikit-learn` | Extracción del árbol de decisión interpretable a reglas auditable IF-THEN. | `export_text(dt_model, feature_names=features)` |
| `plot_tree()` | `scikit-learn` | Renderizado gráfico de los nodos y ramas del árbol de reglas. | `plot_tree(dt_model, filled=True)` |
