import os
import sys

# Asegurar que el directorio de DAGs esté en sys.path para importaciones locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timezone
from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from train_and_save import train_and_save_model

with DAG(
    dag_id="fraud_model_retraining_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",  # Se ejecuta una vez a la semana de forma automática
    catchup=False,
) as dag:

    tarea_reentrenamiento = PythonOperator(
        task_id="Reentrenar_Y_Actualizar_Modelos",
        python_callable=train_and_save_model
    )
    
    tarea_reentrenamiento
