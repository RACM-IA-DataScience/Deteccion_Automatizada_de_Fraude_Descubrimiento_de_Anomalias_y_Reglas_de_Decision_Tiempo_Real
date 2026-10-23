import pandas as pd
import uuid
import random
from datetime import datetime
from db.db_connector import get_engine

def simulate_transactions():
    print("Conectando a la base de datos para simular flujo transaccional...")
    engine = get_engine()
    
    # Determinar el tamaño del lote (entre 10 y 50 transacciones)
    batch_size = random.randint(10, 50)
    print(f"Generando un lote de {batch_size} nuevas transacciones...")

    # Extraer una muestra rápida de la tabla original para usarla de plantilla
    # Tomamos 500 registros recientes o aleatorios y luego seleccionamos la cantidad del batch
    query = "SELECT * FROM raw.transactions LIMIT 500;"
    df_template = pd.read_sql(query, con=engine)
    
    if df_template.empty:
        print("No hay datos en raw.transactions para usar como plantilla.")
        return
        
    df_new = df_template.sample(n=batch_size, replace=True).copy()

    # Asignar nuevos identificadores únicos y la fecha/hora actual
    df_new["transaction_id"] = [f"TXN-{uuid.uuid4().hex[:8].upper()}" for _ in range(batch_size)]
    df_new["transaction_datetime"] = datetime.now()

    # (Opcional) Introducir una ligera variación aleatoria en los montos para que no sean idénticos
    df_new["transaction_amount"] = df_new["transaction_amount"] * [random.uniform(0.95, 1.05) for _ in range(batch_size)]
    df_new["transaction_amount"] = df_new["transaction_amount"].round(2)

    # Insertar las nuevas transacciones en raw.transactions
    print("Insertando transacciones en raw.transactions...")
    df_new.to_sql(
        name="transactions",
        schema="raw",
        con=engine,
        if_exists="append",
        index=False
    )
    
    print(f"¡Simulación exitosa! Se inyectaron {batch_size} nuevas transacciones.")

if __name__ == "__main__":
    simulate_transactions()