import psycopg2
import sqlalchemy

PGHOST = "localhost"
PGPORT = "5432"
PGUSER = "postgres"
PGPASSWORD = "R0bb34l3x"
PGDATABASE = "fraud_detection_kdd"

def get_engine():
    return sqlalchemy.create_engine(f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}")    

def test_connection():
    try:
        connection = psycopg2.connect(
            host = PGHOST,
            port = PGPORT,
            user = PGUSER,
            password = PGPASSWORD,
            database = PGDATABASE
        )
        print("Connection to PostgreSQL database successful")
        connection.close()
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")

if __name__ == "__main__":
  test_connection()