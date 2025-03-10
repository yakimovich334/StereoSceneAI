import psycopg2
from psycopg2 import sql

# Налаштування підключення
DB_CONFIG = {
    "dbname": "StereoAI",
    "user": "Admin",
    "password": "admin",
    "host": "localhost",  # або IP-адреса сервера
    "port": "5432"  # стандартний порт PostgreSQL
}

def get_connection():
    """Функція для створення підключення до бази даних."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Помилка підключення до БД: {e}")
        return None

def close_connection(conn):
    """Функція для закриття підключення до бази даних."""
    if conn:
        conn.close()
        print("Підключення до БД закрито.")

# Тестове підключення
if __name__ == "__main__":
    connection = get_connection()
    if connection:
        print("Підключення успішне!")
        close_connection(connection)