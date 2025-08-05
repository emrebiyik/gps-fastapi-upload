import sqlite3

DB_FILE = "gps_data.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    return conn