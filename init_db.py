import sqlite3

conn = sqlite3.connect("gps_data.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS gps_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        latitude REAL,
        longitude REAL
    )
''')

conn.commit()
conn.close()