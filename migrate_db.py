import sqlite3
import os
from backend.config import BASE_DIR

db_path = f"{BASE_DIR}/db/nuvant.db"

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE defect_logs ADD COLUMN embedding JSON")
        print("Column 'embedding' added successfully to 'defect_logs'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("Column 'embedding' already exists.")
        else:
            print(f"Error: {e}")
    conn.commit()
    conn.close()
else:
    print("Database not found. It will be created with the new schema on next start.")
