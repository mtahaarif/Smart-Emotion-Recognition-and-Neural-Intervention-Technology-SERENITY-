from fastapi import FastAPI
import sqlite3

app = FastAPI()

@app.get("/")
def home():
    # test SQLite connection
    try:
        conn = sqlite3.connect("serenity.db")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.commit()
        conn.close()
        return {"message": "Hello from SERENITY Database!", "db_status": "Connected & Working"}
    except Exception as e:
        return {"message": "Database connection failed", "error": str(e)}
