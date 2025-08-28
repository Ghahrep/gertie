import psycopg2

DATABASE_URL = "postgresql://postgres:%40b%24ythNwTkVxqAkNHCZD8sVdg%21iAEp33D%21yUq%5ERChyUv@localhost/gertie_db"

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    print("Making beta column nullable...")
    cur.execute("ALTER TABLE portfolio_risk_snapshots ALTER COLUMN beta DROP NOT NULL;")
    conn.commit()
    print("Beta column is now nullable!")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()