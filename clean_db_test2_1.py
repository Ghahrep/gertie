import psycopg2

conn = psycopg2.connect("postgresql://postgres:%40b%24ythNwTkVxqAkNHCZD8sVdg%21iAEp33D%21yUq%5ERChyUv@localhost/gertie_db")
cur = conn.cursor()

# Correct order: delete in reverse dependency order
cleanup_sql = [
    "DELETE FROM holdings WHERE asset_id IN (SELECT id FROM assets WHERE ticker IN ('AAPL', 'GOOGL', 'MSFT'));",
    "DELETE FROM portfolios WHERE name = 'Task 2.1 Test Portfolio';", 
    "DELETE FROM users WHERE email = 'task21_test@example.com';",
    "DELETE FROM assets WHERE ticker IN ('AAPL', 'GOOGL', 'MSFT');"
]

for sql in cleanup_sql:
    try:
        cur.execute(sql)
        print(f"Executed: {sql}")
    except Exception as e:
        print(f"Error executing {sql}: {e}")

conn.commit()
conn.close()
print("Cleanup completed")