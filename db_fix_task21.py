import psycopg2
from config.settings import settings

def run_database_fixes():
    try:
        # Connect using your existing connection string
        conn = psycopg2.connect(settings.DATABASE_URL)
        cur = conn.cursor()
        
        print("Running database schema fixes...")
        
        # Add asset_type column
        cur.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS asset_type VARCHAR DEFAULT 'stock';")
        print("✅ Added asset_type column")
        
        # Add quantity column
        cur.execute("ALTER TABLE holdings ADD COLUMN IF NOT EXISTS quantity FLOAT;")
        print("✅ Added quantity column")
        
        # Update quantity values
        cur.execute("UPDATE holdings SET quantity = shares WHERE quantity IS NULL AND shares IS NOT NULL;")
        print("✅ Updated quantity values")
        
        # Update asset_type values
        cur.execute("UPDATE assets SET asset_type = 'stock' WHERE asset_type IS NULL;")
        print("✅ Updated asset_type values")
        
        conn.commit()
        print("✅ All fixes applied successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_database_fixes()