# in create_tables.py

# Import the correct, shared Base object and the engine from our session manager
from db.session import Base, engine

# We must also import the models module. Even though we don't use it directly
# in this script, importing it is what tells SQLAlchemy that classes like
# User, Portfolio, etc., exist and are attached to our Base object.
from db import models

def main():
    """
    Creates all database tables defined in db/models.py.
    """
    print("Connecting to the database...")
    print(f"Target: {engine.url}")
    print("\nCreating tables based on models in db/models.py...")
    
    try:
        # The create_all command will now see User, Asset, Portfolio, and Holding
        # because they inherited from the imported Base object.
        Base.metadata.create_all(bind=engine)
        print("\nSUCCESS: Tables created successfully!")
        print("Please verify with the '\\dt' command in psql.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()