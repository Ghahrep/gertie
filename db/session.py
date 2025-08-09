# in db/session.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Import our settings from the core config file
from core.config import settings

# 1. Create the SQLAlchemy engine
# This is the central object that connects to our PostgreSQL database.
# The 'pool_pre_ping' argument helps handle stale connections.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True
)

# 2. Create a SessionLocal class
# Each instance of a SessionLocal class will be a new database session.
# This is a factory for creating sessions.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 3. Create a Base class for our ORM models
# All of our database models (like User, Portfolio, etc.) will inherit
# from this class. This is how SQLAlchemy's ORM discovers them.
Base = declarative_base()


# 4. Create the Database Dependency for FastAPI
def get_db():
    """
    A FastAPI dependency that creates and yields a new database session
    for each incoming API request. It ensures the session is always
    closed, even if errors occur.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()