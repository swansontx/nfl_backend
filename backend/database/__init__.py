"""Database module for PostgreSQL integration.

Provides SQLAlchemy ORM models, session management, and CRUD operations.

Environment Variables Required:
    DATABASE_URL: Full PostgreSQL connection string
        Format: postgresql://user:password@host:port/dbname

Example:
    from backend.database import get_db, engine
    from backend.database.models import User, BetHistory

    # Get database session
    session = next(get_db())

    # Query users
    users = session.query(User).filter(User.is_active == True).all()
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

# Database URL from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://nfl_props_user:password@localhost:5432/nfl_props'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=int(os.getenv('DB_POOL_SIZE', 10)),
    max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 20)),
    pool_pre_ping=True,  # Verify connections before using
    echo=os.getenv('DB_ECHO', 'false').lower() == 'true'  # SQL logging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session (FastAPI dependency).

    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session (direct use).

    Usage:
        session = get_db_session()
        users = session.query(User).all()
        session.close()
    """
    return SessionLocal()


def init_db():
    """Initialize database (create all tables).

    Run this once to create all tables defined in models.py
    """
    from backend.database import models  # Import to register models
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


def drop_db():
    """Drop all tables (WARNING: Destructive!)."""
    from backend.database import models
    Base.metadata.drop_all(bind=engine)
    print("⚠ All tables dropped")
