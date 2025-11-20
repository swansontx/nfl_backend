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

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from typing import Generator
import os

# Base class for ORM models (always available)
Base = declarative_base()

# Lazy-load PostgreSQL connection only when needed
_engine = None
_SessionLocal = None


def _init_postgres():
    """Initialize PostgreSQL connection (lazy-loaded)."""
    global _engine, _SessionLocal

    if _engine is not None:
        return

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'postgresql://nfl_props_user:password@localhost:5432/nfl_props'
    )

    _engine = create_engine(
        DATABASE_URL,
        pool_size=int(os.getenv('DB_POOL_SIZE', 10)),
        max_overflow=int(os.getenv('DB_MAX_OVERFLOW', 20)),
        pool_pre_ping=True,
        echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
    )

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_engine():
    """Get SQLAlchemy engine (lazy-loaded)."""
    _init_postgres()
    return _engine


def get_session_local():
    """Get session factory (lazy-loaded)."""
    _init_postgres()
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Get database session (FastAPI dependency).

    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    _init_postgres()
    db = _SessionLocal()
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
    _init_postgres()
    return _SessionLocal()


def init_db():
    """Initialize database (create all tables).

    Run this once to create all tables defined in models.py
    """
    _init_postgres()
    from backend.database import models  # Import to register models
    Base.metadata.create_all(bind=_engine)
    print("✓ Database tables created")


def drop_db():
    """Drop all tables (WARNING: Destructive!)."""
    _init_postgres()
    from backend.database import models
    Base.metadata.drop_all(bind=_engine)
    print("⚠ All tables dropped")
