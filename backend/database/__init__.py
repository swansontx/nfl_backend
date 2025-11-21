"""Database module for NFL betting data.

Two database options available:
1. SQLite (local_db.py) - Simple, no dependencies, used by default
2. PostgreSQL (optional) - For production, requires SQLAlchemy

For SQLite (default):
    from backend.database.local_db import init_database, get_database_status

For PostgreSQL (optional):
    from backend.database import get_db, engine
"""

from typing import Generator
import os

# Lazy-load SQLAlchemy only when PostgreSQL is used
_Base = None

def _get_base():
    """Get SQLAlchemy Base (lazy-loaded)."""
    global _Base
    if _Base is None:
        try:
            from sqlalchemy.ext.declarative import declarative_base
            _Base = declarative_base()
        except ImportError:
            raise ImportError("SQLAlchemy required for PostgreSQL. Install with: pip install sqlalchemy")
    return _Base

@property
def Base():
    """Backward-compatible Base accessor."""
    return _get_base()

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


def get_db():
    """Get database session (FastAPI dependency).

    Usage:
        @app.get("/users")
        def get_users(db = Depends(get_db)):
            return db.query(User).all()
    """
    _init_postgres()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
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
