"""API dependencies for authentication and database"""
from typing import Optional
from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.session import get_db_session


async def verify_admin_token(
    authorization: Optional[str] = Header(None)
) -> bool:
    """
    Verify admin token from Authorization header

    Expected format: "Bearer <token>"
    """
    if not settings.admin_token:
        # No token configured, allow access (dev mode)
        return True

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token != settings.admin_token.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token"
        )

    return True


def get_db():
    """Database session dependency"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()


# Dependency to get database session
DatabaseDep = Depends(get_db)

# Dependency for admin authentication
AdminAuthDep = Depends(verify_admin_token)
