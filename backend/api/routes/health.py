"""Health check and status routes"""
from fastapi import APIRouter
from datetime import datetime
from typing import Dict

from backend.config import settings
from backend.database.session import get_db
from backend.database.models import Game, Player
from backend.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    """
    Root endpoint

    Returns basic API information.
    """
    return {
        "service": "NFL Props API",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@router.get("/health")
def health_check():
    """
    Health check endpoint

    Tests database connectivity and returns system status.
    """
    try:
        with get_db() as session:
            # Test database connection
            game_count = session.query(Game).count()
            player_count = session.query(Player).count()

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database": "connected",
                "games": game_count,
                "players": player_count,
            }

    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "disconnected",
            "error": str(e),
        }


@router.get("/status")
def status() -> Dict:
    """
    Detailed system status

    Returns configuration and system information.
    """
    return {
        "service": "NFL Props API",
        "version": "1.0.0",
        "environment": "production",  # Would come from env var
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "recommendations": True,
            "parlays": True,
            "backtest": True,
            "odds": True,  # If API key configured
            "news": True,  # If API keys configured
            "weather": True,  # If API key configured
        },
        "supported_markets": settings.supported_markets if hasattr(settings, 'supported_markets') else [],
    }
