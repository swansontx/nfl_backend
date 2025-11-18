"""Data ingestion module for NFL data and odds"""

from .nflverse_client import NFLverseClient
from .odds_client import OddsAPIClient

__all__ = ["NFLverseClient", "OddsAPIClient"]
