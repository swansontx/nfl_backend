"""Pydantic models for API responses"""
from .recommendations import (
    PropRecommendationResponse,
    ParlayRecommendationResponse,
    RecommendationsListResponse,
    ParlaysListResponse,
)
from .games import GameResponse, GamesListResponse
from .backtest import BacktestResponse, SignalAnalysisResponse
from .odds import PropOddsResponse, OddsListResponse

__all__ = [
    "PropRecommendationResponse",
    "ParlayRecommendationResponse",
    "RecommendationsListResponse",
    "ParlaysListResponse",
    "GameResponse",
    "GamesListResponse",
    "BacktestResponse",
    "SignalAnalysisResponse",
    "PropOddsResponse",
    "OddsListResponse",
]
