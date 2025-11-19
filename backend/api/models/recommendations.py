"""Pydantic models for recommendation responses"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PropRecommendationResponse(BaseModel):
    """Single prop recommendation"""
    player_id: str
    player_name: str
    position: str
    team: str
    game_id: str
    market: str

    # Projection
    line: float = Field(..., description="Projected line value")
    model_prob: float = Field(..., ge=0, le=1, description="Raw model probability")
    calibrated_prob: float = Field(..., ge=0, le=1, description="Calibrated probability")

    # Signals (0-1 scores)
    base_signal: float = Field(..., ge=0, le=1)
    matchup_signal: float = Field(..., ge=0, le=1)
    trend_signal: float = Field(..., ge=0, le=1)
    news_signal: float = Field(..., ge=0, le=1)
    roster_signal: float = Field(..., ge=0, le=1)

    # Overall
    overall_score: float = Field(..., ge=0, le=1, description="Combined score")
    recommendation_strength: str = Field(..., description="elite/strong/moderate/weak/avoid")
    confidence: float = Field(..., ge=0, le=1)

    # Context
    reasoning: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)

    # Market (optional)
    market_line: Optional[float] = None
    market_odds: Optional[int] = None
    edge: Optional[float] = Field(None, description="Model prob - implied prob from odds")

    class Config:
        schema_extra = {
            "example": {
                "player_id": "mahomes_patrick",
                "player_name": "Patrick Mahomes",
                "position": "QB",
                "team": "KC",
                "game_id": "2024_17_KC_LV",
                "market": "player_pass_yds",
                "line": 287.5,
                "model_prob": 0.65,
                "calibrated_prob": 0.62,
                "base_signal": 0.68,
                "matchup_signal": 0.55,
                "trend_signal": 0.72,
                "news_signal": 0.60,
                "roster_signal": 0.90,
                "overall_score": 0.742,
                "recommendation_strength": "strong",
                "confidence": 0.85,
                "reasoning": [
                    "Model probability: 62.0%",
                    "Hot streak: 5 game streak",
                    "Favorable matchup vs LV (rank 28)"
                ],
                "flags": [],
                "market_line": 285.5,
                "market_odds": -110,
                "edge": 0.10
            }
        }


class PropInParlayResponse(BaseModel):
    """Simplified prop for parlay display"""
    player_name: str
    market: str
    line: float
    probability: float
    score: float


class ParlayRecommendationResponse(BaseModel):
    """Parlay recommendation"""
    props: List[PropInParlayResponse]

    # Parlay pricing
    raw_probability: float = Field(..., description="Assuming independence")
    adjusted_probability: float = Field(..., description="With correlation adjustment")
    adjustment_factor: float

    # Scoring
    overall_score: float = Field(..., ge=0, le=1)
    correlation_impact: str = Field(..., description="positive/negative/neutral")
    confidence: float = Field(..., ge=0, le=1)

    # Context
    reasoning: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)

    # Market (optional)
    market_odds: Optional[int] = None
    edge: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "props": [
                    {
                        "player_name": "Patrick Mahomes",
                        "market": "player_pass_yds",
                        "line": 287.5,
                        "probability": 0.62,
                        "score": 0.74
                    },
                    {
                        "player_name": "Travis Kelce",
                        "market": "player_rec_yds",
                        "line": 68.5,
                        "probability": 0.58,
                        "score": 0.69
                    }
                ],
                "raw_probability": 0.36,
                "adjusted_probability": 0.42,
                "adjustment_factor": 1.17,
                "overall_score": 0.715,
                "correlation_impact": "positive",
                "confidence": 0.78,
                "reasoning": [
                    "Parlay of 2 props",
                    "Correlation: +0.65 (positive)",
                    "QB-TE stack on same team"
                ],
                "flags": []
            }
        }


class RecommendationsListResponse(BaseModel):
    """List of prop recommendations"""
    game_id: str
    game_time: Optional[datetime] = None
    home_team: str
    away_team: str

    recommendations: List[PropRecommendationResponse]
    total_count: int

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    markets_analyzed: List[str]
    min_confidence: float


class ParlaysListResponse(BaseModel):
    """List of parlay recommendations"""
    game_id: str
    parlays: List[ParlayRecommendationResponse]
    total_count: int

    # Settings
    parlay_size: int
    min_correlation: float
    max_correlation: float

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
