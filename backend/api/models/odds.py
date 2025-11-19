"""Pydantic models for odds responses"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PropOddsResponse(BaseModel):
    """Single prop market odds"""
    sportsbook: str
    player_id: str
    player_name: str
    market: str

    line: float
    over_odds: int = Field(..., description="American odds")
    under_odds: int = Field(..., description="American odds")

    # Implied probabilities
    over_prob: float = Field(..., ge=0, le=1)
    under_prob: float = Field(..., ge=0, le=1)
    vig: float = Field(..., description="Vigorish/juice")

    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "sportsbook": "draftkings",
                "player_id": "mahomes_patrick",
                "player_name": "Patrick Mahomes",
                "market": "player_pass_yds",
                "line": 285.5,
                "over_odds": -110,
                "under_odds": -110,
                "over_prob": 0.524,
                "under_prob": 0.524,
                "vig": 0.048,
                "timestamp": "2024-12-24T10:00:00Z"
            }
        }


class AggregatedPropOddsResponse(BaseModel):
    """Aggregated odds across sportsbooks"""
    player_id: str
    player_name: str
    market: str

    # Best available
    best_over_line: float
    best_over_odds: int
    best_over_book: str

    best_under_line: float
    best_under_odds: int
    best_under_book: str

    # Consensus
    consensus_line: float
    consensus_prob: float

    # Metadata
    n_books: int
    line_spread: float = Field(..., description="Max - min line across books")
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "player_id": "mahomes_patrick",
                "player_name": "Patrick Mahomes",
                "market": "player_pass_yds",
                "best_over_line": 285.5,
                "best_over_odds": -105,
                "best_over_book": "fanduel",
                "best_under_line": 287.5,
                "best_under_odds": -108,
                "best_under_book": "draftkings",
                "consensus_line": 286.5,
                "consensus_prob": 0.520,
                "n_books": 5,
                "line_spread": 2.0,
                "timestamp": "2024-12-24T10:00:00Z"
            }
        }


class OddsListResponse(BaseModel):
    """List of prop odds"""
    game_id: Optional[str] = None
    odds: List[PropOddsResponse]
    aggregated_odds: List[AggregatedPropOddsResponse]

    total_count: int
    sportsbooks: List[str]

    # Metadata
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
