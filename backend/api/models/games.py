"""Pydantic models for game responses"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class GameResponse(BaseModel):
    """Single game information"""
    game_id: str
    season: int
    week: int
    game_type: str = Field(..., description="REG/POST/PRE")

    home_team: str
    away_team: str

    game_date: datetime
    game_time: Optional[str] = None
    stadium: Optional[str] = None

    # Scores (if completed)
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    completed: bool = False

    # Weather (if available)
    weather: Optional[dict] = None

    class Config:
        schema_extra = {
            "example": {
                "game_id": "2024_17_KC_LV",
                "season": 2024,
                "week": 17,
                "game_type": "REG",
                "home_team": "KC",
                "away_team": "LV",
                "game_date": "2024-12-25T13:00:00Z",
                "game_time": "1:00 PM ET",
                "stadium": "Arrowhead Stadium",
                "home_score": None,
                "away_score": None,
                "completed": False,
                "weather": {
                    "temperature": 32,
                    "wind_speed": 12,
                    "condition": "clear"
                }
            }
        }


class GamesListResponse(BaseModel):
    """List of games"""
    games: List[GameResponse]
    total_count: int

    # Filters applied
    season: Optional[int] = None
    week: Optional[int] = None
    team: Optional[str] = None
    date: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "games": [
                    {
                        "game_id": "2024_17_KC_LV",
                        "season": 2024,
                        "week": 17,
                        "game_type": "REG",
                        "home_team": "KC",
                        "away_team": "LV",
                        "game_date": "2024-12-25T13:00:00Z",
                        "completed": False
                    }
                ],
                "total_count": 1,
                "season": 2024,
                "week": 17
            }
        }
