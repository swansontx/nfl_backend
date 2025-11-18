"""Data validation utilities"""
from typing import Any, Dict
from pydantic import BaseModel, validator, Field


class PlayerStatLine(BaseModel):
    """Validation schema for player stat lines"""

    player_id: str
    week: int = Field(ge=1, le=22)
    passing_yards: float = Field(ge=0, le=650)
    rushing_yards: float = Field(ge=0, le=350)
    receiving_yards: float = Field(ge=0, le=400)
    passing_tds: int = Field(ge=0, le=10)
    rushing_tds: int = Field(ge=0, le=6)
    receiving_tds: int = Field(ge=0, le=5)
    receptions: int = Field(ge=0, le=25)
    targets: int = Field(ge=0, le=30)
    rush_attempts: int = Field(ge=0, le=50)
    pass_attempts: int = Field(ge=0, le=80)

    @validator('targets')
    def targets_gte_receptions(cls, v, values):
        """Ensure targets >= receptions"""
        if 'receptions' in values and v < values['receptions']:
            raise ValueError("Targets cannot be less than receptions")
        return v


def validate_player_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate player statistics and return cleaned data

    Args:
        stats: Raw player statistics dictionary

    Returns:
        Validated statistics dictionary

    Raises:
        ValueError: If validation fails
    """
    try:
        validated = PlayerStatLine(**stats)
        return validated.dict()
    except Exception as e:
        raise ValueError(f"Invalid player stats: {e}")


def sanitize_stat_value(value: Any, min_val: float = 0.0, max_val: float = 1000.0) -> float:
    """
    Sanitize a stat value to reasonable bounds

    Args:
        value: Raw stat value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Sanitized float value
    """
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(min_val, min(float_val, max_val))
